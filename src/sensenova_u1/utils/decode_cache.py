"""Reusable CUDA-graph text decode cache for native U1.5 inference."""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import torch
from torch import Tensor
from transformers.cache_utils import Cache, CacheLayerMixin


class PreallocatedDecodeLayer(CacheLayerMixin):
    """A growable Transformers cache layer with fixed backing addresses."""

    is_compileable = False

    def __init__(
        self,
        keys: Tensor,
        values: Tensor,
        *,
        capacity: int,
        flash_decode_seqlens: Tensor,
    ) -> None:
        super().__init__()
        if keys.shape != values.shape or keys.ndim != 4:
            raise ValueError("decode KV layers must be matching rank-4 tensors")
        length = int(keys.shape[-2])
        if capacity < length:
            raise ValueError("decode cache capacity is smaller than its prefix")
        self.max_batch_size = int(keys.shape[0])
        self.num_heads = int(keys.shape[1])
        self.head_dim = int(keys.shape[-1])
        self.max_cache_len = int(capacity)
        self.dtype = keys.dtype
        self.device = keys.device
        self._length = length
        self.flash_decode_seqlens = flash_decode_seqlens
        self.flash_decode_k_cache = torch.empty(
            (keys.shape[0], capacity, keys.shape[1], keys.shape[-1]),
            dtype=keys.dtype,
            device=keys.device,
        )
        self.flash_decode_v_cache = torch.empty_like(self.flash_decode_k_cache)
        self.flash_decode_k_cache[:, :length].copy_(keys.transpose(1, 2))
        self.flash_decode_v_cache[:, :length].copy_(values.transpose(1, 2))
        self._refresh_views()
        self.is_initialized = True

    def _refresh_views(self) -> None:
        self.keys = self.flash_decode_k_cache[:, : self._length].transpose(1, 2)
        self.values = self.flash_decode_v_cache[:, : self._length].transpose(1, 2)

    def can_load(self, keys: Tensor, values: Tensor) -> bool:
        return (
            keys.shape == values.shape
            and keys.ndim == 4
            and int(keys.shape[0]) == self.max_batch_size
            and int(keys.shape[1]) == self.num_heads
            and int(keys.shape[-1]) == self.head_dim
            and int(keys.shape[-2]) <= self.max_cache_len
            and keys.dtype == self.dtype
            and values.dtype == self.dtype
            and keys.device == self.device
            and values.device == self.device
        )

    def load_prefix(self, keys: Tensor, values: Tensor) -> None:
        if not self.can_load(keys, values):
            raise ValueError("prefix does not fit the reusable decode cache")
        length = int(keys.shape[-2])
        self.flash_decode_k_cache[:, :length].copy_(keys.transpose(1, 2))
        self.flash_decode_v_cache[:, :length].copy_(values.transpose(1, 2))
        self._length = length
        self._refresh_views()

    def load_row(self, slot: int, keys: Tensor, values: Tensor) -> None:
        """Load one unpadded prefix into a persistent physical batch slot."""

        if type(slot) is not int or not 0 <= slot < self.max_batch_size:
            raise ValueError("decode cache slot is out of range")
        if (
            keys.shape != values.shape
            or keys.ndim != 4
            or int(keys.shape[0]) != 1
            or int(keys.shape[1]) != self.num_heads
            or int(keys.shape[-1]) != self.head_dim
            or keys.dtype != self.dtype
            or values.dtype != self.dtype
            or keys.device != self.device
            or values.device != self.device
        ):
            raise ValueError("decode cache row prefix is incompatible")
        length = int(keys.shape[-2])
        if length < 1 or length > self.max_cache_len:
            raise ValueError("decode cache row prefix does not fit")
        self.flash_decode_k_cache[slot, :length].copy_(
            keys[0].transpose(0, 1)
        )
        self.flash_decode_v_cache[slot, :length].copy_(
            values[0].transpose(0, 1)
        )
        self._length = max(self._length, length)
        self._refresh_views()

    def advance_length(self, added: int) -> None:
        if type(added) is not int or added < 1:
            raise ValueError("decode must append a positive token count")
        end = self._length + added
        if end > self.max_cache_len:
            raise RuntimeError(
                f"decode cache exhausted at {end}/{self.max_cache_len} tokens"
            )
        self._length = end
        self._refresh_views()

    def grow_capacity(self, capacity: int) -> None:
        """Grow the backing tensors without changing their logical contents."""

        if type(capacity) is not int or capacity <= self.max_cache_len:
            raise ValueError("new decode cache capacity must be larger")
        old_capacity = self.max_cache_len
        old_keys = self.flash_decode_k_cache
        old_values = self.flash_decode_v_cache
        new_keys = torch.empty(
            (self.max_batch_size, capacity, self.num_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        new_keys[:, :old_capacity].copy_(old_keys)
        # Drop the public view before replacing its backing storage.  This
        # lets the allocator reuse the old K block while reserving V.
        self.keys = None
        self.flash_decode_k_cache = new_keys
        del old_keys
        new_values = torch.empty_like(new_keys)
        new_values[:, :old_capacity].copy_(old_values)
        self.values = None
        self.flash_decode_v_cache = new_values
        del old_values
        self.max_cache_len = capacity
        self._refresh_views()

    def compact_rows(
        self,
        rows: Tensor,
        *,
        capacity: int,
        length: int,
    ) -> None:
        """Keep selected physical rows and release storage for finished rows."""

        if rows.ndim != 1 or rows.dtype != torch.long or not rows.numel():
            raise ValueError("decode cache compaction rows must be non-empty long")
        if rows.device != self.device:
            raise ValueError("decode cache compaction rows are on the wrong device")
        if length < 0 or capacity < length:
            raise ValueError("decode cache compaction capacity is too small")
        row_values = rows.detach().cpu().tolist()
        old_keys = self.flash_decode_k_cache
        old_values = self.flash_decode_v_cache
        new_shape = (len(row_values), capacity, self.num_heads, self.head_dim)
        new_keys = torch.empty(new_shape, dtype=self.dtype, device=self.device)
        for destination, source in enumerate(row_values):
            new_keys[destination, :length].copy_(old_keys[source, :length])
        self.keys = None
        self.flash_decode_k_cache = new_keys
        del old_keys
        new_values = torch.empty_like(new_keys)
        for destination, source in enumerate(row_values):
            new_values[destination, :length].copy_(old_values[source, :length])
        self.values = None
        self.flash_decode_v_cache = new_values
        del old_values
        self.max_batch_size = len(row_values)
        self.max_cache_len = capacity
        self._length = length
        self._refresh_views()

    def lazy_initialization(self, key_states: Tensor) -> None:
        raise RuntimeError("preallocated decode layers initialize eagerly")

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor]:
        del cache_kwargs
        if key_states.shape != value_states.shape or key_states.ndim != 4:
            raise ValueError("decode KV updates must be matching rank-4 tensors")
        added = int(key_states.shape[-2])
        end = self._length + added
        if end > self.max_cache_len:
            raise RuntimeError(
                f"decode cache exhausted at {end}/{self.max_cache_len} tokens"
            )
        self.flash_decode_k_cache[:, self._length : end].copy_(
            key_states.transpose(1, 2)
        )
        self.flash_decode_v_cache[:, self._length : end].copy_(
            value_states.transpose(1, 2)
        )
        self._length = end
        self._refresh_views()
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: Tensor) -> tuple[int, int]:
        return self._length + int(cache_position.shape[0]), 0

    def get_seq_length(self) -> int:
        return self._length

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def reset(self) -> None:
        self._length = 0
        self._refresh_views()


class PreallocatedDecodeCache(Cache):
    """Fixed-address cache shared by FlashAttention and image transitions."""

    @classmethod
    def from_cache(cls, cache: Cache, *, capacity: int) -> PreallocatedDecodeCache:
        if not cache.layers:
            raise ValueError("cannot reserve an empty decode cache")
        prefix = int(cache.get_seq_length())
        first = cache.layers[0]
        seqlens = torch.tensor(
            [prefix], dtype=torch.int32, device=first.keys.device
        )
        layers = [
            PreallocatedDecodeLayer(
                layer.keys,
                layer.values,
                capacity=capacity,
                flash_decode_seqlens=seqlens,
            )
            for layer in cache.layers
        ]
        reserved = cls(layers=layers)
        reserved.flash_decode_seqlens = seqlens
        return reserved

    def can_load(self, cache: Cache, *, required_capacity: int) -> bool:
        return (
            required_capacity <= self.max_cache_len
            and len(cache.layers) == len(self.layers)
            and all(
                target.can_load(source.keys, source.values)
                for target, source in zip(self.layers, cache.layers, strict=True)
            )
        )

    def load_prefix(self, cache: Cache, *, required_capacity: int) -> None:
        if not self.can_load(cache, required_capacity=required_capacity):
            raise ValueError("prefix does not fit the reusable decode cache")
        for target, source in zip(self.layers, cache.layers, strict=True):
            target.load_prefix(source.keys, source.values)
        self.sync_flash_decode_length()

    def commit_flash_decode(self, added: int) -> None:
        for layer in self.layers:
            layer.advance_length(added)
        self.flash_decode_seqlens.add_(added)

    def sync_flash_decode_length(self) -> None:
        length = int(self.layers[0].get_seq_length())
        if any(layer.get_seq_length() != length for layer in self.layers):
            raise RuntimeError("decode KV cache layers have inconsistent lengths")
        self.flash_decode_seqlens.fill_(length)

    def reset(self) -> None:
        super().reset()
        self.flash_decode_seqlens.zero_()


class BatchedFlashDecodeCache(Cache):
    """Preallocated dense KV storage with one logical length per batch row.

    Multimodal prefills are left padded so they can share one eager prefill
    call.  FlashAttention KV decode cannot consume those dummy prefix slots,
    therefore ``from_cache`` compacts every row into the beginning of its own
    fixed-capacity cache and records the real lengths in ``int32``.  Decode can
    then select only unfinished rows via ``cache_batch_idx`` without copying
    their KV tensors or running finished rows through the model.
    """

    @classmethod
    def from_cache(
        cls,
        cache: Cache,
        *,
        key_valid: Tensor,
        capacity: int,
        max_capacity: int | None = None,
    ) -> BatchedFlashDecodeCache:
        if not cache.layers:
            raise ValueError("cannot reserve an empty batched decode cache")
        if key_valid.ndim != 2 or key_valid.dtype != torch.bool:
            raise ValueError("batched decode key_valid must be a rank-2 bool tensor")
        batch_size, padded_length = key_valid.shape
        if capacity < padded_length:
            raise ValueError("batched decode capacity is smaller than its prefix")
        first = cache.layers[0]
        if first.keys is None or int(first.keys.shape[0]) != batch_size:
            raise ValueError("batched decode cache and key_valid batch sizes disagree")
        if int(first.keys.shape[-2]) != padded_length:
            raise ValueError("batched decode cache and key_valid lengths disagree")

        seqlens = key_valid.sum(dim=1, dtype=torch.int32).contiguous()
        hard_capacity = capacity if max_capacity is None else max_capacity
        if hard_capacity < capacity:
            raise ValueError("batched decode max capacity is smaller than capacity")
        layers = []
        for source in cache.layers:
            keys = source.keys
            values = source.values
            if (
                keys is None
                or values is None
                or keys.shape != values.shape
                or int(keys.shape[0]) != batch_size
                or int(keys.shape[-2]) != padded_length
            ):
                raise ValueError("batched decode prefix layers are inconsistent")
            target = PreallocatedDecodeLayer(
                keys,
                values,
                capacity=capacity,
                flash_decode_seqlens=seqlens,
            )
            for row in range(batch_size):
                positions = torch.nonzero(key_valid[row], as_tuple=False).flatten()
                length = int(positions.numel())
                target.flash_decode_k_cache[row, :length].copy_(
                    keys[row].index_select(1, positions).transpose(0, 1)
                )
                target.flash_decode_v_cache[row, :length].copy_(
                    values[row].index_select(1, positions).transpose(0, 1)
                )
            layers.append(target)
            # The source DynamicCache will be discarded after conversion.
            # Releasing converted layers keeps peak memory close to the new
            # cache instead of retaining two complete prefix caches.
            source.keys = None
            source.values = None

        reserved = cls(layers=layers)
        reserved.flash_decode_seqlens = seqlens
        reserved._physical_length = int(seqlens.max().item())
        reserved._max_capacity = hard_capacity
        reserved._logical_batch_size = batch_size
        reserved._row_slots = torch.arange(
            batch_size, device=seqlens.device, dtype=torch.long
        )
        reserved._active_indices = None
        reserved._next_physical_length = None
        return reserved

    def _compact_active_rows(
        self,
        indices: Tensor,
        *,
        capacity: int,
        length: int,
    ) -> None:
        physical_rows = self._row_slots.index_select(0, indices)
        if bool(physical_rows.lt(0).any().item()):
            raise RuntimeError("cannot reactivate a released decode cache row")
        for layer in self.layers:
            layer.compact_rows(
                physical_rows,
                capacity=capacity,
                length=length,
            )
        row_slots = torch.full_like(self._row_slots, -1)
        row_slots.index_copy_(
            0,
            indices,
            torch.arange(indices.numel(), device=indices.device, dtype=torch.long),
        )
        self._row_slots = row_slots
        self._physical_length = length

    def ensure_capacity(self, indices: Tensor, added: int = 1) -> None:
        """Reserve enough room for the selected rows' next KV update."""

        if type(added) is not int or added < 1:
            raise ValueError("batched Flash decode growth must be positive")
        needed = int(
            self.flash_decode_seqlens.index_select(0, indices).max().item()
        ) + added
        active_length = needed - added
        should_compact = (
            int(indices.numel()) * 2 <= self.layers[0].max_batch_size
        )
        if should_compact:
            next_physical_length = needed
        else:
            next_physical_length = max(self._physical_length, needed)
        if next_physical_length > self._max_capacity:
            raise RuntimeError(
                f"batched Flash decode cache exhausted at {next_physical_length}/"
                f"{self._max_capacity} tokens"
            )
        new_capacity = self.max_cache_len
        if next_physical_length > new_capacity:
            new_capacity = min(
                self._max_capacity,
                max(next_physical_length, new_capacity + 2048),
            )
        if should_compact:
            self._compact_active_rows(
                indices,
                capacity=new_capacity,
                length=active_length,
            )
        elif new_capacity > self.max_cache_len:
            for layer in self.layers:
                layer.grow_capacity(new_capacity)
        self._next_physical_length = next_physical_length

    def activate(self, indices: Tensor) -> None:
        if (
            not isinstance(indices, Tensor)
            or indices.ndim != 1
            or indices.dtype != torch.long
            or not indices.numel()
        ):
            raise ValueError("batched Flash decode indices must be non-empty long")
        if self._active_indices is not None:
            raise RuntimeError("batched Flash decode cache is already active")
        if indices.device != self.flash_decode_seqlens.device:
            raise ValueError("batched Flash decode indices are on the wrong device")
        self.ensure_capacity(indices)
        cache_batch_idx = self._row_slots.index_select(0, indices).to(
            dtype=torch.int32
        )
        selected_lengths = self.flash_decode_seqlens.index_select(
            0, indices
        ).contiguous()
        self._active_indices = indices
        for layer in self.layers:
            layer.flash_decode_seqlens = selected_lengths
            layer.flash_decode_cache_batch_idx = cache_batch_idx

    def commit_active(self) -> None:
        indices = self._active_indices
        if indices is None:
            raise RuntimeError("batched Flash decode cache has no active rows")
        self.flash_decode_seqlens.index_add_(
            0,
            indices,
            torch.ones_like(indices, dtype=torch.int32),
        )
        if self._next_physical_length is None:
            raise RuntimeError("batched Flash decode capacity was not prepared")
        self._physical_length = self._next_physical_length
        for layer in self.layers:
            layer._length = self._physical_length
            layer._refresh_views()
            layer.flash_decode_seqlens = self.flash_decode_seqlens
            delattr(layer, "flash_decode_cache_batch_idx")
        self._active_indices = None
        self._next_physical_length = None

    def cancel_active(self) -> None:
        if self._active_indices is None:
            return
        for layer in self.layers:
            layer.flash_decode_seqlens = self.flash_decode_seqlens
            if hasattr(layer, "flash_decode_cache_batch_idx"):
                delattr(layer, "flash_decode_cache_batch_idx")
        self._active_indices = None
        self._next_physical_length = None

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return int(self._physical_length)

    def reset(self) -> None:
        self.cancel_active()
        super().reset()
        self.flash_decode_seqlens.zero_()
        self._physical_length = 0
        self._next_physical_length = None


class PagedDecodeLayer(CacheLayerMixin):
    """One model layer backed by FlashAttention's paged KV layout."""

    is_compileable = False

    def __init__(
        self,
        prototype: Tensor,
        *,
        num_blocks: int,
        page_size: int,
        flash_decode_seqlens: Tensor,
        flash_decode_block_table: Tensor,
    ) -> None:
        super().__init__()
        if prototype.ndim != 4 or int(prototype.shape[0]) != 1:
            raise ValueError("paged decode prototype must be a one-row KV tensor")
        self.num_heads = int(prototype.shape[1])
        self.head_dim = int(prototype.shape[-1])
        self.page_size = page_size
        self.max_cache_len = int(flash_decode_block_table.shape[1]) * page_size
        shape = (num_blocks, page_size, self.num_heads, self.head_dim)
        self.flash_decode_k_cache = prototype.new_empty(shape)
        self.flash_decode_v_cache = prototype.new_empty(shape)
        self.flash_decode_seqlens = flash_decode_seqlens
        self.flash_decode_block_table = flash_decode_block_table
        self.keys = None
        self.values = None
        self.is_initialized = True

    def load_prefix(
        self,
        block_ids: list[int],
        keys: Tensor,
        values: Tensor,
    ) -> None:
        if (
            keys.shape != values.shape
            or keys.ndim != 4
            or int(keys.shape[0]) != 1
            or int(keys.shape[1]) != self.num_heads
            or int(keys.shape[-1]) != self.head_dim
            or keys.dtype != self.flash_decode_k_cache.dtype
            or values.dtype != self.flash_decode_v_cache.dtype
            or keys.device != self.flash_decode_k_cache.device
            or values.device != self.flash_decode_v_cache.device
        ):
            raise ValueError("paged decode prefix is incompatible")
        length = int(keys.shape[-2])
        if len(block_ids) != math.ceil(length / self.page_size):
            raise ValueError("paged decode prefix block count is inconsistent")
        keys_flash = keys[0].transpose(0, 1)
        values_flash = values[0].transpose(0, 1)
        for page, block in enumerate(block_ids):
            start = page * self.page_size
            end = min(length, start + self.page_size)
            self.flash_decode_k_cache[block, : end - start].copy_(
                keys_flash[start:end]
            )
            self.flash_decode_v_cache[block, : end - start].copy_(
                values_flash[start:end]
            )

    def lazy_initialization(self, key_states: Tensor) -> None:
        del key_states
        raise RuntimeError("paged decode layers initialize eagerly")

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor]:
        del key_states, value_states, cache_kwargs
        raise RuntimeError("paged decode updates require flash_attn_with_kvcache")

    def get_mask_sizes(self, cache_position: Tensor) -> tuple[int, int]:
        return self.get_seq_length() + int(cache_position.shape[0]), 0

    def get_seq_length(self) -> int:
        return int(self.flash_decode_seqlens.max().item())

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def reset(self) -> None:
        self.flash_decode_seqlens.zero_()


class ContinuousFlashDecodeCache(Cache):
    """Reusable paged KV pool for native continuous text batching."""

    @classmethod
    def from_prefix(
        cls,
        cache: Cache,
        *,
        max_batch_size: int,
        max_capacity: int,
        max_kv_tokens: int,
        page_size: int = 256,
        slot: int = 0,
    ) -> ContinuousFlashDecodeCache:
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("continuous decode max_batch_size must be positive")
        if type(max_capacity) is not int or max_capacity < 2:
            raise ValueError("continuous decode max_capacity must exceed one")
        if type(page_size) is not int or page_size < 256 or page_size % 256:
            raise ValueError("continuous decode page_size must be a multiple of 256")
        if type(max_kv_tokens) is not int or max_kv_tokens < page_size:
            raise ValueError("continuous decode max_kv_tokens is too small")
        if not cache.layers:
            raise ValueError("cannot initialize continuous decode from an empty cache")
        first = cache.layers[0]
        if first.keys is None:
            raise ValueError("continuous decode prefix has no keys")
        num_blocks = max_kv_tokens // page_size
        prefix = int(cache.get_seq_length())
        if prefix < 1 or prefix >= max_capacity:
            raise ValueError("continuous decode prefix length is out of range")
        if math.ceil(prefix / page_size) > num_blocks:
            raise ValueError("continuous decode KV pool cannot hold its first prefix")
        max_blocks_per_sequence = math.ceil(max_capacity / page_size)
        device = first.keys.device
        seqlens = torch.zeros(max_batch_size, dtype=torch.int32, device=device)
        block_table = torch.full(
            (max_batch_size, max_blocks_per_sequence),
            -1,
            dtype=torch.int32,
            device=device,
        )
        layers = []
        for source in cache.layers:
            if source.keys is None or source.values is None:
                raise ValueError("continuous decode prefix layers are incomplete")
            layers.append(
                PagedDecodeLayer(
                    source.keys,
                    num_blocks=num_blocks,
                    page_size=page_size,
                    flash_decode_seqlens=seqlens,
                    flash_decode_block_table=block_table,
                )
            )
        result = cls(layers=layers)
        result.flash_decode_seqlens = seqlens
        result.flash_decode_block_table = block_table
        result.page_size = page_size
        result._max_capacity = max_capacity
        result._slot_blocks = [[] for _ in range(max_batch_size)]
        result._free_blocks = deque(range(num_blocks))
        result._active_slots = None
        result._active_token_count = 0
        result.load_prefix(slot, cache)
        return result

    @property
    def max_batch_size(self) -> int:
        return int(self.flash_decode_seqlens.numel())

    @property
    def free_kv_tokens(self) -> int:
        return len(self._free_blocks) * self.page_size

    def can_admit(self, prefix_tokens: int) -> bool:
        if (
            type(prefix_tokens) is not int
            or prefix_tokens < 1
            or prefix_tokens >= self._max_capacity
        ):
            return False
        return math.ceil(prefix_tokens / self.page_size) <= len(self._free_blocks)

    def _allocate_blocks(self, slot: int, required_length: int) -> None:
        if required_length > self._max_capacity:
            raise RuntimeError(
                "continuous paged KV sequence exceeds max capacity: "
                f"{required_length}>{self._max_capacity}"
            )
        required = math.ceil(required_length / self.page_size)
        current = len(self._slot_blocks[slot])
        missing = required - current
        if missing <= 0:
            return
        if missing > len(self._free_blocks):
            raise RuntimeError(
                "continuous paged KV pool exhausted: "
                f"need {missing} blocks, have {len(self._free_blocks)}"
            )
        for _ in range(missing):
            block = self._free_blocks.popleft()
            page = len(self._slot_blocks[slot])
            self._slot_blocks[slot].append(block)
            self.flash_decode_block_table[slot, page] = block

    def load_prefix(self, slot: int, cache: Cache) -> None:
        if self._active_slots is not None:
            raise RuntimeError("cannot admit a prefix during a decode forward")
        if type(slot) is not int or not 0 <= slot < self.max_batch_size:
            raise ValueError("continuous decode slot is out of range")
        if self._slot_blocks[slot]:
            raise RuntimeError("continuous decode slot is already occupied")
        if len(cache.layers) != len(self.layers):
            raise ValueError("continuous decode prefix layer count changed")
        prefix = int(cache.get_seq_length())
        if prefix < 1 or prefix >= self._max_capacity:
            raise ValueError("continuous decode prefix length is out of range")
        self._allocate_blocks(slot, prefix)
        block_ids = self._slot_blocks[slot]
        for target, source in zip(self.layers, cache.layers, strict=True):
            if source.keys is None or source.values is None:
                raise ValueError("continuous decode prefix layer is empty")
            target.load_prefix(block_ids, source.keys, source.values)
        self.flash_decode_seqlens[slot] = prefix

    def release(self, slot: int) -> None:
        if self._active_slots is not None:
            raise RuntimeError("cannot release a slot during a decode forward")
        if type(slot) is not int or not 0 <= slot < self.max_batch_size:
            raise ValueError("continuous decode slot is out of range")
        blocks = self._slot_blocks[slot]
        for block in blocks:
            self._free_blocks.append(block)
        self.flash_decode_block_table[slot].fill_(-1)
        blocks.clear()
        self.flash_decode_seqlens[slot] = 0

    def can_reserve(self, slots: Tensor, token_count: int) -> bool:
        """Return whether ``slots`` can append a same-length token block.

        The check accounts for pages already owned by each slot.  It does not
        mutate the free list, so schedulers can shrink an image batch before
        activating it instead of discovering pool pressure inside a model
        forward.
        """

        if (
            not isinstance(slots, Tensor)
            or slots.ndim != 1
            or slots.dtype != torch.long
            or not slots.numel()
            or slots.device != self.flash_decode_seqlens.device
        ):
            return False
        if type(token_count) is not int or token_count < 1:
            return False
        if int(torch.unique(slots).numel()) != int(slots.numel()):
            return False
        lengths = self.flash_decode_seqlens.index_select(0, slots)
        if bool(lengths.le(0).any().item()):
            return False
        slot_values = slots.detach().cpu().tolist()
        length_values = lengths.detach().cpu().tolist()
        missing = 0
        for slot, length in zip(slot_values, length_values, strict=True):
            required_length = int(length) + token_count
            if required_length > self._max_capacity:
                return False
            missing += max(
                0,
                math.ceil(required_length / self.page_size)
                - len(self._slot_blocks[int(slot)]),
            )
        return missing <= len(self._free_blocks)

    def activate(self, slots: Tensor, *, token_count: int = 1) -> None:
        if (
            not isinstance(slots, Tensor)
            or slots.ndim != 1
            or slots.dtype != torch.long
            or not slots.numel()
        ):
            raise ValueError("continuous decode slots must be non-empty long")
        if self._active_slots is not None:
            raise RuntimeError("continuous decode cache is already active")
        if slots.device != self.flash_decode_seqlens.device:
            raise ValueError("continuous decode slots are on the wrong device")
        if int(torch.unique(slots).numel()) != int(slots.numel()):
            raise ValueError("continuous decode slots must be unique")
        if type(token_count) is not int or token_count < 1:
            raise ValueError("continuous decode token_count must be positive")
        selected = self.flash_decode_seqlens.index_select(0, slots)
        if bool(selected.le(0).any().item()):
            raise RuntimeError("continuous decode selected an empty slot")
        lengths = selected.detach().cpu().tolist()
        slot_values = slots.detach().cpu().tolist()
        needed = sum(
            max(
                0,
                math.ceil((int(length) + token_count) / self.page_size)
                - len(self._slot_blocks[int(slot)]),
            )
            for slot, length in zip(slot_values, lengths, strict=True)
        )
        if needed > len(self._free_blocks):
            raise RuntimeError("continuous paged KV pool has no decode page")
        for slot, length in zip(slot_values, lengths, strict=True):
            self._allocate_blocks(int(slot), int(length) + token_count)
        selected_table = self.flash_decode_block_table.index_select(
            0, slots
        ).contiguous()
        self._active_slots = slots
        self._active_token_count = token_count
        for layer in self.layers:
            layer.flash_decode_seqlens = selected.contiguous()
            layer.flash_decode_block_table = selected_table

    def commit_active(self) -> None:
        slots = self._active_slots
        if slots is None:
            raise RuntimeError("continuous decode cache has no active forward")
        self.flash_decode_seqlens.index_add_(
            0,
            slots,
            torch.full_like(
                slots, self._active_token_count, dtype=torch.int32
            ),
        )
        for layer in self.layers:
            layer.flash_decode_seqlens = self.flash_decode_seqlens
            layer.flash_decode_block_table = self.flash_decode_block_table
        self._active_slots = None
        self._active_token_count = 0

    def cancel_active(self) -> None:
        if self._active_slots is None:
            return
        for layer in self.layers:
            layer.flash_decode_seqlens = self.flash_decode_seqlens
            layer.flash_decode_block_table = self.flash_decode_block_table
        self._active_slots = None
        self._active_token_count = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return int(self.flash_decode_seqlens.max().item())

    def reset(self) -> None:
        self.cancel_active()
        for slot in range(self.max_batch_size):
            self.release(slot)


class CudaGraphDecodeWorkspace:
    """One reusable single-token CUDA graph and its fixed-address inputs."""

    def __init__(self, *, model: Any, cache: PreallocatedDecodeCache) -> None:
        self.model = model
        self.cache = cache
        self.device = cache.layers[0].keys.device
        self.graph: torch.cuda.CUDAGraph | None = None
        self.logits: Tensor | None = None
        self.token = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.indexes = torch.zeros((3, 1), dtype=torch.long, device=self.device)
        self.cache_position = torch.zeros(1, dtype=torch.long, device=self.device)

    @property
    def capacity(self) -> int:
        return int(self.cache.max_cache_len)

    def load_prefix(self, cache: Cache, *, required_capacity: int) -> None:
        self.cache.load_prefix(cache, required_capacity=required_capacity)

    def _forward(self):
        return self.model.language_model(
            input_ids=self.token,
            indexes=self.indexes,
            attention_mask={"full_attention": None},
            past_key_values=self.cache,
            use_cache=True,
            cache_position=self.cache_position,
        )

    def _capture(self, *, t_index: int) -> None:
        with torch.cuda.device(self.device):
            self.indexes[0, 0] = t_index + 1
            self.cache_position[0] = self.cache.get_seq_length()
            current = torch.cuda.current_stream(self.device)
            warmup = torch.cuda.Stream(device=self.device)
            warmup.wait_stream(current)
            with torch.cuda.stream(warmup):
                for _ in range(3):
                    self._forward()
            current.wait_stream(warmup)
            torch.cuda.synchronize(self.device)
            capture_stream = torch.cuda.Stream(device=self.device)
            capture_stream.wait_stream(current)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                outputs = self._forward()
            current.wait_stream(capture_stream)
            torch.cuda.synchronize(self.device)
        self.graph = graph
        self.logits = outputs.logits[:, -1, :]

    def replay(self, token_ids: Tensor, *, t_index: int) -> Tensor:
        with torch.cuda.device(self.device):
            if self.graph is None:
                self._capture(t_index=t_index)
            self.token.copy_(token_ids.reshape(1, 1))
            self.indexes.zero_()
            self.indexes[0, 0] = t_index + 1
            self.cache_position[0] = self.cache.get_seq_length()
            self.graph.replay()
            self.cache.commit_flash_decode(1)
        if self.logits is None:
            raise RuntimeError("CUDA graph produced no logits buffer")
        return self.logits


def reserve_cuda_graph_decode(
    owner: Any,
    *,
    model: Any,
    cache: Cache,
    capacity: int,
) -> CudaGraphDecodeWorkspace:
    """Reuse or create ``owner``'s native CUDA decode workspace."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA graph decode requires CUDA")
    prefix = int(cache.get_seq_length())
    if type(capacity) is not int or capacity <= prefix:
        raise ValueError("decode capacity must exceed the prompt prefix")
    context_limit = int(model.language_model.config.max_position_embeddings)
    if capacity > context_limit:
        raise ValueError(
            f"decode cache needs {capacity} tokens, context is {context_limit}"
        )
    workspace = getattr(owner, "_cuda_graph_decode_workspace", None)
    if workspace is None or not workspace.cache.can_load(
        cache, required_capacity=capacity
    ):
        workspace = CudaGraphDecodeWorkspace(
            model=model,
            cache=PreallocatedDecodeCache.from_cache(cache, capacity=capacity),
        )
        setattr(owner, "_cuda_graph_decode_workspace", workspace)
    else:
        workspace.load_prefix(cache, required_capacity=capacity)
    prepare_rotary = getattr(
        model.language_model.model, "prepare_rotary_inference_cache", None
    )
    if not callable(prepare_rotary):
        raise TypeError("model has no inference RoPE cache")
    device = workspace.cache.layers[0].keys.device
    dtype = workspace.cache.layers[0].keys.dtype
    prepare_rotary(
        max_temporal_position=min(context_limit, capacity + 1),
        max_spatial_position=int(
            model.language_model.config.max_position_embeddings_hw
        ),
        device=device,
        dtype=dtype,
    )
    return workspace


__all__ = [
    "BatchedFlashDecodeCache",
    "ContinuousFlashDecodeCache",
    "CudaGraphDecodeWorkspace",
    "PreallocatedDecodeCache",
    "PreallocatedDecodeLayer",
    "reserve_cuda_graph_decode",
]
