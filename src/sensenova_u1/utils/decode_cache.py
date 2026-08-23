"""Reusable CUDA-graph text decode cache for native U1.5 inference."""

from __future__ import annotations

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
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = self._forward()
        self.graph = graph
        self.logits = outputs.logits[:, -1, :]

    def replay(self, token_ids: Tensor, *, t_index: int) -> Tensor:
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
    "CudaGraphDecodeWorkspace",
    "PreallocatedDecodeCache",
    "PreallocatedDecodeLayer",
    "reserve_cuda_graph_decode",
]
