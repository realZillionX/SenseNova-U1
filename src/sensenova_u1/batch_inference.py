"""Native dynamic batching for SenseNova-U1 text and interleaved rollouts.

The public ``NEOChatModel.batch_chat`` entry point is intentionally not used:
it is a historical stub and the model's Transformers ``generate`` path still
assumes one row.  This module exposes the lower-level operation needed by both
evaluation and RL rollout: one multimodal prefill per row, followed by a shared
batched KV cache with independently advancing/eos-ing rows.

``ContiguousTextBatchSession`` is the fixed-batch compatibility primitive.
``ContinuousTextBatchEngine`` adds chunked prefill and reusable paged KV slots
for TI2T. ``ContinuousInterleaveBatchEngine`` extends that pool with independent
text-ready and image-ready queues: one text step is followed by one homogeneous
image batch, and completed image rows immediately rejoin text decoding.  The
legacy ``batch_interleave_gen`` path remains available unchanged.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from torch import Tensor

from .models.neo_unify.conversation import get_conv_template
from .models.neo_unify.modeling_qwen3 import (
    create_block_causal_mask,
    effective_attn_backend,
)
from .models.neo_unify.utils import load_image_native
from .utils.decode_cache import (
    BatchedFlashDecodeCache,
    ContinuousFlashDecodeCache,
)


@dataclass(frozen=True)
class TextBatchRequest:
    """One prompt row for :class:`ContiguousTextBatchSession`."""

    prompt: str
    images: tuple[Any, ...] = ()
    system_message: str = ""
    assistant_prefix: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str):
            raise TypeError("SenseNova batch prompt must be a string")
        if not isinstance(self.images, tuple):
            raise TypeError("SenseNova batch images must be a tuple")
        if not isinstance(self.system_message, str):
            raise TypeError("SenseNova batch system message must be a string")
        if not isinstance(self.assistant_prefix, str):
            raise TypeError("SenseNova batch assistant prefix must be a string")


@dataclass(frozen=True)
class TextBatchResult:
    """One completed TI2T response, in request order."""

    text: str
    finish_reason: str
    generated_tokens: int


@dataclass(frozen=True)
class InterleaveBatchRequest:
    """One row accepted by :func:`batch_interleave_gen`."""

    prompt: str
    images: tuple[Any, ...] = ()
    system_message: str = ""
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str):
            raise TypeError("SenseNova interleave prompt must be a string")
        if not isinstance(self.images, tuple):
            raise TypeError("SenseNova interleave images must be a tuple")
        if not isinstance(self.system_message, str):
            raise TypeError("SenseNova interleave system message must be a string")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("SenseNova interleave seed must be a non-negative int")


@dataclass(frozen=True)
class InterleaveBatchResult:
    """One completed interleaved response, in request order."""

    text: str
    images: tuple[Tensor, ...]
    finish_reason: str
    generated_tokens: int


class _InterleaveState(Enum):
    TEXT_READY = "text_ready"
    IMAGE_READY = "image_ready"
    FINISHED = "finished"


class _ContinuousInterleavePhase(Enum):
    TEXT_READY = "text_ready"
    IMAGE_READY = "image_ready"


def _same_image_input(left: Any, right: Any) -> bool:
    if left is right:
        return True
    if isinstance(left, (str, bytes)) and isinstance(right, type(left)):
        return left == right
    from pathlib import Path

    if isinstance(left, Path) and isinstance(right, Path):
        return left == right
    return False


def _requests_share_prefix(requests: Sequence[TextBatchRequest]) -> bool:
    first = requests[0]
    for request in requests[1:]:
        if (
            request.prompt != first.prompt
            or request.system_message != first.system_message
            or request.assistant_prefix != first.assistant_prefix
            or len(request.images) != len(first.images)
            or any(
                not _same_image_input(left, right)
                for left, right in zip(first.images, request.images)
            )
        ):
            return False
    return True


def _repeat_cache_batch(past_key_values: Any, repeats: int) -> Any:
    """Expand a one-row prefix cache after computing that prefix exactly once."""

    if type(repeats) is not int or repeats < 1:
        raise ValueError("SenseNova cache repeat count must be a positive int")
    if repeats == 1:
        return past_key_values
    layers = getattr(past_key_values, "layers", None)
    if not isinstance(layers, list):
        raise TypeError("SenseNova prefix cache does not expose mutable layers")
    for layer in layers:
        for name in ("keys", "values"):
            value = getattr(layer, name, None)
            if value is not None:
                if not isinstance(value, Tensor) or value.shape[0] != 1:
                    raise ValueError(
                        f"SenseNova shared prefix cache {name} must have batch size one"
                    )
                setattr(
                    layer,
                    name,
                    value.expand(repeats, *value.shape[1:]).contiguous(),
                )
    return past_key_values


def _select_cache_batch(past_key_values: Any, indices: Tensor) -> Any:
    """Return an independent cache containing only ``indices`` batch rows."""

    if not isinstance(indices, Tensor) or indices.ndim != 1 or indices.dtype != torch.long:
        raise TypeError("SenseNova cache indices must be a one-dimensional long tensor")
    selected = copy.deepcopy(past_key_values)
    layers = getattr(selected, "layers", None)
    if not isinstance(layers, list):
        raise TypeError("SenseNova prefix cache does not expose mutable layers")
    for layer in layers:
        for name in ("keys", "values"):
            value = getattr(layer, name, None)
            if value is not None:
                setattr(layer, name, value.index_select(0, indices.to(value.device)))
        for name in (
            "flash_k_cache",
            "flash_v_cache",
            "flash_decode_k_cache",
            "flash_decode_v_cache",
            "flash_decode_seqlens",
        ):
            if hasattr(layer, name):
                delattr(layer, name)
    return selected


def _validate_prefill_mask(mask: Tensor, *, length: int) -> Tensor:
    if not isinstance(mask, Tensor) or mask.shape != (1, 1, length, length):
        raise ValueError(
            "SenseNova prefill mask must have shape "
            f"(1, 1, {length}, {length}), got {getattr(mask, 'shape', None)!r}"
        )
    return mask[0, 0]


def _left_pad_prefills(
    input_embeds: Sequence[Tensor],
    indexes: Sequence[Tensor],
    masks: Sequence[Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Merge single-row multimodal prefills without changing their geometry.

    Rows are left padded so every row's final real prompt token occupies the
    same physical cache position.  Real queries retain the upstream block
    causal mask exactly.  Padding queries attend only to their own diagonal,
    preventing all-masked softmax rows; real queries can never see padding
    keys.  The returned ``key_valid`` mask is carried into every decode step.
    """

    count = len(input_embeds)
    if count < 1 or len(indexes) != count or len(masks) != count:
        raise ValueError("SenseNova batch prefill lists must have equal non-zero length")
    lengths = []
    hidden_size = None
    device = None
    dtype = None
    for row, (embeds, row_indexes) in enumerate(zip(input_embeds, indexes)):
        if not isinstance(embeds, Tensor) or embeds.ndim != 3 or embeds.shape[0] != 1:
            raise ValueError(
                f"SenseNova prefill row {row} embeds must have shape (1, sequence, hidden)"
            )
        length = int(embeds.shape[1])
        if length < 1:
            raise ValueError(f"SenseNova prefill row {row} is empty")
        if not isinstance(row_indexes, Tensor) or row_indexes.shape != (3, length):
            raise ValueError(
                f"SenseNova prefill row {row} indexes must have shape (3, {length})"
            )
        if row_indexes.dtype != torch.long:
            raise TypeError(f"SenseNova prefill row {row} indexes must be torch.long")
        if hidden_size is None:
            hidden_size = int(embeds.shape[2])
            device = embeds.device
            dtype = embeds.dtype
        elif (
            int(embeds.shape[2]) != hidden_size
            or embeds.device != device
            or embeds.dtype != dtype
        ):
            raise ValueError("SenseNova batch prefill rows must share hidden size/device/dtype")
        lengths.append(length)

    assert hidden_size is not None and device is not None and dtype is not None
    max_length = max(lengths)
    padded_embeds = torch.zeros(
        count, max_length, hidden_size, device=device, dtype=dtype
    )
    padded_indexes = torch.zeros(
        count, 3, max_length, device=device, dtype=torch.long
    )
    attention_mask = torch.full(
        (count, 1, max_length, max_length),
        -torch.inf,
        device=device,
        dtype=torch.float32,
    )
    key_valid = torch.zeros(count, max_length, device=device, dtype=torch.bool)

    for row, (embeds, row_indexes, row_mask, length) in enumerate(
        zip(input_embeds, indexes, masks, lengths)
    ):
        start = max_length - length
        padded_embeds[row, start:] = embeds[0]
        padded_indexes[row, :, start:] = row_indexes.to(device=device)
        attention_mask[row, 0, start:, start:] = _validate_prefill_mask(
            row_mask, length=length
        ).to(device=device, dtype=torch.float32)
        key_valid[row, start:] = True
        if start:
            pad_positions = torch.arange(start, device=device)
            attention_mask[row, 0, pad_positions, pad_positions] = 0.0

    return padded_embeds, padded_indexes, attention_mask, key_valid


def _prepare_text_request(
    model: Any,
    tokenizer: Any,
    request: TextBatchRequest,
    *,
    device: torch.device,
    dtype: torch.dtype,
    image_start_token: str,
    image_context_token: str,
    image_end_token: str,
) -> tuple[Tensor, Tensor]:
    """Render one official prompt and build embeddings without a dense mask."""

    prompt = request.prompt
    images = list(request.images)
    image_count = prompt.count("<image>")
    if len(images) < image_count:
        raise ValueError("SenseNova prompt has more image placeholders than images")
    if len(images) > image_count:
        prompt = "<image>\n" * (len(images) - image_count) + prompt

    pixel_values: list[Tensor] = []
    grid_hw: list[Tensor] = []
    for image in images:
        pixels, grid = load_image_native(
            image,
            model.patch_size,
            model.downsample_ratio,
            min_pixels=512 * 512,
            max_pixels=min(
                2048 * 2048,
                (4096 * 4096) // max(1, len(images)),
            ),
            upscale=False,
        )
        pixel_values.append(pixels.to(device, dtype=dtype))
        grid_hw.append(grid.to(device))

    template = get_conv_template(model.template)
    template.system_message = request.system_message
    template.append_message(template.roles[0], prompt)
    template.append_message(template.roles[1], None)
    query = template.get_prompt() + request.assistant_prefix
    for grid in grid_hw:
        context_tokens = int(
            grid[0, 0] * grid[0, 1] * float(model.downsample_ratio) ** 2
        )
        image_span = (
            image_start_token
            + image_context_token * context_tokens
            + image_end_token
        )
        query = query.replace("<image>", image_span, 1)

    pixels_tensor = torch.cat(pixel_values) if pixel_values else None
    grid_tensor = torch.cat(grid_hw) if grid_hw else None
    return model._build_it2i_embeddings(
        tokenizer,
        query,
        pixels_tensor,
        grid_tensor,
    )


class ContiguousTextBatchSession:
    """A native U1/U1.5 contiguous-batch text inference session.

    The constructor performs one batched multimodal prefill.  Each later
    ``advance``/``commit`` call performs one batched decode step for the rows
    selected by ``accepted``.  The caller owns token selection, so the same
    session can serve greedy evaluation or stochastic RL rollout while sharing
    the exact prompt/image preprocessing and KV-cache implementation.

    With FlashAttention enabled, only accepted rows enter the model forward and
    ``cache_batch_idx`` maps them into contiguous preallocated KV storage.  The
    eager fallback keeps a static physical batch and masks rejected rows.  In
    both modes each logical row advances and stops independently.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        requests: Sequence[TextBatchRequest],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        image_start_token: str = "<img>",
        image_context_token: str = "<IMG_CONTEXT>",
        image_end_token: str = "</img>",
        allow_image_actions: bool = False,
        prefix_sharing: bool = False,
        flash_decode_tokens: int = 0,
    ) -> None:
        rows = tuple(requests)
        if not rows:
            raise ValueError("SenseNova text batch must contain at least one request")
        if not all(isinstance(row, TextBatchRequest) for row in rows):
            raise TypeError("SenseNova text batch contains an invalid request")

        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = dtype
        self.batch_size = len(rows)
        self.allow_image_actions = bool(allow_image_actions)
        self.image_start_token_id = int(
            tokenizer.convert_tokens_to_ids(image_start_token)
        )
        self.image_context_token_id = int(
            tokenizer.convert_tokens_to_ids(image_context_token)
        )
        self.image_end_token_id = int(tokenizer.convert_tokens_to_ids(image_end_token))
        model.img_context_token_id = self.image_context_token_id
        model.img_start_token_id = self.image_start_token_id

        if prefix_sharing and not _requests_share_prefix(rows):
            raise ValueError(
                "SenseNova prefix sharing requires identical prompts and image inputs"
            )

        prepared_rows = rows[:1] if prefix_sharing else rows
        row_embeds: list[Tensor] = []
        row_indexes: list[Tensor] = []
        row_masks: list[Tensor] = []
        for request in prepared_rows:
            embeds, indexes, mask = self._prepare_request(
                request,
                image_start_token=image_start_token,
                image_context_token=image_context_token,
                image_end_token=image_end_token,
            )
            row_embeds.append(embeds)
            row_indexes.append(indexes)
            row_masks.append(mask)

        inputs_embeds, indexes, attention_mask, key_valid = _left_pad_prefills(
            row_embeds, row_indexes, row_masks
        )
        outputs = model.language_model(
            inputs_embeds=inputs_embeds,
            indexes=indexes,
            attention_mask={"full_attention": attention_mask},
            use_cache=True,
        )
        if prefix_sharing:
            self.cache = _repeat_cache_batch(outputs.past_key_values, self.batch_size)
            self.key_valid = key_valid.expand(self.batch_size, -1).contiguous()
            shared_t = int(row_indexes[0][0].max().item())
            self.t_indexes = torch.full(
                (self.batch_size,), shared_t, device=self.device, dtype=torch.long
            )
            self.next_logits = outputs.logits[:, -1, :].expand(
                self.batch_size, -1
            ).contiguous()
        else:
            self.cache = outputs.past_key_values
            self.key_valid = key_valid
            self.t_indexes = torch.tensor(
                [
                    int(row_indexes[row][0].max().item())
                    for row in range(self.batch_size)
                ],
                device=self.device,
                dtype=torch.long,
            )
            self.next_logits = outputs.logits[:, -1, :]

        self._flash_decode = False
        if flash_decode_tokens:
            if type(flash_decode_tokens) is not int or flash_decode_tokens < 1:
                raise ValueError("SenseNova flash_decode_tokens must be positive")
            if self.device.type != "cuda" or effective_attn_backend() != "flash":
                raise RuntimeError(
                    "SenseNova batched Flash decode requires CUDA and flash backend"
                )
            self.cache = BatchedFlashDecodeCache.from_cache(
                self.cache,
                key_valid=self.key_valid,
                capacity=int(self.key_valid.shape[1]) + min(flash_decode_tokens, 2048),
                max_capacity=int(self.key_valid.shape[1]) + flash_decode_tokens,
            )
            self._flash_decode = True

    @property
    def decode_backend(self) -> str:
        """Return the active decode implementation for observability/tests."""

        return "flash_kv" if self._flash_decode else "masked_dynamic"

    def _prepare_request(
        self,
        request: TextBatchRequest,
        *,
        image_start_token: str,
        image_context_token: str,
        image_end_token: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        inputs_embeds, indexes = _prepare_text_request(
            self.model,
            self.tokenizer,
            request,
            device=self.device,
            dtype=self.dtype,
            image_start_token=image_start_token,
            image_context_token=image_context_token,
            image_end_token=image_end_token,
        )
        return inputs_embeds, indexes, create_block_causal_mask(indexes[0])

    def constrained_logits(self) -> Tensor:
        # ``Tensor.float()`` aliases an already-float32 tensor.  Always clone so
        # masking the image action cannot corrupt the cached unmodified logits.
        logits = self.next_logits.to(dtype=torch.float32, copy=True)
        if logits.ndim != 2 or logits.shape[0] != self.batch_size:
            raise RuntimeError(
                "SenseNova batch returned invalid next-token logits "
                f"{tuple(logits.shape)}"
            )
        if not self.allow_image_actions:
            logits[:, self.image_start_token_id] = torch.finfo(logits.dtype).min
        return logits

    def commit(self, token_ids: Tensor, accepted: Tensor | None = None) -> None:
        if token_ids.shape != (self.batch_size,) or token_ids.dtype != torch.long:
            raise ValueError(
                "SenseNova text batch token_ids must have shape "
                f"({self.batch_size},) and dtype torch.long"
            )
        if token_ids.device != self.device:
            raise ValueError("SenseNova text batch token_ids are on the wrong device")
        if accepted is None:
            accepted = torch.ones(
                self.batch_size, device=self.device, dtype=torch.bool
            )
        if accepted.shape != (self.batch_size,) or accepted.dtype != torch.bool:
            raise ValueError(
                "SenseNova text batch accepted mask must have shape "
                f"({self.batch_size},) and dtype torch.bool"
            )
        if accepted.device != self.device:
            raise ValueError("SenseNova text batch accepted mask is on the wrong device")

        if self._flash_decode:
            active_indices = torch.nonzero(accepted, as_tuple=False).flatten()
            if not active_indices.numel():
                return
            next_t = self.t_indexes.index_select(0, active_indices) + 1
            zeros = torch.zeros_like(next_t)
            indexes = torch.stack((next_t, zeros, zeros), dim=1).unsqueeze(-1)
            self.cache.activate(active_indices)
            try:
                outputs = self.model.language_model(
                    input_ids=token_ids.index_select(0, active_indices).reshape(-1, 1),
                    indexes=indexes,
                    attention_mask={"full_attention": None},
                    past_key_values=self.cache,
                    use_cache=True,
                )
            except Exception:
                self.cache.cancel_active()
                raise
            self.cache.commit_active()
            self.t_indexes.index_copy_(0, active_indices, next_t)
            self.next_logits.index_copy_(
                0, active_indices, outputs.logits[:, -1, :]
            )
            return

        next_t = self.t_indexes + accepted.to(dtype=torch.long)
        zeros = torch.zeros_like(next_t)
        indexes = torch.stack((next_t, zeros, zeros), dim=1).unsqueeze(-1)
        next_key_valid = torch.cat((self.key_valid, accepted.unsqueeze(1)), dim=1)
        attention_mask = torch.full(
            (self.batch_size, 1, 1, next_key_valid.shape[1]),
            -torch.inf,
            device=self.device,
            dtype=torch.float32,
        )
        attention_mask[:, 0, 0, :] = torch.where(
            next_key_valid,
            torch.zeros((), device=self.device, dtype=torch.float32),
            torch.full((), -torch.inf, device=self.device, dtype=torch.float32),
        )
        outputs = self.model.language_model(
            input_ids=token_ids.reshape(self.batch_size, 1),
            indexes=indexes,
            attention_mask={"full_attention": attention_mask},
            past_key_values=self.cache,
            use_cache=True,
        )
        self.cache = outputs.past_key_values
        self.key_valid = next_key_valid
        self.t_indexes = next_t
        self.next_logits = torch.where(
            accepted.unsqueeze(1), outputs.logits[:, -1, :], self.next_logits
        )

    def advance(self, token_ids: Tensor, accepted: Tensor) -> Tensor:
        """Commit one token for each accepted row and return next-token logits."""

        self.commit(token_ids, accepted)
        return self.constrained_logits()

    def selected_cache(self, indices: Tensor) -> tuple[Any, Tensor, Tensor]:
        """Copy selected rows for a temporary image-generation batch."""

        if indices.device != self.device:
            indices = indices.to(self.device)
        return (
            _select_cache_batch(self.cache, indices),
            self.key_valid.index_select(0, indices),
            self.t_indexes.index_select(0, indices),
        )

    def append_generated_images(
        self,
        image_predictions: Tensor,
        indices: Tensor,
        *,
        image_end_token_id: int,
    ) -> None:
        """Re-encode a homogeneous generated-image batch into the text cache."""

        if indices.device != self.device:
            indices = indices.to(self.device)
        if indices.ndim != 1 or indices.dtype != torch.long or not indices.numel():
            raise ValueError("SenseNova generated-image indices must be non-empty")
        if image_predictions.ndim != 4 or image_predictions.shape[0] != indices.numel():
            raise ValueError("SenseNova generated-image batch and indices disagree")

        count, channels, height, width = image_predictions.shape
        if channels != 3:
            raise ValueError("SenseNova generated images must have three channels")
        patch = int(self.model.patch_size)
        if height % patch or width % patch:
            raise ValueError("SenseNova generated image size must divide by patch size")
        patch_h = height // patch
        patch_w = width // patch
        grid_hw = torch.tensor(
            [[patch_h, patch_w]] * count, device=self.device, dtype=torch.long
        )

        prediction = image_predictions.to(self.device, dtype=torch.bfloat16)
        raw = prediction * 0.5 + 0.5
        mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=raw.dtype, device=self.device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=raw.dtype, device=self.device
        ).view(1, 3, 1, 1)
        normalized = (raw - mean) / std
        flattened = (
            normalized.view(count, 3, patch_h, patch, patch_w, patch)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(count * patch_h * patch_w, 3 * patch**2)
        )
        features = self.model.extract_feature(flattened, grid_hw=grid_hw)
        hidden = int(features.shape[-1])
        features = features.reshape(count, -1, hidden)
        image_tokens = int(features.shape[1])

        end_ids = torch.full(
            (count, 1), int(image_end_token_id), device=self.device, dtype=torch.long
        )
        end_embed = self.model.language_model.get_input_embeddings()(end_ids)
        selected_embeds = torch.cat((features, end_embed), dim=1)
        span = image_tokens + 1
        full_embeds = torch.zeros(
            self.batch_size,
            span,
            hidden,
            device=self.device,
            dtype=selected_embeds.dtype,
        )
        full_embeds.index_copy_(0, indices, selected_embeds)

        next_indexes = torch.zeros(
            self.batch_size, 3, span, device=self.device, dtype=torch.long
        )
        merge = int(1 / float(self.model.downsample_ratio))
        token_h = patch_h // merge
        token_w = patch_w // merge
        if token_h * token_w != image_tokens:
            raise RuntimeError("SenseNova generated-image token geometry is inconsistent")
        spatial_h = (
            torch.arange(token_h, device=self.device)
            .view(token_h, 1)
            .expand(token_h, token_w)
            .reshape(-1)
        )
        spatial_w = (
            torch.arange(token_w, device=self.device)
            .view(1, token_w)
            .expand(token_h, token_w)
            .reshape(-1)
        )
        selected_t = self.t_indexes.index_select(0, indices)
        for local, row in enumerate(indices.tolist()):
            next_indexes[row, 0, :image_tokens] = selected_t[local] + 1
            next_indexes[row, 0, image_tokens] = selected_t[local] + 2
            next_indexes[row, 1, :image_tokens] = spatial_h
            next_indexes[row, 2, :image_tokens] = spatial_w

        past_len = int(self.key_valid.shape[1])
        mask = torch.full(
            (self.batch_size, 1, span, past_len + span),
            -torch.inf,
            device=self.device,
            dtype=torch.float32,
        )
        selected_set = set(indices.tolist())
        for row in range(self.batch_size):
            if row in selected_set:
                valid_prefix = self.key_valid[row]
                mask[row, 0, :, :past_len] = torch.where(
                    valid_prefix,
                    torch.zeros((), device=self.device),
                    torch.full((), -torch.inf, device=self.device),
                )
                mask[row, 0, :, past_len:] = 0.0
                mask[row, 0, :image_tokens, past_len + image_tokens] = -torch.inf
            else:
                diagonal = torch.arange(span, device=self.device)
                mask[row, 0, diagonal, past_len + diagonal] = 0.0

        outputs = self.model.language_model(
            inputs_embeds=full_embeds,
            indexes=next_indexes,
            attention_mask={"full_attention": mask},
            past_key_values=self.cache,
            use_cache=True,
        )
        self.cache = outputs.past_key_values
        ready = torch.zeros(self.batch_size, device=self.device, dtype=torch.bool)
        ready[indices] = True
        self.key_valid = torch.cat(
            (self.key_valid, ready.unsqueeze(1).expand(-1, span)), dim=1
        )
        self.t_indexes = self.t_indexes + ready.to(torch.long) * 2
        self.next_logits = torch.where(
            ready.unsqueeze(1), outputs.logits[:, -1, :], self.next_logits
        )


# Backwards-compatible name used by the first WIP callers.  New integrations
# should use ``ContiguousTextBatchSession`` so the execution contract is clear.
NativeTextBatchSession = ContiguousTextBatchSession


@dataclass(frozen=True)
class ContinuousTextBatch:
    """Decode-ready rows returned by :meth:`ContinuousTextBatchEngine.schedule`."""

    request_ids: tuple[int, ...]
    logits: Tensor


@dataclass(frozen=True)
class ContinuousImageBatch:
    """Image-ready rows selected from a continuous TI2TI engine."""

    request_ids: tuple[int, ...]


@dataclass
class _ContinuousTextState:
    request_id: int
    request: TextBatchRequest
    max_new_tokens: int
    inputs_embeds: Tensor | None = None
    indexes: Tensor | None = None
    prefill_cache: Any | None = None
    prefill_cursor: int = 0
    slot: int | None = None
    t_index: int = 0
    next_logits: Tensor | None = None
    generated: list[int] = field(default_factory=list)


@dataclass
class _ContinuousInterleaveState(_ContinuousTextState):
    interleave_request: InterleaveBatchRequest | None = None
    phase: _ContinuousInterleavePhase = _ContinuousInterleavePhase.TEXT_READY
    pending_tokens: list[int] = field(default_factory=list)
    parts: list[str] = field(default_factory=list)
    generated_images: list[Tensor] = field(default_factory=list)
    generator: torch.Generator | None = None
    image_ready_step: int | None = None


def _chunk_end_without_splitting_block(
    temporal_indexes: Tensor,
    start: int,
    chunk_size: int,
) -> int:
    """Choose a bounded prefill end without splitting a bidirectional block."""

    if temporal_indexes.ndim != 1 or temporal_indexes.dtype != torch.long:
        raise ValueError("SenseNova temporal indexes must be one-dimensional long")
    length = int(temporal_indexes.numel())
    if type(start) is not int or not 0 <= start < length:
        raise ValueError("SenseNova prefill start is out of range")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("SenseNova prefill chunk size must be positive")
    end = min(length, start + chunk_size)
    if end == length:
        return end
    block = temporal_indexes[end - 1]
    while end < length and bool(temporal_indexes[end].eq(block).item()):
        end += 1
    return end


def _chunk_block_causal_mask(
    temporal_indexes: Tensor,
    start: int,
    end: int,
) -> Tensor:
    """Build only the query-chunk rows of the official block-causal mask."""

    if temporal_indexes.ndim != 1 or temporal_indexes.dtype != torch.long:
        raise ValueError("SenseNova temporal indexes must be one-dimensional long")
    length = int(temporal_indexes.numel())
    if not 0 <= start < end <= length:
        raise ValueError("SenseNova prefill chunk range is invalid")
    query_index = temporal_indexes[start:end]
    key_index = temporal_indexes[:end]
    query_positions = torch.arange(start, end, device=temporal_indexes.device)
    key_positions = torch.arange(end, device=temporal_indexes.device)
    allowed = query_index[:, None].eq(key_index[None, :]) | key_positions[
        None, :
    ].le(query_positions[:, None])
    zeros = torch.zeros((), device=temporal_indexes.device, dtype=torch.float32)
    blocked = torch.full(
        (), -torch.inf, device=temporal_indexes.device, dtype=torch.float32
    )
    return torch.where(allowed, zeros, blocked).unsqueeze(0).unsqueeze(0)


class ContinuousTextBatchEngine:
    """Hand-written continuous batching engine for native SenseNova TI2T.

    Requests enter a waiting queue, receive block-safe chunked prefill, and are
    admitted into reusable physical KV slots.  Every decode iteration forwards
    all currently active rows together.  Finished rows release their slots, so
    later requests join without rebuilding or copying surviving rows' caches.

    ``schedule`` exposes float32 logits and ``advance`` accepts caller-selected
    tokens.  This split is intentional: evaluation can use argmax while RL can
    sample and record exact behavior log-probabilities from the same engine.
    KV storage is a fixed-size paged pool, so one long request does not grow
    every physical batch row.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        max_batch_size: int = 32,
        max_model_len: int = 16384,
        prefill_chunk_size: int = 2048,
        default_max_new_tokens: int = 2048,
        max_kv_tokens: int = 98304,
        kv_page_size: int = 256,
        prefix_sharing: bool = True,
        truncate_to_max_model_len: bool = False,
        image_start_token: str = "<img>",
        image_context_token: str = "<IMG_CONTEXT>",
        image_end_token: str = "</img>",
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("SenseNova continuous max_batch_size must be positive")
        if type(max_model_len) is not int or max_model_len < 2:
            raise ValueError("SenseNova continuous max_model_len must exceed one")
        if type(prefill_chunk_size) is not int or prefill_chunk_size < 1:
            raise ValueError("SenseNova continuous prefill_chunk_size must be positive")
        if type(default_max_new_tokens) is not int or default_max_new_tokens < 1:
            raise ValueError("SenseNova default max_new_tokens must be positive")
        if type(max_kv_tokens) is not int or max_kv_tokens < 256:
            raise ValueError("SenseNova continuous max_kv_tokens is too small")
        if type(kv_page_size) is not int or kv_page_size < 256 or kv_page_size % 256:
            raise ValueError(
                "SenseNova continuous kv_page_size must be a multiple of 256"
            )

        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if self.device.type != "cuda" or effective_attn_backend() != "flash":
            raise RuntimeError(
                "SenseNova continuous batching requires CUDA FlashAttention"
            )
        self.model = model
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.max_model_len = max_model_len
        self.prefill_chunk_size = prefill_chunk_size
        self.default_max_new_tokens = default_max_new_tokens
        self.max_kv_tokens = max_kv_tokens
        self.kv_page_size = kv_page_size
        self.prefix_sharing = bool(prefix_sharing)
        self.truncate_to_max_model_len = bool(truncate_to_max_model_len)
        self.image_start_token = image_start_token
        self.image_context_token = image_context_token
        self.image_end_token = image_end_token

        self.image_start_token_id = int(
            tokenizer.convert_tokens_to_ids(image_start_token)
        )
        model.img_context_token_id = int(
            tokenizer.convert_tokens_to_ids(image_context_token)
        )
        model.img_start_token_id = self.image_start_token_id
        template = get_conv_template(model.template)
        self.eos_token_id = int(
            tokenizer.convert_tokens_to_ids(template.sep.strip())
        )

        self._next_request_id = 0
        self._waiting: deque[_ContinuousTextState] = deque()
        self._prefilling: _ContinuousTextState | None = None
        self._active: dict[int, _ContinuousTextState] = {}
        self._completed: deque[tuple[int, TextBatchResult]] = deque()
        self._free_slots: deque[int] = deque(range(max_batch_size))
        self._cache: ContinuousFlashDecodeCache | None = None
        self._scheduled_request_ids: tuple[int, ...] | None = None

    @property
    def decode_backend(self) -> str:
        return "flash_kv_paged_continuous"

    @property
    def has_unfinished_requests(self) -> bool:
        return bool(self._waiting or self._prefilling or self._active)

    @property
    def active_size(self) -> int:
        return len(self._active)

    @property
    def waiting_size(self) -> int:
        return len(self._waiting) + int(self._prefilling is not None)

    @property
    def free_kv_tokens(self) -> int:
        """Return whole-page KV capacity still available to new rows/decode."""

        if self._cache is None:
            return self.max_kv_tokens
        return self._cache.free_kv_tokens

    def submit(
        self,
        request: TextBatchRequest,
        *,
        max_new_tokens: int | None = None,
    ) -> int:
        """Queue one request and return its stable engine-local identifier."""

        if not isinstance(request, TextBatchRequest):
            raise TypeError("SenseNova continuous request must be TextBatchRequest")
        limit = self.default_max_new_tokens if max_new_tokens is None else max_new_tokens
        if type(limit) is not int or limit < 1:
            raise ValueError("SenseNova request max_new_tokens must be positive")
        request_id = self._next_request_id
        self._next_request_id += 1
        self._waiting.append(
            _ContinuousTextState(
                request_id=request_id,
                request=request,
                max_new_tokens=limit,
            )
        )
        return request_id

    def _prepare_prefill(self, state: _ContinuousTextState) -> None:
        inputs_embeds, indexes = _prepare_text_request(
            self.model,
            self.tokenizer,
            state.request,
            device=self.device,
            dtype=self.dtype,
            image_start_token=self.image_start_token,
            image_context_token=self.image_context_token,
            image_end_token=self.image_end_token,
        )
        prefix = int(inputs_embeds.shape[1])
        if prefix + state.max_new_tokens > self.max_model_len:
            available = self.max_model_len - prefix
            if available < 1 or not self.truncate_to_max_model_len:
                raise ValueError(
                    "SenseNova request exceeds max_model_len: "
                    f"{prefix}+{state.max_new_tokens}>{self.max_model_len}"
                )
            state.max_new_tokens = available
        state.inputs_embeds = inputs_embeds
        state.indexes = indexes

    def _admit(
        self,
        state: _ContinuousTextState,
        prefix_cache: Any,
        next_logits: Tensor,
        t_index: int,
    ) -> None:
        if not self._free_slots:
            raise RuntimeError("SenseNova continuous batch has no free KV slot")
        slot = self._free_slots.popleft()
        if self._cache is None:
            self._cache = ContinuousFlashDecodeCache.from_prefix(
                prefix_cache,
                max_batch_size=self.max_batch_size,
                max_capacity=self.max_model_len,
                max_kv_tokens=self.max_kv_tokens,
                page_size=self.kv_page_size,
                slot=slot,
            )
        else:
            self._cache.load_prefix(slot, prefix_cache)
        state.slot = slot
        state.t_index = t_index
        state.next_logits = next_logits.detach().clone()
        state.inputs_embeds = None
        state.indexes = None
        state.prefill_cache = None
        self._active[state.request_id] = state

    def _admit_shared_prefixes(
        self,
        source: _ContinuousTextState,
        prefix_cache: Any,
        next_logits: Tensor,
        t_index: int,
    ) -> None:
        if not self.prefix_sharing or not self._free_slots or not self._waiting:
            return
        remaining: deque[_ContinuousTextState] = deque()
        while self._waiting:
            candidate = self._waiting.popleft()
            prefix = int(prefix_cache.get_seq_length())
            if (
                self._free_slots
                and self._cache is not None
                and self._cache.can_admit(prefix)
                and _requests_share_prefix((source.request, candidate.request))
            ):
                if prefix + candidate.max_new_tokens > self.max_model_len:
                    available = self.max_model_len - prefix
                    if available < 1 or not self.truncate_to_max_model_len:
                        raise ValueError(
                            "SenseNova shared-prefix request exceeds max_model_len"
                        )
                    candidate.max_new_tokens = available
                self._admit(candidate, prefix_cache, next_logits, t_index)
            else:
                remaining.append(candidate)
        self._waiting = remaining

    @torch.no_grad()
    def _prefill_one_chunk(self) -> None:
        if not self._free_slots:
            return
        if self._prefilling is None:
            if not self._waiting:
                return
            self._prefilling = self._waiting.popleft()
            self._prepare_prefill(self._prefilling)

        state = self._prefilling
        assert state is not None and state.inputs_embeds is not None
        assert state.indexes is not None
        prefix_length = int(state.inputs_embeds.shape[1])
        if state.prefill_cursor == prefix_length:
            assert state.prefill_cache is not None
            assert state.next_logits is not None
            if self._cache is not None and not self._cache.can_admit(prefix_length):
                return
            self._prefilling = None
            self._admit(
                state,
                state.prefill_cache,
                state.next_logits,
                state.t_index,
            )
            return
        start = state.prefill_cursor
        temporal_indexes = state.indexes[0]
        end = _chunk_end_without_splitting_block(
            temporal_indexes,
            start,
            self.prefill_chunk_size,
        )
        attention_mask = _chunk_block_causal_mask(temporal_indexes, start, end)
        outputs = self.model.language_model(
            inputs_embeds=state.inputs_embeds[:, start:end],
            indexes=state.indexes[:, start:end],
            attention_mask={"full_attention": attention_mask},
            past_key_values=state.prefill_cache,
            use_cache=True,
        )
        state.prefill_cache = outputs.past_key_values
        state.prefill_cursor = end
        if end < prefix_length:
            return

        prefix_cache = state.prefill_cache
        state.next_logits = outputs.logits[:, -1, :].detach().clone()
        state.t_index = int(temporal_indexes.max().item())
        if self._cache is not None and not self._cache.can_admit(prefix_length):
            return
        self._prefilling = None
        self._admit(state, prefix_cache, state.next_logits, state.t_index)
        self._admit_shared_prefixes(
            state,
            prefix_cache,
            state.next_logits,
            state.t_index,
        )

    @torch.no_grad()
    def schedule(self) -> ContinuousTextBatch | None:
        """Run one prefill chunk and expose all decode-ready row logits."""

        if self._scheduled_request_ids is not None:
            raise RuntimeError("SenseNova continuous batch must be advanced first")
        self._prefill_one_chunk()
        if not self._active:
            return None
        states = sorted(self._active.values(), key=lambda state: int(state.slot))
        request_ids = tuple(state.request_id for state in states)
        logits = torch.cat(
            tuple(state.next_logits for state in states if state.next_logits is not None),
            dim=0,
        ).to(dtype=torch.float32, copy=True)
        logits[:, self.image_start_token_id] = torch.finfo(logits.dtype).min
        self._scheduled_request_ids = request_ids
        return ContinuousTextBatch(request_ids=request_ids, logits=logits)

    def _finish(self, state: _ContinuousTextState, reason: str) -> None:
        if state.slot is None or self._cache is None:
            raise RuntimeError("SenseNova completed request has no KV slot")
        self._cache.release(state.slot)
        self._free_slots.append(state.slot)
        del self._active[state.request_id]
        self._completed.append(
            (
                state.request_id,
                TextBatchResult(
                    text=self.tokenizer.decode(
                        state.generated, skip_special_tokens=True
                    ),
                    finish_reason=reason,
                    generated_tokens=len(state.generated),
                ),
            )
        )

    @torch.no_grad()
    def advance(
        self,
        batch: ContinuousTextBatch,
        token_ids: Tensor,
        *,
        stop_mask: Tensor | None = None,
    ) -> None:
        """Accept caller-selected tokens and execute one active-row decode."""

        if batch.request_ids != self._scheduled_request_ids:
            raise RuntimeError("SenseNova continuous batch is stale or out of order")
        count = len(batch.request_ids)
        if token_ids.shape != (count,) or token_ids.dtype != torch.long:
            raise ValueError("SenseNova continuous token_ids must be batch-shaped long")
        if token_ids.device != self.device:
            raise ValueError("SenseNova continuous token_ids are on the wrong device")
        if stop_mask is None:
            stop_mask = torch.zeros(count, device=self.device, dtype=torch.bool)
        if stop_mask.shape != (count,) or stop_mask.dtype != torch.bool:
            raise ValueError("SenseNova continuous stop_mask must be batch-shaped bool")
        if stop_mask.device != self.device:
            raise ValueError("SenseNova continuous stop_mask is on the wrong device")

        states = [self._active[request_id] for request_id in batch.request_ids]
        token_values = token_ids.detach().cpu().tolist()
        stop_values = stop_mask.detach().cpu().tolist()
        continuing: list[tuple[_ContinuousTextState, int]] = []
        finished: list[tuple[_ContinuousTextState, str]] = []
        for state, token, forced_stop in zip(
            states, token_values, stop_values, strict=True
        ):
            if forced_stop:
                finished.append((state, "stopped"))
            elif int(token) == self.eos_token_id:
                finished.append((state, "eos"))
            elif len(state.generated) + 1 >= state.max_new_tokens:
                finished.append((state, "max_new_tokens"))
            else:
                continuing.append((state, int(token)))

        outputs = None
        if continuing:
            assert self._cache is not None
            slots = torch.tensor(
                [int(state.slot) for state, _ in continuing],
                device=self.device,
                dtype=torch.long,
            )
            tokens = torch.tensor(
                [token for _, token in continuing],
                device=self.device,
                dtype=torch.long,
            )
            next_t = torch.tensor(
                [state.t_index + 1 for state, _ in continuing],
                device=self.device,
                dtype=torch.long,
            )
            zeros = torch.zeros_like(next_t)
            indexes = torch.stack((next_t, zeros, zeros), dim=1).unsqueeze(-1)
            self._cache.activate(slots)
            try:
                outputs = self.model.language_model(
                    input_ids=tokens.reshape(-1, 1),
                    indexes=indexes,
                    attention_mask={"full_attention": None},
                    past_key_values=self._cache,
                    use_cache=True,
                )
            except Exception:
                self._cache.cancel_active()
                raise
            self._cache.commit_active()

        continuing_index = 0
        finish_by_id = {state.request_id: reason for state, reason in finished}
        for state, token in zip(states, token_values, strict=True):
            reason = finish_by_id.get(state.request_id)
            if reason == "stopped" or reason == "eos":
                continue
            state.generated.append(int(token))
            if reason is None:
                assert outputs is not None
                state.t_index += 1
                state.next_logits = outputs.logits[
                    continuing_index : continuing_index + 1, -1, :
                ].detach()
                continuing_index += 1

        for state, reason in finished:
            self._finish(state, reason)
        self._scheduled_request_ids = None

    def pop_completed(self) -> tuple[tuple[int, TextBatchResult], ...]:
        """Return completed results once, in completion order."""

        completed = tuple(self._completed)
        self._completed.clear()
        return completed


@torch.no_grad()
def continuous_batch_text_gen(
    model: Any,
    tokenizer: Any,
    requests: Sequence[TextBatchRequest],
    *,
    generation_config: Any | None = None,
    max_batch_size: int = 32,
    max_model_len: int = 16384,
    prefill_chunk_size: int = 2048,
    max_kv_tokens: int = 98304,
    kv_page_size: int = 256,
    prefix_sharing: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[TextBatchResult, ...]:
    """Greedily drain requests through :class:`ContinuousTextBatchEngine`."""

    rows = tuple(requests)
    if not rows or not all(isinstance(row, TextBatchRequest) for row in rows):
        raise ValueError("SenseNova continuous text batch must contain valid requests")
    max_new_tokens = int(
        getattr(generation_config, "max_new_tokens", None) or 8192
    )
    repetition_penalty = float(
        getattr(generation_config, "repetition_penalty", None) or 1.0
    )
    if repetition_penalty < 1.0:
        raise ValueError("SenseNova repetition_penalty must be at least 1.0")
    runtime_device = torch.device(device if device is not None else model.device)
    engine = ContinuousTextBatchEngine(
        model,
        tokenizer,
        device=runtime_device,
        dtype=dtype,
        max_batch_size=max_batch_size,
        max_model_len=max_model_len,
        prefill_chunk_size=prefill_chunk_size,
        default_max_new_tokens=max_new_tokens,
        max_kv_tokens=max_kv_tokens,
        kv_page_size=kv_page_size,
        prefix_sharing=prefix_sharing,
    )
    request_ids = [engine.submit(row) for row in rows]
    seen_tokens: dict[int, set[int]] = {request_id: set() for request_id in request_ids}
    completed: dict[int, TextBatchResult] = {}
    while engine.has_unfinished_requests:
        batch = engine.schedule()
        if batch is None:
            continue
        logits = batch.logits
        if repetition_penalty != 1.0:
            seen = torch.zeros_like(logits, dtype=torch.bool)
            for row, request_id in enumerate(batch.request_ids):
                if seen_tokens[request_id]:
                    indexes = torch.tensor(
                        sorted(seen_tokens[request_id]),
                        device=runtime_device,
                        dtype=torch.long,
                    )
                    seen[row, indexes] = True
            logits = _apply_repetition_penalty(logits, seen, repetition_penalty)
        next_tokens = torch.argmax(logits, dim=-1).to(
            device=runtime_device, dtype=torch.long
        )
        for request_id, token in zip(
            batch.request_ids, next_tokens.detach().cpu().tolist(), strict=True
        ):
            if int(token) != engine.eos_token_id:
                seen_tokens[request_id].add(int(token))
        engine.advance(batch, next_tokens)
        completed.update(engine.pop_completed())
    completed.update(engine.pop_completed())
    return tuple(completed[request_id] for request_id in request_ids)


def _image_attention_mask(key_valid: Tensor, image_tokens: int) -> Tensor:
    batch, prefix = key_valid.shape
    mask = torch.full(
        (batch, 1, image_tokens, prefix + image_tokens),
        -torch.inf,
        device=key_valid.device,
        dtype=torch.float32,
    )
    mask[:, 0, :, :prefix] = torch.where(
        key_valid[:, None, :],
        torch.zeros((), device=key_valid.device),
        torch.full((), -torch.inf, device=key_valid.device),
    )
    mask[:, 0, :, prefix:] = 0.0
    return mask


def _run_image_sde_batch(
    model: Any,
    cache: Any,
    key_valid: Tensor,
    t_indexes: Tensor,
    generators: Sequence[torch.Generator],
    *,
    image_size: tuple[int, int],
    num_steps: int,
    enable_timestep_shift: bool,
    timestep_shift: float,
    cfg_interval: tuple[float, float],
) -> Tensor:
    """Run one homogeneous no-CFG image SDE batch."""

    count = len(generators)
    if count < 1 or key_valid.shape[0] != count or t_indexes.shape != (count,):
        raise ValueError("SenseNova image-ready rows and cache metadata disagree")
    width, height = image_size
    merge = int(1 / float(model.downsample_ratio))
    patch = int(model.patch_size)
    divisor = patch * merge
    if width <= 0 or height <= 0 or width % divisor or height % divisor:
        raise ValueError("SenseNova image size must divide by patch_size * merge_size")
    token_h = height // divisor
    token_w = width // divisor
    image_tokens = token_h * token_w
    device = key_valid.device

    indexes = torch.stack(
        tuple(
            model._build_t2i_image_indexes(
                token_h, token_w, int(t_index.item()) + 1, device=device
            )
            for t_index in t_indexes
        )
    )
    grid_h = height // patch
    grid_w = width // patch
    grid_hw = torch.tensor(
        [[grid_h, grid_w]] * count, device=device, dtype=torch.long
    )

    noise_scale = float(model.noise_scale)
    if model.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
        base = float(model.noise_scale_base_image_seq_len)
        noise_scale = math.sqrt((grid_h * grid_w) / (merge**2) / base) * noise_scale
        if model.noise_scale_mode == "dynamic_sqrt":
            noise_scale = math.sqrt(noise_scale)
    noise_scale = min(noise_scale, float(model.noise_scale_max_value))
    dtype = next(model.parameters()).dtype
    prediction = torch.cat(
        tuple(
            noise_scale
            * torch.randn(
                (1, 3, height, width),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            for generator in generators
        ),
        dim=0,
    )

    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    if enable_timestep_shift:
        timesteps = model._apply_time_schedule(
            timesteps, image_tokens, timestep_shift
        )
    attention = {"full_attention": _image_attention_mask(key_valid, image_tokens)}
    for step in range(num_steps):
        t = timesteps[step]
        t_next = timesteps[step + 1]
        z = model.patchify(prediction, divisor)
        image_input = model.patchify(prediction, patch, channel_first=True)
        image_embeds = model.extract_feature(
            image_input.reshape(count * grid_h * grid_w, -1),
            gen_model=True,
            grid_hw=grid_hw,
        ).reshape(count, image_tokens, -1)
        expanded_t = t.expand(count * image_tokens)
        timestep_embeddings = model.fm_modules["timestep_embedder"](
            expanded_t
        ).reshape(count, image_tokens, -1)
        if model.add_noise_scale_embedding:
            normalized_noise = noise_scale / float(model.noise_scale_max_value)
            noise_values = torch.full_like(expanded_t, normalized_noise)
            timestep_embeddings = timestep_embeddings + model.fm_modules[
                "noise_scale_embedder"
            ](noise_values).reshape(count, image_tokens, -1)
        image_embeds = image_embeds + timestep_embeddings

        # The current baseline has CFG disabled.  The interval remains in the
        # signature so introducing batched CFG later does not change the API.
        _ = cfg_interval
        velocity = model._t2i_predict_v(
            image_embeds,
            indexes,
            attention,
            cache,
            t,
            z,
            image_token_num=image_tokens,
            timestep_embeddings=timestep_embeddings,
            image_size=image_size,
        )
        z = z + (t_next - t) * velocity
        prediction = model.unpatchify(z, divisor, height, width)
    return prediction


def _continuous_image_geometry(
    model: Any, image_size: tuple[int, int]
) -> tuple[int, int, int, int, int, int, int]:
    """Return validated image/token geometry for the paged TI2TI path."""

    width, height = image_size
    merge = int(1 / float(model.downsample_ratio))
    patch = int(model.patch_size)
    divisor = patch * merge
    if width <= 0 or height <= 0 or width % divisor or height % divisor:
        raise ValueError("SenseNova image size must divide by patch_size * merge_size")
    token_h = height // divisor
    token_w = width // divisor
    return width, height, patch, merge, divisor, token_h, token_w


def _run_paged_image_sde_batch(
    model: Any,
    cache: ContinuousFlashDecodeCache,
    t_indexes: Tensor,
    generators: Sequence[torch.Generator],
    *,
    image_size: tuple[int, int],
    num_steps: int,
    enable_timestep_shift: bool,
    timestep_shift: float,
    cfg_interval: tuple[float, float],
) -> Tensor:
    """Run no-CFG image denoising against an activated paged prefix view.

    Unlike :func:`_run_image_sde_batch`, this path deliberately supplies no
    dense attention mask.  ``ContinuousFlashDecodeCache.activate`` has already
    selected the participating block tables and sequence lengths, so every
    denoising step overwrites the same uncommitted image-token workspace.
    """

    count = len(generators)
    if count < 1 or t_indexes.shape != (count,):
        raise ValueError("SenseNova paged image rows and cache metadata disagree")
    width, height, patch, merge, divisor, token_h, token_w = (
        _continuous_image_geometry(model, image_size)
    )
    image_tokens = token_h * token_w
    device = t_indexes.device
    indexes = torch.stack(
        tuple(
            model._build_t2i_image_indexes(
                token_h, token_w, int(t_index.item()) + 1, device=device
            )
            for t_index in t_indexes
        )
    )
    grid_h = height // patch
    grid_w = width // patch
    grid_hw = torch.tensor(
        [[grid_h, grid_w]] * count, device=device, dtype=torch.long
    )

    noise_scale = float(model.noise_scale)
    if model.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
        base = float(model.noise_scale_base_image_seq_len)
        noise_scale = math.sqrt((grid_h * grid_w) / (merge**2) / base) * noise_scale
        if model.noise_scale_mode == "dynamic_sqrt":
            noise_scale = math.sqrt(noise_scale)
    noise_scale = min(noise_scale, float(model.noise_scale_max_value))
    model_dtype = next(model.parameters()).dtype
    prediction = torch.cat(
        tuple(
            noise_scale
            * torch.randn(
                (1, 3, height, width),
                device=device,
                dtype=model_dtype,
                generator=generator,
            )
            for generator in generators
        ),
        dim=0,
    )

    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    if enable_timestep_shift:
        timesteps = model._apply_time_schedule(
            timesteps, image_tokens, timestep_shift
        )
    attention = {"full_attention": None}
    for step in range(num_steps):
        t = timesteps[step]
        t_next = timesteps[step + 1]
        z = model.patchify(prediction, divisor)
        image_input = model.patchify(prediction, patch, channel_first=True)
        image_embeds = model.extract_feature(
            image_input.reshape(count * grid_h * grid_w, -1),
            gen_model=True,
            grid_hw=grid_hw,
        ).reshape(count, image_tokens, -1)
        expanded_t = t.expand(count * image_tokens)
        timestep_embeddings = model.fm_modules["timestep_embedder"](
            expanded_t
        ).reshape(count, image_tokens, -1)
        if model.add_noise_scale_embedding:
            normalized_noise = noise_scale / float(model.noise_scale_max_value)
            noise_values = torch.full_like(expanded_t, normalized_noise)
            timestep_embeddings = timestep_embeddings + model.fm_modules[
                "noise_scale_embedder"
            ](noise_values).reshape(count, image_tokens, -1)
        image_embeds = image_embeds + timestep_embeddings

        _ = cfg_interval
        velocity = model._t2i_predict_v(
            image_embeds,
            indexes,
            attention,
            cache,
            t,
            z,
            image_token_num=image_tokens,
            timestep_embeddings=timestep_embeddings,
            image_size=image_size,
        )
        z = z + (t_next - t) * velocity
        prediction = model.unpatchify(z, divisor, height, width)
    return prediction


class ContinuousInterleaveBatchEngine(ContinuousTextBatchEngine):
    """Continuous two-queue TI2TI engine backed by one paged KV pool.

    Text-ready rows execute one decode step per scheduler iteration.  A row
    that selects ``<img>`` commits that control token, leaves the text batch,
    and joins an image-ready batch without waiting for unrelated text rows to
    finish.  Generated image tokens are then appended to the same physical KV
    slot and the row rejoins text decoding.

    This is a separate entry point.  The legacy TI2I/interleave functions keep
    their DynamicCache and dense-mask behavior unchanged.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        max_batch_size: int = 32,
        max_image_batch_size: int = 8,
        image_wait_steps: int = 8,
        max_model_len: int = 16384,
        prefill_chunk_size: int = 2048,
        default_max_new_tokens: int = 8192,
        max_kv_tokens: int = 131072,
        kv_page_size: int = 256,
        prefix_sharing: bool = True,
        truncate_to_max_model_len: bool = False,
        max_images: int = 10,
        image_size: tuple[int, int] = (256, 256),
        num_steps: int = 30,
        enable_timestep_shift: bool = True,
        timestep_shift: float = 1.0,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        t_eps: float = 0.02,
        think_mode: bool = False,
        image_start_token: str = "<img>",
        image_context_token: str = "<IMG_CONTEXT>",
        image_end_token: str = "</img>",
    ) -> None:
        if type(max_image_batch_size) is not int or max_image_batch_size < 1:
            raise ValueError("SenseNova max_image_batch_size must be positive")
        if type(image_wait_steps) is not int or image_wait_steps < 0:
            raise ValueError("SenseNova image_wait_steps must be non-negative")
        if type(max_images) is not int or max_images < 0:
            raise ValueError("SenseNova max_images must be a non-negative int")
        if type(num_steps) is not int or num_steps < 1:
            raise ValueError("SenseNova num_steps must be positive")
        if (
            not isinstance(image_size, tuple)
            or len(image_size) != 2
            or any(type(value) is not int or value <= 0 for value in image_size)
        ):
            raise ValueError("SenseNova image_size must be a positive (width, height)")
        _continuous_image_geometry(model, image_size)
        super().__init__(
            model,
            tokenizer,
            device=device,
            dtype=dtype,
            max_batch_size=max_batch_size,
            max_model_len=max_model_len,
            prefill_chunk_size=prefill_chunk_size,
            default_max_new_tokens=default_max_new_tokens,
            max_kv_tokens=max_kv_tokens,
            kv_page_size=kv_page_size,
            prefix_sharing=prefix_sharing,
            truncate_to_max_model_len=truncate_to_max_model_len,
            image_start_token=image_start_token,
            image_context_token=image_context_token,
            image_end_token=image_end_token,
        )
        self.max_image_batch_size = max_image_batch_size
        self.image_wait_steps = image_wait_steps
        self._text_decode_step = 0
        self.max_images = max_images
        self.image_size = image_size
        self.num_steps = num_steps
        self.enable_timestep_shift = bool(enable_timestep_shift)
        self.timestep_shift = float(timestep_shift)
        self.cfg_interval = cfg_interval
        self.assistant_prefix = "" if think_mode else "<think>\n\n</think>\n\n"
        self.image_end_token_id = int(
            tokenizer.convert_tokens_to_ids(image_end_token)
        )
        self.model.config.t_eps = float(t_eps)
        self._completed: deque[tuple[int, InterleaveBatchResult]] = deque()
        self._scheduled_image_request_ids: tuple[int, ...] | None = None

    @property
    def image_token_count(self) -> int:
        *_, token_h, token_w = _continuous_image_geometry(
            self.model, self.image_size
        )
        return token_h * token_w

    def submit(
        self,
        request: InterleaveBatchRequest,
        *,
        max_new_tokens: int | None = None,
    ) -> int:
        if not isinstance(request, InterleaveBatchRequest):
            raise TypeError(
                "SenseNova continuous interleave request must be InterleaveBatchRequest"
            )
        limit = self.default_max_new_tokens if max_new_tokens is None else max_new_tokens
        if type(limit) is not int or limit < 1:
            raise ValueError("SenseNova request max_new_tokens must be positive")
        request_id = self._next_request_id
        self._next_request_id += 1
        text_request = TextBatchRequest(
            prompt=request.prompt,
            images=request.images,
            system_message=request.system_message,
            assistant_prefix=self.assistant_prefix,
        )
        self._waiting.append(
            _ContinuousInterleaveState(
                request_id=request_id,
                request=text_request,
                max_new_tokens=limit,
                interleave_request=request,
                generator=torch.Generator(device=self.device).manual_seed(request.seed),
            )
        )
        return request_id

    def _flush_text(self, state: _ContinuousInterleaveState) -> None:
        if state.pending_tokens:
            state.parts.append(
                self.tokenizer.decode(
                    state.pending_tokens, skip_special_tokens=True
                )
            )
            state.pending_tokens.clear()

    def _finish_interleave(
        self, state: _ContinuousInterleaveState, reason: str
    ) -> None:
        if state.slot is None or self._cache is None:
            raise RuntimeError("SenseNova completed interleave request has no KV slot")
        self._flush_text(state)
        self._cache.release(state.slot)
        self._free_slots.append(state.slot)
        del self._active[state.request_id]
        self._completed.append(
            (
                state.request_id,
                InterleaveBatchResult(
                    text="".join(state.parts),
                    images=tuple(state.generated_images),
                    finish_reason=reason,
                    generated_tokens=len(state.generated),
                ),
            )
        )

    @torch.no_grad()
    def schedule_text(self) -> ContinuousTextBatch | None:
        """Run one prefill chunk and expose only text-ready row logits."""

        if self._scheduled_request_ids is not None:
            raise RuntimeError("SenseNova continuous text batch must be advanced first")
        if self._scheduled_image_request_ids is not None:
            raise RuntimeError("SenseNova continuous image batch must be run first")
        self._prefill_one_chunk()
        states = sorted(
            (
                state
                for state in self._active.values()
                if state.phase is _ContinuousInterleavePhase.TEXT_READY
            ),
            key=lambda state: int(state.slot),
        )
        if not states:
            return None
        request_ids = tuple(state.request_id for state in states)
        logits = torch.cat(
            tuple(state.next_logits for state in states if state.next_logits is not None),
            dim=0,
        ).to(dtype=torch.float32, copy=True)
        self._scheduled_request_ids = request_ids
        return ContinuousTextBatch(request_ids=request_ids, logits=logits)

    def schedule(self) -> ContinuousTextBatch | None:
        """Compatibility alias for the engine's text scheduler."""

        return self.schedule_text()

    @torch.no_grad()
    def advance_text(
        self,
        batch: ContinuousTextBatch,
        token_ids: Tensor,
        *,
        stop_mask: Tensor | None = None,
    ) -> None:
        """Commit one caller-selected token and route image actions."""

        if batch.request_ids != self._scheduled_request_ids:
            raise RuntimeError("SenseNova continuous text batch is stale or out of order")
        count = len(batch.request_ids)
        if token_ids.shape != (count,) or token_ids.dtype != torch.long:
            raise ValueError("SenseNova continuous token_ids must be batch-shaped long")
        if token_ids.device != self.device:
            raise ValueError("SenseNova continuous token_ids are on the wrong device")
        if stop_mask is None:
            stop_mask = torch.zeros(count, device=self.device, dtype=torch.bool)
        if stop_mask.shape != (count,) or stop_mask.dtype != torch.bool:
            raise ValueError("SenseNova continuous stop_mask must be batch-shaped bool")
        if stop_mask.device != self.device:
            raise ValueError("SenseNova continuous stop_mask is on the wrong device")

        decode_step = self._text_decode_step + 1
        states = [
            self._active[request_id] for request_id in batch.request_ids
        ]
        token_values = token_ids.detach().cpu().tolist()
        stop_values = stop_mask.detach().cpu().tolist()
        commits: list[tuple[_ContinuousInterleaveState, int, str]] = []
        finished: list[tuple[_ContinuousInterleaveState, str]] = []
        final_text_tokens: list[tuple[_ContinuousInterleaveState, int]] = []
        assert self._cache is not None
        for raw_state, token, forced_stop in zip(
            states, token_values, stop_values, strict=True
        ):
            state = raw_state
            if not isinstance(state, _ContinuousInterleaveState):
                raise TypeError("SenseNova interleave engine contains a text-only state")
            token = int(token)
            if forced_stop:
                finished.append((state, "stopped"))
            elif token == self.eos_token_id:
                finished.append((state, "eos"))
            elif token == self.image_start_token_id:
                if len(state.generated_images) >= self.max_images:
                    finished.append((state, "max_images"))
                    continue
                if state.slot is None:
                    raise RuntimeError("SenseNova image-ready request has no KV slot")
                current_length = int(
                    self._cache.flash_decode_seqlens[state.slot].item()
                )
                required = current_length + 1 + self.image_token_count + 1
                if required > self.max_model_len:
                    finished.append((state, "max_model_len"))
                else:
                    commits.append((state, token, "image"))
            elif len(state.generated) + 1 >= state.max_new_tokens:
                final_text_tokens.append((state, token))
                finished.append((state, "max_new_tokens"))
            else:
                commits.append((state, token, "text"))

        outputs = None
        if commits:
            slots = torch.tensor(
                [int(state.slot) for state, _, _ in commits],
                device=self.device,
                dtype=torch.long,
            )
            tokens = torch.tensor(
                [token for _, token, _ in commits],
                device=self.device,
                dtype=torch.long,
            )
            next_t = torch.tensor(
                [state.t_index + 1 for state, _, _ in commits],
                device=self.device,
                dtype=torch.long,
            )
            zeros = torch.zeros_like(next_t)
            indexes = torch.stack((next_t, zeros, zeros), dim=1).unsqueeze(-1)
            self._cache.activate(slots)
            try:
                outputs = self.model.language_model(
                    input_ids=tokens.reshape(-1, 1),
                    indexes=indexes,
                    attention_mask={"full_attention": None},
                    past_key_values=self._cache,
                    use_cache=True,
                )
            except Exception:
                self._cache.cancel_active()
                raise
            self._cache.commit_active()

        for index, (state, token, kind) in enumerate(commits):
            state.t_index += 1
            if kind == "image":
                self._flush_text(state)
                state.phase = _ContinuousInterleavePhase.IMAGE_READY
                state.image_ready_step = decode_step
            else:
                state.generated.append(token)
                state.pending_tokens.append(token)
                assert outputs is not None
                state.next_logits = outputs.logits[
                    index : index + 1, -1, :
                ].detach()
        for state, token in final_text_tokens:
            state.generated.append(token)
            state.pending_tokens.append(token)
        for state, reason in finished:
            self._finish_interleave(state, reason)
        self._text_decode_step = decode_step
        self._scheduled_request_ids = None

    def advance(
        self,
        batch: ContinuousTextBatch,
        token_ids: Tensor,
        *,
        stop_mask: Tensor | None = None,
    ) -> None:
        """Compatibility alias that preserves TI2TI image-action routing."""

        self.advance_text(batch, token_ids, stop_mask=stop_mask)

    def schedule_images(
        self, *, max_batch_size: int | None = None
    ) -> ContinuousImageBatch | None:
        """Select a bounded-wait image batch that fits the remaining KV pages."""

        if self._scheduled_request_ids is not None:
            raise RuntimeError("SenseNova continuous text batch must be advanced first")
        if self._scheduled_image_request_ids is not None:
            raise RuntimeError("SenseNova continuous image batch is already scheduled")
        limit = self.max_image_batch_size if max_batch_size is None else max_batch_size
        if type(limit) is not int or limit < 1:
            raise ValueError("SenseNova image schedule limit must be positive")
        ready = sorted(
            (
                state
                for state in self._active.values()
                if state.phase is _ContinuousInterleavePhase.IMAGE_READY
            ),
            key=lambda state: (
                self._text_decode_step
                if state.image_ready_step is None
                else state.image_ready_step,
                int(state.slot),
            ),
        )
        if not ready:
            return None
        text_ready = any(
            state.phase is _ContinuousInterleavePhase.TEXT_READY
            for state in self._active.values()
        )
        all_active_requests_are_image_ready = bool(self._active) and all(
            state.phase is _ContinuousInterleavePhase.IMAGE_READY
            for state in self._active.values()
        )
        oldest_ready_step = ready[0].image_ready_step
        oldest_wait_steps = (
            0
            if oldest_ready_step is None
            else self._text_decode_step - oldest_ready_step
        )
        should_flush = (
            len(ready) >= limit
            or oldest_wait_steps >= self.image_wait_steps
            or not text_ready
            or all_active_requests_are_image_ready
        )
        if not should_flush:
            return None
        assert self._cache is not None
        selected: list[_ContinuousInterleaveState] = []
        for state in ready[:limit]:
            candidate = selected + [state]
            slots = torch.tensor(
                [int(item.slot) for item in candidate],
                device=self.device,
                dtype=torch.long,
            )
            if self._cache.can_reserve(slots, self.image_token_count + 1):
                selected = candidate
            else:
                continue
        if not selected:
            text_can_progress = any(
                state.phase is _ContinuousInterleavePhase.TEXT_READY
                for state in self._active.values()
            )
            if not text_can_progress and not self._waiting and self._prefilling is None:
                raise RuntimeError(
                    "SenseNova continuous paged KV pool cannot reserve one image workspace"
                )
            return None
        request_ids = tuple(state.request_id for state in selected)
        self._scheduled_image_request_ids = request_ids
        return ContinuousImageBatch(request_ids=request_ids)

    def _append_generated_images_paged(
        self,
        states: Sequence[_ContinuousInterleaveState],
        predictions: Tensor,
    ) -> None:
        """Re-encode generated images and commit image block then ``</img>``."""

        count = len(states)
        if predictions.ndim != 4 or int(predictions.shape[0]) != count:
            raise ValueError("SenseNova generated-image batch and states disagree")
        width, height, patch, merge, _, token_h, token_w = (
            _continuous_image_geometry(self.model, self.image_size)
        )
        if tuple(predictions.shape[1:]) != (3, height, width):
            raise ValueError("SenseNova generated images have unexpected geometry")
        grid_h = height // patch
        grid_w = width // patch
        grid_hw = torch.tensor(
            [[grid_h, grid_w]] * count, device=self.device, dtype=torch.long
        )
        prediction = predictions.to(self.device, dtype=torch.bfloat16)
        raw = prediction * 0.5 + 0.5
        mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=raw.dtype, device=self.device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=raw.dtype, device=self.device
        ).view(1, 3, 1, 1)
        normalized = (raw - mean) / std
        flattened = (
            normalized.view(count, 3, grid_h, patch, grid_w, patch)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(count * grid_h * grid_w, 3 * patch**2)
        )
        features = self.model.extract_feature(flattened, grid_hw=grid_hw)
        features = features.reshape(count, -1, int(features.shape[-1]))
        image_tokens = int(features.shape[1])
        if image_tokens != token_h * token_w:
            raise RuntimeError("SenseNova generated-image token geometry is inconsistent")

        temporal = torch.tensor(
            [state.t_index + 1 for state in states],
            device=self.device,
            dtype=torch.long,
        )
        indexes = torch.zeros(
            count, 3, image_tokens, device=self.device, dtype=torch.long
        )
        indexes[:, 0, :] = temporal[:, None]
        spatial_h = (
            torch.arange(token_h, device=self.device)
            .view(token_h, 1)
            .expand(token_h, token_w)
            .reshape(-1)
        )
        spatial_w = (
            torch.arange(token_w, device=self.device)
            .view(1, token_w)
            .expand(token_h, token_w)
            .reshape(-1)
        )
        indexes[:, 1, :] = spatial_h
        indexes[:, 2, :] = spatial_w
        slots = torch.tensor(
            [int(state.slot) for state in states],
            device=self.device,
            dtype=torch.long,
        )
        assert self._cache is not None
        self._cache.activate(slots, token_count=image_tokens)
        try:
            self.model.language_model.model(
                inputs_embeds=features,
                indexes=indexes,
                attention_mask={"full_attention": None},
                past_key_values=self._cache,
                use_cache=True,
                paged_append_causal=False,
            )
        except Exception:
            self._cache.cancel_active()
            raise
        self._cache.commit_active()

        end_ids = torch.full(
            (count, 1), self.image_end_token_id, device=self.device, dtype=torch.long
        )
        end_indexes = torch.zeros(
            count, 3, 1, device=self.device, dtype=torch.long
        )
        end_indexes[:, 0, 0] = temporal + 1
        self._cache.activate(slots)
        try:
            outputs = self.model.language_model(
                input_ids=end_ids,
                indexes=end_indexes,
                attention_mask={"full_attention": None},
                past_key_values=self._cache,
                use_cache=True,
            )
        except Exception:
            self._cache.cancel_active()
            raise
        self._cache.commit_active()
        for index, state in enumerate(states):
            state.t_index += 2
            state.next_logits = outputs.logits[
                index : index + 1, -1, :
            ].detach()

    @torch.no_grad()
    def run_images(self, batch: ContinuousImageBatch) -> Tensor:
        """Denoise and append one previously scheduled image-ready batch."""

        if batch.request_ids != self._scheduled_image_request_ids:
            raise RuntimeError("SenseNova continuous image batch is stale or out of order")
        raw_states = [self._active[request_id] for request_id in batch.request_ids]
        if not all(isinstance(state, _ContinuousInterleaveState) for state in raw_states):
            raise TypeError("SenseNova image batch contains a text-only state")
        states = list(raw_states)
        slots = torch.tensor(
            [int(state.slot) for state in states],
            device=self.device,
            dtype=torch.long,
        )
        t_indexes = torch.tensor(
            [state.t_index for state in states],
            device=self.device,
            dtype=torch.long,
        )
        generators = [state.generator for state in states]
        if any(generator is None for generator in generators):
            raise RuntimeError("SenseNova image-ready request has no generator")
        assert self._cache is not None
        self._cache.activate(slots, token_count=self.image_token_count + 1)
        try:
            predictions = _run_paged_image_sde_batch(
                self.model,
                self._cache,
                t_indexes,
                generators,
                image_size=self.image_size,
                num_steps=self.num_steps,
                enable_timestep_shift=self.enable_timestep_shift,
                timestep_shift=self.timestep_shift,
                cfg_interval=self.cfg_interval,
            )
        finally:
            self._cache.cancel_active()
        self._append_generated_images_paged(states, predictions)
        for index, state in enumerate(states):
            state.parts.append("<image>")
            state.generated_images.append(predictions[index : index + 1].detach())
            state.phase = _ContinuousInterleavePhase.TEXT_READY
            state.image_ready_step = None
        self._scheduled_image_request_ids = None
        return predictions

    def pop_completed(self) -> tuple[tuple[int, InterleaveBatchResult], ...]:
        completed = tuple(self._completed)
        self._completed.clear()
        return completed


def _apply_repetition_penalty(
    logits: Tensor,
    seen_tokens: Tensor,
    penalty: float,
) -> Tensor:
    """Apply LightLLM/HuggingFace-style generated-token repetition penalty."""

    if logits.shape != seen_tokens.shape or seen_tokens.dtype != torch.bool:
        raise ValueError("SenseNova repetition mask must match logits and be bool")
    if penalty < 1.0:
        raise ValueError("SenseNova repetition_penalty must be at least 1.0")
    if penalty == 1.0:
        return logits
    adjusted = torch.where(logits > 0, logits / penalty, logits * penalty)
    return torch.where(seen_tokens, adjusted, logits)


@torch.no_grad()
def batch_text_gen(
    model: Any,
    tokenizer: Any,
    requests: Sequence[TextBatchRequest],
    *,
    generation_config: Any | None = None,
    prefix_sharing: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[TextBatchResult, ...]:
    """Generate one native greedy TI2T batch.

    ``prefix_sharing`` is intended for RL groups.  When enabled, every request
    must carry exactly the same rendered prefix; evaluation batches normally
    leave it disabled and use the varlen left-padded prefill path.
    """

    rows = tuple(requests)
    if not rows or not all(isinstance(row, TextBatchRequest) for row in rows):
        raise ValueError("SenseNova text batch must contain valid requests")
    max_new_tokens = int(
        getattr(generation_config, "max_new_tokens", None) or 8192
    )
    if max_new_tokens < 1:
        raise ValueError("SenseNova max_new_tokens must be positive")
    repetition_penalty = float(
        getattr(generation_config, "repetition_penalty", None) or 1.0
    )
    if repetition_penalty < 1.0:
        raise ValueError("SenseNova repetition_penalty must be at least 1.0")
    runtime_device = torch.device(device if device is not None else model.device)
    session = NativeTextBatchSession(
        model,
        tokenizer,
        rows,
        device=runtime_device,
        dtype=dtype,
        prefix_sharing=prefix_sharing,
        flash_decode_tokens=(
            max_new_tokens
            if runtime_device.type == "cuda" and effective_attn_backend() == "flash"
            else 0
        ),
    )
    runtime_device = session.device
    template = get_conv_template(model.template)
    eos_token_id = int(tokenizer.convert_tokens_to_ids(template.sep.strip()))
    active = [True for _ in rows]
    seen_tokens = torch.zeros(
        len(rows),
        int(session.next_logits.shape[-1]),
        device=runtime_device,
        dtype=torch.bool,
    )
    generated: list[list[int]] = [[] for _ in rows]
    reasons = ["" for _ in rows]
    while any(active):
        logits = _apply_repetition_penalty(
            session.constrained_logits(), seen_tokens, repetition_penalty
        )
        next_tokens = torch.argmax(logits, dim=-1).to(
            device=runtime_device, dtype=torch.long
        )
        token_values = next_tokens.detach().cpu().tolist()
        accepted_indices = []
        for row, is_active in enumerate(active):
            if not is_active:
                continue
            token = int(token_values[row])
            if token == eos_token_id:
                active[row] = False
                reasons[row] = "eos"
                continue
            generated[row].append(token)
            if len(generated[row]) >= max_new_tokens:
                active[row] = False
                reasons[row] = "max_new_tokens"
            else:
                # There is no next token to select after a row reaches its
                # length limit, so avoid one otherwise wasted model forward.
                accepted_indices.append(row)
        if accepted_indices:
            accepted_rows = torch.tensor(
                accepted_indices, device=runtime_device, dtype=torch.long
            )
            accepted = torch.zeros(
                len(rows), device=runtime_device, dtype=torch.bool
            )
            accepted[accepted_rows] = True
            seen_tokens[accepted_rows, next_tokens.index_select(0, accepted_rows)] = True
            session.commit(next_tokens, accepted)
    return tuple(
        TextBatchResult(
            text=tokenizer.decode(tokens, skip_special_tokens=True),
            finish_reason=reasons[row],
            generated_tokens=len(tokens),
        )
        for row, tokens in enumerate(generated)
    )


@torch.no_grad()
def continuous_batch_interleave_gen(
    model: Any,
    tokenizer: Any,
    requests: Sequence[InterleaveBatchRequest],
    *,
    generation_config: Any | None = None,
    cfg_scale: float = 1.0,
    img_cfg_scale: float = 1.0,
    cfg_norm: str = "none",
    max_images: int = 10,
    enable_timestep_shift: bool = True,
    timestep_shift: float = 1.0,
    image_size: tuple[int, int] = (256, 256),
    num_steps: int = 30,
    image_start_token: str = "<img>",
    image_end_token: str = "</img>",
    image_context_token: str = "<IMG_CONTEXT>",
    method: str = "euler",
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    t_eps: float = 0.02,
    think_mode: bool = False,
    max_batch_size: int = 32,
    max_image_batch_size: int = 8,
    image_wait_steps: int = 8,
    max_model_len: int = 16384,
    prefill_chunk_size: int = 2048,
    max_kv_tokens: int = 131072,
    kv_page_size: int = 256,
    prefix_sharing: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[InterleaveBatchResult, ...]:
    """Generate TI2TI responses through paged continuous two-queue batching.

    This new path preserves the existing no-CFG Euler semantics but does not
    reuse or redirect the legacy TI2I/interleave implementation.  Each loop
    performs one text decode step and accumulates image-ready rows into bounded
    batches before returning those rows to text decoding.
    """

    rows = tuple(requests)
    if not rows or not all(isinstance(row, InterleaveBatchRequest) for row in rows):
        raise ValueError("SenseNova interleave batch must contain valid requests")
    if cfg_scale != 1.0 or img_cfg_scale != 1.0 or cfg_norm != "none":
        raise NotImplementedError(
            "SenseNova continuous TI2TI currently supports no-CFG only"
        )
    if method != "euler":
        raise NotImplementedError(
            "SenseNova continuous TI2TI currently supports Euler only"
        )
    max_new_tokens = int(
        getattr(generation_config, "max_new_tokens", None) or 8192
    )
    if max_new_tokens < 1:
        raise ValueError("SenseNova max_new_tokens must be positive")
    repetition_penalty = float(
        getattr(generation_config, "repetition_penalty", None) or 1.0
    )
    if repetition_penalty < 1.0:
        raise ValueError("SenseNova repetition_penalty must be at least 1.0")
    runtime_device = torch.device(device if device is not None else model.device)
    engine = ContinuousInterleaveBatchEngine(
        model,
        tokenizer,
        device=runtime_device,
        dtype=dtype,
        max_batch_size=max_batch_size,
        max_image_batch_size=max_image_batch_size,
        image_wait_steps=image_wait_steps,
        max_model_len=max_model_len,
        prefill_chunk_size=prefill_chunk_size,
        default_max_new_tokens=max_new_tokens,
        max_kv_tokens=max_kv_tokens,
        kv_page_size=kv_page_size,
        prefix_sharing=prefix_sharing,
        max_images=max_images,
        image_size=image_size,
        num_steps=num_steps,
        enable_timestep_shift=enable_timestep_shift,
        timestep_shift=timestep_shift,
        cfg_interval=cfg_interval,
        t_eps=t_eps,
        think_mode=think_mode,
        image_start_token=image_start_token,
        image_context_token=image_context_token,
        image_end_token=image_end_token,
    )
    request_ids = [engine.submit(row) for row in rows]
    seen_tokens: dict[int, set[int]] = {
        request_id: set() for request_id in request_ids
    }
    completed: dict[int, InterleaveBatchResult] = {}
    while engine.has_unfinished_requests:
        text_batch = engine.schedule_text()
        if text_batch is not None:
            logits = text_batch.logits
            if repetition_penalty != 1.0:
                seen = torch.zeros_like(logits, dtype=torch.bool)
                for row, request_id in enumerate(text_batch.request_ids):
                    if seen_tokens[request_id]:
                        indexes = torch.tensor(
                            sorted(seen_tokens[request_id]),
                            device=runtime_device,
                            dtype=torch.long,
                        )
                        seen[row, indexes] = True
                logits = _apply_repetition_penalty(
                    logits, seen, repetition_penalty
                )
            next_tokens = torch.argmax(logits, dim=-1).to(
                device=runtime_device, dtype=torch.long
            )
            for request_id, token in zip(
                text_batch.request_ids,
                next_tokens.detach().cpu().tolist(),
                strict=True,
            ):
                if token not in (engine.eos_token_id, engine.image_start_token_id):
                    seen_tokens[request_id].add(int(token))
            engine.advance_text(text_batch, next_tokens)
            completed.update(engine.pop_completed())

        image_batch = engine.schedule_images()
        if image_batch is not None:
            engine.run_images(image_batch)
        completed.update(engine.pop_completed())
    completed.update(engine.pop_completed())
    return tuple(completed[request_id] for request_id in request_ids)


@torch.no_grad()
def batch_interleave_gen(
    model: Any,
    tokenizer: Any,
    requests: Sequence[InterleaveBatchRequest],
    *,
    generation_config: Any | None = None,
    cfg_scale: float = 1.0,
    img_cfg_scale: float = 1.0,
    cfg_norm: str = "none",
    max_images: int = 10,
    enable_timestep_shift: bool = True,
    timestep_shift: float = 1.0,
    image_size: tuple[int, int] = (256, 256),
    num_steps: int = 30,
    image_start_token: str = "<img>",
    image_end_token: str = "</img>",
    image_context_token: str = "<IMG_CONTEXT>",
    method: str = "euler",
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    t_eps: float = 0.02,
    think_mode: bool = False,
    prefix_sharing: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[InterleaveBatchResult, ...]:
    """Generate TI2TI responses with separate text-ready and image-ready batches.

    The first implementation intentionally matches the project's current
    no-CFG baseline.  Requests with an identical prompt/image prefix compute
    that prefix once and expand its KV cache across the rollout group.
    """

    rows = tuple(requests)
    if not rows or not all(isinstance(row, InterleaveBatchRequest) for row in rows):
        raise ValueError("SenseNova interleave batch must contain valid requests")
    if cfg_scale != 1.0 or img_cfg_scale != 1.0 or cfg_norm != "none":
        raise NotImplementedError("SenseNova batched TI2TI currently supports no-CFG only")
    if method != "euler":
        raise NotImplementedError("SenseNova batched TI2TI currently supports Euler only")
    if type(max_images) is not int or max_images < 0:
        raise ValueError("SenseNova max_images must be a non-negative int")
    if type(num_steps) is not int or num_steps < 1:
        raise ValueError("SenseNova num_steps must be a positive int")
    if (
        not isinstance(image_size, tuple)
        or len(image_size) != 2
        or any(type(value) is not int or value <= 0 for value in image_size)
    ):
        raise ValueError("SenseNova batch image_size must be a positive (width, height)")
    max_new_tokens = int(
        getattr(generation_config, "max_new_tokens", None) or 8192
    )
    if max_new_tokens < 1:
        raise ValueError("SenseNova max_new_tokens must be positive")
    model.config.t_eps = float(t_eps)
    runtime_device = torch.device(device if device is not None else model.device)

    assistant_prefix = "" if think_mode else "<think>\n\n</think>\n\n"
    text_requests = tuple(
        TextBatchRequest(
            prompt=row.prompt,
            images=row.images,
            system_message=row.system_message,
            assistant_prefix=assistant_prefix,
        )
        for row in rows
    )
    share = bool(prefix_sharing and _requests_share_prefix(text_requests))
    session = NativeTextBatchSession(
        model,
        tokenizer,
        text_requests,
        device=runtime_device,
        dtype=dtype,
        image_start_token=image_start_token,
        image_context_token=image_context_token,
        image_end_token=image_end_token,
        allow_image_actions=True,
        prefix_sharing=share,
    )

    template = get_conv_template(model.template)
    eos_token_id = int(tokenizer.convert_tokens_to_ids(template.sep.strip()))
    image_start_token_id = int(tokenizer.convert_tokens_to_ids(image_start_token))
    image_end_token_id = int(tokenizer.convert_tokens_to_ids(image_end_token))
    states = [_InterleaveState.TEXT_READY for _ in rows]
    finish_reasons = ["" for _ in rows]
    generated_tokens = [0 for _ in rows]
    generated_images: list[list[Tensor]] = [[] for _ in rows]
    parts: list[list[str]] = [[] for _ in rows]
    pending_tokens: list[list[int]] = [[] for _ in rows]
    generators = [
        torch.Generator(device=runtime_device).manual_seed(row.seed) for row in rows
    ]

    def flush_text(row: int) -> None:
        if pending_tokens[row]:
            parts[row].append(
                tokenizer.decode(pending_tokens[row], skip_special_tokens=True)
            )
            pending_tokens[row].clear()

    while any(state is not _InterleaveState.FINISHED for state in states):
        # Drain TEXT_READY exactly as one batch, as requested.  Rows that reach
        # an event remain as masked dummy cache slots until the text batch ends.
        while any(state is _InterleaveState.TEXT_READY for state in states):
            next_tokens = torch.argmax(session.constrained_logits(), dim=-1).to(
                device=runtime_device, dtype=torch.long
            )
            accepted = torch.zeros(len(rows), device=runtime_device, dtype=torch.bool)
            stop_after_commit: list[int] = []
            for row, state in enumerate(states):
                if state is not _InterleaveState.TEXT_READY:
                    continue
                token = int(next_tokens[row].item())
                if token == eos_token_id:
                    flush_text(row)
                    states[row] = _InterleaveState.FINISHED
                    finish_reasons[row] = "eos"
                elif token == image_start_token_id:
                    flush_text(row)
                    if len(generated_images[row]) >= max_images:
                        states[row] = _InterleaveState.FINISHED
                        finish_reasons[row] = "max_images"
                    else:
                        states[row] = _InterleaveState.IMAGE_READY
                else:
                    pending_tokens[row].append(token)
                    generated_tokens[row] += 1
                    accepted[row] = True
                    if generated_tokens[row] >= max_new_tokens:
                        stop_after_commit.append(row)
            if bool(accepted.any().item()):
                session.commit(next_tokens, accepted)
            for row in stop_after_commit:
                flush_text(row)
                states[row] = _InterleaveState.FINISHED
                finish_reasons[row] = "max_new_tokens"

        image_rows = [
            row for row, state in enumerate(states) if state is _InterleaveState.IMAGE_READY
        ]
        if not image_rows:
            continue
        image_indices = torch.tensor(
            image_rows, device=runtime_device, dtype=torch.long
        )
        accepted = torch.zeros(len(rows), device=runtime_device, dtype=torch.bool)
        accepted[image_indices] = True
        control_tokens = torch.full(
            (len(rows),), eos_token_id, device=runtime_device, dtype=torch.long
        )
        control_tokens[image_indices] = image_start_token_id
        session.commit(control_tokens, accepted)
        image_cache, image_key_valid, image_t_indexes = session.selected_cache(
            image_indices
        )
        predictions = _run_image_sde_batch(
            model,
            image_cache,
            image_key_valid,
            image_t_indexes,
            [generators[row] for row in image_rows],
            image_size=image_size,
            num_steps=num_steps,
            enable_timestep_shift=enable_timestep_shift,
            timestep_shift=timestep_shift,
            cfg_interval=cfg_interval,
        )
        session.append_generated_images(
            predictions, image_indices, image_end_token_id=image_end_token_id
        )
        for local, row in enumerate(image_rows):
            parts[row].append("<image>")
            generated_images[row].append(predictions[local : local + 1])
            states[row] = _InterleaveState.TEXT_READY

    return tuple(
        InterleaveBatchResult(
            text="".join(parts[row]),
            images=tuple(generated_images[row]),
            finish_reason=finish_reasons[row],
            generated_tokens=generated_tokens[row],
        )
        for row in range(len(rows))
    )


__all__ = [
    "ContiguousTextBatchSession",
    "ContinuousImageBatch",
    "ContinuousInterleaveBatchEngine",
    "ContinuousTextBatch",
    "ContinuousTextBatchEngine",
    "InterleaveBatchRequest",
    "InterleaveBatchResult",
    "NativeTextBatchSession",
    "TextBatchRequest",
    "TextBatchResult",
    "batch_interleave_gen",
    "batch_text_gen",
    "continuous_batch_interleave_gen",
    "continuous_batch_text_gen",
]
