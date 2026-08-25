"""Native dynamic batching for SenseNova-U1 text and interleaved rollouts.

The public ``NEOChatModel.batch_chat`` entry point is intentionally not used:
it is a historical stub and the model's Transformers ``generate`` path still
assumes one row.  This module exposes the lower-level operation needed by both
evaluation and RL rollout: one multimodal prefill per row, followed by a shared
batched KV cache with independently advancing/eos-ing rows.

``NativeTextBatchSession`` is the common cache primitive.  The public
``batch_interleave_gen`` entry point adds the deliberately small two-queue
scheduler used by TI2TI: drain every text-ready row as one batch, then drain
every image-ready row as one homogeneous SDE batch.  More elaborate scheduling
(token quanta, continuous admission, resolution buckets) can be layered on top
without moving model semantics into a serving framework.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch import Tensor

from .models.neo_unify.conversation import get_conv_template
from .models.neo_unify.utils import load_image_native


@dataclass(frozen=True)
class TextBatchRequest:
    """One prompt row for :class:`NativeTextBatchSession`."""

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


class NativeTextBatchSession:
    """A batched native U1/U1.5 text-only KV-cache session.

    ``commit`` accepts one token and one ``accepted`` flag per row.  Rejected
    rows still occupy a physical cache slot (all DynamicCache rows must have
    one shared length), but that slot is masked from every later query and the
    row's THW time index does not advance.  This is what permits independent
    EOS/length stopping without corrupting active rows.
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

    def _prepare_request(
        self,
        request: TextBatchRequest,
        *,
        image_start_token: str,
        image_context_token: str,
        image_end_token: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
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
                self.model.patch_size,
                self.model.downsample_ratio,
                min_pixels=512 * 512,
                max_pixels=min(
                    2048 * 2048,
                    (4096 * 4096) // max(1, len(images)),
                ),
                upscale=False,
            )
            pixel_values.append(pixels.to(self.device, dtype=self.dtype))
            grid_hw.append(grid.to(self.device))

        template = get_conv_template(self.model.template)
        template.system_message = request.system_message
        template.append_message(template.roles[0], prompt)
        template.append_message(template.roles[1], None)
        query = template.get_prompt() + request.assistant_prefix
        for grid in grid_hw:
            context_tokens = int(
                grid[0, 0]
                * grid[0, 1]
                * float(self.model.downsample_ratio) ** 2
            )
            image_span = (
                image_start_token
                + image_context_token * context_tokens
                + image_end_token
            )
            query = query.replace("<image>", image_span, 1)

        pixels_tensor = torch.cat(pixel_values) if pixel_values else None
        grid_tensor = torch.cat(grid_hw) if grid_hw else None
        inputs_embeds, indexes, attention = self.model._build_it2i_inputs(
            self.tokenizer,
            query,
            pixels_tensor,
            grid_tensor,
        )
        return inputs_embeds, indexes, attention["full_attention"]

    def constrained_logits(self) -> Tensor:
        logits = self.next_logits.float()
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
    # FlashAttention can consume the prefix and current image span directly
    # only when every physical prefix slot is real.  Rows that waited while a
    # sibling continued text decoding contain masked dummy cache slots; those
    # retain the explicit-mask fallback until the scheduler compacts/buckets
    # their prefixes.  Identical-prefix rollout groups take this fast path.
    use_flash_kv = bool(key_valid.all().item())
    if use_flash_kv:
        from .models.neo_unify.modeling_neo_chat import (
            clear_flash_kv_cache,
            prepare_flash_kv_cache,
        )

        attention = {"full_attention": None}
    else:
        attention = {
            "full_attention": _image_attention_mask(key_valid, image_tokens)
        }

    try:
        if use_flash_kv:
            prepare_flash_kv_cache(
                cache,
                current_len=image_tokens,
                batch_size=count,
            )
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

            # The current baseline has CFG disabled.  The interval remains in
            # the signature so introducing batched CFG later does not change
            # the API.
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
    finally:
        if use_flash_kv:
            clear_flash_kv_cache(cache)
    return prediction


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
    )
    runtime_device = session.device
    template = get_conv_template(model.template)
    eos_token_id = int(tokenizer.convert_tokens_to_ids(template.sep.strip()))
    active = torch.ones(len(rows), device=runtime_device, dtype=torch.bool)
    seen_tokens = torch.zeros(
        len(rows),
        int(session.next_logits.shape[-1]),
        device=runtime_device,
        dtype=torch.bool,
    )
    generated: list[list[int]] = [[] for _ in rows]
    reasons = ["" for _ in rows]
    while bool(active.any().item()):
        logits = _apply_repetition_penalty(
            session.constrained_logits(), seen_tokens, repetition_penalty
        )
        next_tokens = torch.argmax(logits, dim=-1).to(
            device=runtime_device, dtype=torch.long
        )
        accepted = active & next_tokens.ne(eos_token_id)
        for row in torch.nonzero(active, as_tuple=False).flatten().tolist():
            if not bool(accepted[row].item()):
                active[row] = False
                reasons[row] = "eos"
                continue
            generated[row].append(int(next_tokens[row].item()))
            if len(generated[row]) >= max_new_tokens:
                active[row] = False
                reasons[row] = "max_new_tokens"
        if bool(accepted.any().item()):
            accepted_rows = torch.nonzero(accepted, as_tuple=False).flatten()
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
    "InterleaveBatchRequest",
    "InterleaveBatchResult",
    "NativeTextBatchSession",
    "TextBatchRequest",
    "TextBatchResult",
    "batch_interleave_gen",
    "batch_text_gen",
]
