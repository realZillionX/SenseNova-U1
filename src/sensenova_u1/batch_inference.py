"""Native dynamic batching for SenseNova-U1 text rollouts.

The public ``NEOChatModel.batch_chat`` entry point is intentionally not used:
it is a historical stub and the model's Transformers ``generate`` path still
assumes one row.  This module exposes the lower-level operation needed by both
evaluation and RL rollout: one multimodal prefill per row, followed by a shared
batched KV cache with independently advancing/eos-ing rows.

Only text actions are implemented here.  Interleaved image actions need a
second scheduler that temporarily removes image-producing rows from the text
batch, batches their pixel-head/SDE work, appends the resulting image tokens,
and then rejoins them.  Silently treating that as this simpler TI2T path would
change model semantics, so callers must opt into ``modality='ti2t'``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str):
            raise TypeError("SenseNova batch prompt must be a string")
        if not isinstance(self.images, tuple):
            raise TypeError("SenseNova batch images must be a tuple")
        if not isinstance(self.system_message, str):
            raise TypeError("SenseNova batch system message must be a string")


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
    ) -> None:
        rows = tuple(requests)
        if not rows:
            raise ValueError("SenseNova text batch must contain at least one request")
        if not all(isinstance(row, TextBatchRequest) for row in rows):
            raise TypeError("SenseNova text batch contains an invalid request")

        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.dtype = dtype
        self.batch_size = len(rows)
        self.image_start_token_id = int(
            tokenizer.convert_tokens_to_ids(image_start_token)
        )
        self.image_context_token_id = int(
            tokenizer.convert_tokens_to_ids(image_context_token)
        )
        self.image_end_token_id = int(tokenizer.convert_tokens_to_ids(image_end_token))
        model.img_context_token_id = self.image_context_token_id
        model.img_start_token_id = self.image_start_token_id

        row_embeds: list[Tensor] = []
        row_indexes: list[Tensor] = []
        row_masks: list[Tensor] = []
        for request in rows:
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
        query = template.get_prompt()
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


__all__ = ["NativeTextBatchSession", "TextBatchRequest"]
