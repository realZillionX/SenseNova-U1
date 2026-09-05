"""U1.5 model binding for stochastic interleaved RLVR.

This module adapts the Forge ``NEOChatModel`` cache, pixel head and image
re-encoding helpers to Forge's text-token and image-SDE primitives.
Sampling stores old-policy
likelihoods; replay rebuilds every cache from the prompt and evaluates the
same actions with gradients enabled.
"""

from __future__ import annotations

import copy
import importlib
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Protocol

import torch
from PIL import Image
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import Cache

from .full_parameter import (
    local_parameter_view,
    reshard_full_parameter_policy,
    shard_full_parameter_policy,
    snapshot_reference_shards,
)
from .plan import RlPlan
from .rollout import (
    ImageSdeReplay,
    ImageSdeTrace,
    TextRolloutTrace,
    concatenate_text_spans,
    replay_image_sde_batched,
    sample_image_sde,
    sample_text_span,
    select_sde_indices,
    slice_image_sde_trace,
)
from .runtime import validate_policy_model
from .types import SYSTEM_MESSAGE_BY_MODALITY, CandidateResponse, ImageSegment, TextSegment


@dataclass(frozen=True)
class TextEvent:
    trace: TextRolloutTrace
    stop_token_id: int | None


@dataclass(frozen=True)
class ImageEvent:
    trace: ImageSdeTrace


PolicyEvent = TextEvent | ImageEvent


_TEXT_LOGPROB_CHUNK_SIZE = 256


@dataclass(frozen=True)
class U15PolicyRollout:
    candidate: CandidateResponse
    events: tuple[PolicyEvent, ...]
    text_tokens: int
    generated_images: int
    seconds: float
    finish_reason: str = "stop"
    image_context_tokens: int = 0


@dataclass(frozen=True)
class U15PolicyReplay:
    text_trace: TextRolloutTrace
    text_log_probs: Tensor
    text_ref_log_probs: Tensor | None
    image_replays: tuple[ImageSdeReplay, ...]
    image_reference_velocities: tuple[Tensor, ...]
    ratio_max_error: float
    ratio_mean_error: float
    ratio_action_count: int
    numeric_max_error: float


@dataclass(frozen=True)
class U15PolicyReferenceReplay:
    text_log_probs: tuple[Tensor, ...]
    image_velocities: tuple[Tensor, ...]


@dataclass(frozen=True)
class U15TextReplaySpan:
    trace: TextRolloutTrace
    log_probs: Tensor
    ref_log_probs: Tensor | None
    numeric_max_error: float
    is_last: bool


@dataclass(frozen=True)
class U15PolicyAnchor:
    """Replay-geometry old-policy anchor plus serving/local alignment evidence."""

    rollout: U15PolicyRollout
    behavior_ratio_max_error: float
    behavior_ratio_mean_error: float
    behavior_ratio_action_count: int
    numeric_max_error: float


class U15Policy(Protocol):
    model: Any

    def rollout(
        self,
        *,
        prompt: str,
        prompt_images: tuple[str, ...],
        modality: str,
        system_message: str | None,
        artifact_dir: Path,
        rollout_key: str,
        generator: torch.Generator,
    ) -> U15PolicyRollout: ...

    def replay(
        self,
        *,
        prompt: str,
        prompt_images: tuple[str, ...],
        modality: str,
        system_message: str | None,
        rollout: U15PolicyRollout,
        include_reference: bool = True,
        include_image_policy: bool = True,
    ) -> U15PolicyReplay: ...

    def reference_replay(
        self,
        *,
        prompt: str,
        prompt_images: tuple[str, ...],
        modality: str,
        system_message: str | None,
        rollout: U15PolicyRollout,
        include_text: bool,
        include_images: bool,
    ) -> U15PolicyReferenceReplay: ...

    def iter_text_replay_spans(
        self,
        *,
        prompt: str,
        prompt_images: tuple[str, ...],
        modality: str,
        system_message: str | None,
        rollout: U15PolicyRollout,
        reference_log_probs: tuple[Tensor, ...],
    ) -> Iterator[U15TextReplaySpan]: ...

    def iter_image_action_microbatches(
        self,
        *,
        prompt: str,
        prompt_images: tuple[str, ...],
        modality: str,
        system_message: str | None,
        rollout: U15PolicyRollout,
        microbatch_size: int,
        reference_velocities: tuple[Tensor, ...],
    ) -> Iterator[ImageSdeReplay]: ...

    def anchor_rollout(
        self,
        *,
        prompt: str,
        prompt_images: tuple[str, ...],
        modality: str,
        system_message: str | None,
        rollout: U15PolicyRollout,
    ) -> U15PolicyRollout: ...

    def anchor_rollout_with_metrics(
        self,
        *,
        prompt: str,
        prompt_images: tuple[str, ...],
        modality: str,
        system_message: str | None,
        rollout: U15PolicyRollout,
    ) -> U15PolicyAnchor: ...


def _single_row_tokens(trace: TextRolloutTrace) -> list[int]:
    if trace.batch_size != 1:
        raise ValueError("U1.5 runtime replays one rollout at a time")
    return [int(value) for value in trace.token_ids[0][trace.response_mask[0]].detach().cpu().tolist()]


def _checkpointed_selected_log_probs(
    logits: Tensor,
    target_ids: Tensor,
    *,
    forbidden_token_id: int | None,
    chunk_size: int = _TEXT_LOGPROB_CHUNK_SIZE,
) -> Tensor:
    """Select FP32 token log-probabilities without a full FP32 ``[B,T,V]``.

    The U1.5 causal-LM forward returns BF16 vocabulary logits. Converting
    that complete tensor to FP32 and then materializing a complete log-softmax
    briefly retains two FP32 ``[B,T,V]`` tensors. Replay only consumes one
    target probability per position, so compute those values in bounded token
    chunks. Checkpointing prevents every chunk's FP32 softmax from being kept
    until backward; it is recomputed one chunk at a time instead.
    """

    if logits.ndim != 3:
        raise ValueError(f"text replay logits must be rank 3, got {logits.ndim}")
    if target_ids.shape != logits.shape[:2]:
        raise ValueError(
            "text replay target/logit layouts disagree: "
            f"targets={tuple(target_ids.shape)}, logits={tuple(logits.shape)}"
        )
    if target_ids.dtype != torch.long:
        raise TypeError("text replay targets must use torch.long")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("text replay log-prob chunk size must be positive")
    if forbidden_token_id is not None and not (0 <= forbidden_token_id < logits.shape[-1]):
        raise ValueError("forbidden text token is outside the vocabulary")
    if not target_ids.numel():
        return logits.new_empty(target_ids.shape, dtype=torch.float32)

    def select(chunk_logits: Tensor, chunk_targets: Tensor) -> Tensor:
        # ``copy=True`` keeps the TI2T vocabulary constraint local even when a
        # caller supplies FP32 logits.
        distributions = chunk_logits.to(dtype=torch.float32, copy=True)
        if forbidden_token_id is not None:
            distributions[..., forbidden_token_id] = torch.finfo(distributions.dtype).min
        return torch.log_softmax(distributions, dim=-1).gather(-1, chunk_targets.unsqueeze(-1)).squeeze(-1)

    selected: list[Tensor] = []
    for start in range(0, target_ids.shape[1], chunk_size):
        stop = min(start + chunk_size, target_ids.shape[1])
        chunk_logits = logits[:, start:stop, :]
        chunk_targets = target_ids[:, start:stop]
        if torch.is_grad_enabled() and chunk_logits.requires_grad:
            values = checkpoint(
                select,
                chunk_logits,
                chunk_targets,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            values = select(chunk_logits, chunk_targets)
        selected.append(values)
    return torch.cat(selected, dim=1)


def _ratio_error_stats(current: Tensor, old: Tensor) -> tuple[float, float, int]:
    if current.shape != old.shape:
        raise ValueError("current and old likelihood traces have different shapes")
    if not current.numel():
        return 0.0, 0.0, 0
    ratio = torch.exp(current.detach().float() - old.detach().float())
    if not bool(torch.isfinite(ratio).all()):
        raise ValueError("on-policy likelihood ratio is non-finite")
    errors = (ratio - 1.0).abs()
    return (
        float(errors.max().cpu()),
        float(errors.sum().cpu()),
        errors.numel(),
    )


def _ratio_error(current: Tensor, old: Tensor) -> float:
    maximum, _total, _count = _ratio_error_stats(current, old)
    return maximum


def _decode_without_stop(tokenizer: Any, trace: TextRolloutTrace, stop_token_id: int | None) -> str:
    tokens = _single_row_tokens(trace)
    if stop_token_id is not None and tokens and tokens[-1] == stop_token_id:
        tokens.pop()
    if not tokens:
        return ""
    return str(
        tokenizer.decode(
            tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def _validate_full_parameter_closure(model: Any) -> tuple[str, ...]:
    from torch.distributed.tensor import DTensor

    named = tuple(model.named_parameters())
    if not named:
        raise ValueError("U1.5 policy has no parameters")
    frozen = [name for name, parameter in named if not parameter.requires_grad]
    unsharded = [name for name, parameter in named if not isinstance(parameter, DTensor)]
    adapter = [name for name, _parameter in named if "lora_" in name.lower()]
    if frozen or unsharded or adapter:
        raise ValueError(
            "U1.5 full-parameter closure is invalid: "
            f"frozen={frozen[:8]}, unsharded={unsharded[:8]}, adapter={adapter[:8]}"
        )
    return tuple(name for name, _parameter in named)


class _ExpandedReadOnlyCache(Cache):
    """Batch-expanded KV views for batched image replay.

    Image replay never appends to the prompt/text cache while evaluating a
    denoising transition. Expanding the batch-one prefix as a zero-copy view
    lets all fixed SDE steps share one model forward without duplicating the
    KV allocation.
    """

    @classmethod
    def from_cache(cls, cache: Cache, *, batch: int) -> _ExpandedReadOnlyCache:
        if type(batch) is not int or batch < 1:
            raise ValueError("expanded U1.5 cache batch must be positive")
        layers = []
        for source in cache.layers:
            target = copy.copy(source)
            target.keys = source.keys.expand(batch, -1, -1, -1)
            target.values = source.values.expand(batch, -1, -1, -1)
            layers.append(target)
        return cls(layers=layers)


class _PolicySession:
    """One prompt-local policy cache used for either rollout or replay."""

    def __init__(
        self,
        runtime: U15Policy,
        *,
        prompt: str,
        prompt_images: tuple[str, ...],
        modality: str,
        system_message: str | None = None,
    ) -> None:
        if modality not in {"ti2t", "ti2ti"}:
            raise ValueError(f"unsupported U1.5 modality {modality!r}")
        self.runtime = runtime
        self.model = runtime.core_model
        self.tokenizer = runtime.tokenizer
        self.modality = modality
        self.device = runtime.device
        self.dtype = runtime.dtype
        self.model_module = runtime.model_module
        self.img_start_id = runtime.img_start_id
        self.eos_id = runtime.eos_id

        images = list(prompt_images)
        image_count = prompt.count("<image>")
        if len(images) < image_count:
            raise ValueError("U1.5 prompt has more image placeholders than input images")
        if len(images) > image_count:
            prompt = "<image>\n" * (len(images) - image_count) + prompt

        pixel_values: list[Tensor] = []
        grid_hw: list[Tensor] = []
        loader = getattr(self.model_module, "load_image_native", None)
        if not callable(loader):
            raise TypeError("U1.5 module has no load_image_native helper")
        for image_path in images:
            pixels, grid = loader(
                image_path,
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

        get_template = getattr(self.model_module, "get_conv_template", None)
        if not callable(get_template):
            raise TypeError("U1.5 module has no conversation template helper")
        template = get_template(self.model.template)
        # Condition the rollout on exactly the system turn this modality's SFT
        # rows carry, from the same table the exporter reads.  TI2TI ships the
        # U1.5 interleaved-generation system message because the model
        # corpus does; TI2T may supply a caller-authored text-think analogue, the
        # one place the mode is declared now that the task Prompt no longer
        # describes the response envelope.  Replaying anything else here would
        # optimize the policy against a context its supervision never carried.
        template.system_message = SYSTEM_MESSAGE_BY_MODALITY[modality] if system_message is None else system_message
        template.append_message(template.roles[0], prompt)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        for grid in grid_hw:
            context_tokens = int(grid[0, 0] * grid[0, 1] * float(self.model.downsample_ratio) ** 2)
            image_span = "<img>" + "<IMG_CONTEXT>" * context_tokens + "</img>"
            query = query.replace("<image>", image_span, 1)

        pixels_tensor = torch.cat(pixel_values) if pixel_values else None
        grid_tensor = torch.cat(grid_hw) if grid_hw else None
        # The U1.5 builder performs a direct embedding lookup before it
        # enters the language-model backbone. Gather that FSDP2 execution root
        # before the lookup; its following forward owns the backward hooks and
        # keeps the small non-block parameter group gathered through backward.
        unshard_backbone = getattr(self.model.language_model.model, "unshard", None)
        if not callable(unshard_backbone):
            raise TypeError("FSDP2 language-model backbone has no unshard operation")
        unshard_backbone()
        inputs_embeds, indexes, attention_mask = self.model._build_it2i_inputs(
            self.tokenizer,
            query,
            pixels_tensor,
            grid_tensor,
        )
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            indexes=indexes,
            attention_mask=attention_mask,
            use_cache=True,
        )
        self.cache = outputs.past_key_values
        self._register_cache_pre_backward_hooks()
        self.t_index = int(indexes[0].max().item())
        self.next_logits = outputs.logits[:, -1, :]

    def _register_cache_pre_backward_hooks(self) -> None:
        """Expose custom-cache K/V branches to their owning FSDP blocks.

        Transformers mutates and returns per-layer KV through a custom
        ``Cache`` object. FSDP2's decoder forward hook only sees the ordinary
        hidden-state return, so a later TI2TI image loss can otherwise reach a
        saved K/V projection after its full parameter storage was resharded.
        Registering the same decoder pre-backward hook on the latest cache K/V
        makes that exact differentiable prefix branch participate in FSDP's
        normal all-gather/reduce-scatter lifecycle.
        """

        if not torch.is_grad_enabled():
            return
        from torch.distributed.fsdp import fully_shard

        decoder_layers = tuple(self.model.language_model.model.layers)
        cache_layers = tuple(self.cache.layers)
        if len(cache_layers) != len(decoder_layers):
            raise RuntimeError("U1.5 decoder/cache layer counts disagree during replay")
        for decoder, cache_layer in zip(decoder_layers, cache_layers, strict=True):
            tensors = tuple(
                value
                for value in (
                    getattr(cache_layer, "keys", None),
                    getattr(cache_layer, "values", None),
                )
                if isinstance(value, Tensor) and value.requires_grad
            )
            if tensors:
                fully_shard.state(decoder)._register_pre_backward_hook(tensors)

    def validate_capacity(self) -> None:
        prompt_tokens = int(self.cache.get_seq_length())
        plan_limit = int(self.runtime.plan.max_sequence_length)
        limit = int(self.model.language_model.config.max_position_embeddings)
        if plan_limit > limit:
            raise ValueError(f"RL max_sequence_length={plan_limit} exceeds model context {limit}")
        if prompt_tokens >= plan_limit:
            raise ValueError(f"U1.5 prompt needs {prompt_tokens} cache tokens, RL limit is {plan_limit}")

    def generated_image_context_tokens(self) -> int:
        size = int(self.runtime.plan.image_size)
        merge_size = int(1 / float(self.model.downsample_ratio))
        image_tokens = (size // (int(self.model.patch_size) * merge_size)) ** 2
        return image_tokens + 1

    def constrained_logits(self, *, allow_image: bool = True) -> Tensor:
        logits = self.next_logits.float()
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise RuntimeError(f"U1.5 returned invalid next-token logits {tuple(logits.shape)}")
        if self.modality == "ti2t" or not allow_image:
            logits[:, self.img_start_id] = torch.finfo(logits.dtype).min
        return logits

    def advance(self, token_ids: Tensor, accepted: Tensor, *, allow_image: bool = True) -> Tensor:
        self.commit(token_ids, accepted)
        return self.constrained_logits(allow_image=allow_image)

    def commit(self, token_ids: Tensor, accepted: Tensor | None = None) -> None:
        """Append one action without materialising next-step constraints.

        Evaluation already asks for constrained logits at the start of the
        next decode iteration. Its commit-only path avoids an otherwise
        redundant FP32 vocabulary copy, reduction, and host synchronization.
        RLVR continues to call :meth:`advance` because its sampler consumes the
        returned next-step distribution immediately.
        """

        if token_ids.shape != (1,):
            raise ValueError("U1.5 session advances one token at a time")
        if accepted is not None:
            if accepted.shape != (1,):
                raise ValueError("U1.5 session advances one token at a time")
            if not bool(accepted[0]):
                return
        self.model.language_model.model.current_index = self.t_index
        outputs = self.model.language_model(
            input_ids=token_ids.reshape(1, 1),
            past_key_values=self.cache,
            use_cache=True,
        )
        self.cache = outputs.past_key_values
        self._register_cache_pre_backward_hooks()
        commit_flash = getattr(self.cache, "commit_flash_decode", None)
        if callable(commit_flash):
            commit_flash(1)
        self.t_index += 1
        self.next_logits = outputs.logits[:, -1, :]

    def _forward_replayed_text(
        self,
        event: TextEvent,
        *,
        logits_to_keep: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Teacher-force one sampled span and advance the differentiable KV."""

        tokens = _single_row_tokens(event.trace)
        if not tokens:
            raise ValueError("U1.5 text event contains no sampled action")
        token_ids = torch.tensor(tokens, dtype=torch.long, device=self.device).reshape(1, -1)
        count = int(token_ids.shape[1])
        temporal = torch.arange(
            self.t_index + 1,
            self.t_index + 1 + count,
            dtype=torch.long,
            device=self.device,
        )
        indexes = torch.stack(
            (temporal, torch.zeros_like(temporal), torch.zeros_like(temporal)),
            dim=0,
        )
        # FlashAttention applies a bottom-right causal mask when q_len is
        # shorter than prefix + q_len.  The SDPA/eager fallback has no such
        # implicit geometry: passing ``None`` there would let replay token i
        # attend to later sampled tokens, so old/current could agree with each
        # other while no longer representing the autoregressive behavior
        # policy.  Materialize only the small replay-span mask on that debug
        # path; the formal flash path remains allocation-free.
        replay_mask = None
        if self.runtime.plan.attention_backend != "flash":
            past_length = self.cache.get_seq_length()
            replay_mask = torch.zeros(
                1,
                1,
                count,
                past_length + count,
                device=self.device,
                dtype=torch.float32,
            )
            replay_mask[..., past_length:] = torch.triu(
                torch.full(
                    (count, count),
                    -torch.inf,
                    device=self.device,
                    dtype=torch.float32,
                ),
                diagonal=1,
            )
        outputs = self.model.language_model(
            input_ids=token_ids,
            indexes=indexes,
            attention_mask={"full_attention": replay_mask},
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=logits_to_keep,
        )
        self.cache = outputs.past_key_values
        self._register_cache_pre_backward_hooks()
        self.t_index += count
        self.next_logits = outputs.logits[:, -1, :]
        return token_ids, outputs.logits

    def advance_replayed_text(self, event: TextEvent) -> None:
        """Advance an exact text prefix without building an unused token loss."""

        self._forward_replayed_text(event, logits_to_keep=1)

    def replay_text(self, event: TextEvent) -> Tensor:
        first_logits = self.constrained_logits()
        token_ids, replay_logits = self._forward_replayed_text(event)
        first = torch.log_softmax(first_logits, dim=-1).gather(-1, token_ids[:, :1]).reshape(1, 1)
        following = _checkpointed_selected_log_probs(
            replay_logits[:, :-1, :],
            token_ids[:, 1:],
            forbidden_token_id=(self.img_start_id if self.modality == "ti2t" else None),
        )
        current = torch.cat((first, following), dim=1)
        old = event.trace.old_log_probs[event.trace.response_mask].reshape(1, -1)
        self._last_replay_numeric_error = _ratio_error(current, old)
        return current

    def _image_geometry(
        self,
        *,
        image_height: int | None = None,
        image_width: int | None = None,
    ) -> tuple[int, int, int, int, Tensor, float, Tensor]:
        height = int(image_height or self.runtime.plan.image_size)
        width = int(image_width or self.runtime.plan.image_size)
        merge_size = int(1 / float(self.model.downsample_ratio))
        token_h = height // (int(self.model.patch_size) * merge_size)
        token_w = width // (int(self.model.patch_size) * merge_size)
        grid_h = height // int(self.model.patch_size)
        grid_w = width // int(self.model.patch_size)
        sequence_length = token_h * token_w
        indexes = self.model._build_t2i_image_indexes(
            token_h,
            token_w,
            self.t_index + 1,
            device=self.device,
        )
        noise_scale = float(self.model.noise_scale)
        if self.model.noise_scale_mode in {"resolution", "dynamic", "dynamic_sqrt"}:
            base = float(self.model.noise_scale_base_image_seq_len)
            noise_scale = math.sqrt(sequence_length / base) * float(self.model.noise_scale)
            if self.model.noise_scale_mode == "dynamic_sqrt":
                noise_scale = math.sqrt(noise_scale)
        noise_scale = min(noise_scale, float(self.model.noise_scale_max_value))
        schedule = torch.linspace(
            0.0,
            1.0,
            self.runtime.plan.image_steps + 1,
            device=self.device,
        )
        schedule = self.model._apply_time_schedule(
            schedule,
            sequence_length,
            self.runtime.plan.timestep_shift,
        )
        return token_h, token_w, grid_h, grid_w, indexes, noise_scale, schedule

    def _velocity_predictor(
        self,
        *,
        grid_h: int,
        grid_w: int,
        indexes: Tensor,
        noise_scale: float,
        image_height: int | None = None,
        image_width: int | None = None,
        replay_cache: Cache | None = None,
    ):
        merge_size = int(1 / float(self.model.downsample_ratio))
        patch_size = int(self.model.patch_size)
        height = int(image_height or self.runtime.plan.image_size)
        width = int(image_width or self.runtime.plan.image_size)
        sequence_length = (height // (patch_size * merge_size)) * (width // (patch_size * merge_size))

        def run_velocity(sample_model: Tensor, timestep_values: Tensor) -> Tensor:
            batch = int(sample_model.shape[0])
            pixels = self.model.unpatchify(
                sample_model,
                patch_size * merge_size,
                height,
                width,
            )
            image_input = self.model.patchify(
                pixels,
                patch_size,
                channel_first=True,
            )
            grid = torch.tensor([[grid_h, grid_w]], device=self.device).expand(batch, -1)
            image_embeds = self.model.extract_feature(
                image_input.view(batch * grid_h * grid_w, -1),
                gen_model=True,
                grid_hw=grid,
            ).view(batch, sequence_length, -1)
            expanded = timestep_values.unsqueeze(1).expand(batch, sequence_length).reshape(-1)
            timestep_embeddings = self.model.fm_modules["timestep_embedder"](expanded).view(batch, sequence_length, -1)
            if self.model.add_noise_scale_embedding:
                scale = torch.full_like(
                    expanded,
                    noise_scale / float(self.model.noise_scale_max_value),
                )
                timestep_embeddings = timestep_embeddings + self.model.fm_modules["noise_scale_embedder"](scale).view(
                    batch, sequence_length, -1
                )
            image_embeds = image_embeds + timestep_embeddings
            source_cache = self.cache if replay_cache is None else replay_cache
            current_cache = source_cache
            if batch > 1:
                current_cache = _ExpandedReadOnlyCache.from_cache(source_cache, batch=batch)
            velocity = self.model._t2i_predict_v(
                image_embeds,
                indexes,
                {"full_attention": None},
                current_cache,
                timestep_values.view(batch, 1, 1),
                sample_model,
                image_token_num=sequence_length,
                timestep_embeddings=timestep_embeddings,
                image_size=(height, width),
            )
            if velocity.shape != sample_model.shape:
                raise RuntimeError(
                    f"U1.5 velocity shape {tuple(velocity.shape)} != latent shape {tuple(sample_model.shape)}"
                )
            return velocity.float()

        def predict(sample: Tensor, t: Tensor, _index: int) -> Tensor:
            sample_model = sample.to(dtype=self.dtype)
            batch = int(sample_model.shape[0])
            timestep_values = torch.as_tensor(t, device=self.device, dtype=torch.float32).reshape(-1)
            if timestep_values.numel() == 1:
                timestep_values = timestep_values.expand(batch)
            if timestep_values.numel() != batch:
                raise ValueError("U1.5 velocity timestep batch does not match latent batch")
            return run_velocity(sample_model, timestep_values)

        return predict

    def _prepare_image_cache(self, sequence_length: int) -> None:
        helper = getattr(self.model_module, "prepare_flash_kv_cache", None)
        if not callable(helper):
            raise TypeError("U1.5 module has no flash-cache preparation helper")
        helper(self.cache, current_len=sequence_length, batch_size=1)

    def _clear_image_cache(self) -> None:
        helper = getattr(self.model_module, "clear_flash_kv_cache", None)
        if not callable(helper):
            raise TypeError("U1.5 module has no flash-cache cleanup helper")
        helper(self.cache)

    def _append_generated_image(
        self,
        final_latent: Tensor,
        *,
        image_height: int | None = None,
        image_width: int | None = None,
    ) -> Tensor:
        height = int(image_height or self.runtime.plan.image_size)
        width = int(image_width or self.runtime.plan.image_size)
        patch_size = int(self.model.patch_size)
        merge_size = int(1 / float(self.model.downsample_ratio))
        pixels = self.model.unpatchify(
            final_latent.to(dtype=self.dtype),
            patch_size * merge_size,
            height,
            width,
        )
        raw = pixels * 0.5 + 0.5
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=raw.dtype, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=raw.dtype, device=self.device).view(1, 3, 1, 1)
        normalized = (raw - mean) / std
        channels, height, width = normalized[0].shape
        grid_h = height // patch_size
        grid_w = width // patch_size
        flattened = (
            normalized[0]
            .view(channels, grid_h, patch_size, grid_w, patch_size)
            .permute(1, 3, 0, 2, 4)
            .reshape(grid_h * grid_w, channels * patch_size**2)
        )
        grid = torch.tensor([[grid_h, grid_w]], device=self.device)
        embeddings = self.model.extract_feature(
            flattened,
            grid_hw=grid,
        ).unsqueeze(0)
        # A streamed image-action backward reshares the persistent language
        # root before the next generated image is appended. Directly invoking
        # its embedding leaf in that state returns a DTensor, while the vision
        # encoder output above is an ordinary replicated Tensor. Explicitly
        # restore the root's execution representation before that leaf call;
        # the following language-model forward keeps it gathered through the
        # next backward boundary as usual.
        embedding = self.model.language_model.get_input_embeddings()
        if isinstance(embedding.weight, DTensor):
            language_backbone = self.model.language_model.model
            unshard = getattr(language_backbone, "unshard", None)
            if not callable(unshard):
                raise RuntimeError("FSDP language root cannot be unsharded for image replay")
            unshard(async_op=False)
            embedding = self.model.language_model.get_input_embeddings()
        image_end = embedding(torch.tensor([[self.runtime.img_end_id]], device=self.device))
        inputs = torch.cat([embeddings, image_end], dim=1)
        image_tokens = int(embeddings.shape[1])
        position_helper = getattr(self.model_module, "build_abs_positions_from_grid_hw", None)
        if not callable(position_helper):
            raise TypeError("U1.5 module has no image-position helper")
        absolute_w, absolute_h = position_helper(
            grid // merge_size,
            device=self.device,
        )
        target_length = image_tokens + 1
        t_indexes = torch.zeros(target_length, dtype=torch.long, device=self.device)
        t_indexes[:image_tokens] = self.t_index + 1
        t_indexes[image_tokens] = self.t_index + 2
        h_indexes = torch.zeros_like(t_indexes)
        w_indexes = torch.zeros_like(t_indexes)
        h_indexes[:image_tokens] = absolute_h
        w_indexes[:image_tokens] = absolute_w
        indexes = torch.stack([t_indexes, h_indexes, w_indexes], dim=0)
        past_length = self.cache.get_seq_length()
        mask = torch.zeros(
            1,
            1,
            target_length,
            past_length + target_length,
            device=self.device,
        )
        mask[0, 0, :image_tokens, past_length + image_tokens] = -torch.inf
        outputs = self.model.language_model(
            inputs_embeds=inputs,
            indexes=indexes,
            attention_mask={"full_attention": mask},
            past_key_values=self.cache,
            use_cache=True,
        )
        self.cache = outputs.past_key_values
        self._register_cache_pre_backward_hooks()
        sync_flash = getattr(self.cache, "sync_flash_decode_length", None)
        if callable(sync_flash):
            sync_flash()
        self.t_index += 2
        self.next_logits = outputs.logits[:, -1, :]
        return pixels

    def sample_image(self, *, generator: torch.Generator) -> tuple[Tensor, ImageSdeTrace]:
        token_h, token_w, grid_h, grid_w, indexes, noise_scale, schedule = self._image_geometry()
        sequence_length = token_h * token_w
        size = self.runtime.plan.image_size
        noise = noise_scale * torch.randn(
            (1, 3, size, size),
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        latent = self.model.patchify(
            noise,
            int(self.model.patch_size) * int(1 / float(self.model.downsample_ratio)),
        )
        predictor = self._velocity_predictor(
            grid_h=grid_h,
            grid_w=grid_w,
            indexes=indexes,
            noise_scale=noise_scale,
        )
        self._prepare_image_cache(sequence_length)
        try:
            final_latent, trace = sample_image_sde(
                initial_sample=latent,
                timesteps=schedule,
                predict_velocity=predictor,
                sde_indices=select_sde_indices(
                    total_steps=self.runtime.plan.image_steps,
                    window_start=self.runtime.plan.sde_window_start,
                    window_end=self.runtime.plan.sde_window_end,
                    selected_steps=self.runtime.plan.sde_window_steps,
                ),
                noise_level=self.runtime.plan.image_noise_level,
                generator=generator,
            )
        finally:
            self._clear_image_cache()
        pixels = self._append_generated_image(final_latent)
        return pixels, trace

    def _validated_image_replay_geometry(self, event: ImageEvent) -> tuple[int, int, Tensor, float, int, int]:
        height = int(event.trace.image_height or self.runtime.plan.image_size)
        width = int(event.trace.image_width or self.runtime.plan.image_size)
        _token_h, _token_w, grid_h, grid_w, indexes, noise_scale, schedule = self._image_geometry(
            image_height=height, image_width=width
        )
        if not torch.equal(
            schedule[event.trace.sde_indices].float(),
            event.trace.timesteps.float(),
        ) or not torch.equal(
            schedule[event.trace.sde_indices + 1].float(),
            event.trace.next_timesteps.float(),
        ):
            raise ValueError("U1.5 timestep schedule changed during replay")
        return grid_h, grid_w, indexes, noise_scale, height, width

    def replay_reference_image(self, event: ImageEvent) -> Tensor:
        """Return frozen-reference velocities and advance its interleaved state."""

        grid_h, grid_w, indexes, noise_scale, height, width = self._validated_image_replay_geometry(event)
        cache_snapshot = _ExpandedReadOnlyCache.from_cache(self.cache, batch=1)
        predictor = self._velocity_predictor(
            grid_h=grid_h,
            grid_w=grid_w,
            indexes=indexes,
            noise_scale=noise_scale,
            image_height=height,
            image_width=width,
            replay_cache=cache_snapshot,
        )
        replay = replay_image_sde_batched(
            event.trace,
            predict_velocity=predictor,
            microbatch_size=self.runtime.plan.image_replay_microbatch_size,
        )
        self._append_generated_image(
            event.trace.final_sample.detach(),
            image_height=height,
            image_width=width,
        )
        return replay.velocities.detach()

    def advance_replayed_image(self, event: ImageEvent) -> None:
        """Append a fixed rollout image without evaluating its SDE policy."""

        _grid_h, _grid_w, _indexes, _noise_scale, height, width = self._validated_image_replay_geometry(event)
        self._append_generated_image(
            event.trace.final_sample.detach(),
            image_height=height,
            image_width=width,
        )

    def replay_image(self, event: ImageEvent, *, ref_velocities: Tensor | None) -> ImageSdeReplay:
        grid_h, grid_w, indexes, noise_scale, height, width = self._validated_image_replay_geometry(event)
        cache_snapshot = _ExpandedReadOnlyCache.from_cache(self.cache, batch=1)
        predictor = self._velocity_predictor(
            grid_h=grid_h,
            grid_w=grid_w,
            indexes=indexes,
            noise_scale=noise_scale,
            image_height=height,
            image_width=width,
            replay_cache=cache_snapshot,
        )
        # The U1.5 preallocated flash cache is inference-only: every
        # denoising forward overwrites the same current-token slice in place.
        # Differentiable replay retains all step graphs until backward, so it
        # must use the U1.5 non-preallocated torch.cat fallback instead.
        replay = replay_image_sde_batched(
            event.trace,
            predict_velocity=predictor,
            microbatch_size=self.runtime.plan.image_replay_microbatch_size,
        )
        if ref_velocities is not None:
            ref_velocities = ref_velocities.to(
                device=replay.velocities.device,
                dtype=replay.velocities.dtype,
            )
            if ref_velocities.shape != replay.velocities.shape:
                raise RuntimeError("reference/current image velocity layouts disagree")
            replay = replace(
                replay,
                ref_velocities=ref_velocities.detach(),
            )
        self._last_replay_numeric_error = _ratio_error(replay.log_probs, event.trace.old_log_probs)
        self._append_generated_image(
            event.trace.final_sample.detach(),
            image_height=height,
            image_width=width,
        )
        return replay

    def replay_image_action_slice(
        self,
        event: ImageEvent,
        *,
        action_start: int,
        action_stop: int,
        ref_velocities: Tensor | None,
    ) -> ImageSdeReplay:
        """Replay one bounded SDE-action slice against this prefix cache."""

        trace = slice_image_sde_trace(
            event.trace,
            action_start=action_start,
            action_stop=action_stop,
        )
        grid_h, grid_w, indexes, noise_scale, height, width = self._validated_image_replay_geometry(event)
        cache_snapshot = _ExpandedReadOnlyCache.from_cache(self.cache, batch=1)
        predictor = self._velocity_predictor(
            grid_h=grid_h,
            grid_w=grid_w,
            indexes=indexes,
            noise_scale=noise_scale,
            image_height=height,
            image_width=width,
            replay_cache=cache_snapshot,
        )
        replay = replay_image_sde_batched(
            trace,
            predict_velocity=predictor,
        )
        if ref_velocities is not None:
            selected_reference = ref_velocities[action_start:action_stop].to(
                device=replay.velocities.device,
                dtype=replay.velocities.dtype,
            )
            if selected_reference.shape != replay.velocities.shape:
                raise RuntimeError("reference/current image action-slice layouts disagree")
            replay = replace(
                replay,
                ref_velocities=selected_reference.detach(),
            )
        self._last_replay_numeric_error = _ratio_error(replay.log_probs, trace.old_log_probs)
        return replay


class U15PolicyRuntime:
    """FSDP-sharded full-parameter policy backed by published U1.5."""

    def __init__(
        self,
        *,
        plan: RlPlan,
        model: Any,
        core_model: Any,
        tokenizer: Any,
        model_module: Any,
        reference_parameter_shards: dict[str, Tensor],
        compute_dtype: torch.dtype,
    ) -> None:
        self.plan = plan
        self.model = model
        self.core_model = core_model
        self.tokenizer = tokenizer
        self.model_module = model_module
        self.reference_parameter_shards = reference_parameter_shards
        self.device = next(model.parameters()).device
        self.dtype = compute_dtype
        self.img_start_id = int(tokenizer.convert_tokens_to_ids("<img>"))
        self.img_context_id = int(tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>"))
        self.img_end_id = int(tokenizer.convert_tokens_to_ids("</img>"))
        template = model_module.get_conv_template(core_model.template)
        self.eos_id = int(tokenizer.convert_tokens_to_ids(template.sep.strip()))
        self.pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos_id)
        for label, token_id in (
            ("image start", self.img_start_id),
            ("image context", self.img_context_id),
            ("image end", self.img_end_id),
            ("EOS", self.eos_id),
            ("padding", self.pad_id),
        ):
            if token_id < 0:
                raise ValueError(f"U1.5 tokenizer has no {label} token")

    @contextmanager
    def reference_parameters(self):
        """Expose the frozen SFT policy for no-grad reference forwards.

        Every local FSDP shard is temporarily restored to the SFT initialization.
        Live current-policy shards are copied back before any differentiable
        current forward can run.
        """

        parameter_names = set(dict(self.model.named_parameters()))
        if parameter_names != set(self.reference_parameter_shards):
            raise RuntimeError("U1.5 full reference parameter closure changed")
        # ``language_model.model`` deliberately keeps parameters gathered
        # after forward.  Reference snapshots are local shards, so establish
        # the sharded representation before both the swap and the restore.
        reshard_full_parameter_policy(self.model)
        # FSDP2 re-registers sharded parameters when it releases a gathered
        # view. Never retain pre-reshard Parameter objects: their local storage
        # is no longer a valid shard after this lifecycle transition.
        named = dict(self.model.named_parameters())
        if set(named) != parameter_names:
            raise RuntimeError("U1.5 full reference parameter closure changed while resharding")
        live = {name: local_parameter_view(parameter).detach().clone() for name, parameter in named.items()}
        try:
            with torch.no_grad():
                for name, parameter in named.items():
                    local = local_parameter_view(parameter)
                    local.copy_(self.reference_parameter_shards[name].to(dtype=local.dtype))
            yield
        finally:
            with torch.no_grad():
                reshard_full_parameter_policy(self.model)
                named = dict(self.model.named_parameters())
                if set(named) != parameter_names:
                    raise RuntimeError("U1.5 full reference parameter closure changed while restoring")
                for name, parameter in named.items():
                    local_parameter_view(parameter).copy_(live[name])

    @classmethod
    def load(
        cls,
        plan: RlPlan,
        *,
        device: str | torch.device | None = None,
    ) -> U15PolicyRuntime:
        import sensenova_u1
        from sensenova_u1.utils import load_model_and_tokenizer

        sensenova_u1.register_models()
        attention = importlib.import_module("sensenova_u1.models.neo_unify.modeling_qwen3")
        setter = getattr(attention, "set_attn_backend", None)
        effective = getattr(attention, "effective_attn_backend", None)
        if not callable(setter) or not callable(effective):
            raise TypeError("U1.5 runtime has no attention backend control")
        setter(plan.attention_backend)
        if effective() != plan.attention_backend:
            raise RuntimeError("U1.5 attention backend differs from the plan")

        dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[plan.dtype]
        runtime_device = torch.device(device if device is not None else plan.device)
        if runtime_device.type != "cuda":
            raise ValueError("U1.5 full-parameter policy requires CUDA")
        base_model, tokenizer = load_model_and_tokenizer(
            model_path=str(plan.policy_init),
            dtype=dtype,
            device=runtime_device,
        )
        validate_policy_model(base_model)
        base_model.img_context_token_id = int(tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>"))
        base_model.img_start_token_id = int(tokenizer.convert_tokens_to_ids("<img>"))
        base_model.config.t_eps = plan.t_eps
        model_module = importlib.import_module(type(base_model).__module__)

        base_model.requires_grad_(True)
        # RLVR uses gradients in eval mode: stochastic layer state would break
        # the frozen-old/current likelihood contract across repeated replays.
        base_model.eval()
        shard_full_parameter_policy(
            base_model,
            compute_dtype=dtype,
            master_dtype=torch.float32,
            activation_checkpointing=plan.activation_checkpointing,
        )
        needs_reference = bool(plan.text_kl_beta or (plan.modality == "ti2ti" and plan.velocity_mse_weight))
        reference_parameter_shards = snapshot_reference_shards(base_model) if needs_reference else {}
        validate_policy_model(base_model)
        _validate_full_parameter_closure(base_model)
        return cls(
            plan=plan,
            model=base_model,
            core_model=base_model,
            tokenizer=tokenizer,
            model_module=model_module,
            reference_parameter_shards=reference_parameter_shards,
            compute_dtype=dtype,
        )

    def rollout(
        self,
        *,
        prompt: str,
        prompt_images: tuple[str, ...],
        modality: str,
        system_message: str | None = None,
        artifact_dir: Path,
        rollout_key: str,
        generator: torch.Generator,
    ) -> U15PolicyRollout:
        started = time.perf_counter()
        events: list[PolicyEvent] = []
        items: list[TextSegment | ImageSegment] = []
        text_tokens = 0
        image_count = 0
        image_context_tokens = 0
        finish_reason = "length"
        text_only_tail = modality == "ti2t" or self.plan.max_images == 0
        with torch.no_grad():
            session = _PolicySession(
                self,
                prompt=prompt,
                prompt_images=prompt_images,
                modality=modality,
                system_message=system_message,
            )
            session.validate_capacity()
            # FSDP rollout uses the same ordinary collective model path as
            # replay; CUDA-graph capture would bypass parameter gather hooks.
            while int(session.cache.get_seq_length()) < self.plan.max_sequence_length:
                context_tokens = int(session.cache.get_seq_length())
                can_generate_image = modality == "ti2ti" and not text_only_tail
                image_reserve = session.generated_image_context_tokens() + 1 if can_generate_image else 0
                span_tokens = self.plan.max_sequence_length - context_tokens - image_reserve
                if span_tokens < 1:
                    text_only_tail = True
                    can_generate_image = False
                    image_reserve = 0
                    span_tokens = self.plan.max_sequence_length - context_tokens
                if span_tokens < 1:
                    break
                stops = (self.eos_id,)
                if can_generate_image:
                    stops = (self.eos_id, self.img_start_id)
                trace = sample_text_span(
                    initial_logits=session.constrained_logits(allow_image=can_generate_image),
                    advance=lambda token_ids, accepted, allow_image=can_generate_image: session.advance(
                        token_ids,
                        accepted,
                        allow_image=allow_image,
                    ),
                    stop_token_ids=stops,
                    max_new_tokens=span_tokens,
                    pad_token_id=self.pad_id,
                    generator=generator,
                )
                tokens = _single_row_tokens(trace)
                stop = tokens[-1] if trace.stopped.item() else None
                text = _decode_without_stop(self.tokenizer, trace, stop)
                if text:
                    items.append(TextSegment(text))
                text_tokens += len(tokens)
                events.append(TextEvent(trace=trace, stop_token_id=stop))
                if stop != self.img_start_id:
                    if (
                        stop is None
                        and image_reserve
                        and int(session.cache.get_seq_length()) < self.plan.max_sequence_length
                    ):
                        text_only_tail = True
                        continue
                    if stop is not None:
                        finish_reason = "stop"
                    break
                if not can_generate_image:
                    raise RuntimeError("U1.5 sampled a masked image action")
                image_start_length = int(session.cache.get_seq_length())
                pixels, image_trace = session.sample_image(generator=generator)
                image_context_tokens += int(session.cache.get_seq_length()) - image_start_length
                image_name = f"{rollout_key}-image-{image_count:02d}.png"
                image_path = artifact_dir / image_name
                self._save_png(pixels, image_path)
                items.append(ImageSegment(str(image_path.resolve())))
                events.append(ImageEvent(trace=image_trace))
                image_count += 1
                if int(session.cache.get_seq_length()) > self.plan.max_sequence_length:
                    raise RuntimeError("U1.5 image action exceeded max_sequence_length")
                if image_count >= self.plan.max_images:
                    text_only_tail = True
                # A reasoning image never ends the response: the policy must
                # still close its think block and emit the Answer line, so the
                # loop continues until EOS or the token budget stops it.
        if not events:
            raise RuntimeError("U1.5 rollout emitted no policy action")
        if int(session.cache.get_seq_length()) > self.plan.max_sequence_length:
            raise RuntimeError("U1.5 rollout violated max_sequence_length")
        return U15PolicyRollout(
            candidate=CandidateResponse(modality=modality, items=tuple(items)),
            events=tuple(events),
            text_tokens=text_tokens,
            generated_images=image_count,
            image_context_tokens=image_context_tokens,
            seconds=time.perf_counter() - started,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _save_png(pixels: Tensor, path: Path) -> None:
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        if pixels.shape[0] != 1 or pixels.shape[1] != 3:
            raise ValueError(f"U1.5 generated invalid image tensor {tuple(pixels.shape)}")
        array = (
            ((pixels[0].detach().float().clamp(-1, 1) * 0.5 + 0.5) * 255.0)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        Image.fromarray(array, mode="RGB").save(path, format="PNG")

    def replay(
        self,
        *,
        prompt: str,
        prompt_images: tuple[str, ...],
        modality: str,
        system_message: str | None = None,
        rollout: U15PolicyRollout,
        include_reference: bool = True,
        include_image_policy: bool = True,
    ) -> U15PolicyReplay:
        # Build the frozen SFT reference before retaining any current-policy
        # graph.  Swapping reference flow parameters after the current forward
        # would invalidate autograd parameter version counters.
        if type(include_reference) is not bool:
            raise TypeError("include_reference must be a boolean")
        if type(include_image_policy) is not bool:
            raise TypeError("include_image_policy must be a boolean")
        reference_spans: list[Tensor] = []
        reference_images: list[Tensor] = []
        if include_reference:
            with torch.no_grad(), self.reference_parameters():
                reference_session = _PolicySession(
                    self,
                    prompt=prompt,
                    prompt_images=prompt_images,
                    modality=modality,
                    system_message=system_message,
                )
                for event in rollout.events:
                    if isinstance(event, TextEvent):
                        reference_spans.append(reference_session.replay_text(event))
                    elif isinstance(event, ImageEvent):
                        reference_images.append(reference_session.replay_reference_image(event))
                    else:  # pragma: no cover - frozen policy-event union is closed
                        raise TypeError(f"unknown U1.5 policy event {type(event).__name__}")

        session = _PolicySession(
            self,
            prompt=prompt,
            prompt_images=prompt_images,
            modality=modality,
            system_message=system_message,
        )
        spans: list[TextRolloutTrace] = []
        current_spans: list[Tensor] = []
        images: list[ImageSdeReplay] = []
        errors: list[float] = []
        ratio_error_sum = 0.0
        ratio_action_count = 0
        numeric_errors: list[float] = []
        reference_image_index = 0
        for event in rollout.events:
            if isinstance(event, TextEvent):
                current = session.replay_text(event)
                spans.append(event.trace)
                current_spans.append(current)
                old = event.trace.old_log_probs[event.trace.response_mask].reshape(1, -1)
                maximum, total, count = _ratio_error_stats(current, old)
                errors.append(maximum)
                ratio_error_sum += total
                ratio_action_count += count
                numeric_errors.append(session._last_replay_numeric_error)
            elif isinstance(event, ImageEvent):
                if include_image_policy:
                    replay = session.replay_image(
                        event,
                        ref_velocities=(reference_images[reference_image_index] if include_reference else None),
                    )
                    images.append(replay)
                    maximum, total, count = _ratio_error_stats(replay.log_probs, event.trace.old_log_probs)
                    errors.append(maximum)
                    ratio_error_sum += total
                    ratio_action_count += count
                    numeric_errors.append(session._last_replay_numeric_error)
                else:
                    session.advance_replayed_image(event)
                if include_reference:
                    reference_image_index += 1
            else:  # pragma: no cover - frozen union is closed
                raise TypeError(f"unknown U1.5 policy event {type(event).__name__}")
        if not spans:
            raise ValueError("U1.5 rollout has no text policy trace")
        text_trace = concatenate_text_spans(spans, pad_token_id=self.pad_id)
        text_log_probs = torch.cat(current_spans, dim=1)
        if text_log_probs.shape != text_trace.old_log_probs.shape:
            raise RuntimeError("replayed U1.5 text trace changed action layout")

        text_ref_log_probs = torch.cat(reference_spans, dim=1) if include_reference else None
        if text_ref_log_probs is not None and text_ref_log_probs.shape != text_log_probs.shape:
            raise RuntimeError("reference U1.5 replay changed text action layout")
        if include_reference and reference_image_index != len(reference_images):
            raise RuntimeError("reference U1.5 replay changed image event layout")
        return U15PolicyReplay(
            text_trace=text_trace,
            text_log_probs=text_log_probs,
            text_ref_log_probs=text_ref_log_probs,
            image_replays=tuple(images),
            image_reference_velocities=tuple(reference_images),
            ratio_max_error=max(errors, default=0.0),
            ratio_mean_error=(ratio_error_sum / ratio_action_count if ratio_action_count else 0.0),
            ratio_action_count=ratio_action_count,
            numeric_max_error=max(numeric_errors, default=0.0),
        )

    def reference_replay(
        self,
        *,
        prompt: str,
        prompt_images: tuple[str, ...],
        modality: str,
        system_message: str | None = None,
        rollout: U15PolicyRollout,
        include_text: bool,
        include_images: bool,
    ) -> U15PolicyReferenceReplay:
        """Compute only the frozen-reference branches consumed by the loss."""

        if type(include_text) is not bool or type(include_images) is not bool:
            raise TypeError("reference replay branch selectors must be booleans")
        if not include_text and not include_images:
            return U15PolicyReferenceReplay((), ())
        text: list[Tensor] = []
        images: list[Tensor] = []
        with torch.no_grad(), self.reference_parameters():
            session = _PolicySession(
                self,
                prompt=prompt,
                prompt_images=prompt_images,
                modality=modality,
                system_message=system_message,
            )
            for event in rollout.events:
                if isinstance(event, TextEvent):
                    if include_text:
                        text.append(session.replay_text(event))
                    else:
                        session.advance_replayed_text(event)
                elif isinstance(event, ImageEvent):
                    if include_images:
                        images.append(session.replay_reference_image(event).cpu())
                    else:
                        session.advance_replayed_image(event)
                else:  # pragma: no cover - frozen policy-event union is closed
                    raise TypeError(f"unknown U1.5 policy event {type(event).__name__}")
        return U15PolicyReferenceReplay(tuple(text), tuple(images))

    def iter_text_replay_spans(
        self,
        *,
        prompt: str,
        prompt_images: tuple[str, ...],
        modality: str,
        system_message: str | None = None,
        rollout: U15PolicyRollout,
        reference_log_probs: tuple[Tensor, ...],
    ) -> Iterator[U15TextReplaySpan]:
        """Yield one exact text span so its loss can be backpropagated early."""

        text_event_count = sum(isinstance(event, TextEvent) for event in rollout.events)
        if not text_event_count:
            raise ValueError("U1.5 rollout has no text policy trace")
        if reference_log_probs and len(reference_log_probs) != text_event_count:
            raise ValueError("reference/current U1.5 text event counts disagree")
        session = _PolicySession(
            self,
            prompt=prompt,
            prompt_images=prompt_images,
            modality=modality,
            system_message=system_message,
        )
        text_index = 0
        for event in rollout.events:
            if isinstance(event, TextEvent):
                current = session.replay_text(event)
                yield U15TextReplaySpan(
                    trace=event.trace,
                    log_probs=current,
                    ref_log_probs=(reference_log_probs[text_index] if reference_log_probs else None),
                    numeric_max_error=session._last_replay_numeric_error,
                    is_last=text_index + 1 == text_event_count,
                )
                text_index += 1
            elif isinstance(event, ImageEvent):
                session.advance_replayed_image(event)
            else:  # pragma: no cover - frozen policy-event union is closed
                raise TypeError(f"unknown U1.5 policy event {type(event).__name__}")
        if text_index != text_event_count:
            raise RuntimeError("U1.5 text replay stopped before the final event")

    def iter_image_action_microbatches(
        self,
        *,
        prompt: str,
        prompt_images: tuple[str, ...],
        modality: str,
        system_message: str | None = None,
        rollout: U15PolicyRollout,
        microbatch_size: int,
        reference_velocities: tuple[Tensor, ...],
    ) -> Iterator[ImageSdeReplay]:
        """Replay every image action against one exact interleaved prefix.

        Text spans and earlier generated images advance one shared
        differentiable cache. Yielding one SDE action slice at a time lets the
        runner bound each backward to one image event while retaining the
        shared prefix graph only until the final image. This is an exact
        action-weighted trajectory objective, not truncated backpropagation or
        sample clipping.
        """

        if type(microbatch_size) is not int or microbatch_size < 1:
            raise ValueError("image replay microbatch_size must be positive")
        image_events = tuple(
            (index, event) for index, event in enumerate(rollout.events) if isinstance(event, ImageEvent)
        )
        if not image_events:
            if reference_velocities:
                raise ValueError("reference image velocities exist without image events")
            return
        if reference_velocities and len(reference_velocities) != len(image_events):
            raise ValueError("reference/current image event counts disagree")

        session = _PolicySession(
            self,
            prompt=prompt,
            prompt_images=prompt_images,
            modality=modality,
            system_message=system_message,
        )
        image_index = 0
        last_image_event_index = image_events[-1][0]
        for event_index, event in enumerate(rollout.events[: last_image_event_index + 1]):
            if isinstance(event, TextEvent):
                session.advance_replayed_text(event)
            elif isinstance(event, ImageEvent):
                reference = reference_velocities[image_index] if reference_velocities else None
                for action_start in range(0, event.trace.steps, microbatch_size):
                    action_stop = min(
                        action_start + microbatch_size,
                        event.trace.steps,
                    )
                    yield session.replay_image_action_slice(
                        event,
                        action_start=action_start,
                        action_stop=action_stop,
                        ref_velocities=reference,
                    )
                image_index += 1
                if event_index != last_image_event_index:
                    session.advance_replayed_image(event)
            else:  # pragma: no cover - frozen policy-event union is closed
                raise TypeError(f"unknown U1.5 policy event {type(event).__name__}")
        if image_index != len(image_events):
            raise RuntimeError("image replay stopped before the final image event")

    def anchor_rollout(
        self,
        *,
        prompt: str,
        prompt_images: tuple[str, ...],
        modality: str,
        system_message: str | None = None,
        rollout: U15PolicyRollout,
    ) -> U15PolicyRollout:
        """Freeze old-policy likelihoods in differentiable replay geometry."""

        return self.anchor_rollout_with_metrics(
            prompt=prompt,
            prompt_images=prompt_images,
            modality=modality,
            system_message=system_message,
            rollout=rollout,
        ).rollout

    def anchor_rollout_with_metrics(
        self,
        *,
        prompt: str,
        prompt_images: tuple[str, ...],
        modality: str,
        system_message: str | None = None,
        rollout: U15PolicyRollout,
    ) -> U15PolicyAnchor:
        """Anchor frozen-old likelihoods and retain behavior/replay drift."""

        with torch.no_grad():
            replay = self.replay(
                prompt=prompt,
                prompt_images=prompt_images,
                modality=modality,
                system_message=system_message,
                rollout=rollout,
                include_reference=False,
            )
        text_offset = 0
        image_offset = 0
        anchored_events: list[PolicyEvent] = []
        for event in rollout.events:
            if isinstance(event, TextEvent):
                trace = event.trace
                count = int(trace.response_mask.sum().item())
                selected = replay.text_log_probs[:, text_offset : text_offset + count]
                old = torch.zeros_like(trace.old_log_probs)
                old[trace.response_mask] = selected.reshape(-1).detach()
                anchored_events.append(replace(event, trace=replace(trace, old_log_probs=old)))
                text_offset += count
            elif isinstance(event, ImageEvent):
                image = replay.image_replays[image_offset]
                anchored_events.append(
                    replace(
                        event,
                        trace=replace(
                            event.trace,
                            old_log_probs=image.log_probs.detach(),
                            old_means=image.means.detach(),
                        ),
                    )
                )
                image_offset += 1
            else:  # pragma: no cover - frozen policy-event union is closed
                raise TypeError(f"unknown U1.5 policy event {type(event).__name__}")
        if text_offset != replay.text_log_probs.shape[1] or image_offset != len(replay.image_replays):
            raise RuntimeError("old-policy anchor did not consume the complete replay")
        return U15PolicyAnchor(
            rollout=replace(rollout, events=tuple(anchored_events)),
            behavior_ratio_max_error=replay.ratio_max_error,
            behavior_ratio_mean_error=replay.ratio_mean_error,
            behavior_ratio_action_count=replay.ratio_action_count,
            numeric_max_error=replay.numeric_max_error,
        )


__all__ = [
    "ImageEvent",
    "PolicyEvent",
    "TextEvent",
    "U15Policy",
    "U15PolicyAnchor",
    "U15PolicyRuntime",
    "U15PolicyReplay",
    "U15PolicyRollout",
]
