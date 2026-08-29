"""SenseNova-U1.5 rollout traces and the ODE-to-SDE policy boundary.

The public U1.5 sampler evolves flow time ``t`` from 0 (noise) to 1 (data),
whereas :mod:`sensenova_u1.rl.flow` follows the Flow-GRPO convention in which
``sigma`` descends from 1 to 0.  This module is the single adapter between the
two conventions: ``sigma = 1 - t`` and the U1.5 velocity is negated before it
is passed to the shared SDE algebra.

The functions here intentionally accept predictor callbacks instead of
copying the model implementation.  A runtime binding owns U1.5's KV
caches, pixel-head calls, CFG branches, patchify/unpatchify operations and
generated-image re-encoding.  Forge owns stochastic action sampling, the
stored old-policy likelihoods, and differentiable replay.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .flow import (
    ImageBranchInputs,
    SdeTransition,
    TextBranchInputs,
    sde_log_prob,
    sde_sample_step,
    sde_transition,
)

TextAdvance = Callable[[Tensor, Tensor], Tensor]
VelocityPredictor = Callable[[Tensor, Tensor, int], Tensor]


def select_sde_indices(*, total_steps: int, window_start: int, window_end: int, selected_steps: int) -> tuple[int, ...]:
    """Select deterministic, evenly spaced SDE actions inside ``[start, end)``."""

    if not 0 <= window_start < window_end <= total_steps:
        raise ValueError("SDE window must be a non-empty subset of total steps")
    width = window_end - window_start
    if not 1 <= selected_steps <= width:
        raise ValueError("selected SDE step count exceeds its window")
    if selected_steps == 1:
        return (window_start,)
    result = tuple(
        window_start + (position * (width - 1)) // (selected_steps - 1) for position in range(selected_steps)
    )
    if len(set(result)) != selected_steps:
        raise RuntimeError("SDE window selection produced duplicate steps")
    return result


def _require_float_tensor(tensor: Tensor, *, name: str, ndim: int | None = None) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")


def _prepare_u15_timesteps(timesteps: Tensor | Sequence[float], device: torch.device) -> Tensor:
    values = torch.as_tensor(timesteps, device=device, dtype=torch.float32)
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError("U1.5 timesteps must be a rank-1 sequence with at least two values")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("U1.5 timesteps contain non-finite values")
    if not torch.isclose(values[0], values.new_tensor(0.0), atol=1e-7, rtol=0.0):
        raise ValueError("U1.5 SDE rollout must start at flow time t=0")
    if bool((values < 0).any()) or bool((values > 1).any()):
        raise ValueError("U1.5 flow timesteps must lie in [0, 1]")
    if bool((values[1:] <= values[:-1]).any()):
        raise ValueError("U1.5 flow timesteps must be strictly increasing")
    if not bool(values[-1] == values.new_tensor(1.0)):
        raise ValueError("U1.5 SDE rollout must end at the data endpoint t=1")
    # The shared SDE formula substitutes the largest *interior* sigma at the
    # mandatory sigma==1 first step.  With sigma=1-t this is 1-t[1].
    if not 0.0 < float(1.0 - values[1]) < 1.0:
        raise ValueError("the first interior U1.5 timestep must lie strictly inside (0, 1)")
    return values


def u15_sde_transition(
    *,
    velocity: Tensor,
    sample: Tensor,
    t: Tensor | float,
    t_next: Tensor | float,
    sigma_max: float,
    noise_level: float,
) -> SdeTransition:
    """Map one U1.5 flow step onto the shared reverse-SDE formula.

    At ``noise_level=0`` the returned mean is exactly the model Euler step
    ``sample + (t_next - t) * velocity``.  A zero-noise transition has no
    density, so it is useful only for this deterministic identity; RLVR
    rollouts must use a positive noise level.
    """

    return sde_transition(
        model_output=-velocity,
        sample=sample,
        sigma=1.0 - torch.as_tensor(t, device=sample.device, dtype=torch.float32),
        sigma_prev=1.0 - torch.as_tensor(t_next, device=sample.device, dtype=torch.float32),
        sigma_max=sigma_max,
        noise_level=noise_level,
    )


@dataclass(frozen=True)
class TextRolloutTrace:
    """Padded sampled text actions and their old-policy token log-probabilities."""

    token_ids: Tensor
    old_log_probs: Tensor
    response_mask: Tensor
    stopped: Tensor

    def __post_init__(self) -> None:
        if self.token_ids.ndim != 2 or self.token_ids.dtype != torch.long:
            raise ValueError("token_ids must be int64 with shape (batch, sequence)")
        _require_float_tensor(self.old_log_probs, name="old_log_probs", ndim=2)
        if self.response_mask.shape != self.token_ids.shape or self.response_mask.dtype != torch.bool:
            raise ValueError("response_mask must be bool and match token_ids")
        if self.old_log_probs.shape != self.token_ids.shape:
            raise ValueError("old_log_probs must match token_ids")
        if self.stopped.shape != (self.token_ids.shape[0],) or self.stopped.dtype != torch.bool:
            raise ValueError("stopped must be bool with shape (batch,)")
        if bool((~self.response_mask).all(dim=1).any()):
            raise ValueError("every text rollout must contain at least one sampled action")
        if bool((self.old_log_probs > 0).any()):
            raise ValueError("old_log_probs must be log probabilities no greater than 0")

    @property
    def batch_size(self) -> int:
        return int(self.token_ids.shape[0])

    def branch_inputs(
        self,
        *,
        log_probs: Tensor,
        advantages: Tensor,
        ref_log_probs: Tensor | None = None,
        clip_range: float = 0.2,
    ) -> TextBranchInputs:
        if log_probs.shape != self.old_log_probs.shape:
            raise ValueError("replayed text log_probs do not match the rollout trace")
        _require_float_tensor(log_probs, name="log_probs", ndim=2)
        if bool((log_probs > 0).any()):
            raise ValueError("log_probs must be log probabilities no greater than 0")
        if ref_log_probs is not None and ref_log_probs.shape != log_probs.shape:
            raise ValueError("reference text log_probs do not match current log_probs")
        if ref_log_probs is not None:
            _require_float_tensor(ref_log_probs, name="ref_log_probs", ndim=2)
            if bool((ref_log_probs > 0).any()):
                raise ValueError("ref_log_probs must be log probabilities no greater than 0")
        return TextBranchInputs(
            log_probs=log_probs,
            old_log_probs=self.old_log_probs,
            advantages=advantages,
            response_mask=self.response_mask,
            ref_log_probs=ref_log_probs,
            clip_range=clip_range,
        )


def sample_text_span(
    *,
    initial_logits: Tensor,
    advance: TextAdvance,
    stop_token_ids: Sequence[int],
    max_new_tokens: int,
    pad_token_id: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> TextRolloutTrace:
    """Sample one autoregressive text span and store every action likelihood.

    ``advance(tokens, accepted_mask)`` is the U1.5 runtime hook.  It must append
    the sampled actions to the appropriate KV caches and return the next logits
    with shape ``(batch, vocab)``.  The hook is called for stop actions too, so
    an ``<img>`` action can be committed before the runtime enters its visual
    branch.  Rows that stopped earlier carry ``accepted_mask=False`` and must
    not mutate their cache.

    Stop actions (EOS or ``<img>``) are part of the policy trace and therefore
    receive likelihood-ratio credit.  Padding after an earlier stop never does.
    """

    _require_float_tensor(initial_logits, name="initial_logits", ndim=2)
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if isinstance(temperature, bool) or not math.isfinite(temperature):
        raise ValueError("temperature must be a finite number")
    if temperature != 1.0:
        raise ValueError(
            "on-policy text replay currently requires temperature=1.0; transformed likelihood replay is not implemented"
        )
    if isinstance(top_p, bool) or not math.isfinite(top_p):
        raise ValueError("top_p must be a finite number")
    if top_p != 1.0:
        raise ValueError(
            "on-policy text replay currently requires top_p=1.0; transformed likelihood replay is not implemented"
        )
    stops = {int(token) for token in stop_token_ids}
    if not stops:
        raise ValueError("stop_token_ids must not be empty")

    logits = initial_logits
    batch = logits.shape[0]
    active = torch.ones(batch, dtype=torch.bool, device=logits.device)
    token_columns: list[Tensor] = []
    log_prob_columns: list[Tensor] = []
    mask_columns: list[Tensor] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            accepted = active.clone()
            sampling_logits = logits.float()
            _require_float_tensor(sampling_logits, name="sampling logits", ndim=2)
            log_probs = torch.log_softmax(sampling_logits, dim=-1)
            probabilities = torch.softmax(sampling_logits, dim=-1)
            sampled = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
            selected = log_probs.gather(1, sampled.unsqueeze(1)).squeeze(1)

            pad = torch.full_like(sampled, int(pad_token_id))
            sampled = torch.where(accepted, sampled, pad)
            selected = torch.where(accepted, selected, torch.zeros_like(selected))
            token_columns.append(sampled)
            log_prob_columns.append(selected)
            mask_columns.append(accepted)

            stop_mask = torch.zeros_like(active)
            for token_id in stops:
                stop_mask |= sampled.eq(token_id)
            active = active & ~stop_mask
            next_logits = advance(sampled, accepted)
            _require_float_tensor(next_logits, name="advance result", ndim=2)
            if next_logits.shape != logits.shape:
                raise ValueError(f"advance returned {tuple(next_logits.shape)}, expected {tuple(logits.shape)}")
            logits = next_logits
            if not bool(active.any()):
                break

    return TextRolloutTrace(
        token_ids=torch.stack(token_columns, dim=1).detach(),
        old_log_probs=torch.stack(log_prob_columns, dim=1).float().detach(),
        response_mask=torch.stack(mask_columns, dim=1).detach(),
        stopped=(~active).detach(),
    )


def concatenate_text_spans(spans: Sequence[TextRolloutTrace], *, pad_token_id: int) -> TextRolloutTrace:
    """Concatenate text spans separated by generated images, removing span padding."""

    if not spans:
        raise ValueError("at least one text span is required")
    batch = spans[0].batch_size
    device = spans[0].token_ids.device
    if any(span.batch_size != batch or span.token_ids.device != device for span in spans):
        raise ValueError("all text spans must share batch size and device")

    per_row_tokens: list[Tensor] = []
    per_row_log_probs: list[Tensor] = []
    lengths: list[int] = []
    for row in range(batch):
        row_tokens = torch.cat([span.token_ids[row][span.response_mask[row]] for span in spans], dim=0)
        row_log_probs = torch.cat([span.old_log_probs[row][span.response_mask[row]] for span in spans], dim=0)
        per_row_tokens.append(row_tokens)
        per_row_log_probs.append(row_log_probs)
        lengths.append(int(row_tokens.numel()))
    width = max(lengths)
    tokens = torch.full((batch, width), int(pad_token_id), dtype=torch.long, device=device)
    old = torch.zeros((batch, width), dtype=torch.float32, device=device)
    mask = torch.zeros((batch, width), dtype=torch.bool, device=device)
    for row, length in enumerate(lengths):
        tokens[row, :length] = per_row_tokens[row]
        old[row, :length] = per_row_log_probs[row]
        mask[row, :length] = True
    return TextRolloutTrace(
        token_ids=tokens,
        old_log_probs=old,
        response_mask=mask,
        stopped=torch.stack([span.stopped for span in spans]).any(dim=0),
    )


@dataclass(frozen=True)
class ImageSdeTrace:
    """One homogeneous U1.5 generated-image event across a rollout batch."""

    samples: Tensor
    prev_samples: Tensor
    old_log_probs: Tensor
    old_means: Tensor
    timesteps: Tensor
    next_timesteps: Tensor
    sde_indices: Tensor
    final_sample: Tensor
    sigma_max: float
    noise_level: float
    image_height: int | None = None
    image_width: int | None = None

    def __post_init__(self) -> None:
        _require_float_tensor(self.samples, name="samples")
        _require_float_tensor(self.prev_samples, name="prev_samples")
        _require_float_tensor(self.old_log_probs, name="old_log_probs", ndim=2)
        _require_float_tensor(self.old_means, name="old transition means")
        if self.samples.ndim < 3:
            raise ValueError("samples must have shape (steps, batch, *latent)")
        if self.samples.shape != self.prev_samples.shape:
            raise ValueError("samples and prev_samples must have the same shape")
        if self.old_means.shape != self.samples.shape:
            raise ValueError("old transition means must match sampled latents")
        if self.old_log_probs.shape != self.samples.shape[:2]:
            raise ValueError("old_log_probs must have shape (steps, batch)")
        if self.timesteps.shape != (self.samples.shape[0],):
            raise ValueError("timesteps must contain one value per SDE action")
        if self.next_timesteps.shape != self.timesteps.shape:
            raise ValueError("next_timesteps must align with SDE actions")
        if self.sde_indices.shape != self.timesteps.shape or self.sde_indices.dtype != torch.long:
            raise ValueError("sde_indices must be int64 and align with SDE actions")
        if bool((self.next_timesteps <= self.timesteps).any()):
            raise ValueError("each selected SDE action must advance flow time")
        if bool((self.sde_indices[1:] <= self.sde_indices[:-1]).any()):
            raise ValueError("sde_indices must be strictly increasing")
        _require_float_tensor(self.final_sample, name="final image latent")
        if self.final_sample.shape != self.samples.shape[1:]:
            raise ValueError("final image latent must match trace batch/latent shape")
        if not 0.0 < self.sigma_max < 1.0:
            raise ValueError("sigma_max must lie inside (0, 1)")
        if not math.isfinite(self.noise_level) or self.noise_level <= 0:
            raise ValueError("RLVR image rollout requires a positive finite noise_level")
        if (self.image_height is None) != (self.image_width is None):
            raise ValueError("image replay geometry requires both height and width")
        if self.image_height is not None and (
            type(self.image_height) is not int
            or type(self.image_width) is not int
            or self.image_height <= 0
            or self.image_width <= 0
        ):
            raise ValueError("image replay height and width must be positive integers")

    @property
    def steps(self) -> int:
        return int(self.samples.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self.samples.shape[1])


@dataclass(frozen=True)
class ImageSdeReplay:
    """Differentiable current-policy replay of an :class:`ImageSdeTrace`."""

    trace: ImageSdeTrace
    log_probs: Tensor
    means: Tensor
    scales: Tensor
    velocities: Tensor
    ref_velocities: Tensor | None

    def __post_init__(self) -> None:
        _require_float_tensor(self.log_probs, name="replayed image log_probs", ndim=2)
        _require_float_tensor(self.means, name="replayed transition means")
        _require_float_tensor(self.scales, name="expanded transition scales")
        _require_float_tensor(self.velocities, name="current velocities")
        if self.ref_velocities is not None:
            _require_float_tensor(self.ref_velocities, name="reference velocities")
        if self.log_probs.shape != self.trace.old_log_probs.shape:
            raise ValueError("replayed image log_probs do not match the rollout trace")
        if self.means.shape != self.trace.samples.shape:
            raise ValueError("replayed transition means do not match the rollout trace")
        if self.scales.shape != self.means.shape:
            raise ValueError("expanded transition scales must match transition means")
        if self.velocities.shape != self.means.shape:
            raise ValueError("current velocities must match transition means")
        if self.ref_velocities is not None and self.ref_velocities.shape != self.velocities.shape:
            raise ValueError("reference velocities do not match current velocities")

    def branch_inputs(
        self,
        *,
        advantages: Tensor,
        clip_range: float = 1e-4,
    ) -> ImageBranchInputs:
        if advantages.shape != (self.trace.batch_size,):
            raise ValueError(f"advantages must have shape ({self.trace.batch_size},), got {tuple(advantages.shape)}")
        steps, batch = self.trace.old_log_probs.shape
        latent_shape = self.means.shape[2:]
        repeated_advantages = advantages.unsqueeze(0).expand(steps, batch).reshape(-1)
        return ImageBranchInputs(
            log_prob=self.log_probs.reshape(-1),
            old_log_prob=self.trace.old_log_probs.reshape(-1),
            advantages=repeated_advantages,
            mean=self.means.reshape(steps * batch, *latent_shape),
            old_mean=self.trace.old_means.reshape(steps * batch, *latent_shape),
            scale=self.scales.reshape(steps * batch, *latent_shape),
            velocity=self.velocities.reshape(steps * batch, *latent_shape),
            ref_velocity=(
                None if self.ref_velocities is None else self.ref_velocities.reshape(steps * batch, *latent_shape)
            ),
            clip_range=clip_range,
        )


def sample_image_sde(
    *,
    initial_sample: Tensor,
    timesteps: Tensor | Sequence[float],
    predict_velocity: VelocityPredictor,
    sde_indices: Sequence[int] | None = None,
    noise_level: float = 0.7,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, ImageSdeTrace]:
    """Sample one U1.5 visual action with SDE transitions and old log-probs.

    ``initial_sample`` is the patchified U1.5 pixel latent ``z``.  The runtime
    remains responsible for initial pixel-noise scaling and for decoding the
    returned final latent with the U1.5 pixel head geometry.
    """

    _require_float_tensor(initial_sample, name="initial_sample")
    if initial_sample.ndim < 2:
        raise ValueError("initial_sample must have shape (batch, *latent)")
    if not math.isfinite(noise_level) or noise_level <= 0:
        raise ValueError("RLVR image rollout requires a positive finite noise_level")
    schedule = _prepare_u15_timesteps(timesteps, initial_sample.device)
    sigma_max = float(1.0 - schedule[1])

    provided_indices = (
        tuple(range(schedule.numel() - 1)) if sde_indices is None else tuple(int(index) for index in sde_indices)
    )
    selected = set(provided_indices)
    if not selected or min(selected) < 0 or max(selected) >= schedule.numel() - 1:
        raise ValueError("sde_indices must select at least one valid denoising step")
    if len(selected) != len(provided_indices):
        raise ValueError("sde_indices must not contain duplicates")

    samples: list[Tensor] = []
    prev_samples: list[Tensor] = []
    old_log_probs: list[Tensor] = []
    old_means: list[Tensor] = []
    selected_timesteps: list[Tensor] = []
    selected_next_timesteps: list[Tensor] = []
    selected_indices: list[int] = []
    sample = initial_sample.float().detach()
    with torch.no_grad():
        for index in range(schedule.numel() - 1):
            t = schedule[index]
            t_next = schedule[index + 1]
            velocity = predict_velocity(sample, t, index)
            if velocity.shape != sample.shape:
                raise ValueError(
                    f"velocity at step {index} has shape {tuple(velocity.shape)}, expected {tuple(sample.shape)}"
                )
            if index in selected:
                prev_sample, old_log_prob, transition = sde_sample_step(
                    model_output=-velocity,
                    sample=sample,
                    sigma=1.0 - t,
                    sigma_prev=1.0 - t_next,
                    sigma_max=sigma_max,
                    noise_level=noise_level,
                    generator=generator,
                )
                samples.append(sample.detach())
                prev_samples.append(prev_sample.detach())
                old_log_probs.append(old_log_prob.detach())
                old_means.append(transition.mean.detach())
                selected_timesteps.append(t.detach())
                selected_next_timesteps.append(t_next.detach())
                selected_indices.append(index)
            else:
                prev_sample = sample + (t_next - t) * velocity.float()
            sample = prev_sample.detach()

    trace = ImageSdeTrace(
        samples=torch.stack(samples),
        prev_samples=torch.stack(prev_samples),
        old_log_probs=torch.stack(old_log_probs),
        old_means=torch.stack(old_means),
        timesteps=torch.stack(selected_timesteps),
        next_timesteps=torch.stack(selected_next_timesteps),
        sde_indices=torch.tensor(selected_indices, dtype=torch.long, device=schedule.device),
        final_sample=sample.detach(),
        sigma_max=sigma_max,
        noise_level=noise_level,
    )
    return sample, trace


def replay_image_sde(
    trace: ImageSdeTrace,
    *,
    predict_velocity: VelocityPredictor,
    predict_reference_velocity: VelocityPredictor | None = None,
) -> ImageSdeReplay:
    """Recompute image likelihoods under current and optional reference policy."""

    log_probs: list[Tensor] = []
    means: list[Tensor] = []
    scales: list[Tensor] = []
    velocities: list[Tensor] = []
    ref_velocities: list[Tensor] = []
    for index in range(trace.steps):
        sample = trace.samples[index].detach()
        prev_sample = trace.prev_samples[index].detach()
        t = trace.timesteps[index]
        t_next = trace.next_timesteps[index]
        velocity = predict_velocity(sample, t, index)
        if velocity.shape != sample.shape:
            raise ValueError(
                f"current velocity at step {index} has shape {tuple(velocity.shape)}, expected {tuple(sample.shape)}"
            )
        transition = u15_sde_transition(
            velocity=velocity,
            sample=sample,
            t=t,
            t_next=t_next,
            sigma_max=trace.sigma_max,
            noise_level=trace.noise_level,
        )
        log_probs.append(sde_log_prob(prev_sample=prev_sample, transition=transition))
        means.append(transition.mean)
        scales.append(transition.scale.expand_as(transition.mean))
        velocities.append(velocity)

        if predict_reference_velocity is not None:
            with torch.no_grad():
                reference_velocity = predict_reference_velocity(sample, t, index)
                if reference_velocity.shape != sample.shape:
                    raise ValueError(
                        f"reference velocity at step {index} has shape "
                        f"{tuple(reference_velocity.shape)}, expected {tuple(sample.shape)}"
                    )
                ref_velocities.append(reference_velocity.detach())

    return ImageSdeReplay(
        trace=trace,
        log_probs=torch.stack(log_probs),
        means=torch.stack(means),
        scales=torch.stack(scales),
        velocities=torch.stack(velocities),
        ref_velocities=(torch.stack(ref_velocities) if ref_velocities else None),
    )


def replay_image_sde_batched(
    trace: ImageSdeTrace,
    *,
    predict_velocity: VelocityPredictor,
    predict_reference_velocity: VelocityPredictor | None = None,
    microbatch_size: int | None = None,
) -> ImageSdeReplay:
    """Replay fixed SDE transitions in bounded model microbatches.

    Unlike sampling, replay already owns every ``x_t`` and timestep, so the
    velocity evaluations are independent.  Flattening ``steps * batch`` still
    permits efficient batching, while ``microbatch_size`` bounds the model
    batch width of each callback. Peak activation is bounded only if callers
    either backward each slice separately or make the callback recomputation-
    backed before joining slice losses. ``-1`` marks that there is no single
    scalar denoising-step index.
    """

    steps, batch = trace.samples.shape[:2]
    latent_shape = trace.samples.shape[2:]
    samples = trace.samples.detach().reshape(steps * batch, *latent_shape)
    prev_samples = trace.prev_samples.detach().reshape(steps * batch, *latent_shape)
    timesteps = trace.timesteps.repeat_interleave(batch)
    next_timesteps = trace.next_timesteps.repeat_interleave(batch)
    action_count = int(samples.shape[0])
    if microbatch_size is None:
        microbatch_size = action_count
    if type(microbatch_size) is not int or not 1 <= microbatch_size <= action_count:
        raise ValueError("image replay microbatch_size must lie inside the action batch")
    velocities: list[Tensor] = []
    reference_velocities: list[Tensor] = []
    for start in range(0, action_count, microbatch_size):
        stop = min(start + microbatch_size, action_count)
        sample_chunk = samples[start:stop]
        timestep_chunk = timesteps[start:stop]
        velocity_chunk = predict_velocity(sample_chunk, timestep_chunk, -1)
        if velocity_chunk.shape != sample_chunk.shape:
            raise ValueError(
                f"batched current velocity has shape {tuple(velocity_chunk.shape)}, "
                f"expected {tuple(sample_chunk.shape)}"
            )
        velocities.append(velocity_chunk)

        if predict_reference_velocity is not None:
            with torch.no_grad():
                reference_chunk = predict_reference_velocity(sample_chunk, timestep_chunk, -1)
            if reference_chunk.shape != sample_chunk.shape:
                raise ValueError(
                    "batched reference velocity has shape "
                    f"{tuple(reference_chunk.shape)}, expected "
                    f"{tuple(sample_chunk.shape)}"
                )
            reference_velocities.append(reference_chunk.detach())

    velocity = torch.cat(velocities, dim=0)
    if velocity.shape != samples.shape:
        raise ValueError(f"batched current velocity has shape {tuple(velocity.shape)}, expected {tuple(samples.shape)}")
    transition = u15_sde_transition(
        velocity=velocity,
        sample=samples,
        t=timesteps,
        t_next=next_timesteps,
        sigma_max=trace.sigma_max,
        noise_level=trace.noise_level,
    )
    ref_velocities = (
        torch.cat(reference_velocities, dim=0).reshape(steps, batch, *latent_shape) if reference_velocities else None
    )
    return ImageSdeReplay(
        trace=trace,
        log_probs=sde_log_prob(prev_sample=prev_samples, transition=transition).reshape(steps, batch),
        means=transition.mean.reshape(steps, batch, *latent_shape),
        scales=transition.scale.expand_as(transition.mean).reshape(steps, batch, *latent_shape),
        velocities=velocity.reshape(steps, batch, *latent_shape),
        ref_velocities=ref_velocities,
    )


def slice_image_sde_trace(
    trace: ImageSdeTrace,
    *,
    action_start: int,
    action_stop: int,
) -> ImageSdeTrace:
    """Return one exact action slice of a single-rollout image trace.

    The U1.5 policy runtime replays one rollout at a time, so an action is
    one selected SDE step.  Keeping this restriction explicit prevents an
    accidental slice through the batch axis from changing the loss reduction.
    """

    if trace.batch_size != 1:
        raise ValueError("image action slicing requires a single-rollout trace")
    if (
        type(action_start) is not int
        or type(action_stop) is not int
        or not 0 <= action_start < action_stop <= trace.steps
    ):
        raise ValueError("invalid image SDE action slice")
    selected = slice(action_start, action_stop)
    return ImageSdeTrace(
        samples=trace.samples[selected],
        prev_samples=trace.prev_samples[selected],
        old_log_probs=trace.old_log_probs[selected],
        old_means=trace.old_means[selected],
        timesteps=trace.timesteps[selected],
        next_timesteps=trace.next_timesteps[selected],
        sde_indices=trace.sde_indices[selected],
        final_sample=trace.final_sample,
        sigma_max=trace.sigma_max,
        noise_level=trace.noise_level,
        image_height=trace.image_height,
        image_width=trace.image_width,
    )


__all__ = [
    "ImageSdeReplay",
    "ImageSdeTrace",
    "TextRolloutTrace",
    "concatenate_text_spans",
    "replay_image_sde",
    "replay_image_sde_batched",
    "sample_image_sde",
    "sample_text_span",
    "select_sde_indices",
    "slice_image_sde_trace",
    "u15_sde_transition",
]
