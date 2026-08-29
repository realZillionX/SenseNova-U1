"""Flow-GDPO and the UniGDPO loss core (SenseNova-U1.5 line).

SenseNova-U1.5 (NEO-unify) is pixel-space flow matching: the LLM itself is the
denoiser and text tokens and image patch tokens share one sequence and one
backward pass.  UniGDPO therefore optimizes, on the *same* rollout, a
token-level GDPO objective on the text branch and a Flow-GDPO objective on the
image branch (Flow-GRPO's ODE-to-SDE conversion with per-step Gaussian
log-probs, but carrying GDPO advantages), then weights and sums the two branch
losses into one optimizer step.

This module owns only the loss algebra.  Advantage estimation -- per-dimension
group normalization, the weighted sum, and one optimizer-batch normalization --
lives in ``training/rlvr/advantage.py``; nothing here reads or produces rewards.

The four coupling rules in the training README's RLVR and GDPO contract bind this module:

1. advantage batch normalization uses all prompt groups in the independent
   model arm's optimizer batch -- owned by ``advantage.py``, not here;
2. the two branches are never merged into a single likelihood ratio.  The flow
   log-prob is a per-latent-element *mean* while the text log-prob is a
   per-token sum-scale quantity, so the ratios live on incompatible scales and
   each branch carries its own clip range (image 1e-4, text 0.2).  This is
   enforced structurally: the branch inputs are two distinct dataclasses, each
   holding its own ``clip_range``, ``uni_gdpo_loss`` has no clip-range
   parameter of its own, and the two policy-loss functions accept disjoint
   tensor ranks so a concatenated tensor cannot be routed through either;
3. reference regularization is outside reward: text uses token-level k3 KL and
   image uses UniGRPO's unweighted velocity-MSE;
4. an absent image branch contributes exactly zero; TI2TI is allowed to emit
   no image, while TI2T never exposes the visual action space.

Conventions fixed by this module:

*fp32*: every SDE quantity is computed in float32 regardless of the incoming
dtype.  bf16 overflows when forming ``prev_sample_mean`` (the ``1/(2*sigma)``
factor blows up as sigma approaches the terminal step).

*sigma*: sigmas descend along the sampling trajectory, so ``dt = sigma_prev -
sigma`` is negative and ``sqrt(-dt)`` is the real per-step time scale.  Sigmas
are passed in explicitly rather than read off a scheduler object, which keeps
the module usable with the U1 sampler and with any diffusers scheduler without
importing diffusers.

*trajectory mean*: each response first averages over its own sampled text
actions, then the optimizer batch averages over trajectories.  This is the
sealed ``seq-mean-token-mean`` reduction shared by TI2T and TI2TI; variable
response length must not silently change a rollout's total weight.

*reference regularization*: the text branch uses Schulman's k3 low-variance estimator,
``exp(ref - cur) - (ref - cur) - 1``, which is non-negative per token and
unbiased for the forward KL.  The image branch directly matches current and
frozen-reference velocity fields without timestep-dependent KL weighting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor

NOISE_LEVEL_DEFAULT: float = 0.7
IMAGE_CLIP_RANGE_DEFAULT: float = 1e-4
TEXT_CLIP_RANGE_DEFAULT: float = 0.2

ScalarLike = Union[float, Tensor]

_LOG_SQRT_2PI: float = 0.5 * math.log(2.0 * math.pi)


def _as_broadcast(value: ScalarLike, like: Tensor, name: str) -> Tensor:
    """Broadcast a per-step scalar or per-sample vector against ``like``.

    ``like`` has shape ``(batch, *latent)``.  A python float or a 0-dim tensor
    becomes a full-rank constant; a ``(batch,)`` tensor becomes
    ``(batch, 1, ..., 1)``; anything already broadcastable is passed through.

    Dtype follows ``like`` rather than being forced here: the fp32 policy has a
    single decision point, the cast at the top of ``sde_transition``.
    """

    if isinstance(value, Tensor):
        tensor = value.to(device=like.device, dtype=like.dtype)
        if tensor.ndim == 0:
            return tensor.reshape(*([1] * like.ndim))
        if tensor.ndim == 1:
            if tensor.shape[0] != like.shape[0]:
                raise ValueError(f"{name} has batch size {tensor.shape[0]}, expected {like.shape[0]}")
            return tensor.reshape(-1, *([1] * (like.ndim - 1)))
        if tensor.ndim != like.ndim:
            raise ValueError(
                f"{name} must be a scalar, a (batch,) vector or already rank-{like.ndim}, "
                f"got shape {tuple(tensor.shape)}"
            )
        return tensor
    return torch.full((1,) * like.ndim, float(value), device=like.device, dtype=like.dtype)


def _mean_over_latent(tensor: Tensor) -> Tensor:
    """Reduce ``(batch, *latent)`` to ``(batch,)`` by the mean over latent dims."""

    if tensor.ndim < 2:
        raise ValueError(
            f"expected a (batch, *latent) tensor with at least one latent dim, got shape {tuple(tensor.shape)}"
        )
    return tensor.mean(dim=tuple(range(1, tensor.ndim)))


@dataclass(frozen=True)
class SdeTransition:
    """One reverse-SDE transition ``p(x_{t-1} | x_t)``, a diagonal Gaussian.

    Shapes: ``mean`` is ``(batch, *latent)``; ``std`` and ``sqrt_neg_dt`` are
    broadcastable against it (typically ``(batch, 1, ..., 1)``).

    ``std`` is the SDE diffusion coefficient ``std_dev_t``; the standard
    deviation of the transition itself is ``scale = std * sqrt_neg_dt``.
    """

    mean: Tensor
    std: Tensor
    sqrt_neg_dt: Tensor

    @property
    def scale(self) -> Tensor:
        """Standard deviation of the transition distribution, ``(batch, ...)``."""

        return self.std * self.sqrt_neg_dt


@dataclass(frozen=True)
class LossTerm:
    """A scalar loss and its detached diagnostics."""

    value: Tensor
    metrics: Dict[str, float] = field(default_factory=dict)


def sde_transition(
    *,
    model_output: Tensor,
    sample: Tensor,
    sigma: ScalarLike,
    sigma_prev: ScalarLike,
    sigma_max: float,
    noise_level: float = NOISE_LEVEL_DEFAULT,
) -> SdeTransition:
    """Convert the deterministic flow ODE step into its SDE counterpart.

    Args:
        model_output: predicted velocity, ``(batch, *latent)``.
        sample: current latent ``x_t``, ``(batch, *latent)``.
        sigma: sigma at the current step, scalar or ``(batch,)``.
        sigma_prev: sigma at the next (lower) step, scalar or ``(batch,)``.
        sigma_max: the scheduler's largest *interior* sigma -- ``sigmas[1]``,
            not ``sigmas[0]``.  It substitutes for ``sigma`` in the diffusion
            coefficient at ``sigma == 1``, where ``1 - sigma`` vanishes, so it
            must itself lie strictly inside ``(0, 1)``.
        noise_level: SDE noise scale; ``0.0`` degenerates to the Euler ODE step,
            whose transition is a point mass and therefore has no log-prob (see
            :func:`sde_log_prob`).

    Returns:
        ``SdeTransition`` with ``mean`` of shape ``(batch, *latent)``.

    All arithmetic is float32.  ``dt = sigma_prev - sigma`` is negative because
    sigmas descend; a non-negative ``dt`` means the two sigmas were swapped and
    raises rather than silently producing NaNs from ``sqrt(-dt)``.
    """

    if noise_level < 0.0:
        raise ValueError(f"noise_level must be non-negative, got {noise_level}")
    if not 0.0 < float(sigma_max) < 1.0:
        raise ValueError(
            f"sigma_max must lie strictly inside (0, 1), got {sigma_max}. It is the "
            "scheduler's largest interior sigma -- FlowMatchEulerDiscreteScheduler has "
            "sigmas[0] == 1.0 exactly, and passing that back divides by 1 - sigma_max "
            "at the first step, which is the mandatory sigma == 1 step of every rollout"
        )
    if model_output.shape != sample.shape:
        raise ValueError(
            f"model_output shape {tuple(model_output.shape)} does not match sample shape {tuple(sample.shape)}"
        )

    # The single fp32 decision point: every downstream scalar inherits this
    # dtype through _as_broadcast, so the whole transition is computed in fp32.
    model_output = model_output.float()
    sample = sample.float()

    sigma_t = _as_broadcast(sigma, sample, "sigma")
    sigma_p = _as_broadcast(sigma_prev, sample, "sigma_prev")
    dt = sigma_p - sigma_t
    if bool((dt >= 0).any()):
        raise ValueError(
            "sigma_prev must be strictly smaller than sigma (sigmas descend along "
            "the sampling trajectory); received a non-negative dt"
        )

    sigma_max_t = torch.as_tensor(float(sigma_max), device=sample.device, dtype=sample.dtype)
    std_dev_t = torch.sqrt(sigma_t / (1 - torch.where(sigma_t == 1, sigma_max_t, sigma_t)))
    std_dev_t = std_dev_t * noise_level

    mean = (
        sample * (1 + std_dev_t**2 / (2 * sigma_t) * dt)
        + model_output * (1 + std_dev_t**2 * (1 - sigma_t) / (2 * sigma_t)) * dt
    )

    return SdeTransition(mean=mean, std=std_dev_t, sqrt_neg_dt=torch.sqrt(-dt))


def sde_log_prob(*, prev_sample: Tensor, transition: SdeTransition) -> Tensor:
    """Log-prob of ``prev_sample`` under ``transition``, reduced to ``(batch,)``.

    Args:
        prev_sample: the drawn latent ``x_{t-1}``, ``(batch, *latent)``.
        transition: the Gaussian produced by ``sde_transition``.

    Returns:
        ``(batch,)`` -- the *mean* of the per-element Gaussian log density over
        all latent dimensions, exactly as Flow-GRPO reduces it.  This mean (not
        sum) reduction is why the image clip range is ``1e-4`` and not ``0.2``.

    ``prev_sample`` is detached: gradient flows only through
    ``transition.mean``, matching the reference implementation.  Recomputing
    this during training with unchanged parameters must reproduce the rollout
    log-prob bit-for-bit, which is what makes the on-policy first step have
    ratio exactly one.

    A zero scale (``noise_level == 0``) is the Euler ODE step: a point mass with
    no density, whose log-prob would come out ``NaN`` and propagate silently
    into the loss.  It raises instead.
    """

    prev = prev_sample.detach().float()
    if prev.shape != transition.mean.shape:
        raise ValueError(
            f"prev_sample shape {tuple(prev.shape)} does not match transition mean shape {tuple(transition.mean.shape)}"
        )

    scale = transition.scale
    if not bool((scale > 0).all()):
        raise ValueError(
            "transition scale must be strictly positive to have a log-prob; a zero "
            "scale is the deterministic Euler ODE step (noise_level == 0), which has "
            "no density -- use transition.mean directly and take no log-prob"
        )
    log_prob = -((prev - transition.mean) ** 2) / (2 * scale**2) - torch.log(scale) - _LOG_SQRT_2PI
    return _mean_over_latent(log_prob.expand_as(transition.mean))


def sde_sample_step(
    *,
    model_output: Tensor,
    sample: Tensor,
    sigma: ScalarLike,
    sigma_prev: ScalarLike,
    sigma_max: float,
    noise_level: float = NOISE_LEVEL_DEFAULT,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor, SdeTransition]:
    """Draw one reverse-SDE step, for rollout.

    Args and shapes are those of ``sde_transition``.

    Returns:
        ``(prev_sample, log_prob, transition)`` with ``prev_sample`` of shape
        ``(batch, *latent)`` and ``log_prob`` of shape ``(batch,)``.

    The returned ``log_prob`` is the rollout ``old_log_prob``; storing it and
    re-deriving it later through ``sde_transition`` + ``sde_log_prob`` under the
    same sigma is the self-check that the sampler and the trainer agree on the
    sigma/timestep convention.
    """

    transition = sde_transition(
        model_output=model_output,
        sample=sample,
        sigma=sigma,
        sigma_prev=sigma_prev,
        sigma_max=sigma_max,
        noise_level=noise_level,
    )
    noise = torch.randn(
        transition.mean.shape,
        generator=generator,
        device=transition.mean.device,
        dtype=torch.float32,
    )
    prev_sample = transition.mean + transition.scale * noise
    return (
        prev_sample,
        sde_log_prob(prev_sample=prev_sample, transition=transition),
        transition,
    )


def flow_policy_loss(
    *,
    log_prob: Tensor,
    old_log_prob: Tensor,
    advantages: Tensor,
    clip_range: float,
    mean: Tensor,
    old_mean: Tensor,
    scale: Tensor,
) -> LossTerm:
    """Clipped surrogate on the image branch, for one denoising step.

    Args:
        log_prob: recomputed step log-prob, ``(batch,)``.
        old_log_prob: rollout step log-prob, ``(batch,)``.
        advantages: GDPO advantage per rollout, ``(batch,)``; a per-sample
            advantage is broadcast across denoising steps by the caller.
        clip_range: image-branch clip range, ``1e-4`` at the U1 starting point.
            Required, with no default, so it can never be inherited from the
            text branch.
    Returns:
        ``LossTerm`` whose ``value`` is a 0-dim tensor.

    Rank is the structural guard for coupling rule 2: all three tensors must be
    rank 1.  Text log-probs are ``(batch, seq)`` and are rejected here, so the
    two branches cannot be routed through one ratio.
    """

    for name, tensor in (
        ("log_prob", log_prob),
        ("old_log_prob", old_log_prob),
        ("advantages", advantages),
    ):
        if tensor.ndim != 1:
            raise ValueError(
                f"{name} must be rank 1 (batch,) on the image branch; got shape "
                f"{tuple(tensor.shape)}. Token-level text tensors belong to "
                f"text_policy_loss -- the two branches never share a ratio."
            )
    if not (log_prob.shape == old_log_prob.shape == advantages.shape):
        raise ValueError(
            f"image-branch shapes disagree: log_prob {tuple(log_prob.shape)}, "
            f"old_log_prob {tuple(old_log_prob.shape)}, "
            f"advantages {tuple(advantages.shape)}"
        )
    if not clip_range > 0.0:
        raise ValueError(f"clip_range must be positive, got {clip_range}")
    for name, tensor in (("log_prob", log_prob), ("old_log_prob", old_log_prob)):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(
                f"{name} carries non-finite values; a NaN here silently NaNs the whole "
                "optimizer step, so it is refused rather than differentiated"
            )

    adv = advantages.detach().float()
    log_ratio = ratio_normalized_log_ratio(
        log_prob=log_prob,
        old_log_prob=old_log_prob,
        mean=mean,
        old_mean=old_mean,
        scale=scale,
    )
    ratio = torch.exp(log_ratio)

    unclipped = -adv * ratio
    clipped = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    value = torch.mean(torch.maximum(unclipped, clipped))

    with torch.no_grad():
        metrics = {
            "ratio_mean": float(ratio.mean()),
            "raw_ratio_mean": float(torch.exp(log_prob.float() - old_log_prob.float()).mean()),
            "clipfrac": float((ratio - 1.0).abs().gt(clip_range).float().mean()),
            "approx_kl": float(0.5 * (log_ratio**2).mean()),
        }
    return LossTerm(value=value, metrics=metrics)


def ratio_normalized_log_ratio(
    *,
    log_prob: Tensor,
    old_log_prob: Tensor,
    mean: Tensor,
    old_mean: Tensor,
    scale: Tensor,
) -> Tensor:
    """Return GRPO-Guard RatioNorm log-ratios for mean-reduced flow log-probs.

    The additive mean-drift term removes the negative Gaussian log-ratio bias;
    multiplying by the transition scale makes ratio variance comparable across
    SDE timesteps.  ``scale`` is the sampled transition standard deviation,
    including ``sqrt(dt)``.
    """

    if log_prob.ndim != 1 or old_log_prob.shape != log_prob.shape:
        raise ValueError("flow RatioNorm log-probs must share shape (actions,)")
    if mean.shape != old_mean.shape or mean.shape != scale.shape:
        raise ValueError("flow RatioNorm means and scales must share a shape")
    if mean.shape[0] != log_prob.shape[0] or mean.ndim < 2:
        raise ValueError("flow RatioNorm action and latent batches disagree")
    if not bool((scale > 0).all()):
        raise ValueError("flow RatioNorm transition scales must be positive")
    latent_dims = tuple(range(1, mean.ndim))
    scale_per_action = scale.float().mean(dim=latent_dims)
    if not bool(
        torch.allclose(
            scale.float(),
            scale_per_action.reshape(-1, *([1] * (mean.ndim - 1))).expand_as(scale),
        )
    ):
        raise ValueError("flow RatioNorm requires isotropic transition scales")
    drift = ((old_mean.float() - mean.float()) ** 2).mean(dim=latent_dims)
    raw = log_prob.float() - old_log_prob.float()
    return scale_per_action * (raw + drift / (2.0 * scale_per_action**2))


def velocity_mse_loss(*, velocity: Tensor, ref_velocity: Tensor) -> Tensor:
    """Unweighted velocity-field reference regularization from UniGRPO."""

    if velocity.shape != ref_velocity.shape or velocity.ndim < 2:
        raise ValueError("current/reference velocities must share an action-latent shape")
    return ((velocity.float() - ref_velocity.float()) ** 2).mean()


def _broadcast_token_advantages(advantages: Tensor, like: Tensor) -> Tensor:
    if advantages.ndim == 1:
        if advantages.shape[0] != like.shape[0]:
            raise ValueError(f"advantages batch {advantages.shape[0]} does not match log_probs batch {like.shape[0]}")
        return advantages.float().unsqueeze(-1).expand_as(like)
    if advantages.shape != like.shape:
        raise ValueError(f"advantages shape {tuple(advantages.shape)} must be (batch,) or {tuple(like.shape)}")
    return advantages.float()


def _masked_trajectory_mean(tensor: Tensor, mask: Tensor) -> Tensor:
    """Mean over actions inside each trajectory, then mean over trajectories."""

    counts = mask.sum(dim=1)
    if bool((counts <= 0).any()):
        raise ValueError("response_mask selects no tokens for at least one trajectory")
    per_trajectory = (tensor * mask).sum(dim=1) / counts
    return per_trajectory.mean()


def _check_token_shapes(log_probs: Tensor, other: Tensor, other_name: str, response_mask: Tensor) -> None:
    if log_probs.ndim != 2:
        raise ValueError(
            f"log_probs must be rank 2 (batch, seq) on the text branch; got shape "
            f"{tuple(log_probs.shape)}. Per-step image log-probs belong to "
            f"flow_policy_loss -- the two branches never share a ratio."
        )
    if other.shape != log_probs.shape:
        raise ValueError(
            f"{other_name} shape {tuple(other.shape)} does not match log_probs shape {tuple(log_probs.shape)}"
        )
    if response_mask.shape != log_probs.shape:
        raise ValueError(
            f"response_mask shape {tuple(response_mask.shape)} does not match log_probs shape {tuple(log_probs.shape)}"
        )


def text_policy_loss(
    *,
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    response_mask: Tensor,
    clip_range: float = TEXT_CLIP_RANGE_DEFAULT,
) -> LossTerm:
    """Token-level clipped surrogate on the text branch of UniGDPO.

    Args:
        log_probs: recomputed per-token log-probs, ``(batch, seq)``.
        old_log_probs: rollout per-token log-probs, ``(batch, seq)``.
        advantages: GDPO advantage, ``(batch,)`` broadcast across tokens, or
            already ``(batch, seq)``.
        response_mask: 1 on generated tokens, 0 on prompt and padding,
            ``(batch, seq)``.
        clip_range: text-branch clip range, ``0.2`` at the U1 starting point.

    Returns:
        ``LossTerm`` whose ``value`` is a 0-dim tensor.

    Aggregation first takes the sampled-token mean inside each trajectory and
    then the trajectory mean across the batch.  Both branches consume the same
    detached trajectory advantage without branch-specific clipping.
    """

    _check_token_shapes(log_probs, old_log_probs, "old_log_probs", response_mask)
    if not clip_range > 0.0:
        raise ValueError(f"clip_range must be positive, got {clip_range}")

    mask = response_mask.float()
    adv = _broadcast_token_advantages(advantages, log_probs)
    log_ratio = log_probs.float() - old_log_probs.float()
    ratio = torch.exp(log_ratio)

    unclipped = -adv * ratio
    clipped = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    value = _masked_trajectory_mean(torch.maximum(unclipped, clipped), mask)

    with torch.no_grad():
        metrics = {
            "ratio_mean": float(_masked_trajectory_mean(ratio, mask)),
            "clipfrac": float(_masked_trajectory_mean((ratio - 1.0).abs().gt(clip_range).float(), mask)),
            "approx_kl": float(_masked_trajectory_mean(0.5 * log_ratio**2, mask)),
        }
    return LossTerm(value=value, metrics=metrics)


def text_kl_loss(*, log_probs: Tensor, ref_log_probs: Tensor, response_mask: Tensor) -> Tensor:
    """Token-level KL to the reference policy, exact k3, trajectory-mean reduced.

    Args:
        log_probs: current per-token log-probs, ``(batch, seq)``.
        ref_log_probs: reference per-token log-probs, ``(batch, seq)``.
        response_mask: ``(batch, seq)``.

    Returns:
        0-dim tensor.

    Estimator: Schulman's k3, ``exp(r) - r - 1`` with ``r = ref - cur``.  It is
    non-negative for every token and unbiased for ``KL(cur || ref)``, unlike the
    raw ``cur - ref`` difference which is zero-mean only in expectation and can
    go negative on a batch.  The sealed objective does not clip ``r``: a
    non-finite result aborts the update instead of silently changing the KL.
    """

    _check_token_shapes(log_probs, ref_log_probs, "ref_log_probs", response_mask)

    log_ratio = ref_log_probs.float() - log_probs.float()
    per_token = torch.expm1(log_ratio) - log_ratio
    if not bool(torch.isfinite(per_token).all()):
        raise ValueError("text k3 KL is non-finite")
    return _masked_trajectory_mean(per_token, response_mask.float())


@dataclass(frozen=True)
class ImageBranchInputs:
    """Everything the image branch of one UniGDPO step needs.

    Shapes: ``log_prob``/``old_log_prob``/``advantages`` are ``(batch,)``;
    flow means/scales/velocities are ``(batch, *latent)``. ``clip_range``
    belongs to this dataclass and to no other, which is
    how coupling rule 2 is made unrepresentable rather than merely documented.
    """

    log_prob: Tensor
    old_log_prob: Tensor
    advantages: Tensor
    mean: Optional[Tensor] = None
    old_mean: Optional[Tensor] = None
    scale: Optional[Tensor] = None
    velocity: Optional[Tensor] = None
    ref_velocity: Optional[Tensor] = None
    clip_range: float = IMAGE_CLIP_RANGE_DEFAULT


@dataclass(frozen=True)
class TextBranchInputs:
    """Everything the text branch of one UniGDPO step needs.

    Shapes: ``log_probs``/``old_log_probs``/``ref_log_probs``/``response_mask``
    are ``(batch, seq)``; ``advantages`` is ``(batch,)`` or ``(batch, seq)``.
    ``clip_range`` is this branch's own and defaults to ``0.2``.
    """

    log_probs: Tensor
    old_log_probs: Tensor
    advantages: Tensor
    response_mask: Tensor
    ref_log_probs: Optional[Tensor] = None
    clip_range: float = TEXT_CLIP_RANGE_DEFAULT


@dataclass(frozen=True)
class BranchWeights:
    """Weights on the two branch policy losses summed into one optimizer step."""

    image: float = 1.0
    text: float = 1.0


@dataclass(frozen=True)
class RegularizationWeights:
    """Independent reference regularizers for text and visual policies."""

    image_velocity_mse: float = 0.0
    text_kl: float = 0.0


@dataclass(frozen=True)
class UniGdpoLoss:
    """The joint objective and its per-branch decomposition.

    ``value`` is the 0-dim tensor to call ``backward()`` on.  Every component
    is a 0-dim tensor; an absent branch contributes an exact zero, never NaN.
    """

    value: Tensor
    image_policy: Tensor
    image_velocity_mse: Tensor
    text_policy: Tensor
    text_kl: Tensor
    metrics: Dict[str, float] = field(default_factory=dict)


def uni_gdpo_loss(
    *,
    image: Optional[ImageBranchInputs],
    text: Optional[TextBranchInputs],
    weights: BranchWeights,
    regularization: RegularizationWeights,
) -> UniGdpoLoss:
    """Weighted sum of the image and text branch objectives, one optimizer step.

    Args:
        image: image-branch inputs, or ``None`` for a text-only rollout.
        text: text-branch inputs, or ``None`` for an image-only rollout.
        weights: branch weights on the policy losses.
        regularization: independent text-KL and image-velocity-MSE weights.

    Returns:
        ``UniGdpoLoss``.

    The two branches keep separate ratios, separate clip ranges (carried by
    their own inputs dataclass) and separate reference regularizers. There is deliberately no
    ``clip_range`` parameter here: a caller cannot supply one range for both.
    A branch that is ``None`` contributes an exact zero so single-modality
    rollouts train the branch they actually produced without poisoning the sum.
    """

    if image is None and text is None:
        raise ValueError("uni_gdpo_loss needs at least one branch; both were None")
    if regularization.image_velocity_mse < 0.0 or regularization.text_kl < 0.0:
        raise ValueError(f"regularization weights must be non-negative, got {regularization}")

    reference = image.log_prob if image is not None else text.log_probs  # type: ignore[union-attr]
    zero = torch.zeros((), device=reference.device, dtype=torch.float32)

    metrics: Dict[str, float] = {}
    image_policy = zero
    image_velocity_mse = zero
    text_policy = zero
    text_kl = zero

    if image is not None:
        if image.mean is None or image.old_mean is None or image.scale is None:
            raise ValueError("image RatioNorm needs current/old transition means and scales")
        term = flow_policy_loss(
            log_prob=image.log_prob,
            old_log_prob=image.old_log_prob,
            advantages=image.advantages,
            clip_range=image.clip_range,
            mean=image.mean,
            old_mean=image.old_mean,
            scale=image.scale,
        )
        image_policy = term.value
        metrics.update({f"image/{key}": val for key, val in term.metrics.items()})
        metrics["image/clip_range"] = image.clip_range

        if image.ref_velocity is not None:
            if image.velocity is None:
                raise ValueError("image velocity-MSE needs current and reference velocities")
            image_velocity_mse = velocity_mse_loss(
                velocity=image.velocity,
                ref_velocity=image.ref_velocity,
            )
            metrics["image/velocity_mse"] = float(image_velocity_mse.detach())
        elif regularization.image_velocity_mse > 0.0:
            raise ValueError(
                "positive image velocity-MSE weight requires a frozen reference "
                "velocity; regularization is never folded into reward"
            )

    if text is not None:
        term = text_policy_loss(
            log_probs=text.log_probs,
            old_log_probs=text.old_log_probs,
            advantages=text.advantages,
            response_mask=text.response_mask,
            clip_range=text.clip_range,
        )
        text_policy = term.value
        metrics.update({f"text/{key}": val for key, val in term.metrics.items()})
        metrics["text/clip_range"] = text.clip_range

        if text.ref_log_probs is not None:
            text_kl = text_kl_loss(
                log_probs=text.log_probs,
                ref_log_probs=text.ref_log_probs,
                response_mask=text.response_mask,
            )
            metrics["text/kl"] = float(text_kl.detach())
        elif regularization.text_kl > 0.0:
            raise ValueError(
                "positive text KL weight requires reference log-probs; KL is a loss "
                "term and is never folded into a reward"
            )

    value = (
        weights.image * image_policy
        + regularization.image_velocity_mse * image_velocity_mse
        + weights.text * text_policy
        + regularization.text_kl * text_kl
    )
    metrics["loss"] = float(value.detach())
    return UniGdpoLoss(
        value=value,
        image_policy=image_policy,
        image_velocity_mse=image_velocity_mse,
        text_policy=text_policy,
        text_kl=text_kl,
        metrics=metrics,
    )


__all__ = [
    "IMAGE_CLIP_RANGE_DEFAULT",
    "NOISE_LEVEL_DEFAULT",
    "TEXT_CLIP_RANGE_DEFAULT",
    "BranchWeights",
    "ImageBranchInputs",
    "LossTerm",
    "SdeTransition",
    "TextBranchInputs",
    "UniGdpoLoss",
    "RegularizationWeights",
    "flow_policy_loss",
    "sde_log_prob",
    "sde_sample_step",
    "sde_transition",
    "text_kl_loss",
    "text_policy_loss",
    "uni_gdpo_loss",
    "ratio_normalized_log_ratio",
    "velocity_mse_loss",
]
