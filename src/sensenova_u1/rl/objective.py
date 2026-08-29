"""SenseNova-U1.5 reward-to-UniGDPO orchestration.

The shared RLVR modules own verifier execution, conditioned rewards, GDPO
normalization and branch loss algebra.  This adapter wires their typed outputs
to U1.5 rollout traces without merging text-token and image-SDE likelihood
ratios.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .advantage import GdpoAdvantageResult, compute_gdpo_advantage
from .conditioned import RewardCondition, apply_conditioned_rewards
from .flow import (
    BranchWeights,
    RegularizationWeights,
    UniGdpoLoss,
    uni_gdpo_loss,
)
from .rollout import (
    ImageSdeReplay,
    TextRolloutTrace,
)
from .types import RewardBatch


def _validate_reward_batch(reward_batch: RewardBatch) -> int:
    """Validate only the optimizer-facing reward ABI.

    Dimension meaning, ranges and hard gates belong to the downstream reward
    provider.  Forge requires a complete, finite and consistently shaped batch.
    """

    rows, columns = reward_batch.matrix.shape
    if columns != len(reward_batch.dimension_names):
        raise ValueError("reward dimensions do not match the matrix")
    failures = [f"row {index}: {error}" for index, error in enumerate(reward_batch.errors) if error is not None]
    if failures or not bool(reward_batch.scored.all()):
        raise ValueError(
            "reward provider returned unscorable rows; retry or abort the complete "
            "rollout group:\n" + "\n".join(failures[:20])
        )
    if not bool(reward_batch.availability.any(axis=1).all()):
        raise ValueError("every scored rollout must expose at least one reward dimension")
    return rows


def _require_finite_scalar(value: float, *, name: str, minimum: float, maximum: float | None = None) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar, not bool")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if numeric < minimum or (maximum is not None and numeric >= maximum):
        interval = f"[{minimum}, infinity)" if maximum is None else f"[{minimum}, {maximum})"
        raise ValueError(f"{name} must lie in {interval}, got {numeric}")
    return numeric


def _require_positive_scalar(value: float, *, name: str) -> float:
    numeric = _require_finite_scalar(value, name=name, minimum=0.0)
    if numeric == 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return numeric


def _require_finite_tensor(tensor: Tensor, *, name: str) -> None:
    if not isinstance(tensor, Tensor) or not tensor.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")


def _require_text_log_probs(tensor: Tensor, *, name: str) -> None:
    _require_finite_tensor(tensor, name=name)
    if bool((tensor > 0).any()):
        raise ValueError(f"{name} must contain log probabilities no greater than 0")


def compute_reward_advantages(
    reward_batch: RewardBatch,
    *,
    modality: str | None = None,
    weights: Mapping[str, float] | Sequence[float] | Tensor | None = None,
    conditions: Sequence[RewardCondition] = (),
) -> GdpoAdvantageResult:
    """Condition downstream rewards and compute GDPO advantages.

    Verifier exceptions are refused instead of being turned into negative
    rewards or silently removed: dropping rows here would desynchronize policy
    traces and could leave a rollout group with a different normalization
    population from the one the caller believes it sampled.
    """

    del modality  # retained for source compatibility with existing callers
    _validate_reward_batch(reward_batch)

    rewards = torch.as_tensor(reward_batch.matrix, dtype=torch.float32)
    conditioned = (
        apply_conditioned_rewards(
            rewards,
            dimension_names=reward_batch.dimension_names,
            conditions=conditions,
        )
        if conditions
        else rewards
    )
    prepared_weights: Sequence[float] | Tensor | None
    if isinstance(weights, Mapping):
        names = set(reward_batch.dimension_names)
        if set(weights) != names:
            raise ValueError(
                "named reward weights must match the verifier dimensions exactly: "
                f"missing={sorted(names - set(weights))}, "
                f"unexpected={sorted(set(weights) - names)}"
            )
        prepared_weights = [float(weights[name]) for name in reward_batch.dimension_names]
    else:
        prepared_weights = weights
    return compute_gdpo_advantage(
        conditioned,
        rollout_group_ids=reward_batch.group_ids,
        weights=prepared_weights,
        dimension_names=reward_batch.dimension_names,
        kl_in_reward=False,
    )


def compute_uni_gdpo_loss(
    *,
    advantages: Tensor,
    image_replay: ImageSdeReplay | Sequence[ImageSdeReplay] | None,
    text_trace: TextRolloutTrace | None,
    text_log_probs: Tensor | None = None,
    text_ref_log_probs: Tensor | None = None,
    weights: BranchWeights | None = None,
    regularization: RegularizationWeights | None = None,
    image_clip_range: float = 1e-4,
    text_clip_range: float = 0.2,
) -> UniGdpoLoss:
    """Build separate U1.5 policy branches and return one joint objective.

    A TI2TI response may contain several generated images.  Each replay may
    have a different latent resolution, so image events are evaluated
    separately and their policy/KL terms are action-count weighted *inside the
    trajectory*.  The trainer then averages trajectories, matching the text
    branch's sequence-mean/token-mean reduction without concatenating
    incompatible latent tensors.
    """

    if not isinstance(advantages, Tensor):
        raise TypeError("advantages must be a torch.Tensor")
    if advantages.ndim != 1 or not advantages.is_floating_point():
        raise ValueError("advantages must be floating point with shape (batch,)")
    _require_finite_tensor(advantages, name="advantages")
    if image_replay is None:
        image_replays: tuple[ImageSdeReplay, ...] = ()
    elif isinstance(image_replay, ImageSdeReplay):
        image_replays = (image_replay,)
    else:
        image_replays = tuple(image_replay)
        if not all(isinstance(replay, ImageSdeReplay) for replay in image_replays):
            raise TypeError("image_replay sequence must contain only ImageSdeReplay values")

    if not image_replays and text_trace is None:
        raise ValueError("U1.5 UniGDPO needs at least one policy branch")
    resolved_weights = weights if weights is not None else BranchWeights()
    resolved_regularization = regularization if regularization is not None else RegularizationWeights()
    if not isinstance(resolved_weights, BranchWeights):
        raise TypeError("weights must be BranchWeights")
    if not isinstance(resolved_regularization, RegularizationWeights):
        raise TypeError("regularization must be RegularizationWeights")
    image_weight = _require_finite_scalar(resolved_weights.image, name="weights.image", minimum=0.0)
    text_weight = _require_finite_scalar(resolved_weights.text, name="weights.text", minimum=0.0)
    image_velocity_mse_weight = _require_finite_scalar(
        resolved_regularization.image_velocity_mse,
        name="regularization.image_velocity_mse",
        minimum=0.0,
    )
    text_kl_weight = _require_finite_scalar(
        resolved_regularization.text_kl,
        name="regularization.text_kl",
        minimum=0.0,
    )
    image_clip_range = _require_positive_scalar(image_clip_range, name="image_clip_range")
    text_clip_range = _require_positive_scalar(text_clip_range, name="text_clip_range")
    if image_clip_range >= 1.0:
        raise ValueError("image_clip_range must be strictly smaller than 1")
    if text_clip_range >= 1.0:
        raise ValueError("text_clip_range must be strictly smaller than 1")
    batch_sizes: list[int] = []
    image_branches = []
    for replay in image_replays:
        batch_sizes.append(replay.trace.batch_size)
        _require_finite_tensor(replay.log_probs, name="current image log_probs")
        _require_finite_tensor(replay.trace.old_log_probs, name="old image log_probs")
        _require_finite_tensor(replay.means, name="current image transition means")
        _require_finite_tensor(replay.trace.old_means, name="old image transition means")
        _require_finite_tensor(replay.scales, name="image transition scales")
        _require_finite_tensor(replay.velocities, name="current image velocities")
        if not bool((replay.scales > 0).all()):
            raise ValueError("image transition scales must be strictly positive")
        if replay.ref_velocities is not None:
            _require_finite_tensor(replay.ref_velocities, name="reference image velocities")
        if replay.log_probs.device != advantages.device:
            raise ValueError("image replay and advantages must be on the same device")
        image_branches.append(
            replay.branch_inputs(
                advantages=advantages,
                clip_range=image_clip_range,
            )
        )

    text = None
    if text_trace is not None:
        batch_sizes.append(text_trace.batch_size)
        if text_log_probs is None:
            raise ValueError("a text rollout requires differentiable current text_log_probs")
        _require_text_log_probs(text_log_probs, name="current text_log_probs")
        _require_text_log_probs(text_trace.old_log_probs, name="old text log_probs")
        if text_ref_log_probs is not None:
            _require_text_log_probs(text_ref_log_probs, name="reference text_log_probs")
        if (
            text_log_probs.device != advantages.device
            or text_trace.old_log_probs.device != advantages.device
            or (text_ref_log_probs is not None and text_ref_log_probs.device != advantages.device)
        ):
            raise ValueError("text replay, rollout old log-probabilities and advantages must share a device")
        text = text_trace.branch_inputs(
            log_probs=text_log_probs,
            advantages=advantages,
            ref_log_probs=text_ref_log_probs,
            clip_range=text_clip_range,
        )
    elif text_log_probs is not None or text_ref_log_probs is not None:
        raise ValueError("text log-probabilities were supplied without a text rollout trace")

    if any(batch != advantages.shape[0] for batch in batch_sizes):
        raise ValueError(f"policy branch batch sizes {batch_sizes} do not match advantages batch {advantages.shape[0]}")

    # The common primitive remains the sole implementation of each branch's
    # PPO/reference-regularization algebra.  We only aggregate several
    # heterogeneous image events.
    image_parts: list[tuple[int, UniGdpoLoss]] = []
    for replay, branch in zip(image_replays, image_branches, strict=True):
        actions = replay.trace.steps * replay.trace.batch_size
        part = uni_gdpo_loss(
            image=branch,
            text=None,
            weights=BranchWeights(image=1.0, text=0.0),
            regularization=RegularizationWeights(
                image_velocity_mse=image_velocity_mse_weight,
                text_kl=0.0,
            ),
        )
        image_parts.append((actions, part))

    text_part = (
        None
        if text is None
        else uni_gdpo_loss(
            image=None,
            text=text,
            weights=BranchWeights(image=0.0, text=1.0),
            regularization=RegularizationWeights(
                image_velocity_mse=0.0,
                text_kl=text_kl_weight,
            ),
        )
    )

    zero = torch.zeros((), device=advantages.device, dtype=torch.float32)
    image_policy = zero
    image_velocity_mse = zero
    metrics: dict[str, float] = {}
    if image_parts:
        total_actions = sum(actions for actions, _part in image_parts)
        image_policy = sum(part.image_policy * (actions / total_actions) for actions, part in image_parts)
        image_velocity_mse = sum(part.image_velocity_mse * (actions / total_actions) for actions, part in image_parts)
        image_metric_names = {
            name for _actions, part in image_parts for name in part.metrics if name.startswith("image/")
        }
        for name in image_metric_names:
            if name == "image/clip_range":
                metrics[name] = image_clip_range
            else:
                metrics[name] = sum(
                    part.metrics.get(name, 0.0) * (actions / total_actions) for actions, part in image_parts
                )
        metrics["image/events"] = float(len(image_parts))
        metrics["image/actions"] = float(total_actions)

    text_policy = zero if text_part is None else text_part.text_policy
    text_kl = zero if text_part is None else text_part.text_kl
    if text_part is not None:
        metrics.update({name: value for name, value in text_part.metrics.items() if name.startswith("text/")})
    value = (
        image_weight * image_policy
        + image_velocity_mse_weight * image_velocity_mse
        + text_weight * text_policy
        + text_kl_weight * text_kl
    )
    for name, component in (
        ("joint loss", value),
        ("image policy loss", image_policy),
        ("image velocity-MSE loss", image_velocity_mse),
        ("text policy loss", text_policy),
        ("text KL loss", text_kl),
    ):
        _require_finite_tensor(component, name=name)
    non_finite_metrics = [name for name, metric in metrics.items() if not math.isfinite(float(metric))]
    if non_finite_metrics:
        raise ValueError("UniGDPO produced non-finite metrics: " + ", ".join(sorted(non_finite_metrics)))
    metrics["loss"] = float(value.detach())
    return UniGdpoLoss(
        value=value,
        image_policy=image_policy,
        image_velocity_mse=image_velocity_mse,
        text_policy=text_policy,
        text_kl=text_kl,
        metrics=metrics,
    )


__all__ = ["compute_reward_advantages", "compute_uni_gdpo_loss"]
