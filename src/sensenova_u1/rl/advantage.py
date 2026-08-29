"""GDPO advantage estimation, backend independent.

The pipeline contract fixes the order of operations, and this module is the
single implementation available to native model backends:

1. per reward dimension, per rollout group: subtract the group mean and divide
   by the group standard deviation;
2. weighted sum of the per-dimension *normalized* advantages;
3. one batch-level normalization of that aggregate over every rollout in the
   optimizer batch, exactly as GDPO defines it.  Independent model arms make a
   SenseNova batch single-modality by construction; task/Family/domain labels
   must not split the final statistic.

Summing raw rewards first and normalizing once is not GDPO.  It exists here
only as :func:`compute_rawsum_grpo_advantage`, a separately named ablation arm,
and never as a configuration flag on the GDPO path.

Availability follows the canonical manifest: a reward dimension that does not
apply to a sample's modality arrives as ``NaN``, never as a constant dummy.  A
``NaN`` entry contributes exactly ``0`` to the aggregate and never poisons
another sample's statistics.

Numerical semantics, fixed here because the two reference implementations do
not agree on all of them: standard deviations are Bessel corrected and NaN
aware; a group with at most one available value contributes exactly ``0``
rather than passing the raw reward through; the group stage divides by
``std + eps_group`` and the batch stage by ``std + eps_batch``; all
moments are accumulated in float64 regardless of the input dtype, and the
returned tensors are cast back to it.

Rewards must be the raw final-answer reward vector produced by the ``data/``
verifiers.  KL belongs in the loss and must never be folded into a reward, so
this module refuses a KL-adjusted input surface: it accepts no ``beta`` or
``kl_values`` argument at all, and the one ``kl_in_reward`` keyword it does
accept raises on anything but ``False``.  The keyword is kept precisely so that
code ported from ms-swift -- which passes ``kl_in_reward`` and merely warns
when it conflicts with GDPO -- fails loudly here instead of silently dropping
the penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

ESTIMATOR_GDPO = "gdpo"
ESTIMATOR_RAWSUM_GRPO = "rawsum_grpo"

_Weights = Union[Tensor, Sequence[float], None]


@dataclass(frozen=True)
class GroupStatistic:
    """Group-wise moments used by the group normalization stage.

    On the GDPO path there is one record per ``(group, reward dimension)``.  On
    the ablation path the group stage normalizes the summed reward instead of
    each dimension, so there is one record per group and both dimension fields
    are ``None``.
    """

    group_id: Hashable
    dimension_index: Optional[int]
    dimension_name: Optional[str]
    member_count: int
    available_count: int
    mean: float
    std: float
    normalized: bool


@dataclass(frozen=True)
class DomainStatistic:
    """Batch-normalization moments of one domain, for per-run reporting."""

    domain_id: Hashable
    member_count: int
    mean: float
    std: float
    normalized: bool


@dataclass(frozen=True)
class GdpoAdvantageResult:
    """Advantages plus every statistic a training loop must report.

    ``per_dimension_advantages`` holds the group-normalized advantage of each
    reward dimension on the GDPO path.  On the ablation path there is no
    per-dimension normalization stage, so the field holds the weighted raw
    contributions instead; ``estimator`` says which one it is.

    ``aggregate`` is the pre-batch-normalization weighted sum, kept because it
    is the quantity the two estimators actually disagree about.
    """

    estimator: str
    advantages: Tensor
    per_dimension_advantages: Tensor
    aggregate: Tensor
    availability: Tensor
    weights: Tensor
    dimension_names: Optional[Tuple[str, ...]]
    group_statistics: Tuple[GroupStatistic, ...]
    domain_statistics: Tuple[DomainStatistic, ...]


def reject_kl_in_reward(kl_in_reward: bool) -> None:
    """Refuse any attempt to fold a KL penalty into the reward vector."""

    if kl_in_reward:
        raise ValueError(
            "kl_in_reward is not supported: the KL penalty belongs in the loss, "
            "never in the reward. Subtracting "
            "it before normalization would make it one more unnamed reward "
            "dimension and silently break the availability mask."
        )


def _as_index(
    ids: Sequence[Hashable],
    expected: int,
    label: str,
    device: torch.device,
) -> Tuple[Tensor, Tuple[Hashable, ...]]:
    values = list(ids)
    if len(values) != expected:
        raise ValueError(f"{label} has {len(values)} entries, expected {expected}")
    order: List[Hashable] = []
    lookup: dict = {}
    for value in values:
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(f"{label} entries must be hashable, got {value!r}") from exc
        if value not in lookup:
            lookup[value] = len(order)
            order.append(value)
    index = torch.tensor([lookup[value] for value in values], dtype=torch.long, device=device)
    return index, tuple(order)


def _prepare_rewards(rewards: Tensor) -> Tuple[Tensor, Tensor]:
    """Validate the reward matrix and promote it to the float64 working dtype.

    Both normalization stages subtract a mean before dividing by a standard
    deviation, so a group or domain whose values are (near) identical is a
    catastrophic-cancellation site: in float32 the leftover rounding error is
    of the same order as ``eps``, and dividing one by the other turns pure
    noise into advantages of order 1.  float64 pushes the residual eight orders
    of magnitude below ``eps``, which is what actually makes a zero-variance
    domain collapse to zero.  The cost is negligible -- this is an ``(N, D)``
    reduction, not a model forward.
    """

    if not isinstance(rewards, Tensor):
        raise TypeError(f"rewards must be a torch.Tensor, got {type(rewards).__name__}")
    if rewards.dim() != 2:
        raise ValueError(f"rewards must be 2-D (N, D), got shape {tuple(rewards.shape)}")
    if rewards.numel() == 0:
        raise ValueError("rewards must contain at least one sample and one dimension")
    if not rewards.is_floating_point():
        raise TypeError("rewards must be a floating point tensor; NaN marks unavailability")
    values = rewards.detach().to(torch.float64)
    available = ~torch.isnan(values)
    if torch.isinf(values[available]).any():
        raise ValueError("rewards contain infinite values; only finite values or NaN are allowed")
    return values, available


def _prepare_weights(weights: _Weights, dimensions: int, reference: Tensor) -> Tensor:
    if weights is None:
        return torch.ones(dimensions, dtype=reference.dtype, device=reference.device)
    if isinstance(weights, Tensor):
        prepared = weights.detach().to(dtype=reference.dtype, device=reference.device)
    else:
        prepared = torch.tensor(list(weights), dtype=reference.dtype, device=reference.device)
    if prepared.dim() != 1 or prepared.numel() != dimensions:
        raise ValueError(f"weights must have shape ({dimensions},), got {tuple(prepared.shape)}")
    if not torch.isfinite(prepared).all():
        raise ValueError("weights must all be finite")
    return prepared


def _prepare_dimension_names(
    dimension_names: Optional[Sequence[str]],
    dimensions: int,
) -> Optional[Tuple[str, ...]]:
    if dimension_names is None:
        return None
    names = tuple(str(name) for name in dimension_names)
    if len(names) != dimensions:
        raise ValueError(f"dimension_names has {len(names)} entries, expected {dimensions}")
    if len(set(names)) != len(names):
        raise ValueError("dimension_names must be unique")
    return names


def _moments(
    values: Tensor,
    available: Tensor,
    index: Tensor,
    num_groups: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """NaN-aware, Bessel-corrected per-group mean and std.

    Groups with no available value, or exactly one, yield a zero std; the
    caller must gate on ``counts`` rather than on the std, because a zero std
    also legitimately arises from a group whose values are all identical.
    """

    dimensions = values.shape[1]
    zeros = torch.zeros_like(values)
    counts = values.new_zeros((num_groups, dimensions))
    counts.index_add_(0, index, available.to(values.dtype))
    filled = torch.where(available, values, zeros)
    totals = values.new_zeros((num_groups, dimensions))
    totals.index_add_(0, index, filled)
    mean = totals / counts.clamp(min=1.0)
    deviation = torch.where(available, filled - mean[index], zeros)
    squares = values.new_zeros((num_groups, dimensions))
    squares.index_add_(0, index, deviation * deviation)
    std = torch.sqrt(squares / (counts - 1.0).clamp(min=1.0))
    return counts, mean, std


def _member_counts(index: Tensor, num_groups: int, reference: Tensor) -> Tensor:
    counts = reference.new_zeros(num_groups)
    counts.index_add_(0, index, torch.ones_like(index, dtype=reference.dtype))
    return counts


def _per_dimension_group_statistics(
    group_ids: Tuple[Hashable, ...],
    dimension_names: Optional[Tuple[str, ...]],
    member_counts: Tensor,
    counts: Tensor,
    mean: Tensor,
    std: Tensor,
) -> Tuple[GroupStatistic, ...]:
    members = member_counts.tolist()
    available = counts.tolist()
    means = mean.tolist()
    stds = std.tolist()
    records: List[GroupStatistic] = []
    for position, group_id in enumerate(group_ids):
        for dimension in range(counts.shape[1]):
            available_count = int(available[position][dimension])
            records.append(
                GroupStatistic(
                    group_id=group_id,
                    dimension_index=dimension,
                    dimension_name=None if dimension_names is None else dimension_names[dimension],
                    member_count=int(members[position]),
                    available_count=available_count,
                    mean=float(means[position][dimension]),
                    std=float(stds[position][dimension]),
                    normalized=available_count > 1,
                )
            )
    return tuple(records)


def _aggregate_group_statistics(
    group_ids: Tuple[Hashable, ...],
    counts: Tensor,
    mean: Tensor,
    std: Tensor,
) -> Tuple[GroupStatistic, ...]:
    members = counts.tolist()
    means = mean.tolist()
    stds = std.tolist()
    return tuple(
        GroupStatistic(
            group_id=group_id,
            dimension_index=None,
            dimension_name=None,
            member_count=int(members[position]),
            available_count=int(members[position]),
            mean=float(means[position]),
            std=float(stds[position]),
            normalized=members[position] > 1,
        )
        for position, group_id in enumerate(group_ids)
    )


def _normalize_batch(
    aggregate: Tensor,
    eps_batch: float,
) -> Tuple[Tensor, Tuple[DomainStatistic, ...]]:
    """Standardize the aggregate once over the complete optimizer batch.

    A batch with one member has no dispersion and yields advantage 0.  Formal
    SenseNova plans require multiple prompt groups, but keeping this numerical
    boundary explicit makes the estimator fail predictably in probes.

    ``eps_batch`` is added to the batch standard deviation, matching GDPO Eq. 6
    and ms-swift's GDPO implementation.  Analytically constant aggregates have
    an exact zero numerator under float64 accumulation and therefore remain
    zero; no alternative variance-floor convention is introduced.
    """

    count = aggregate.numel()
    mean = aggregate.mean()
    std = aggregate.std(correction=1) if count > 1 else aggregate.new_zeros(())
    advantages = (aggregate - mean) / (std + eps_batch) if count > 1 else torch.zeros_like(aggregate)
    statistics = (
        DomainStatistic(
            domain_id="optimizer_batch",
            member_count=count,
            mean=float(mean),
            std=float(std),
            normalized=count > 1,
        ),
    )
    return advantages, statistics


def compute_gdpo_advantage(
    rewards: Tensor,
    *,
    rollout_group_ids: Sequence[Hashable],
    weights: _Weights = None,
    dimension_names: Optional[Sequence[str]] = None,
    eps_group: float = 1e-8,
    eps_batch: float = 1e-8,
    kl_in_reward: bool = False,
) -> GdpoAdvantageResult:
    """Compute GDPO advantages from an unaggregated raw reward matrix.

    ``rewards`` is ``(N, D)``; ``NaN`` means the dimension is unavailable for
    that sample.  ``rollout_group_ids`` scopes step 1 (one entry per sample,
    normally the manifest's rollout group key).  Step 3 always uses every row
    in ``rewards``; a caller cannot partition it by task or modality.

    A ``(group, dimension)`` cell with at most one available value has no
    within-group contrast and contributes exactly ``0``, matching ms-swift's
    ``nanstd``/``nan_to_num`` behaviour.  verl's per-dimension normalizer
    instead substitutes ``mean=0, std=1`` for a singleton group, which passes
    the raw reward through as an advantage; that convention is rejected here
    because it makes an uncontrasted sample's absolute reward scale leak into
    the aggregate.

    ``eps_group`` and ``eps_batch`` are added to their respective standard
    deviations, matching the GDPO paper's normalization order and ms-swift.
    """

    reject_kl_in_reward(kl_in_reward)
    values, available = _prepare_rewards(rewards)
    num_samples, dimensions = values.shape
    names = _prepare_dimension_names(dimension_names, dimensions)
    weight_vector = _prepare_weights(weights, dimensions, values)
    index, group_ids = _as_index(rollout_group_ids, num_samples, "rollout_group_ids", values.device)

    counts, mean, std = _moments(values, available, index, len(group_ids))
    usable = available & (counts[index] > 1.0)
    filled = torch.where(available, values, torch.zeros_like(values))
    per_dimension = torch.where(
        usable,
        (filled - mean[index]) / (std[index] + eps_group),
        torch.zeros_like(values),
    )
    aggregate = (per_dimension * weight_vector).sum(dim=1)
    advantages, domain_statistics = _normalize_batch(aggregate, eps_batch)

    out_dtype = rewards.dtype
    return GdpoAdvantageResult(
        estimator=ESTIMATOR_GDPO,
        advantages=advantages.to(out_dtype),
        per_dimension_advantages=per_dimension.to(out_dtype),
        aggregate=aggregate.to(out_dtype),
        availability=available,
        weights=weight_vector.to(out_dtype),
        dimension_names=names,
        group_statistics=_per_dimension_group_statistics(
            group_ids,
            names,
            _member_counts(index, len(group_ids), values),
            counts,
            mean,
            std,
        ),
        domain_statistics=domain_statistics,
    )


def compute_rawsum_grpo_advantage(
    rewards: Tensor,
    *,
    rollout_group_ids: Sequence[Hashable],
    weights: _Weights = None,
    dimension_names: Optional[Sequence[str]] = None,
    eps_group: float = 1e-8,
    kl_in_reward: bool = False,
) -> GdpoAdvantageResult:
    """The controlled ablation contrast. **This is not GDPO.**

    It sums the weighted raw rewards first (``nansum`` semantics: unavailable
    dimensions drop out of the sum) and then applies a single group-wise
    normalization -- the classic GRPO scalar-reward estimator.  This ordering
    is explicitly *not* GDPO and must never be reachable as an option on the
    GDPO path.

    The arm deliberately has one normalization stage, so it takes no batch
    normalization arguments and returns no domain statistics.  Adding a batch
    stage would make it a second variant rather than the contrast the ablation
    is defined as.
    """

    reject_kl_in_reward(kl_in_reward)
    values, available = _prepare_rewards(rewards)
    num_samples, dimensions = values.shape
    names = _prepare_dimension_names(dimension_names, dimensions)
    weight_vector = _prepare_weights(weights, dimensions, values)
    index, group_ids = _as_index(rollout_group_ids, num_samples, "rollout_group_ids", values.device)

    contributions = torch.where(available, values * weight_vector, torch.zeros_like(values))
    aggregate = contributions.sum(dim=1)

    column = aggregate.unsqueeze(1)
    column_available = torch.ones_like(column, dtype=torch.bool)
    counts, mean, std = _moments(column, column_available, index, len(group_ids))
    usable = counts[index, 0] > 1.0
    advantages = torch.where(
        usable,
        (aggregate - mean[index, 0]) / (std[index, 0] + eps_group),
        torch.zeros_like(aggregate),
    )

    out_dtype = rewards.dtype
    return GdpoAdvantageResult(
        estimator=ESTIMATOR_RAWSUM_GRPO,
        advantages=advantages.to(out_dtype),
        per_dimension_advantages=contributions.to(out_dtype),
        aggregate=aggregate.to(out_dtype),
        availability=available,
        weights=weight_vector.to(out_dtype),
        dimension_names=names,
        group_statistics=_aggregate_group_statistics(group_ids, counts[:, 0], mean[:, 0], std[:, 0]),
        domain_statistics=(),
    )


__all__ = [
    "DomainStatistic",
    "ESTIMATOR_GDPO",
    "ESTIMATOR_RAWSUM_GRPO",
    "GdpoAdvantageResult",
    "GroupStatistic",
    "compute_gdpo_advantage",
    "compute_rawsum_grpo_advantage",
    "reject_kl_in_reward",
]
