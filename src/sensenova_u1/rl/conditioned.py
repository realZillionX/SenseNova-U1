"""Conditioned rewards: how downstream hard gates enter GDPO.

The GDPO paper pays a secondary reward dimension only once the dimension it
depends on has cleared a threshold.  A downstream verifier may use that form
hard gates -- preferred over weight tuning, because
a weight cannot express "this score is not even measurable yet", while a
condition can.

Conditioning happens on the raw reward matrix, before any GDPO normalization.
It only ever writes ``0`` into a dimension that was available and unearned; an
unavailable dimension stays ``NaN`` (availability masks, never dummies).

Every condition is evaluated against the *original* matrix in one pass, so the
result does not depend on condition order.  That is why
:func:`conditions_from_hard_gates` emits full, already-transitive prerequisite
lists instead of relying on a cascade.

Derivation rule for a downstream modality's declared hard gates -- two layers, each
conditioned on every declared gate in the layers below it:

* **Layer 0, availability and structure** -- ``text_parse`` (the delivery
  envelope carries one parseable typed payload).  Unconditioned: gating it
  would leave a policy that has not yet learned the delivery envelope with no
  gradient at all.
* **Layer 1, delivery semantics** -- ``text_semantic``.  Semantic credit is
  paid only on something that is formally a delivery: an unparseable payload
  has no semantics to score.

Both modalities deliver the same single typed ``Answer:`` line, so both derive
the same conditioning: the semantic reward is withheld while the answer line is
malformed.  Under decoupled normalization it would otherwise hand out positive
advantage for a response that is not a legal final delivery at all, which is
precisely the partial-credit hack conditioned rewards exist to close.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

import torch
from torch import Tensor

_LAYER_0: Tuple[str, ...] = ("text_parse",)
_LAYER_1: Tuple[str, ...] = ("text_semantic",)


@dataclass(frozen=True)
class RewardCondition:
    """``dimension`` is paid only when every dimension in ``requires`` reaches ``threshold``."""

    dimension: str
    requires: Tuple[str, ...]
    threshold: float = 1.0

    def __post_init__(self) -> None:
        if not self.requires:
            raise ValueError(f"{self.dimension}: a condition needs at least one prerequisite")
        if self.dimension in self.requires:
            raise ValueError(f"{self.dimension}: a dimension cannot be its own prerequisite")
        if len(set(self.requires)) != len(self.requires):
            raise ValueError(f"{self.dimension}: duplicate prerequisite")


def apply_conditioned_rewards(
    rewards: Tensor,
    *,
    dimension_names: Sequence[str],
    conditions: Sequence[RewardCondition],
) -> Tensor:
    """Zero every conditioned dimension whose prerequisites are unmet.

    A prerequisite counts as met only when its value is present and at least
    ``threshold``; a ``NaN`` prerequisite is unmet, so the conditioned
    dimension fails closed rather than inheriting missing evidence.  Entries
    that are themselves ``NaN`` keep their unavailability.
    """

    if not isinstance(rewards, Tensor):
        raise TypeError(f"rewards must be a torch.Tensor, got {type(rewards).__name__}")
    if rewards.dim() != 2:
        raise ValueError(f"rewards must be 2-D (N, D), got shape {tuple(rewards.shape)}")
    if not rewards.is_floating_point():
        raise TypeError("rewards must be a floating point tensor; NaN marks unavailability")

    names = tuple(str(name) for name in dimension_names)
    if len(names) != rewards.shape[1]:
        raise ValueError(f"dimension_names has {len(names)} entries, expected {rewards.shape[1]}")
    if len(set(names)) != len(names):
        raise ValueError("dimension_names must be unique")
    position = {name: column for column, name in enumerate(names)}

    conditioned = rewards.detach().clone()
    seen: Set[str] = set()
    for condition in conditions:
        if condition.dimension not in position:
            raise ValueError(f"unknown conditioned dimension {condition.dimension!r}")
        if condition.dimension in seen:
            raise ValueError(f"duplicate condition for dimension {condition.dimension!r}")
        seen.add(condition.dimension)
        column = position[condition.dimension]
        met = torch.ones(rewards.shape[0], dtype=torch.bool, device=rewards.device)
        for requirement in condition.requires:
            if requirement not in position:
                raise ValueError(f"{condition.dimension}: unknown prerequisite {requirement!r}")
            source = rewards[:, position[requirement]]
            met &= ~torch.isnan(source) & (source >= condition.threshold)
        available = ~torch.isnan(rewards[:, column])
        conditioned[:, column] = torch.where(
            available & ~met,
            torch.zeros_like(conditioned[:, column]),
            conditioned[:, column],
        )
    return conditioned


def conditions_from_hard_gates(
    hard_gates: Sequence[str],
    dimension_names: Sequence[str],
) -> Tuple[RewardCondition, ...]:
    """Derive one modality's conditions from its declared final-answer hard gates.

    ``hard_gates`` is the modality's entry in the manifest's
    ``reward_schema.hard_gates``.  Only declared gates participate, both as
    conditioned dimensions and as prerequisites, and the emitted ``requires``
    tuples follow ``dimension_names`` order so the result is deterministic.
    Any gate this module does not recognize raises: a new reward dimension must
    be placed in a layer deliberately, never defaulted into one.
    """

    names = tuple(str(name) for name in dimension_names)
    if len(set(names)) != len(names):
        raise ValueError("dimension_names must be unique")
    known = set(names)
    order = {name: column for column, name in enumerate(names)}

    gates = tuple(str(gate) for gate in hard_gates)
    if len(set(gates)) != len(gates):
        raise ValueError("hard_gates must not repeat a dimension")
    for gate in gates:
        if gate not in known:
            raise ValueError(f"hard gate {gate!r} is not a declared reward dimension")
        if gate not in _LAYER_0 + _LAYER_1:
            raise ValueError(f"hard gate {gate!r} has no declared conditioning layer")

    declared = set(gates)
    layer_0 = tuple(gate for gate in _LAYER_0 if gate in declared)

    conditions: List[RewardCondition] = []
    for gate in gates:
        if gate in _LAYER_0:
            continue
        prerequisites = tuple(sorted((name for name in layer_0 if name != gate), key=order.__getitem__))
        if not prerequisites:
            continue
        conditions.append(RewardCondition(dimension=gate, requires=prerequisites))
    return tuple(conditions)


__all__ = [
    "RewardCondition",
    "apply_conditioned_rewards",
    "conditions_from_hard_gates",
]
