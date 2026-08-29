"""Backend-neutral rollout and reward types for SenseNova-U1.5 RL.

Forge owns the optimizer-facing shape of a reward batch, but it deliberately
does not own task semantics or verifier execution.  Downstream projects may
use any reward dimensions as long as they return this typed matrix contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, TypeAlias

import numpy as np

MODALITIES = ("ti2t", "ti2ti")


@dataclass(frozen=True)
class TextSegment:
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text segment must contain a string")


@dataclass(frozen=True)
class ImageSegment:
    path: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("image segment path must be non-empty")


ResponseItem: TypeAlias = TextSegment | ImageSegment


@dataclass(frozen=True)
class CandidateResponse:
    modality: str
    items: tuple[ResponseItem, ...] = ()

    def __post_init__(self) -> None:
        if self.modality not in MODALITIES:
            raise ValueError(f"unsupported modality {self.modality!r}")
        if not isinstance(self.items, tuple):
            raise TypeError("response items must be a tuple")
        if any(not isinstance(item, (TextSegment, ImageSegment)) for item in self.items):
            raise TypeError("response items must be text or image segments")

    @property
    def text(self) -> str:
        return "".join(item.text for item in self.items if isinstance(item, TextSegment))

    def to_dict(self) -> dict[str, object]:
        return {
            "modality": self.modality,
            "items": [
                {"type": "text", "text": item.text}
                if isinstance(item, TextSegment)
                else {"type": "image", "path": item.path}
                for item in self.items
            ],
        }


@dataclass(frozen=True)
class RewardBatch:
    """Raw downstream reward evidence in rollout order.

    Unavailable cells are represented by ``NaN`` and ``availability=False``.
    Any row with a non-``None`` error is unscorable and must abort or be
    retried before advantage computation.
    """

    matrix: np.ndarray
    dimension_names: tuple[str, ...]
    availability: np.ndarray
    group_ids: tuple[str, ...]
    errors: tuple[str | None, ...]
    diagnostics: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.matrix, np.ndarray) or self.matrix.ndim != 2:
            raise TypeError("reward matrix must be a rank-2 numpy array")
        rows, columns = self.matrix.shape
        if rows < 1 or columns < 1:
            raise ValueError("reward matrix must be non-empty")
        if len(self.dimension_names) != columns or len(set(self.dimension_names)) != columns:
            raise ValueError("reward dimension names must be unique and match the matrix")
        if self.availability.shape != self.matrix.shape or self.availability.dtype != np.bool_:
            raise ValueError("reward availability must be a boolean matrix of the same shape")
        if len(self.group_ids) != rows or len(self.errors) != rows:
            raise ValueError("reward row metadata must match the matrix")
        if self.diagnostics and len(self.diagnostics) != rows:
            raise ValueError("reward diagnostics must be empty or have one entry per row")
        if not np.array_equal(np.isnan(self.matrix), ~self.availability):
            raise ValueError("reward NaNs must exactly match unavailable cells")
        if not np.isfinite(self.matrix[self.availability]).all():
            raise ValueError("available reward values must be finite")

    def __len__(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def scored(self) -> np.ndarray:
        return np.asarray([error is None for error in self.errors], dtype=bool)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RewardBatch":
        required = {"matrix", "dimension_names", "availability", "group_ids", "errors"}
        if not required <= set(payload):
            raise ValueError(f"reward payload is missing {sorted(required - set(payload))}")
        matrix = np.asarray(payload["matrix"], dtype=np.float32)
        availability = np.asarray(payload["availability"], dtype=np.bool_)
        diagnostics_raw = payload.get("diagnostics") or [{} for _ in range(len(matrix))]
        if not isinstance(diagnostics_raw, Sequence):
            raise TypeError("reward diagnostics must be a sequence")
        return cls(
            matrix=matrix,
            dimension_names=tuple(str(value) for value in payload["dimension_names"]),
            availability=availability,
            group_ids=tuple(str(value) for value in payload["group_ids"]),
            errors=tuple(None if value is None else str(value) for value in payload["errors"]),
            diagnostics=tuple(dict(value) for value in diagnostics_raw),
        )


# Defaults model the two public reasoning modes.  Callers may override the
# system message per prompt row without changing Forge.
TEXT_THINK_SYSTEM_MESSAGE = (
    "You are a multimodal assistant capable of reasoning in text. When "
    "reasoning is needed, place it inside <think></think>, use text alone, "
    "and then provide a concise final answer."
)
INTERLEAVED_THINK_SYSTEM_MESSAGE = (
    "You are a multimodal assistant capable of reasoning with text and images. "
    "When reasoning is needed, place it inside <think></think>; generated "
    "reasoning images may appear only inside that block. Then provide a "
    "concise final answer."
)
SYSTEM_MESSAGE_BY_MODALITY = {
    "ti2t": TEXT_THINK_SYSTEM_MESSAGE,
    "ti2ti": INTERLEAVED_THINK_SYSTEM_MESSAGE,
}


__all__ = [
    "CandidateResponse",
    "ImageSegment",
    "INTERLEAVED_THINK_SYSTEM_MESSAGE",
    "MODALITIES",
    "RewardBatch",
    "SYSTEM_MESSAGE_BY_MODALITY",
    "TEXT_THINK_SYSTEM_MESSAGE",
    "TextSegment",
]
