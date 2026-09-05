"""Per-run budget ledger.

Doctrine 5 requires the paper to prove "stronger" and "more efficient"
separately: a visual method that buys accuracy with a larger generation and
verification budget may claim a capability gain but not an efficiency gain.
That is only checkable if every run records what it spent, so this ledger is a
mandatory run output rather than an optional instrumentation nicety.

It counts what a run actually pays for and nothing else -- consumed samples,
generated rollouts and their token/image volume, verifier invocations,
and wall clock split between generating rollouts and verifying them.  No
derived rates, no efficiency scores: those are analysis, and computing them
here would invite reporting a number no run ever filled in.

The code that spends a resource records it: ``score_rollouts`` records verifier
route calls, while a backend rollout loop records samples, completions, media,
and generation time.  Backends that delegate execution to an external training
framework must translate that framework's run metrics into this schema before
comparing efficiency.

Summaries are written outside the repository.  Run artifacts are never
committed, so :meth:`BudgetLedger.write_json` refuses a path inside it.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class BudgetLedger:
    """Additive tally of one run's generation and verification spend."""

    samples: int = 0
    rollouts: int = 0
    generated_text_tokens: int = 0
    generated_image_context_tokens: int = 0
    generated_images: int = 0
    length_truncated_rollouts: int = 0
    image_limit_hits: int = 0
    rollout_seconds: float = 0.0
    verifier_seconds: float = 0.0
    verifier_invocations: int = 0

    def record_samples(self, count: int = 1) -> None:
        """Count prompts consumed, independent of how many rollouts each got."""

        self.samples += int(count)

    def record_rollouts(
        self,
        count: int = 1,
        *,
        text_tokens: int = 0,
        image_context_tokens: int = 0,
        images: int = 0,
        seconds: float = 0.0,
        truncated: bool = False,
        image_limit_hit: bool = False,
    ) -> None:
        """Count generated rollouts and everything their generation produced."""

        self.rollouts += int(count)
        self.generated_text_tokens += int(text_tokens)
        self.generated_image_context_tokens += int(image_context_tokens)
        self.generated_images += int(images)
        self.length_truncated_rollouts += int(truncated)
        self.image_limit_hits += int(image_limit_hit)
        self.rollout_seconds += float(seconds)

    def record_verification(self, invocations: int = 1, *, seconds: float = 0.0) -> None:
        """Count verifier-route calls and the time they took.

        ``score_rollouts`` fills this in itself, once per route call, when it is
        handed a ledger.
        """

        self.verifier_invocations += int(invocations)
        self.verifier_seconds += float(seconds)

    @contextmanager
    def time_rollouts(self) -> Iterator[None]:
        """Add the enclosed wall clock to ``rollout_seconds``."""

        start = time.perf_counter()
        try:
            yield
        finally:
            self.rollout_seconds += time.perf_counter() - start

    @contextmanager
    def time_verification(self) -> Iterator[None]:
        """Add the enclosed wall clock to ``verifier_seconds``."""

        start = time.perf_counter()
        try:
            yield
        finally:
            self.verifier_seconds += time.perf_counter() - start

    def merge(self, other: "BudgetLedger") -> None:
        """Fold another worker's ledger into this one."""

        self.samples += other.samples
        self.rollouts += other.rollouts
        self.generated_text_tokens += other.generated_text_tokens
        self.generated_image_context_tokens += other.generated_image_context_tokens
        self.generated_images += other.generated_images
        self.length_truncated_rollouts += other.length_truncated_rollouts
        self.image_limit_hits += other.image_limit_hits
        self.rollout_seconds += other.rollout_seconds
        self.verifier_seconds += other.verifier_seconds
        self.verifier_invocations += other.verifier_invocations

    @classmethod
    def merged(cls, ledgers: Iterable["BudgetLedger"]) -> "BudgetLedger":
        """Return the sum of per-worker ledgers.

        Wall clock adds across workers, so the seconds fields report aggregate
        compute time, not the run's elapsed time.
        """

        total = cls()
        for ledger in ledgers:
            total.merge(ledger)
        return total

    def as_dict(self) -> Dict[str, Any]:
        """Return the plain summary dict."""

        return asdict(self)

    def write_json(self, path: Path) -> Path:
        """Write the summary outside the repository and return the path."""

        destination = Path(path).expanduser().resolve()
        if destination == _REPO_ROOT or _REPO_ROOT in destination.parents:
            raise ValueError(
                f"budget summaries are run artifacts and must be written outside {_REPO_ROOT}, got {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return destination


__all__ = ["BudgetLedger"]
