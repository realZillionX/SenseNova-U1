"""Global sample batches are defined before rank sharding or physical packing."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SampleBatch:
    epoch: int
    sample_start: int
    sample_count: int
    rows: tuple[int, ...]


def sample_batches(*, rows, batch_samples, max_samples, seed, rank=0, world_size=1,
                   worker_id=0, num_workers=1):
    """Yield a worker's ordered rank-local share of global sample batches.

    Each epoch is a fresh global permutation. Batch membership depends only on
    that permutation and sample budgets, never on token lengths or modalities.
    DataLoader's ordered worker interleave reconstructs the global batch order.
    """
    for name, value in (("rows", rows), ("batch_samples", batch_samples),
                        ("max_samples", max_samples), ("world_size", world_size),
                        ("num_workers", num_workers)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not 0 <= rank < world_size or not 0 <= worker_id < num_workers:
        raise ValueError("sample batch rank/worker is outside its topology")
    batch_index = 0
    for epoch in range((max_samples + rows - 1) // rows):
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(rows)
        limit = min(rows, max_samples - epoch * rows)
        for start in range(0, limit, batch_samples):
            end = min(start + batch_samples, limit)
            if batch_index % num_workers == worker_id:
                yield SampleBatch(epoch, epoch * rows + start, end - start,
                                  tuple(int(row) for row in order[start:end][rank::world_size]))
            batch_index += 1


def optimizer_updates(*, rows, batch_samples, max_samples):
    full_epochs, remainder = divmod(max_samples, rows)
    return full_epochs * ((rows + batch_samples - 1) // batch_samples) + (remainder + batch_samples - 1) // batch_samples
