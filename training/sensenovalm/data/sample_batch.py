"""Global sample batches are defined before rank sharding or physical packing."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SampleBatch:
    epoch: int
    sample_start: int
    sample_count: int
    rows: tuple[int, ...]
    global_rows: tuple[int, ...]


def sample_batches(*, rows, batch_samples, max_samples, seed, rank=0, world_size=1,
                   worker_id=0, num_workers=1, start_samples=0):
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
    if type(start_samples) is not int or not 0 <= start_samples < max_samples:
        raise ValueError("start_samples must be inside the declared sample budget")
    if start_samples % rows % batch_samples:
        raise ValueError("start_samples must be a completed optimizer batch boundary")
    batch_index = 0
    for epoch in range(start_samples // rows, (max_samples + rows - 1) // rows):
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(rows)
        limit = min(rows, max_samples - epoch * rows)
        first = start_samples % rows if epoch == start_samples // rows else 0
        for start in range(first, limit, batch_samples):
            end = min(start + batch_samples, limit)
            if batch_index % num_workers == worker_id:
                yield SampleBatch(epoch, epoch * rows + start, end - start,
                                  tuple(int(row) for row in order[start:end][rank::world_size]),
                                  tuple(int(row) for row in order[start:end]))
            batch_index += 1


def optimizer_updates(*, rows, batch_samples, max_samples):
    full_epochs, remainder = divmod(max_samples, rows)
    return full_epochs * ((rows + batch_samples - 1) // batch_samples) + (remainder + batch_samples - 1) // batch_samples


def balance_rows(rows, costs, world_size):
    """Assign the fixed batch to ranks by estimated token work, without loss weighting."""
    if len(rows) != len(costs) or world_size < 1:
        raise ValueError("invalid sample scheduling costs or world size")
    assignments = [[] for _ in range(world_size)]
    loads = [0] * world_size
    for row, cost in sorted(zip(rows, costs), key=lambda pair: -pair[1]):
        rank = min(range(world_size), key=lambda index: (loads[index], len(assignments[index]), index))
        assignments[rank].append(row)
        loads[rank] += max(1, cost)
    return assignments
