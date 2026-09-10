"""Deterministic global prompt permutations, shared by both training arms."""

from functools import lru_cache
import random


@lru_cache(maxsize=2)
def _epoch_order(count: int, seed: int, epoch: int) -> tuple[int, ...]:
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    return tuple(order)


def prompt_batch_indices(
    batch_index: int, *, count: int, prompts_per_batch: int, seed: int
) -> tuple[int, ...]:
    """Draw full batches without replacement; reshuffle the entire pool each epoch.

    A remainder smaller than one batch is dropped before the next permutation.
    The order depends only on the shared seed, epoch and input row positions,
    never on modality, rank, run identity or rollout RNG consumption.
    """
    if type(batch_index) is not int or batch_index < 0:
        raise ValueError("batch_index must be a non-negative integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if prompts_per_batch < 2 or count < prompts_per_batch:
        raise ValueError("GDPO requires a complete batch of at least two distinct prompts")
    batches_per_epoch = count // prompts_per_batch
    epoch, position = divmod(batch_index, batches_per_epoch)
    start = position * prompts_per_batch
    return _epoch_order(count, seed, epoch)[start:start + prompts_per_batch]
