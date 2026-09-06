"""Deterministic global JSONL ordering before data-rank/worker sharding."""

from array import array
import mmap

import numpy as np


def annotation_offsets(path):
    """Index byte boundaries without keeping JSON records or copying media."""
    offsets = array("Q", [0])
    with open(path, "rb") as stream:
        while stream.readline():
            offsets.append(stream.tell())
    return offsets


def shard_order(row_count, *, seed, epoch, shard_id, shard_count, shuffle=True):
    if row_count < 1 or shard_count < 1 or not 0 <= shard_id < shard_count:
        raise ValueError("invalid annotation size or reader shard")
    if row_count < shard_count:
        raise ValueError("every reader shard must own at least one annotation row")
    if shuffle:
        # Independent of file bytes, response lengths and the arm's modality.
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(row_count)
    else:
        order = np.arange(row_count)
    return order[shard_id::shard_count]


def indexed_lines(path, offsets, order):
    with open(path, "rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        for row in order:
            start, end = offsets[int(row)], offsets[int(row) + 1]
            yield mapped[start:end]
