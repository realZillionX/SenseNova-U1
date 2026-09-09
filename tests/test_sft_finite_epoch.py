"""Finite packing must flush its tail once, including uneven rank workloads."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sensenovavl.data.dataset_interleaved_iterable as packing
import torch
from sensenovavl.data.dataset_interleaved_iterable import PackedDataset
from sensenovavl.data.multimodal_dataset import LazySupervisedDataset


class Reader(LazySupervisedDataset):
    def __init__(self, ids, factor):
        self.ids, self.factor = ids, factor
        self.ds_name, self.dataset_type = "fixture", "multimodal"
        self.length = len(ids)
        self._state_dict = {}
        self.starts = 0

    def __iter__(self):
        self.starts += 1
        return iter(self.ids)

    def get_sample(self, index):
        size = self.factor * (index % 3 + 2)
        return dict(
            input_ids=torch.full((size,), 7),
            labels=torch.full((size,), 7),
            type_ids=torch.zeros(size, dtype=torch.long),
            pixel_values=None,
            sample_ids=[str(index)],
        )


class FiniteEpochTest(unittest.TestCase):
    def test_two_modalities_flush_uneven_rank_tails_without_restarting(self):
        tokenizer = SimpleNamespace(unk_token_id=-1, convert_tokens_to_ids=lambda value: hash(value))
        for factor in (1, 3):
            union = []
            packed_counts = []
            for rank in range(3):
                expected = list(range(rank, 49, 3))
                reader = Reader(expected, factor)
                with (
                    patch.object(packing, "gpc", SimpleNamespace(config=SimpleNamespace(data={}))),
                    patch.object(packing, "get_rank", return_value=rank),
                    patch.object(packing, "get_world_size", return_value=3),
                ):
                    dataset = PackedDataset(
                        tokenizer,
                        rank,
                        3,
                        [reader],
                        max_packed_tokens=20,
                        max_buffer_size=10,
                        replacement=False,
                        allow_overflow=False,
                    )
                    rows = list(dataset)
                actual = [identity for row in rows for identity in row["sample_ids"]]
                self.assertEqual(sorted(actual, key=int), list(map(str, expected)))
                self.assertEqual(reader.starts, 1)
                self.assertTrue(all(len(row["input_ids"]) <= 20 for row in rows))
                union.extend(actual)
                packed_counts.append(len(rows))
            self.assertEqual(sorted(union, key=int), list(map(str, range(49))))
            self.assertGreater(len(set(packed_counts)), 1)

    def test_exhaustion_does_not_restart_when_replacement_is_disabled(self):
        reader = Reader([], 1)
        packer = object.__new__(PackedDataset)
        packer.datasets, packer.dataset_iter_list = [reader], [iter(reader)]
        packer.replacement = False
        packer.dataset_weight = packer.dataset_weight_orig = [1.0]
        self.assertIsNone(packer.next_data(0))
        self.assertEqual(reader.starts, 1)
        self.assertEqual(packer.dataset_weight, [0.0])


if __name__ == "__main__":
    unittest.main()
