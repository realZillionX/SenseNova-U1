from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sensenovalm.data.annotation_order import annotation_offsets, indexed_lines, shard_order
from sensenovavl.data.dataset_interleaved_iterable import PackedDataset
from sensenovavl.data.multimodal_dataset import LazySupervisedDataset


class AnnotationOrderTest(unittest.TestCase):
    def test_global_permutation_covers_both_arms_once_before_sharding(self):
        with tempfile.TemporaryDirectory() as folder:
            orders = []
            for modality in ("ti2t", "ti2ti"):
                path = Path(folder) / f"{modality}.jsonl"
                path.write_text(
                    "\n".join(json.dumps({"sample_id": i, "response": modality * (i + 1)}) for i in range(101)),
                    encoding="utf-8",
                )
                offsets = annotation_offsets(path)
                shards = []
                for shard in range(8):
                    order = shard_order(101, seed=42, epoch=0, shard_id=shard, shard_count=8)
                    shards.append([json.loads(row)["sample_id"] for row in indexed_lines(path, offsets, order)])
                self.assertEqual(sorted(i for shard in shards for i in shard), list(range(101)))
                reconstructed = [shards[i % 8][i // 8] for i in range(101)]
                self.assertNotEqual(reconstructed, list(range(101)))
                orders.append(reconstructed)
            self.assertEqual(orders[0], orders[1])
            next_epoch = shard_order(101, seed=42, epoch=1, shard_id=0, shard_count=1)
            self.assertNotEqual(orders[0], next_epoch.tolist())

    def test_production_reader_uses_global_order_without_decoding_twice(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            path.write_text("\n".join(json.dumps({"sample_id": i}) for i in range(12)))
            ds = object.__new__(LazySupervisedDataset)
            ds.annotation_file = str(path)
            ds._annotation_offsets = annotation_offsets(path)
            ds.meta = {"length": 12}
            ds.shuffle_seed, ds.annotation_epoch = 42, 0
            ds.data_rank, ds.data_world_size = 1, 2
            ds.repeat_time, ds.force_shuffle = 1, True
            ds.distributed_mode = False
            ds.reset()
            with patch.object(ds, "get_sample", side_effect=AssertionError("decode belongs to packer")):
                first = [json.loads(row)["sample_id"] for row in ds]
                second = [json.loads(row)["sample_id"] for row in ds]
            self.assertEqual(first, shard_order(12, seed=42, epoch=0, shard_id=1, shard_count=2).tolist())
            self.assertNotEqual(first, second)

    def test_bad_sample_propagates_instead_of_restarting_dataset(self):
        ds = object.__new__(LazySupervisedDataset)
        packer = object.__new__(PackedDataset)
        packer.datasets = [ds]
        packer.dataset_iter_list = [iter([b"invalid"])]
        packer.replacement = True
        with patch.object(ds, "get_sample", side_effect=ValueError("invalid supervision")):
            with self.assertRaisesRegex(ValueError, "invalid supervision"):
                packer.next_data(0)


if __name__ == "__main__":
    unittest.main()
