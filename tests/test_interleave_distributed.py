from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from examples.interleave.inference import (
    _distributed_batch_indices,
    _merge_rank_results,
    _rank_results_path,
)


class InterleaveDistributedTest(unittest.TestCase):
    def test_rank_assignment_covers_batch_once(self) -> None:
        assignments = [
            list(_distributed_batch_indices(10, rank=rank, world_size=3))
            for rank in range(3)
        ]
        self.assertEqual(assignments, [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]])
        self.assertEqual(
            sorted(index for rows in assignments for index in rows),
            list(range(10)),
        )

    def test_result_shards_merge_in_input_order_and_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            shards = (
                [{"index": 2}, {"index": 0}],
                [{"index": 3}, {"index": 1}],
            )
            for rank, rows in enumerate(shards):
                path = _rank_results_path(output_dir, rank)
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )

            result = _merge_rank_results(output_dir, world_size=2, total=4)

            merged = [json.loads(line) for line in result.read_text().splitlines()]
            self.assertEqual([row["index"] for row in merged], [0, 1, 2, 3])
            self.assertFalse(_rank_results_path(output_dir, 0).exists())
            self.assertFalse(_rank_results_path(output_dir, 1).exists())

    def test_invalid_rank_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank/world"):
            _distributed_batch_indices(1, rank=2, world_size=2)


if __name__ == "__main__":
    unittest.main()
