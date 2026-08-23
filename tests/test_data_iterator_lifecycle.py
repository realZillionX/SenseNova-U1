from __future__ import annotations

import unittest

from sensenovavl.data.dataset_interleaved_iterable import PackedDataset


class DataIteratorLifecycleTest(unittest.TestCase):
    def test_normal_iterator_exhaustion_is_not_a_data_error(self) -> None:
        self.assertTrue(PackedDataset._is_dataset_exhaustion(StopIteration()))
        self.assertTrue(
            PackedDataset._is_dataset_exhaustion(
                RuntimeError("generator raised StopIteration")
            )
        )
        self.assertFalse(
            PackedDataset._is_dataset_exhaustion(RuntimeError("decode failed"))
        )


if __name__ == "__main__":
    unittest.main()
