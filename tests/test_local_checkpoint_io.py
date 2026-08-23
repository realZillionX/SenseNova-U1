from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from sensenovalm.utils.storage_manager import StorageManager


class LocalCheckpointIOTest(unittest.TestCase):
    def test_zip_checkpoint_can_load_through_mmap_manager(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.pt"
            expected = {"tensor": torch.arange(32, dtype=torch.float32)}
            torch.save(expected, path)
            manager = type.__call__(
                StorageManager,
                False,
                async_mode=False,
                local_mmap_load=True,
            )

            observed = manager.load(str(path), weights_only=False)

            self.assertTrue(torch.equal(observed["tensor"], expected["tensor"]))

    def test_mmap_rejects_legacy_write_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires zip serialization"):
            type.__call__(
                StorageManager,
                False,
                async_mode=False,
                local_legacy_serialization=True,
                local_mmap_load=True,
            )


if __name__ == "__main__":
    unittest.main()
