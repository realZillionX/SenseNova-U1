from __future__ import annotations

import unittest

import torch
from transformers.cache_utils import DynamicCache

from sensenova_u1.utils.decode_cache import PreallocatedDecodeCache


class DecodeCacheTest(unittest.TestCase):
    def _prefix(self, length: int = 3) -> DynamicCache:
        cache = DynamicCache()
        for layer in range(2):
            keys = torch.arange(2 * length * 4, dtype=torch.float32).reshape(
                1, 2, length, 4
            )
            cache.update(keys + layer, keys + layer + 1, layer)
        return cache

    def test_preallocated_cache_loads_and_reuses_prefix(self) -> None:
        prefix = self._prefix()
        cache = PreallocatedDecodeCache.from_cache(prefix, capacity=8)

        self.assertEqual(cache.get_seq_length(), 3)
        self.assertEqual(int(cache.flash_decode_seqlens.item()), 3)
        self.assertTrue(cache.can_load(prefix, required_capacity=7))

        shorter = self._prefix(length=2)
        cache.load_prefix(shorter, required_capacity=7)
        self.assertEqual(cache.get_seq_length(), 2)
        self.assertEqual(int(cache.flash_decode_seqlens.item()), 2)

    def test_image_style_updates_can_resynchronize_flash_length(self) -> None:
        cache = PreallocatedDecodeCache.from_cache(self._prefix(), capacity=8)
        key = torch.ones(1, 2, 2, 4)
        for layer in range(2):
            cache.update(key, key, layer)

        self.assertEqual(cache.get_seq_length(), 5)
        self.assertEqual(int(cache.flash_decode_seqlens.item()), 3)
        cache.sync_flash_decode_length()
        self.assertEqual(int(cache.flash_decode_seqlens.item()), 5)


if __name__ == "__main__":
    unittest.main()
