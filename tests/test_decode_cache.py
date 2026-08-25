from __future__ import annotations

import unittest
import warnings
from types import SimpleNamespace

import torch
from transformers.cache_utils import DynamicCache

from sensenova_u1.utils.decode_cache import (
    BatchedFlashDecodeCache,
    ContinuousFlashDecodeCache,
    CudaGraphDecodeWorkspace,
    PreallocatedDecodeCache,
)


class DecodeCacheTest(unittest.TestCase):
    def _prefix(
        self, length: int = 3, device: str | torch.device = "cpu"
    ) -> DynamicCache:
        cache = DynamicCache()
        for layer in range(2):
            keys = torch.arange(
                2 * length * 4, dtype=torch.float32, device=device
            ).reshape(1, 2, length, 4)
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

    def test_batched_cache_compacts_prefix_and_selects_active_rows(self) -> None:
        prefix = DynamicCache()
        keys = torch.arange(2 * 2 * 4 * 3, dtype=torch.float32).reshape(
            2, 2, 4, 3
        )
        prefix.update(keys, keys + 100, 0)
        valid = torch.tensor(
            [[False, False, True, True], [True, True, True, True]]
        )

        cache = BatchedFlashDecodeCache.from_cache(
            prefix, key_valid=valid, capacity=8
        )

        self.assertEqual(cache.flash_decode_seqlens.tolist(), [2, 4])
        expected = keys[0, :, 2:].transpose(0, 1)
        self.assertTrue(
            torch.equal(cache.layers[0].flash_decode_k_cache[0, :2], expected)
        )

        indices = torch.tensor([1], dtype=torch.long)
        cache.activate(indices)
        self.assertEqual(cache.layers[0].flash_decode_seqlens.tolist(), [4])
        self.assertEqual(
            cache.layers[0].flash_decode_cache_batch_idx.tolist(), [0]
        )
        self.assertEqual(cache.layers[0].max_batch_size, 1)
        cache.commit_active()
        self.assertEqual(cache.flash_decode_seqlens.tolist(), [2, 5])
        self.assertFalse(
            hasattr(cache.layers[0], "flash_decode_cache_batch_idx")
        )

        cache.ensure_capacity(indices)
        self.assertEqual(cache.max_cache_len, 8)

    def test_continuous_cache_reuses_released_physical_slot(self) -> None:
        first = DynamicCache()
        first_keys = torch.arange(1 * 2 * 4 * 3, dtype=torch.float32).reshape(
            1, 2, 4, 3
        )
        first.update(first_keys, first_keys + 100, 0)
        cache = ContinuousFlashDecodeCache.from_prefix(
            first,
            max_batch_size=3,
            max_capacity=1024,
            max_kv_tokens=1024,
            page_size=256,
            slot=0,
        )

        second = DynamicCache()
        second_keys = torch.full((1, 2, 2, 3), 7.0)
        second.update(second_keys, second_keys + 100, 0)
        cache.load_prefix(2, second)
        self.assertEqual(cache.flash_decode_seqlens.tolist(), [4, 0, 2])

        slots = torch.tensor([2, 0], dtype=torch.long)
        cache.activate(slots)
        self.assertEqual(cache.layers[0].flash_decode_seqlens.tolist(), [2, 4])
        self.assertEqual(
            cache.layers[0].flash_decode_block_table[:, 0].tolist(), [1, 0]
        )
        cache.commit_active()
        self.assertEqual(cache.flash_decode_seqlens.tolist(), [5, 0, 3])

        cache.release(0)
        cache.load_prefix(0, second)
        self.assertEqual(cache.flash_decode_seqlens.tolist(), [2, 0, 3])
        block = int(cache.flash_decode_block_table[0, 0].item())
        self.assertTrue(
            torch.equal(
                cache.layers[0].flash_decode_k_cache[block, :2],
                second_keys[0].transpose(0, 1),
            )
        )

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
    def test_graph_capture_uses_each_workspace_device(self) -> None:
        class LanguageModel(torch.nn.Module):
            def forward(
                self,
                *,
                input_ids,
                indexes,
                cache_position,
                **_kwargs,
            ):
                values = input_ids.float() + indexes[0] + cache_position
                return SimpleNamespace(logits=values.unsqueeze(-1).expand(-1, -1, 4))

        model = SimpleNamespace(language_model=LanguageModel())
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            for index in range(2):
                device = torch.device(f"cuda:{index}")
                cache = PreallocatedDecodeCache.from_cache(
                    self._prefix(device=device), capacity=8
                )
                workspace = CudaGraphDecodeWorkspace(model=model, cache=cache)
                output = workspace.replay(
                    torch.tensor([2], device=device), t_index=3
                )
                torch.cuda.synchronize(device)
                self.assertEqual(output.device, device)
                self.assertEqual(cache.get_seq_length(), 4)
        self.assertFalse(
            any("CUDA Graph is empty" in str(warning.message) for warning in observed)
        )


if __name__ == "__main__":
    unittest.main()
