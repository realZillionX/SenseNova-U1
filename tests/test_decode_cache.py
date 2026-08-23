from __future__ import annotations

import unittest
import warnings
from types import SimpleNamespace

import torch
from transformers.cache_utils import DynamicCache

from sensenova_u1.utils.decode_cache import (
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
