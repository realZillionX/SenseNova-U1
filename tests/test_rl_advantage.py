from __future__ import annotations

import unittest

import torch

from sensenova_u1.rl.advantage import compute_gdpo_advantage


class GdpoAdvantageTest(unittest.TestCase):
    def test_group_then_batch_normalization(self) -> None:
        rewards = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=torch.float32,
        )
        result = compute_gdpo_advantage(
            rewards,
            rollout_group_ids=("a", "a", "b", "b"),
            weights=(1.0, 1.0),
            dimension_names=("parse", "semantic"),
        )
        self.assertEqual(result.advantages.shape, (4,))
        torch.testing.assert_close(result.advantages.mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
        torch.testing.assert_close(result.advantages.std(correction=1), torch.tensor(1.0), atol=1e-5, rtol=0)

    def test_singleton_dimension_has_zero_group_contribution(self) -> None:
        result = compute_gdpo_advantage(
            torch.tensor([[1.0], [1.0]], dtype=torch.float32),
            rollout_group_ids=("a", "b"),
            dimension_names=("score",),
        )
        torch.testing.assert_close(result.advantages, torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
