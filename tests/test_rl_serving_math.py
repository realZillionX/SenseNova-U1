from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

from sensenova_u1.rl.flow import sde_log_prob
from sensenova_u1.rl.rollout import select_sde_indices, u15_sde_transition

ROOT = Path(__file__).resolve().parents[1]
SERVING_SDE = ROOT / "serving/third_party/LightX2V/lightx2v/rl/sde.py"


def load_serving_sde():
    spec = importlib.util.spec_from_file_location("forge_test_lightx2v_sde", SERVING_SDE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned LightX2V SDE implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RlServingMathTest(unittest.TestCase):
    def test_rollout_and_replay_share_exact_sde_physics(self) -> None:
        serving = load_serving_sde()
        generator = torch.Generator().manual_seed(7)
        sample = torch.randn((2, 3, 4, 4), generator=generator)
        velocity = torch.randn((2, 3, 4, 4), generator=generator)
        transition = u15_sde_transition(
            velocity=velocity,
            sample=sample,
            t=0.2,
            t_next=0.35,
            sigma_max=0.9,
            noise_level=0.7,
        )
        serving_mean, serving_scale = serving.transition(
            velocity,
            sample,
            0.2,
            0.35,
            0.9,
            0.7,
        )
        torch.testing.assert_close(transition.mean, serving_mean, atol=0, rtol=0)
        torch.testing.assert_close(transition.scale, serving_scale, atol=0, rtol=0)

        next_sample = serving_mean + serving_scale * torch.randn(serving_mean.shape, generator=generator)
        trainer_log_prob = sde_log_prob(prev_sample=next_sample, transition=transition)
        serving_log_prob = serving.recompute_log_prob(next_sample, serving_mean, serving_scale)
        torch.testing.assert_close(trainer_log_prob, serving_log_prob, atol=0, rtol=0)

    def test_rollout_and_replay_select_the_same_sde_window(self) -> None:
        serving = load_serving_sde()
        expected = select_sde_indices(total_steps=30, window_start=3, window_end=27, selected_steps=8)
        observed = serving.select_sde_indices(30, 3, 27, 8)
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
