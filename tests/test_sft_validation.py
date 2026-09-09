import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tools.sft_validation import aggregate_samples, validation_state


class SftValidationTest(unittest.TestCase):
    def test_validation_releases_inference_only_fsdp_views(self):
        calls = []

        class Sharded(torch.nn.Module):
            def __init__(self, name):
                torch.nn.Module.__init__(self)
                self.name = name

            def reshard(self):
                calls.append(self.name)

        model = Sharded("root")
        model.child = Sharded("child")
        with patch("torch.distributed.fsdp.FSDPModule", Sharded):
            with validation_state(model, []):
                self.assertFalse(model.training)
        self.assertEqual(calls, ["child", "root"])
        self.assertTrue(model.training)

    def test_validation_restores_rng_mode_and_keeps_parameters_and_gradients(self):
        model = torch.nn.Linear(3, 2).train()
        model.weight.grad = torch.ones_like(model.weight)
        weights = model.weight.detach().clone()

        def seed():
            random.seed(17)
            np.random.seed(17)
            torch.manual_seed(17)

        seed()
        expected = (random.random(), np.random.rand(), torch.rand(1))
        seed()
        with self.assertRaisesRegex(RuntimeError, "validation stopped"):
            with validation_state(model, []):
                self.assertFalse(model.training)
                self.assertFalse(torch.is_grad_enabled())
                random.random()
                np.random.rand()
                torch.rand(9)
                model(torch.ones(1, 3))
                raise RuntimeError("validation stopped")
        actual = (random.random(), np.random.rand(), torch.rand(1))
        self.assertEqual(expected[:2], actual[:2])
        torch.testing.assert_close(expected[2], actual[2])
        self.assertTrue(model.training)
        torch.testing.assert_close(model.weight, weights)
        torch.testing.assert_close(model.weight.grad, torch.ones_like(model.weight))

    def test_each_sample_and_noise_seed_has_equal_weight(self):
        rows = [
            dict(sample_id=identity, noise_seed=seed, text_loss=text, image_loss=image)
            for identity, seed, text, image in [
                ("b", 2, 10.0, 4.0),
                ("a", 1, 2.0, 1.0),
                ("b", 1, 6.0, 2.0),
                ("a", 2, 4.0, 3.0),
            ]
        ]
        report = aggregate_samples(rows, ["a", "b"], [1, 2])
        self.assertEqual(report["means"], dict(text_loss=5.5, image_loss=2.5, loss=8.0))
        self.assertEqual(report["examples"][0]["loss"], 5.0)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            aggregate_samples(rows + [rows[0]], ["a", "b"], [1, 2])
        with self.assertRaisesRegex(ValueError, "coverage"):
            aggregate_samples(rows[:-1], ["a", "b"], [1, 2])


if __name__ == "__main__":
    unittest.main()
