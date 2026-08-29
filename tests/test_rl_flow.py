from __future__ import annotations

import unittest

import torch

from sensenova_u1.rl.flow import sde_log_prob, sde_transition, text_policy_loss


class RlFlowTest(unittest.TestCase):
    def test_sde_log_prob_is_differentiable_through_current_mean(self) -> None:
        velocity = torch.full((2, 3, 4, 4), 0.25, requires_grad=True)
        sample = torch.zeros_like(velocity)
        transition = sde_transition(
            model_output=velocity,
            sample=sample,
            sigma=0.8,
            sigma_prev=0.7,
            sigma_max=0.9,
            noise_level=0.7,
        )
        previous = transition.mean.detach() + 0.05
        log_prob = sde_log_prob(prev_sample=previous, transition=transition)
        self.assertEqual(log_prob.shape, (2,))
        log_prob.sum().backward()
        self.assertIsNotNone(velocity.grad)
        self.assertTrue(torch.isfinite(velocity.grad).all())

    def test_text_policy_first_update_uses_ratio_one(self) -> None:
        current = torch.tensor([[-1.0, -2.0], [-1.5, -2.5]], requires_grad=True)
        old = current.detach().clone()
        result = text_policy_loss(
            log_probs=current,
            old_log_probs=old,
            advantages=torch.tensor([1.0, -1.0]),
            response_mask=torch.ones_like(current, dtype=torch.bool),
            clip_range=0.2,
        )
        self.assertAlmostEqual(result.metrics["ratio_mean"], 1.0)
        result.value.backward()
        self.assertTrue(torch.isfinite(current.grad).all())


if __name__ == "__main__":
    unittest.main()
