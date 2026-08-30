from __future__ import annotations

import unittest

import torch

from sensenova_u1.rl.flow import sde_log_prob, sde_transition, text_kl_loss, text_policy_loss


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

    def test_text_losses_average_actions_then_trajectories(self) -> None:
        current = torch.zeros((2, 3), requires_grad=True)
        old = current.detach().clone()
        mask = torch.tensor([[True, False, False], [True, True, True]])
        result = text_policy_loss(
            log_probs=current,
            old_log_probs=old,
            advantages=torch.tensor([1.0, 3.0]),
            response_mask=mask,
            clip_range=0.2,
        )
        self.assertAlmostEqual(float(result.value.detach()), -2.0)

        ref = torch.zeros_like(current)
        kl_current = torch.tensor([[-21.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        kl = text_kl_loss(log_probs=kl_current, ref_log_probs=ref, response_mask=mask)
        self.assertGreater(float(kl), 6.0e8)

    def test_action_weighted_span_stream_matches_full_text_objective(self) -> None:
        current = torch.tensor([[-1.0, -1.5, -2.0, -2.5, -3.0]])
        old = current + torch.tensor([[0.02, -0.01, 0.03, -0.02, 0.01]])
        ref = current + torch.tensor([[0.1, -0.2, 0.05, -0.1, 0.2]])
        advantage = torch.tensor([0.7])
        mask = torch.ones_like(current, dtype=torch.bool)
        full_policy = text_policy_loss(
            log_probs=current,
            old_log_probs=old,
            advantages=advantage,
            response_mask=mask,
            clip_range=0.2,
        ).value
        full_kl = text_kl_loss(
            log_probs=current,
            ref_log_probs=ref,
            response_mask=mask,
        )

        split_policy = current.new_zeros(())
        split_kl = current.new_zeros(())
        for start, stop in ((0, 2), (2, 5)):
            weight = (stop - start) / current.shape[1]
            split_policy = split_policy + weight * text_policy_loss(
                log_probs=current[:, start:stop],
                old_log_probs=old[:, start:stop],
                advantages=advantage,
                response_mask=mask[:, start:stop],
                clip_range=0.2,
            ).value
            split_kl = split_kl + weight * text_kl_loss(
                log_probs=current[:, start:stop],
                ref_log_probs=ref[:, start:stop],
                response_mask=mask[:, start:stop],
            )

        torch.testing.assert_close(split_policy, full_policy)
        torch.testing.assert_close(split_kl, full_kl)


if __name__ == "__main__":
    unittest.main()
