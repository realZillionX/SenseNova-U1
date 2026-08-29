from __future__ import annotations

import types
import unittest

import torch

from sensenova_u1.rl.policy_runtime import (
    TextEvent,
    U15PolicyReplay,
    U15PolicyRollout,
    U15PolicyRuntime,
)
from sensenova_u1.rl.rollout import TextRolloutTrace
from sensenova_u1.rl.types import CandidateResponse, TextSegment


class PolicyAnchorTest(unittest.TestCase):
    def test_anchor_replaces_behavior_likelihood_once_with_replay_geometry(self) -> None:
        runtime = object.__new__(U15PolicyRuntime)
        original = TextRolloutTrace(
            token_ids=torch.tensor([[11, 12]]),
            old_log_probs=torch.tensor([[-5.0, -6.0]]),
            response_mask=torch.tensor([[True, True]]),
            stopped=torch.tensor([True]),
        )
        rollout = U15PolicyRollout(
            candidate=CandidateResponse(modality="ti2t", items=(TextSegment("Answer: A"),)),
            events=(TextEvent(trace=original, stop_token_id=12),),
            text_tokens=2,
            generated_images=0,
            seconds=0.1,
        )
        replay = U15PolicyReplay(
            text_trace=original,
            text_log_probs=torch.tensor([[-1.0, -2.0]]),
            text_ref_log_probs=None,
            image_replays=(),
            image_reference_velocities=(),
            ratio_max_error=0.25,
            ratio_mean_error=0.125,
            ratio_action_count=2,
            numeric_max_error=1e-5,
        )

        def fake_replay(self, **kwargs):
            self.asserted_kwargs = kwargs
            return replay

        runtime.replay = types.MethodType(fake_replay, runtime)
        anchor = runtime.anchor_rollout_with_metrics(
            prompt="prompt",
            prompt_images=(),
            modality="ti2t",
            rollout=rollout,
        )

        anchored_event = anchor.rollout.events[0]
        self.assertIsInstance(anchored_event, TextEvent)
        torch.testing.assert_close(anchored_event.trace.old_log_probs, replay.text_log_probs)
        torch.testing.assert_close(original.old_log_probs, torch.tensor([[-5.0, -6.0]]))
        self.assertEqual(anchor.behavior_ratio_action_count, 2)
        self.assertAlmostEqual(anchor.behavior_ratio_max_error, 0.25)
        self.assertFalse(runtime.asserted_kwargs["include_reference"])


if __name__ == "__main__":
    unittest.main()
