from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import torch

import sensenova_u1.rl.policy_runtime as policy_runtime_module
from sensenova_u1.rl.policy_runtime import (
    TextEvent,
    U15PolicyReplay,
    U15PolicyRollout,
    U15PolicyRuntime,
)
from sensenova_u1.rl.rollout import TextRolloutTrace
from sensenova_u1.rl.types import CandidateResponse, TextSegment


class PolicyAnchorTest(unittest.TestCase):
    def test_reference_swap_reshards_gathered_parameters_before_copy_and_restore(self) -> None:
        runtime = object.__new__(U15PolicyRuntime)
        parameter = object()

        class FakeModel:
            def named_parameters(self):
                return (("weight", parameter),)

        runtime.model = FakeModel()
        runtime.reference_parameter_shards = {"weight": torch.tensor([9.0, 8.0])}
        local_shard = torch.tensor([1.0, 2.0])
        gathered = torch.zeros(8)
        state = {"sharded": False, "reshards": 0}

        def fake_reshard(_model):
            state["sharded"] = True
            state["reshards"] += 1

        def fake_local_view(_parameter):
            return local_shard if state["sharded"] else gathered

        with (
            patch.object(policy_runtime_module, "reshard_full_parameter_policy", fake_reshard),
            patch.object(policy_runtime_module, "local_parameter_view", fake_local_view),
        ):
            with runtime.reference_parameters():
                torch.testing.assert_close(local_shard, torch.tensor([9.0, 8.0]))
                # Simulate a reference forward through a root configured with
                # reshard_after_forward=False.
                state["sharded"] = False

        self.assertEqual(state["reshards"], 2)
        torch.testing.assert_close(local_shard, torch.tensor([1.0, 2.0]))

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
