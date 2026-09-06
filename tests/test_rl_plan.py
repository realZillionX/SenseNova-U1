from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sensenova_u1.rl.plan import RlPlan, TorchrunSpec
from sensenova_u1.rl.types import RewardBatch


class RlPlanTest(unittest.TestCase):
    def test_round_trip_and_digest_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prompts = root / "prompts.jsonl"
            checkpoint = root / "checkpoint"
            prompts.write_text("{}\n", encoding="utf-8")
            checkpoint.mkdir()
            plan = RlPlan(
                run_dir=root / "run",
                prompts=prompts,
                policy_init=checkpoint,
                modality="ti2t",
                reward_command=("python", "reward.py"),
                reward_dimension_names=("correct",),
                reward_weights=(1.0,),
                max_steps=2,
                prompts_per_batch=2,
                rollout_api_base_urls=(
                    "http://127.0.0.1:8000",
                    "http://127.0.0.1:8001",
                ),
                torchrun=TorchrunSpec(nproc_per_node=2),
            )
            payload = plan.to_dict()
            restored = RlPlan.from_dict(json.loads(json.dumps(payload)))
            self.assertEqual(restored.digest, plan.digest)
            self.assertEqual(restored.torchrun.world_size, 2)
            self.assertEqual(len(restored.rollout_api_base_urls), 2)
            self.assertEqual(restored.max_sequence_length, 8192)
            self.assertNotIn("max_new_tokens", payload)
            for retired in ("max_new_tokens", "max_tokens", "max_completion_length"):
                with self.subTest(retired=retired), self.assertRaises((TypeError, ValueError)):
                    RlPlan.from_dict({**payload, retired: 6144})
            self.assertEqual(restored.max_images, 10)
            self.assertEqual(restored.image_size, 512)
            self.assertEqual(restored.save_every_steps, 50)
            self.assertEqual(restored.optimizer_cpu_offload_min_images, 6)

    def test_modality_defaults_are_resolved_but_explicit_ablation_survives(self) -> None:
        common = dict(
            run_dir=Path("run"), prompts=Path("prompts"), policy_init=Path("model"),
            reward_command=("reward",), reward_dimension_names=("correct",),
            reward_weights=(1.0,), max_steps=2,
        )
        for modality, image, mse in (("ti2t", 0.0, 0.0), ("ti2ti", 1.0, 0.01)):
            plan = RlPlan(**common, modality=modality)
            restored = RlPlan.from_dict(plan.to_dict())
            self.assertEqual((restored.image_objective_weight, restored.velocity_mse_weight), (image, mse))
            self.assertEqual(restored.text_kl_beta, 0.04)
            self.assertTrue(restored.activation_checkpointing)
        ablation = RlPlan(**common, modality="ti2ti", text_kl_beta=0.0,
                          velocity_mse_weight=0.0, activation_checkpointing=False)
        restored = RlPlan.from_dict(ablation.to_dict())
        self.assertEqual((restored.text_kl_beta, restored.velocity_mse_weight), (0.0, 0.0))
        self.assertFalse(restored.activation_checkpointing)

    def test_plan_rejects_partial_update_batches_and_rank_mismatch(self) -> None:
        common = dict(
            run_dir=Path("run"),
            prompts=Path("prompts"),
            policy_init=Path("model"),
            modality="ti2t",
            reward_command=("reward",),
            reward_dimension_names=("correct",),
            reward_weights=(1.0,),
        )
        with self.assertRaisesRegex(ValueError, "frozen-old"):
            RlPlan(**common, max_steps=3)
        with self.assertRaisesRegex(ValueError, "one prompt group"):
            RlPlan(
                **common,
                max_steps=2,
                prompts_per_batch=2,
                torchrun=TorchrunSpec(nproc_per_node=3),
            )

    def test_ti2t_rejects_image_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "TI2T"):
            RlPlan(
                run_dir=Path("run"),
                prompts=Path("prompts"),
                policy_init=Path("model"),
                modality="ti2t",
                reward_command=("reward",),
                reward_dimension_names=("correct",),
                reward_weights=(1.0,),
                max_steps=2,
                image_objective_weight=1.0,
            )

    def test_plan_fixes_weight_decay_and_rl_resolution(self) -> None:
        common = dict(
            run_dir=Path("run"),
            prompts=Path("prompts"),
            policy_init=Path("model"),
            modality="ti2t",
            reward_command=("reward",),
            reward_dimension_names=("correct",),
            reward_weights=(1.0,),
            max_steps=2,
        )
        with self.assertRaisesRegex(ValueError, "weight_decay=0"):
            RlPlan(**common, weight_decay=0.01)
        with self.assertRaisesRegex(ValueError, "32-pixel"):
            RlPlan(**common, image_size=500)

    def test_plan_accepts_arbitrary_positive_training_node_count(self) -> None:
        plan = RlPlan(
            run_dir=Path("run"),
            prompts=Path("prompts"),
            policy_init=Path("model"),
            modality="ti2t",
            reward_command=("reward",),
            reward_dimension_names=("correct",),
            reward_weights=(1.0,),
            max_steps=2,
            prompts_per_batch=24,
            torchrun=TorchrunSpec(
                nproc_per_node=8,
                nnodes=3,
                master_addr="10.0.0.1",
            ),
        )
        self.assertEqual(plan.torchrun.world_size, 24)

    def test_reward_batch_requires_nan_for_unavailable_cells(self) -> None:
        with self.assertRaisesRegex(ValueError, "NaNs"):
            RewardBatch(
                matrix=np.asarray([[0.0, 1.0]], dtype=np.float32),
                dimension_names=("a", "b"),
                availability=np.asarray([[True, False]], dtype=np.bool_),
                group_ids=("g",),
                errors=(None,),
            )


if __name__ == "__main__":
    unittest.main()
