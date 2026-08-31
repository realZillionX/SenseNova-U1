from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from sensenova_u1.rl.api_rollout import _first_input_geometry
from sensenova_u1.rl.plan import RlPlan, TorchrunSpec
from sensenova_u1.rl.types import RewardBatch


class RlPlanTest(unittest.TestCase):
    def test_first_input_geometry_preserves_rectangular_source(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.png"
            Image.new("RGB", (640, 384)).save(path)
            self.assertEqual(_first_input_geometry((str(path),)), (384, 640))

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
            self.assertEqual(restored.max_new_tokens, 6144)
            self.assertEqual(restored.max_images, 10)
            self.assertEqual(restored.image_resolution_mode, "first_input")
            self.assertEqual(restored.save_every_steps, 100)
            self.assertEqual(restored.checkpoint_keep_last, 2)
            self.assertEqual(restored.optimizer_cpu_offload_min_images, 6)

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

    def test_plan_fixes_weight_decay_and_first_input_resolution(self) -> None:
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
        with self.assertRaisesRegex(ValueError, "first prompt image"):
            RlPlan(**common, image_resolution_mode="fixed")

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
