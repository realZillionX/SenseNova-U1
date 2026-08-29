from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sensenova_u1.rl.plan import RlPlan
from sensenova_u1.rl.trainer import PersistentRewardClient
from sensenova_u1.rl.types import CandidateResponse, TextSegment

ROOT = Path(__file__).resolve().parents[1]


class RewardProviderTest(unittest.TestCase):
    def test_persistent_ndjson_provider_is_verifier_agnostic(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prompts = root / "prompts.jsonl"
            prompts.write_text("{}\n", encoding="utf-8")
            checkpoint = root / "model"
            checkpoint.mkdir()
            plan = RlPlan(
                run_dir=root / "run",
                prompts=prompts,
                policy_init=checkpoint,
                modality="ti2t",
                reward_command=(
                    sys.executable,
                    str(ROOT / "tests/fixtures/reward_provider.py"),
                ),
                reward_dimension_names=("synthetic_correct",),
                reward_weights=(1.0,),
                max_steps=2,
            )
            client = PersistentRewardClient(plan)
            try:
                batch, budget = client(
                    plan,
                    "ti2t",
                    "sample-1",
                    "group-1",
                    (
                        SimpleNamespace(
                            candidate=CandidateResponse(modality="ti2t", items=(TextSegment("Answer: A"),))
                        ),
                        SimpleNamespace(
                            candidate=CandidateResponse(modality="ti2t", items=(TextSegment("Answer: B"),))
                        ),
                    ),
                    root,
                )
            finally:
                client.close()
            self.assertEqual(batch.dimension_names, ("synthetic_correct",))
            self.assertEqual(batch.group_ids, ("group-1", "group-1"))
            self.assertEqual(batch.matrix[:, 0].tolist(), [0.0, 1.0])
            self.assertEqual(budget, {})


if __name__ == "__main__":
    unittest.main()
