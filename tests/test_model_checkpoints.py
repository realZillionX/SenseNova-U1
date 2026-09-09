"""Exercise the production checkpoint writers with small CPU model weights."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import train_sensenovau1_fsdp2 as sft
from sensenovalm.data.sample_progress import SampleProgress
from torch.distributed.checkpoint import FileSystemReader

from sensenova_u1.rl import trainer
from sensenova_u1.rl.budget import BudgetLedger
from sensenova_u1.rl.full_parameter import DistributedContext, load_dcp


class ModelCheckpointTest(unittest.TestCase):
    def assert_model_roundtrip(self, path, model):
        metadata = FileSystemReader(path / "dcp").read_metadata()
        self.assertEqual(
            set(metadata.state_dict_metadata),
            {f"model.{name}" for name in model.state_dict()},
        )
        expected = {name: value.clone() for name, value in model.state_dict().items()}
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(100)
        load_dcp(path / "dcp", model)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_sft_writes_only_model_and_sample_metadata(self):
        model = torch.nn.Linear(3, 2)
        progress = SampleProgress(100, 100)
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(torch.distributed, "get_rank", return_value=0),
            patch.object(torch.distributed, "get_world_size", return_value=1),
            patch.object(torch.distributed, "barrier"),
            patch.object(sft, "_distributed_max", side_effect=lambda value: value),
            patch.object(
                sft,
                "gpc",
                SimpleNamespace(
                    config=SimpleNamespace(
                        data=SimpleNamespace(batch_samples=10),
                        model=SimpleNamespace(
                            vit_cfg=SimpleNamespace(num_hidden_layers=1), num_layers=1, moe_kwargs={}
                        ),
                    )
                ),
            ),
        ):
            for count in range(10, 101, 10):
                target = progress.advance(10)
                path = sft._save_training_checkpoint(
                    root=Path(folder),
                    progress=progress,
                    checkpoint_target_samples=target,
                    model=model,
                )
                self.assertEqual(set(p.name for p in path.iterdir()), {"dcp", "checkpoint.json"})
                payload = json.loads((path / "checkpoint.json").read_text())
                self.assertIs(payload["model_only"], True)
                self.assertEqual(payload["consumed_samples"], count)
                self.assertEqual(payload["batch_samples"], 10)
                self.assertEqual(
                    payload["conversion_config"],
                    {
                        "vit_cfg": {"num_hidden_layers": 1},
                        "num_layers": 1,
                        "moe_kwargs": {"first_k_dense_replace": 0, "num_experts": 1, "gen_num_experts": 1},
                    },
                )
                self.assert_model_roundtrip(path, model)
            self.assertEqual(len(list(Path(folder).iterdir())), 10)

    def test_rl_writes_only_model_and_budget_metadata(self):
        model = torch.nn.Linear(3, 2)
        context = DistributedContext(0, 0, 1, torch.device("cpu"))
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(trainer, "_broadcast_primary", side_effect=lambda _ctx, call: call()),
            patch.object(torch.distributed, "barrier"),
        ):
            plan = SimpleNamespace(run_dir=Path(folder), digest="producer-plan")
            path = trainer.save_checkpoint(
                plan=plan,
                step=50,
                model=model,
                ledger=BudgetLedger(),
                policy_version="policy-50",
                context=context,
            )
            self.assertEqual(set(p.name for p in path.iterdir()), {"dcp", "trainer_state.json", "COMMIT"})
            saved = trainer.latest_checkpoint(plan, context=context)
            self.assertEqual(saved.step, 50)
            self.assertEqual(saved.policy_version, "policy-50")
            self.assert_model_roundtrip(path, model)

    def test_rl_rejects_restarting_a_directory_with_checkpoints_before_loading_model(self):
        context = DistributedContext(0, 0, 1, torch.device("cpu"))
        plan = SimpleNamespace(torchrun=SimpleNamespace(world_size=1))
        with (
            patch.object(trainer, "latest_checkpoint", return_value=object()),
            patch.object(trainer.U15PolicyRuntime, "load") as load,
            self.assertRaisesRegex(ValueError, "start a fresh run directory"),
        ):
            trainer._run_plan(plan, context)
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
