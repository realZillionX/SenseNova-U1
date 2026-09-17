import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from sensenovalm.data.sample_batch import sample_batches
from sensenovalm.data.sample_progress import SampleProgress
from tools.sft_restart import OPTIONS, load_model_checkpoint, load_recovery, save_recovery
from torch import nn
from torch.distributed.checkpoint.state_dict import get_model_state_dict


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.unused = nn.Parameter(torch.ones(2))

    def forward(self, x):
        return self.linear(x)


def update(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    model(torch.arange(12, dtype=torch.float32).reshape(4, 3)).square().mean().backward()
    optimizer.step()


def save_model(root, model, consumed, updates):
    progress = SampleProgress(32, 64, consumed, updates, 4)
    path = root / f"samples-{consumed:012d}"
    dcp.save({"model": get_model_state_dict(model, options=OPTIONS)}, checkpoint_id=path / "dcp")
    payload = dict(
        schema="sensenova.u15.forge.sft.checkpoint.v3",
        model_only=True,
        samples_per_epoch=32,
        max_samples=64,
        consumed_samples=consumed,
        optimizer_updates=updates,
        last_update_samples=4,
        checkpoint_target_samples=consumed,
        world_size=1,
        batch_samples=4,
    )
    (path / "checkpoint.json").write_text(json.dumps(payload))
    return path, progress


class SftRestartTest(unittest.TestCase):
    def test_batch_suffix_keeps_order_and_restarts_worker_assignment(self):
        options = dict(rows=101, batch_samples=16, max_samples=303, seed=42)
        full = list(sample_batches(**options))
        for offset in (32, 96, 101, 117, 202):
            expected = [b for b in full if b.sample_start >= offset]
            actual = list(sample_batches(**options, start_samples=offset))
            self.assertEqual(actual, expected)
            workers = [
                list(sample_batches(**options, start_samples=offset, worker_id=i, num_workers=3)) for i in range(3)
            ]
            merged = [workers[i % 3][i // 3] for i in range(len(expected))]
            self.assertEqual(merged, expected)
        with self.assertRaisesRegex(ValueError, "boundary"):
            list(sample_batches(**options, start_samples=33))

    def test_recovery_restores_adam_rng_and_survives_rolling_retirement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.manual_seed(7)
            np.random.seed(7)
            random.seed(7)
            model = Net()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.1)
            update(model, optimizer)
            update(model, optimizer)
            checkpoint, progress = save_model(root, model, 8, 2)
            generation = save_recovery(
                root=root / "recovery",
                model=model,
                optimizer=optimizer,
                progress=progress,
                model_checkpoint=checkpoint,
                seed=42,
                batch_samples=4,
            )
            snapshot = root / "input"
            shutil.copytree(generation, snapshot, copy_function=__import__("os").link)
            expected_random = (torch.rand(4), np.random.random(4), random.random())
            update(model, optimizer)
            expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
            restored = Net()
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.003, weight_decay=0.1)
            restored_progress = load_model_checkpoint(restored, checkpoint)
            load_recovery(
                model=restored,
                optimizer=restored_optimizer,
                recovery=snapshot,
                progress=restored_progress,
                seed=42,
                batch_samples=4,
            )
            torch.testing.assert_close(torch.rand(4), expected_random[0])
            np.testing.assert_equal(np.random.random(4), expected_random[1])
            self.assertEqual(random.random(), expected_random[2])
            self.assertEqual(restored_optimizer.state.get(restored.unused, {}), {})
            update(restored, restored_optimizer)
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            update(model, optimizer)
            checkpoint2, progress2 = save_model(root, model, 16, 4)
            save_recovery(
                root=root / "recovery",
                model=model,
                optimizer=optimizer,
                progress=progress2,
                model_checkpoint=checkpoint2,
                seed=42,
                batch_samples=4,
            )
            self.assertFalse(generation.exists())
            self.assertTrue((snapshot / "optimizer/.metadata").is_file())
            self.assertEqual(
                json.loads((root / "recovery/latest.json").read_text())["generation"], "samples-000000000016"
            )

    def test_failed_recovery_does_not_retire_previous_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = Net()
            optimizer = torch.optim.AdamW(model.parameters())
            update(model, optimizer)
            update(model, optimizer)
            checkpoint, progress = save_model(root, model, 8, 2)
            previous = save_recovery(
                root=root / "recovery",
                model=model,
                optimizer=optimizer,
                progress=progress,
                model_checkpoint=checkpoint,
                seed=42,
                batch_samples=4,
            )
            update(model, optimizer)
            update(model, optimizer)
            checkpoint2, progress2 = save_model(root, model, 16, 4)
            with patch("tools.sft_restart.dcp.save", side_effect=OSError("disk quota")):
                with self.assertRaisesRegex(OSError, "disk quota"):
                    save_recovery(
                        root=root / "recovery",
                        model=model,
                        optimizer=optimizer,
                        progress=progress2,
                        model_checkpoint=checkpoint2,
                        seed=42,
                        batch_samples=4,
                    )
            self.assertTrue(previous.is_dir())
            self.assertEqual(json.loads((root / "recovery/latest.json").read_text())["generation"], previous.name)


if __name__ == "__main__":
    unittest.main()
