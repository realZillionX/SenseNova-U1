"""CPU DCP export preserves tensor values and only publishes complete outputs."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file, save_file
from tools.export_sft_checkpoint import export_checkpoint, latest_checkpoint


class CheckpointExportTest(unittest.TestCase):
    def checkpoint(self, root, count=251):
        path = root / f"samples-{count:012d}"
        path.mkdir()
        weights = {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4), "bias": torch.tensor([1.0, 2.0, 3.0])}
        dcp.save({"model": weights}, checkpoint_id=path / "dcp")
        metadata = dict(
            schema="sensenova.u15.forge.sft.checkpoint.v3",
            trainer="torch_fsdp2",
            samples_per_epoch=1000,
            max_samples=1000,
            consumed_samples=count,
            optimizer_updates=10,
            last_update_samples=31,
            checkpoint_target_samples=(count // 250) * 250,
            world_size=8,
            model_only=True,
            conversion_config={"vit_cfg": {"num_hidden_layers": 1}, "num_layers": 1},
        )
        (path / "checkpoint.json").write_text(json.dumps(metadata))
        return path, weights

    @staticmethod
    def convert(*, src, tgt, typ, extras_from):
        weights = torch.load(Path(src) / "model_wp0_pp0.pt", weights_only=True)
        target = Path(tgt)
        target.mkdir()
        save_file(weights, target / "model.safetensors")
        (target / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {name: "model.safetensors" for name in weights}})
        )

    def test_cpu_roundtrip_and_latest_ignore_incomplete_staging(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint, expected = self.checkpoint(root)
            (root / ".samples-000000000900.staging").mkdir()
            self.assertEqual(latest_checkpoint(root), checkpoint)
            with patch("tools.export_sft_checkpoint.convert", side_effect=self.convert):
                result = export_checkpoint(checkpoint, root / "hf", root / "base")
            self.assertEqual(result["consumed_samples"], 251)
            self.assertEqual(result["max_samples"], 1000)
            actual = load_file(root / "hf/model.safetensors")
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key])
            self.assertFalse(list(root.glob(".hf.export-*")))
            with self.assertRaises(FileExistsError):
                export_checkpoint(checkpoint, root / "hf", root / "base")

    def test_hf_converter_publishes_fp32_masters_as_bf16(self):
        from tools.publish_hf import convert

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.mkdir()
            key = "vision_model.embeddings.class_embedding"
            value = torch.tensor([0.125, 0.25, 0.5], dtype=torch.float32)
            torch.save({key: value}, source / "model_wp0_pp0.pt")
            torch.save({"vit_cfg": {"num_hidden_layers": 3}, "num_layers": 10}, source / "model_config.pt")
            convert(src=str(source), tgt=str(root / "hf"), typ="neo++_mot", extras_from=None)
            index = json.loads((root / "hf/model.safetensors.index.json").read_text())
            self.assertEqual(len(index["weight_map"]), 1)
            name, shard = next(iter(index["weight_map"].items()))
            actual = load_file(root / "hf" / shard)[name]
            self.assertEqual(actual.dtype, torch.bfloat16)
            torch.testing.assert_close(actual, value.to(torch.bfloat16))

    def test_failed_conversion_never_publishes_partial_target(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint, _ = self.checkpoint(root)
            with patch("tools.export_sft_checkpoint.convert", side_effect=ValueError("incomplete conversion")):
                with self.assertRaisesRegex(ValueError, "incomplete conversion"):
                    export_checkpoint(checkpoint, root / "hf", root / "base")
            self.assertFalse((root / "hf").exists())
            self.assertFalse(list(root.glob(".hf.export-*")))


if __name__ == "__main__":
    unittest.main()
