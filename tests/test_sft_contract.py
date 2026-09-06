from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SftContractTest(unittest.TestCase):
    def test_public_launcher_is_full_parameter_u15(self) -> None:
        launcher = (ROOT / "training/shell/train_u1/U1.5_8B_SFT.sh").read_text()
        self.assertIn("freeze_llm=false", launcher)
        self.assertIn("freeze_backbone=false", launcher)
        self.assertIn("enable_und_loss='true'", launcher)
        self.assertIn("SenseNova-U1.5-8B-MoT", launcher)
        self.assertIn("max_samples=${max_samples:-$samples_per_epoch}", launcher)
        self.assertNotIn("checkpoint_every", launcher)
        self.assertIn("seq_len=${seq_len:-8192}", launcher)
        self.assertIn("cfg_txt_uncond_drop_prob=0", launcher)
        self.assertIn("cfg_img_uncond_drop_prob=0", launcher)
        self.assertIn("cfg_txtimg_uncond_drop_prob=0", launcher)
        self.assertIn("JOB_NAME=${JOB_NAME:?", launcher)
        self.assertIn('RUN_ROOT=${RUN_ROOT:-"RUN"}', launcher)
        self.assertIn("model.safetensors.index.json", launcher)
        self.assertIn("train_sensenovau1_fsdp2.py", launcher)
        self.assertIn("FSDP2_PREFETCH_DEPTH", launcher)
        self.assertIn("SFT_CHECKPOINT_ROOT", launcher)
        self.assertIn("SFT_HF_OUTPUT", launcher)
        self.assertIn("SFT_BENCHMARK_ONLY", launcher)
        self.assertIn("unset SFT_HF_OUTPUT", launcher)
        config = (ROOT / "training/configs/sensenovavl_qwen3_gen/sensenovau1_5_8b_mot_sft.py").read_text()
        self.assertIn("enable_save_ckpt=False", config)
        self.assertIn('tensor_parallel_mode = "mtp"', config)
        self.assertNotIn("enabel_und_loss", config)

    def test_only_u15_public_presets_remain(self) -> None:
        configs = sorted(path.name for path in (ROOT / "training/configs/sensenovavl_qwen3_gen").glob("*.py"))
        launchers = sorted(path.name for path in (ROOT / "training/shell/train_u1").glob("*.sh"))
        self.assertEqual(configs, ["sensenovau1_5_8b_mot_sft.py"])
        self.assertEqual(launchers, ["U1.5_8B_SFT.sh"])

    def test_fsdp2_is_the_only_sft_trainer(self) -> None:
        launcher = (ROOT / "training/shell/train_u1/U1.5_8B_SFT.sh").read_text()
        runner = (ROOT / "training/train_sensenovau1_fsdp2.py").read_text()
        self.assertIn("tensor_parallel_mode=mtp", launcher)
        self.assertNotIn("SFT_PER_RANK_LOSS_REDUCTION", launcher)
        self.assertIn("FSDP2_PREFETCH_DEPTH", runner)
        self.assertIn("reduce_dtype=torch.bfloat16", runner)
        self.assertIn("torch.distributed.checkpoint", runner)
        self.assertIn("supports only NVIDIA H200", runner)
        self.assertIn("SFT_BENCHMARK_ONLY", runner)
        self.assertIn("benchmark-only SFT cannot publish an HF checkpoint", runner)
        self.assertFalse((ROOT / "training/train_sensenovau1.py").exists())
        self.assertFalse(tuple((ROOT / "training/shell/ablation").glob("*.sh")))
        self.assertFalse((ROOT / "training/pyproject.toml").exists())


if __name__ == "__main__":
    unittest.main()
