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
        self.assertIn("total_steps=${total_steps:-200000}", launcher)
        self.assertIn("seq_len=${seq_len:-8192}", launcher)
        self.assertIn("cfg_txt_uncond_drop_prob=0", launcher)
        self.assertIn("cfg_img_uncond_drop_prob=0", launcher)
        self.assertIn("cfg_txtimg_uncond_drop_prob=0", launcher)
        self.assertIn("JOB_NAME=${JOB_NAME:?", launcher)
        self.assertIn('RUN_ROOT=${RUN_ROOT:-"RUN"}', launcher)
        self.assertIn("model.safetensors.index.json", launcher)
        self.assertIn("WORLD_SIZE % MODEL_PARALLEL_SIZE", launcher)
        config = (ROOT / "training/configs/sensenovavl_qwen3_gen/sensenovau1_5_8b_mot_sft.py").read_text()
        self.assertIn('f"/dev/shm/sensenovalm_tmp_ckpt/{JOB_NAME}"', config)
        self.assertIn('Path(os.environ.get("RUN_ROOT", "RUN"))', config)
        self.assertNotIn("enabel_und_loss", config)

    def test_only_u15_public_presets_remain(self) -> None:
        configs = sorted(path.name for path in (ROOT / "training/configs/sensenovavl_qwen3_gen").glob("*.py"))
        launchers = sorted(path.name for path in (ROOT / "training/shell/train_u1").glob("*.sh"))
        self.assertEqual(configs, ["sensenovau1_5_8b_mot_sft.py"])
        self.assertEqual(launchers, ["U1.5_8B_SFT.sh"])


if __name__ == "__main__":
    unittest.main()
