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

    def test_only_u15_public_presets_remain(self) -> None:
        configs = sorted(path.name for path in (ROOT / "training/configs/sensenovavl_qwen3_gen").glob("*.py"))
        launchers = sorted(path.name for path in (ROOT / "training/shell/train_u1").glob("*.sh"))
        self.assertEqual(configs, ["sensenovau1_5_8b_mot_sft.py"])
        self.assertEqual(launchers, ["U1.5_8B_SFT.sh"])


if __name__ == "__main__":
    unittest.main()
