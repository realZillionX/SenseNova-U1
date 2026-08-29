from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "serving" / "configs" / "neopp_u15_forge_512.json"


class ServingContractTest(unittest.TestCase):
    def test_launcher_uses_the_sealed_forge_profile(self) -> None:
        launcher = (ROOT / "scripts" / "rl_engine" / "launch_server.sh").read_text()
        self.assertIn("serving/configs/neopp_u15_forge_512.json", launcher)
        self.assertIn('--x2v_gen_model_config "$X2V_CONFIG"', launcher)
        config = json.loads(CONFIG.read_text())
        self.assertEqual(config["infer_steps"], 30)
        self.assertEqual(config["timestep_shift"], 1.0)
        self.assertEqual(config["min_pixels"], 512 * 512)
        self.assertEqual(config["max_pixels"], 512 * 512)
        self.assertEqual(config["attn_type"], "flash_attn3")


if __name__ == "__main__":
    unittest.main()
