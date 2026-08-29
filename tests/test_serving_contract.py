from __future__ import annotations

import hashlib
import json
import subprocess
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
        lightllm = ROOT / "serving/third_party/LightLLM"
        overlay = ROOT / "serving/patches/lightllm-sensenova-policy.patch"
        check = subprocess.run(
            ["git", "-C", str(lightllm), "apply", "--check", str(overlay)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(check.returncode, 0, check.stderr)
        patch = overlay.read_text()
        self.assertGreaterEqual(patch.count("scheduler.infer_steps = int(param.steps)"), 2)
        self.assertIn("_cfg_norm: CfgNormType = CfgNormType.NONE", patch)
        self.assertIn('"repetition_penalty": 1.0', patch)
        self.assertIn('"presence_penalty": 0.0', patch)
        self.assertIn('"frequency_penalty": 0.0', patch)
        contract = json.loads((ROOT / "docker/rl-engine/runtime_contract.json").read_text())
        expected_sha = contract["overlays"]["lightllm_sensenova_policy"]["sha256"]
        self.assertEqual(hashlib.sha256(overlay.read_bytes()).hexdigest(), expected_sha)
        smoke = (ROOT / "examples" / "serving" / "rl_smoke.py").read_text()
        self.assertIn('"image_steps": 30', smoke)
        self.assertIn('"timestep_shift": 1.0', smoke)
        self.assertIn('"sde_window_end": 30', smoke)

    def test_client_separates_official_inference_from_rl_sampling(self) -> None:
        client = (ROOT / "examples/serving/client.py").read_text()
        self.assertIn('"temperature": 0.6', client)
        self.assertIn('"top_p": 0.95', client)
        self.assertIn('"top_k": 20', client)
        self.assertIn('"repetition_penalty": 1.05', client)
        self.assertIn('"steps": 50', client)
        self.assertIn('"cfg_norm": "none"', client)
        launcher = (ROOT / "scripts/rl_engine/launch_server.sh").read_text()
        self.assertIn("INPUT_PENALTY", launcher)
        self.assertIn("MAX_REQ_TOTAL_LEN:-16384", launcher)


if __name__ == "__main__":
    unittest.main()
