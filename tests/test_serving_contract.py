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
        self.assertEqual(contract["platform"]["gpu"], "NVIDIA H200")
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
        self.assertIn("LIGHTLLM_MEM_FRACTION:-0.80", launcher)
        self.assertIn('--mem_fraction "$LIGHTLLM_MEM_FRACTION"', launcher)
        self.assertIn("LIGHTLLM_TRITON_AUTOTUNE_LEVEL:-1", launcher)
        self.assertIn("FORGE_SERVING_REPLICAS", launcher)
        self.assertIn("FORGE_SERVING_PORT_BASE", launcher)
        self.assertIn("FORGE_SERVING_REPLICA_ID_OFFSET", launcher)
        self.assertIn('device_pair="${VISIBLE_GPUS[$((2 * local_index))]},${VISIBLE_GPUS[$((2 * local_index + 1))]}"', launcher)
        self.assertIn('MOVA_RL_TRACE_DIR="$TRACE_ROOT/replica-$replica_id"', launcher)
        self.assertIn('--port "$port"', launcher)
        manager = (
            ROOT
            / "serving/third_party/LightLLM/lightllm/server/httpserver/manager.py"
        ).read_text()
        self.assertIn('replica_count = int(payload.get("replica_count", 1))', manager)
        self.assertIn('"language_rank_base": 1 + replica_index', manager)
        self.assertIn('expected_world = 1 + 3 * replica_count', manager)
        self.assertIn('"replica_id": int(os.getenv("MOVA_RL_REPLICA_ID", "0"))', manager)
        self.assertIn("async def commit_weights_update", manager)
        api_http = (
            ROOT / "serving/third_party/LightLLM/lightllm/server/api_http.py"
        ).read_text()
        self.assertIn('@app.post("/commit_weights_update")', api_http)
        api_start = (
            ROOT / "serving/third_party/LightLLM/lightllm/server/api_start.py"
        ).read_text()
        self.assertIn('internal_port_start = 10000 + replica_id * 2048', api_start)
        self.assertIn('from_port_num=internal_port_start', api_start)
        preflight = (ROOT / "scripts" / "rl_engine" / "preflight.py").read_text()
        self.assertIn("supports only NVIDIA H200", preflight)
        self.assertIn('parser.add_argument("--require-rdma", action="store_true")', preflight)
        self.assertIn('ctypes.CDLL("libibverbs.so.1")', preflight)
        dockerfile = (ROOT / "docker/rl-engine/Dockerfile").read_text()
        self.assertIn("libibverbs1 ibverbs-providers librdmacm1 rdma-core", dockerfile)


if __name__ == "__main__":
    unittest.main()
