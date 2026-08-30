"""Small command surface for SFT, RL and production serving."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .rl.plan import RlPlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(command: list[str], *, cwd: Path) -> int:
    return subprocess.call(command, cwd=cwd, env=os.environ.copy())


def _plan_rl(config: Path) -> int:
    payload = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RL plan config must be a JSON object")
    plan = RlPlan.from_dict(payload)
    plan.write()
    print(plan.plan_path)
    return 0


def _run_rl(path: Path) -> int:
    plan = RlPlan.read(path.resolve())
    spec = plan.torchrun
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc-per-node={spec.nproc_per_node}",
        f"--nnodes={spec.nnodes}",
        f"--node-rank={spec.node_rank}",
        f"--master-addr={spec.master_addr}",
        f"--master-port={spec.master_port}",
        "-m",
        "sensenova_u1.rl.trainer",
        str(path.resolve()),
    ]
    return _run(command, cwd=_repo_root())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sensenova-forge",
        description="High-performance SFT, RL and serving for SenseNova-U1.5-8B-MoT.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    sft = subcommands.add_parser("sft", help="Launch the FSDP2 full-parameter SFT preset.")
    sft.add_argument("extra", nargs=argparse.REMAINDER)
    serve = subcommands.add_parser("serve", help="Launch LightLLM + LightX2V serving.")
    serve.add_argument("extra", nargs=argparse.REMAINDER)
    rl_plan = subcommands.add_parser("rl-plan", help="Validate and seal an RL plan from JSON.")
    rl_plan.add_argument("config", type=Path)
    rl_run = subcommands.add_parser("rl-run", help="Run a sealed FSDP2 RL plan.")
    rl_run.add_argument("plan", type=Path)
    args = parser.parse_args()

    root = _repo_root()
    if args.command == "sft":
        raise SystemExit(
            _run(
                ["bash", "training/shell/train_u1/U1.5_8B_SFT.sh", *args.extra],
                cwd=root,
            )
        )
    if args.command == "serve":
        raise SystemExit(_run(["bash", "scripts/rl_engine/launch_server.sh", *args.extra], cwd=root))
    if args.command == "rl-plan":
        raise SystemExit(_plan_rl(args.config))
    if args.command == "rl-run":
        raise SystemExit(_run_rl(args.plan))
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
