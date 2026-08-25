#!/usr/bin/env python3
"""Fail-fast runtime/provenance check for the two-GPU RL serving stack."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, packages_distributions, version
from hashlib import sha256
from pathlib import Path


EXPECTED_BASE_DIGEST = "sha256:bb1900389c320b37dbcfe51fdf4db76a198d38a10c4c80d8b9b0726f1fb43ac7"
EXPECTED_PYTHON = "/opt/mostar-u1-py312/bin/python"
EXPECTED_DISTRIBUTIONS = {
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "torchaudio": "2.8.0",
    "triton": "3.4.0",
    "transformers": "4.57.1",
    "tokenizers": "0.22.1",
    "huggingface-hub": "0.36.2",
    "numpy": "2.5.2",
    "protobuf": "7.35.1",
    "flash-attn-3": "3.0.0+20260817.cu128torch280cxx11abitrue.25110",
}
MANIFEST_DIR = Path("/opt/mova-runtime-manifests/lightllm-x2v")


def _version(module_name: str) -> dict[str, str | bool | None]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return {"available": False, "version": None, "path": None}
    distribution_version = None
    top_level = module_name.split(".", 1)[0]
    for distribution in packages_distributions().get(top_level, ()):
        try:
            distribution_version = version(distribution)
            break
        except PackageNotFoundError:
            continue
    return {
        "available": True,
        "version": distribution_version or "source-tree",
        "path": str(spec.origin or spec.submodule_search_locations or "unknown"),
    }


def _git_commit(path: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--expected-gpus", type=int, default=2)
    parser.add_argument("--allow-no-gpu", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    import torch

    distributions = {}
    for name, expected in EXPECTED_DISTRIBUTIONS.items():
        try:
            actual = version(name)
        except PackageNotFoundError:
            actual = None
        distributions[name] = {"expected": expected, "actual": actual}

    neo_runner = {"available": False, "class": None, "error": None}
    try:
        if args.allow_no_gpu:
            os.environ.setdefault("SKIP_PLATFORM_CHECK", "1")
        from lightx2v.pipeline import _ensure_runner_registered
        from lightx2v.utils.registry_factory import RUNNER_REGISTER

        _ensure_runner_registered("neopp")
        runner_class = RUNNER_REGISTER["neopp"]
        neo_runner = {
            "available": True,
            "class": f"{runner_class.__module__}.{runner_class.__name__}",
            "error": None,
        }
    except Exception as exc:
        neo_runner["error"] = f"{type(exc).__name__}: {exc}"

    modules = {
        name: _version(name)
        for name in ("lightllm", "lightx2v", "safetensors", "fastapi", "pydantic", "zmq", "rpyc")
    }
    flash_attention = {
        name: importlib.util.find_spec(name) is not None
        for name in ("flash_attn", "flash_attn_interface", "flash_attn_3", "flashinfer")
    }
    nccl_version = None
    if torch.cuda.is_available():
        try:
            nccl_version = list(torch.cuda.nccl.version())
        except (AttributeError, TypeError):
            nccl_version = str(torch.cuda.nccl.version())

    payload = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "nccl": nccl_version,
        "flash_attention": flash_attention,
        "torch_flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "modules": modules,
        "neo_runner": neo_runner,
        "distributions": distributions,
        "runtime_manifest": {
            "directory": str(MANIFEST_DIR),
            "requirements_sha256": _sha256(MANIFEST_DIR / "requirements.lock"),
            "contract_sha256": _sha256(MANIFEST_DIR / "runtime_contract.json"),
            "pip_freeze_exists": (MANIFEST_DIR / "pip-freeze.txt").is_file(),
        },
        "provenance": {
            "requested_base_digest": EXPECTED_BASE_DIGEST,
            "runtime_image_digest": os.getenv("MOVA_IMAGE_DIGEST", "unknown"),
            "platform_parent_image": os.getenv("MOVA_PLATFORM_PARENT_IMAGE", "unknown"),
            "sensenova_commit": os.getenv("MOVA_SENSENOVA_COMMIT", "unknown"),
            "lightllm_commit": os.getenv("MOVA_LIGHTLLM_COMMIT", "unknown"),
            "lightx2v_commit": os.getenv("MOVA_LIGHTX2V_COMMIT", "unknown"),
            "lightllm_checkout": _git_commit(os.getenv("MOVA_LIGHTLLM_ROOT", "/workspace/LightLLM")),
            "lightx2v_checkout": _git_commit(os.getenv("MOVA_LIGHTX2V_ROOT", "/workspace/LightX2V")),
        },
        "model_path": args.model_path,
        "model_exists": bool(args.model_path and Path(args.model_path).is_dir()),
    }

    errors = []
    if Path(sys.executable).resolve() != Path(EXPECTED_PYTHON).resolve():
        errors.append(f"expected interpreter {EXPECTED_PYTHON}, found {sys.executable}")
    for name, info in distributions.items():
        if info["actual"] != info["expected"]:
            errors.append(f"expected {name} {info['expected']}, found {info['actual']}")
    if not str(torch.__version__).startswith("2.8.0"):
        errors.append(f"expected Torch 2.8.0, found {torch.__version__}")
    if torch.version.cuda != "12.8":
        errors.append(f"expected CUDA 12.8 Torch build, found {torch.version.cuda}")
    if not args.allow_no_gpu and torch.cuda.device_count() < args.expected_gpus:
        errors.append(f"expected at least {args.expected_gpus} GPUs, found {torch.cuda.device_count()}")
    if args.model_path and not payload["model_exists"]:
        errors.append(f"model path does not exist: {args.model_path}")
    for name, info in modules.items():
        if not info["available"]:
            errors.append(f"required module is unavailable: {name}")
    if not payload["flash_attention"]["flash_attn_interface"]:
        errors.append("FA3 module flash_attn_interface is unavailable")
    if not neo_runner["available"]:
        errors.append(f"NeoPP import closure failed: {neo_runner['error']}")
    if payload["runtime_manifest"]["requirements_sha256"] is None:
        errors.append(f"runtime manifest is missing under {MANIFEST_DIR}")
    payload["errors"] = errors
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
