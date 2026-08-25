#!/usr/bin/env python3
"""Fail-fast runtime/provenance check for the two-GPU RL serving stack."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


EXPECTED_BASE_DIGEST = "sha256:bb1900389c320b37dbcfe51fdf4db76a198d38a10c4c80d8b9b0726f1fb43ac7"


def _version(module_name: str) -> dict[str, str | bool | None]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return {"available": False, "version": None, "path": None}
    module = importlib.import_module(module_name)
    return {
        "available": True,
        "version": str(getattr(module, "__version__", "unknown")),
        "path": str(getattr(module, "__file__", "unknown")),
    }


def _git_commit(path: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--expected-gpus", type=int, default=2)
    parser.add_argument("--allow-no-gpu", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    import torch

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
    payload["errors"] = errors
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
