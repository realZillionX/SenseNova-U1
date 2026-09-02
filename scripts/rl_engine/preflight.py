#!/usr/bin/env python3
"""Fail-fast runtime/provenance check for one RL serving GPU pair."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import inspect
import json
import os
import platform
import subprocess
import sys
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, packages_distributions, version
from pathlib import Path

EXPECTED_PYTHON = "/opt/sensenova-forge-py312/bin/python"
EXPECTED_DISTRIBUTIONS = {
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "torchaudio": "2.8.0",
    "triton": "3.4.0",
    "transformers": "4.57.1",
    "tokenizers": "0.22.1",
    "huggingface-hub": "0.36.2",
    "numpy": "2.5.2",
    "dill": "0.4.1",
    "einops": "0.8.1",
    "imageio": "2.37.4",
    "opencv-python": "5.0.0.93",
    "tensorboard": "2.20.0",
    "timm": "1.0.28",
    "httpx": "0.28.1",
    "protobuf": "7.35.1",
    "nvidia-cuda-runtime-cu12": "12.8.90",
    "nvidia-cusolver-cu12": "11.7.3.90",
    "nvidia-nccl-cu12": "2.27.3",
    "flash-attn-3": "3.0.0",
}
MANIFEST_DIR = Path("/opt/sensenova-forge/manifests/rl-serving")
LIGHTLLM_POLICY_MARKERS = {
    "lightllm/server/api_rl.py": (
        '"repetition_penalty": 1.0',
        '"guidance_scale": 1.0',
    ),
    "lightllm/server/core/objs/x2i_params.py": ("_cfg_norm: CfgNormType = CfgNormType.NONE",),
    "lightllm/server/x2i_server/manager.py": ("scheduler.infer_steps = int(param.steps)",),
}


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


def _source_commit(path: str, label: str) -> str | None:
    """Resolve a checkout revision on hosts and the sealed build revision in images."""
    checkout = _git_commit(path)
    if checkout is not None:
        return checkout
    manifest = MANIFEST_DIR / "source-commits.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    commit = payload.get(label)
    return commit if isinstance(commit, str) and commit else None


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_manifest(model_path: str | None) -> dict[str, object]:
    result: dict[str, object] = {
        "index": None,
        "tensor_count": 0,
        "shard_count": 0,
        "missing_shards": [],
        "error": None,
    }
    if not model_path:
        return result
    root = Path(model_path)
    try:
        indexes = sorted(root.glob("*.safetensors.index.json"))
        if len(indexes) != 1:
            raise ValueError(f"expected one safetensors index, found {len(indexes)}")
        payload = json.loads(indexes[0].read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("safetensors index has no non-empty weight_map")
        shards = sorted(set(weight_map.values()))
        missing = [name for name in shards if not (root / name).is_file()]
        result.update(
            index=str(indexes[0]),
            tensor_count=len(weight_map),
            shard_count=len(shards),
            missing_shards=missing,
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _lightllm_policy_overlay() -> dict[str, object]:
    root = Path(os.getenv("FORGE_LIGHTLLM_ROOT", "/workspace/LightLLM"))
    missing = []
    for relative, markers in LIGHTLLM_POLICY_MARKERS.items():
        path = root / relative
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        for marker in markers:
            if marker not in text:
                missing.append(f"{relative}: {marker}")
    return {"available": not missing, "missing": missing}


def _rdma_status() -> dict[str, object]:
    devices_root = Path("/sys/class/infiniband")
    try:
        devices = sorted(path.name for path in devices_root.iterdir())
    except OSError:
        devices = []
    verbs_error = None
    try:
        ctypes.CDLL("libibverbs.so.1")
        verbs_available = True
    except OSError as exc:
        verbs_available = False
        verbs_error = str(exc)
    return {
        "available": bool(verbs_available and devices),
        "libibverbs": verbs_available,
        "devices": devices,
        "error": verbs_error,
    }


def _serving_x2v_config(path: str | None) -> dict[str, object]:
    result: dict[str, object] = {"path": path, "config": None, "error": None}
    if not path:
        result["error"] = "LightX2V serving config path is missing"
        return result
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("config must be a JSON object")
        result["config"] = payload
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--x2v-config")
    parser.add_argument("--expected-gpus", type=int, default=2)
    parser.add_argument("--allow-no-gpu", action="store_true")
    parser.add_argument("--require-rdma", action="store_true")
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

    http_server = {"available": False, "module": None, "error": None}
    try:
        if args.allow_no_gpu:
            spec = importlib.util.find_spec("lightllm.server.api_http")
            if spec is None:
                raise ModuleNotFoundError("lightllm.server.api_http")
            http_server.update(available=True, module=spec.name)
        else:
            import lightllm.server.api_http as api_http

            http_server.update(available=True, module=api_http.__name__)
    except Exception as exc:
        http_server["error"] = f"{type(exc).__name__}: {exc}"

    modules = {
        name: _version(name)
        for name in (
            "lightllm",
            "lightx2v",
            "safetensors",
            "fastapi",
            "hypercorn",
            "websockets",
            "pydantic",
            "zmq",
            "rpyc",
        )
    }
    flash_attention = {
        name: importlib.util.find_spec(name) is not None
        for name in ("flash_attn", "flash_attn_interface", "flash_attn_3", "flashinfer")
    }
    neo_fa3 = {
        "available": False,
        "image_token_end": False,
        "lightllm_backend_imported": False,
        "path": None,
        "error": None,
    }
    try:
        import flash_attn_interface
        from flash_attn_interface import flash_attn_with_kvcache

        has_scoped_abi = "image_token_end" in inspect.signature(flash_attn_with_kvcache).parameters
        neo_fa3.update(
            available=has_scoped_abi,
            image_token_end=has_scoped_abi,
            path=str(Path(flash_attn_interface.__file__).resolve()),
        )
        # Importing the full LightLLM attention package initializes GPU-specific
        # Triton autotune state.  Verify that adapter on GPU; the CPU builder
        # still verifies the installed native ABI directly above.
        if torch.cuda.is_available():
            from lightllm.common.basemodel.attention.fa3.fp import (
                FA3_NEO_ARGUMENT,
                HAS_FLASH_ATTN_INTERFACE,
            )

            neo_fa3["available"] = bool(
                has_scoped_abi and HAS_FLASH_ATTN_INTERFACE and FA3_NEO_ARGUMENT == "image_token_end"
            )
            neo_fa3["lightllm_backend_imported"] = True
    except Exception as exc:
        neo_fa3["error"] = f"{type(exc).__name__}: {exc}"
    nccl_version = None
    if torch.cuda.is_available():
        try:
            nccl_version = list(torch.cuda.nccl.version())
        except (AttributeError, TypeError):
            nccl_version = str(torch.cuda.nccl.version())

    checkpoint_manifest = _checkpoint_manifest(args.model_path)
    lightllm_policy_overlay = _lightllm_policy_overlay()
    rdma = _rdma_status()
    serving_x2v_config = _serving_x2v_config(args.x2v_config)
    try:
        trace_ttl_seconds = int(os.getenv("MOVA_RL_TRACE_TTL", "3600"))
    except ValueError:
        trace_ttl_seconds = 0
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
        "neo_fa3": neo_fa3,
        "torch_flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "modules": modules,
        "neo_runner": neo_runner,
        "http_server": http_server,
        "distributions": distributions,
        "runtime_manifest": {
            "directory": str(MANIFEST_DIR),
            "requirements_sha256": _sha256(MANIFEST_DIR / "requirements.lock"),
            "contract_sha256": _sha256(MANIFEST_DIR / "runtime_contract.json"),
            "pip_freeze_exists": (MANIFEST_DIR / "pip-freeze.txt").is_file(),
        },
        "provenance": {
            "runtime_image": os.getenv("FORGE_RUNTIME_IMAGE", "unknown"),
            "forge_commit": os.getenv("FORGE_COMMIT", "unknown"),
            "forge_checkout": _source_commit(os.getenv("FORGE_ROOT", ""), "forge"),
            "lightllm_commit": os.getenv("FORGE_LIGHTLLM_COMMIT", "unknown"),
            "lightx2v_commit": os.getenv("FORGE_LIGHTX2V_COMMIT", "unknown"),
            "lightllm_checkout": _source_commit(os.getenv("FORGE_LIGHTLLM_ROOT", "/workspace/LightLLM"), "lightllm"),
            "lightx2v_checkout": _source_commit(os.getenv("FORGE_LIGHTX2V_ROOT", "/workspace/LightX2V"), "lightx2v"),
        },
        "model_path": args.model_path,
        "model_exists": bool(args.model_path and Path(args.model_path).is_dir()),
        "model_checkpoint": checkpoint_manifest,
        "lightllm_policy_overlay": lightllm_policy_overlay,
        "serving_x2v_config": serving_x2v_config,
        "rl_trace": {
            "root": os.getenv("MOVA_RL_TRACE_DIR", "/dev/shm/mova_rl_traces"),
            "ttl_seconds": trace_ttl_seconds,
        },
        "rdma": rdma,
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
    if not args.allow_no_gpu:
        unsupported = [name for name in payload["gpu_names"] if "H200" not in name.upper()]
        if unsupported:
            errors.append(f"SenseNova runtime supports only NVIDIA H200, found {unsupported}")
    if args.model_path and not payload["model_exists"]:
        errors.append(f"model path does not exist: {args.model_path}")
    if args.model_path and (
        checkpoint_manifest["error"] or checkpoint_manifest["missing_shards"] or not checkpoint_manifest["tensor_count"]
    ):
        errors.append(f"model safetensors closure is incomplete: {checkpoint_manifest}")
    for name, info in modules.items():
        if not info["available"]:
            errors.append(f"required module is unavailable: {name}")
    if not payload["flash_attention"]["flash_attn_interface"]:
        errors.append("FA3 module flash_attn_interface is unavailable")
    if not neo_fa3["available"] or not neo_fa3["image_token_end"]:
        errors.append(f"LightLLM FA3-Neo is unavailable or lacks scoped image_token_end support: {neo_fa3['error']}")
    if not neo_runner["available"]:
        errors.append(f"NeoPP import closure failed: {neo_runner['error']}")
    if not http_server["available"]:
        errors.append(f"LightLLM HTTP server import closure failed: {http_server['error']}")
    if not lightllm_policy_overlay["available"]:
        errors.append(f"SenseNova LightLLM policy overlay is incomplete: {lightllm_policy_overlay['missing']}")
    if trace_ttl_seconds <= 0:
        errors.append("RL trace TTL must be a positive integer")
    x2v_payload = serving_x2v_config["config"]
    if serving_x2v_config["error"] or not isinstance(x2v_payload, dict):
        errors.append(f"LightX2V serving config is unavailable: {serving_x2v_config['error']}")
    elif x2v_payload.get("enable_cfg") is not True or x2v_payload.get("cfg_scale") != 4.0:
        errors.append("ordinary LightX2V serving config must preserve the U1.5 CFG profile")
    if args.require_rdma and not rdma["available"]:
        errors.append(f"multi-node RL serving requires libibverbs.so.1 and a visible InfiniBand/RoCE device: {rdma}")
    if payload["runtime_manifest"]["requirements_sha256"] is None:
        errors.append(f"runtime manifest is missing under {MANIFEST_DIR}")
    provenance = payload["provenance"]
    for label in ("forge", "lightllm", "lightx2v"):
        declared = provenance[f"{label}_commit"]
        observed = provenance[f"{label}_checkout"]
        if declared in {None, "", "unknown", "UNSET"}:
            errors.append(f"{label} source commit is not declared")
        elif observed != declared:
            errors.append(f"{label} checkout revision {observed!r} does not match declared {declared!r}")
    payload["errors"] = errors
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
