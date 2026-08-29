"""Pinned Hugging Face checkpoint loading for Forge replay and inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from . import accel


def _resolve_local_model_path(model_path: str) -> str:
    if Path(model_path).exists():
        return model_path
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(model_path, local_files_only=True))
    except Exception:
        return model_path
    return str(snapshot) if _has_complete_model_weights(snapshot) else model_path


def _has_complete_model_weights(snapshot: Path) -> bool:
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = snapshot / index_name
        if not index_path.is_file():
            continue
        try:
            shard_names = set(json.loads(index_path.read_text())["weight_map"].values())
        except (KeyError, TypeError, OSError, json.JSONDecodeError):
            return False
        return bool(shard_names) and all((snapshot / shard_name).is_file() for shard_name in shard_names)
    return (snapshot / "model.safetensors").is_file() or (snapshot / "pytorch_model.bin").is_file()


def load_model_and_tokenizer(
    model_path: str,
    *,
    dtype: torch.dtype,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, Any]:
    """Load the exact U1.5 model/tokenizer pair used by differentiable replay."""

    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from .. import check_checkpoint_compatibility, register_models
    from ..models.neo_unify.transformers_compat import pretrained_dtype_kwargs

    target = accel.best_available_device() if device is None else torch.device(device)
    resolved = _resolve_local_model_path(model_path)
    register_models()
    config = AutoConfig.from_pretrained(resolved)
    check_checkpoint_compatibility(config)
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    model = AutoModel.from_pretrained(
        resolved,
        config=config,
        **pretrained_dtype_kwargs(dtype),
    ).eval()
    return model.to(target), tokenizer


__all__ = ["load_model_and_tokenizer"]
