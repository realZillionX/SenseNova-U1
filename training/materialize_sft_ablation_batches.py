#!/usr/bin/env python
"""Materialize world-size-independent packed batches for SFT trainer ablations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from sensenovalm.initialize import initialize_distributed_env
from sensenovalm.utils.common import parse_args
from sensenovavl.data import build_train_loader_with_data_type
from sensenovavl.utils.utils import init_pil


def _hash_value(digest: Any, value: Any) -> None:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, dict):
        digest.update(b"dict\0")
        for name in sorted(value):
            digest.update(str(name).encode())
            digest.update(b"\0")
            _hash_value(digest, value[name])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"list\0")
        for item in value:
            _hash_value(digest, item)
        return
    digest.update(repr(value).encode())
    digest.update(b"\0")


def _microbatch(value: Any, index: int) -> Any:
    if isinstance(value, Tensor):
        return value[index : index + 1]
    if isinstance(value, (list, tuple)):
        return value[index : index + 1]
    if isinstance(value, bool):
        return value
    raise TypeError(f"unsupported fixed-batch value: {type(value)!r}")


def _batch_digests(batch: tuple[dict[str, Any], Any], microbatches: int) -> list[str]:
    data, labels = batch
    result: list[str] = []
    for index in range(microbatches):
        digest = hashlib.sha256()
        selected = {
            name: _microbatch(value, index)
            for name, value in data.items()
            if name not in {"num_samples", "num_padding_tokens"}
        }
        _hash_value(digest, selected)
        if isinstance(labels, Tensor):
            _hash_value(digest, labels[index : index + 1])
        elif isinstance(labels, dict):
            _hash_value(
                digest,
                {name: value[index : index + 1] if value.dim() else value for name, value in labels.items()},
            )
        else:
            _hash_value(digest, labels)
        result.append(digest.hexdigest())
    return result


def main() -> None:
    if dist.get_world_size() != 1:
        raise ValueError("fixed SFT batches must be materialized with world_size=1")
    output = Path(os.environ["SFT_ABLATION_BATCH_OUTPUT"]).resolve()
    sidecar = output.with_suffix(output.suffix + ".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite fixed SFT batch artifacts")
    batch_count = int(os.environ.get("SFT_ABLATION_BATCH_COUNT", "13"))
    if batch_count < 1:
        raise ValueError("SFT_ABLATION_BATCH_COUNT must be positive")
    microbatches = int(gpc.config.data.micro_num)
    if microbatches < 1:
        raise ValueError("fixed SFT batch width must be positive")

    loader, _dataset_types = build_train_loader_with_data_type()
    iterator = iter(loader)
    batches: list[tuple[dict[str, Any], Any]] = []
    ordered_digests: list[list[str]] = []
    for _step in range(batch_count):
        batch = next(iterator)
        data, labels = batch
        data.pop("worker_state_key_list", None)
        data.pop("worker_state_dict_list", None)
        data.pop("worker_state_custom_infos_list", None)
        if data.pop("is_empty_data_list", False):
            raise RuntimeError("dataset exhausted while materializing fixed batches")
        ordered_digests.append(_batch_digests((data, labels), microbatches))
        batches.append((data, labels))

    payload = {
        "schema": "sensenova_u15.sft_ablation_batches.v1",
        "seed": int(os.environ.get("SEED", "42")),
        "sequence_length": int(gpc.config.data.seq_len),
        "microbatches_per_optimizer_step": microbatches,
        "data_meta": str(gpc.config.data.meta_path),
        "ordered_microbatch_sha256": ordered_digests,
        "batches": batches,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    file_digest = hashlib.sha256()
    with output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            file_digest.update(chunk)
    identity = {
        key: value for key, value in payload.items() if key != "batches"
    }
    identity.update(
        {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": file_digest.hexdigest(),
        }
    )
    sidecar.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "fixed_batches_materialized", **identity}, sort_keys=True), flush=True)


if __name__ == "__main__":
    from sensenovalm.core.context import global_context as gpc

    args = parse_args()
    initialize_distributed_env(config=args.config, launcher=args.launcher, master_port=args.port, seed=args.seed)
    init_pil()
    main()
