#!/usr/bin/env python
"""Production Torch 2.8 FSDP2 SFT runner for SenseNova-U1.5-8B-MoT.

The model, native-resolution packed loader, forward, and losses remain the
checkpoint-specific U1.5 implementations. FSDP2 is the sole trainer and owns
full-parameter sharding, optimizer state in memory, model-only checkpointing,
and publication.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import socket
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from sensenovalm.core.context import global_context as gpc
from sensenovalm.data.utils import packed_data_normalizer
from sensenovalm.data.sample_progress import SampleProgress
from sensenovalm.initialize.launch import initialize_distributed_env
from sensenovalm.model.losses.ce_loss import FlashGPTLMLoss
from sensenovalm.train.fsdp import initialize_unit_mtp_communicators, training_profile
from sensenovalm.utils.common import move_to_device, parse_args
from sensenovavl.data import build_train_loader_with_data_type
from sensenovavl.train.pipeline import get_model
from sensenovavl.utils.utils import check_image_fn, init_pil
from torch import Tensor, nn

_BLOCK_CLASS_NAMES = frozenset({"Qwen3MoeMoTDecoder", "InternVisionEncoderLayer"})


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_h200() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("SenseNova training requires NVIDIA H200 GPUs")
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", dist.get_world_size()))
    if torch.cuda.device_count() != local_world_size:
        raise RuntimeError("visible GPU count differs from the declared local training ranks")
    name = torch.cuda.get_device_name(torch.cuda.current_device())
    if "H200" not in name.upper():
        raise RuntimeError(f"SenseNova training supports only NVIDIA H200, found {name!r}")
    return name


def _reshard_after_forward() -> bool | int:
    raw = os.environ.get("FSDP2_RESHARD_AFTER_FORWARD", "true").strip().lower()
    if raw in {"true", "1", "yes", "on"}:
        return True
    if raw in {"false", "0", "no", "off"}:
        return False
    value = int(raw)
    world_size = dist.get_world_size()
    if value <= 1 or value >= world_size or world_size % value:
        raise ValueError(
            "integer FSDP2_RESHARD_AFTER_FORWARD must be a non-trivial divisor of world_size"
        )
    return value


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _slice_microbatch(value: Any, offset: int) -> Any:
    if isinstance(value, Tensor):
        return value[offset : offset + 1]
    if isinstance(value, (list, tuple)):
        return value[offset : offset + 1]
    if isinstance(value, bool):
        return value
    raise TypeError(f"unsupported packed microbatch value: {type(value)!r}")


def _prepare_microbatch(batch: tuple[dict[str, Any], Any], offset: int) -> tuple[dict[str, Any], Any]:
    data, labels = batch
    micro_data = {name: _slice_microbatch(value, offset) for name, value in data.items()}
    if isinstance(labels, Tensor):
        micro_labels: Any = labels[offset : offset + 1]
    elif isinstance(labels, dict):
        micro_labels = {
            name: value[offset : offset + 1] if value.dim() else value
            for name, value in labels.items()
        }
    else:
        micro_labels = labels
    micro_data, micro_labels = packed_data_normalizer(micro_data, micro_labels)
    return check_image_fn(micro_data, micro_labels)


def _shard_model(model: nn.Module) -> tuple[nn.Module, tuple[str, ...]]:
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import DTensor

    if not dist.is_initialized() or dist.get_world_size() < 2:
        raise ValueError("FSDP2 SFT requires a multi-rank torchrun process group")
    model.requires_grad_(True)
    model.to(dtype=torch.float32)
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        output_dtype=None,
        cast_forward_inputs=True,
    )
    reshard_after_forward = _reshard_after_forward()
    wrapped: list[tuple[str, FSDPModule]] = []
    seen: set[int] = set()
    for name, module in tuple(model.named_modules()):
        if type(module).__name__ not in _BLOCK_CLASS_NAMES or id(module) in seen:
            continue
        seen.add(id(module))
        fully_shard(
            module,
            mp_policy=policy,
            reshard_after_forward=reshard_after_forward,
        )
        if not isinstance(module, FSDPModule):
            raise TypeError(f"FSDP2 did not instrument {name}")
        wrapped.append((name, module))
    if not wrapped:
        raise ValueError("U1.5 exposes no recognized FSDP2 SFT blocks")

    # The complete SFT forward always enters this root, unlike RL replay's
    # direct submodule calls.  Keep root parameters resident after forward as
    # recommended by FSDP2; child resharding remains an explicit tuning knob.
    fully_shard(model, mp_policy=policy, reshard_after_forward=False)
    if not isinstance(model, FSDPModule):
        raise TypeError("FSDP2 did not instrument the U1.5 SFT root")

    prefetch_depth = _env_int("FSDP2_PREFETCH_DEPTH", 1)
    if prefetch_depth:
        for index, (_name, module) in enumerate(wrapped):
            following = [candidate for _, candidate in wrapped[index + 1 : index + 1 + prefetch_depth]]
            if following:
                module.set_modules_to_forward_prefetch(following)

    unsharded = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not isinstance(parameter, DTensor)
    ]
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    if unsharded or frozen:
        raise RuntimeError(
            "FSDP2 SFT full-parameter closure is incomplete: "
            f"unsharded={unsharded[:8]}, frozen={frozen[:8]}"
        )
    return model, tuple(name for name, _ in wrapped)


def _numeric_extra_losses(values: tuple[Any, ...]) -> Tensor | float:
    total: Tensor | float = 0.0
    for value in values:
        if value is None or isinstance(value, dict):
            continue
        if isinstance(value, list):
            if value:
                total = total + sum(value)
            continue
        if isinstance(value, (int, float, Tensor)):
            total = total + value
            continue
        raise TypeError(f"unsupported SFT auxiliary loss: {type(value)!r}")
    return total


def _local_parameter_view(value: Tensor) -> Tensor:
    to_local = getattr(value, "to_local", None)
    local = to_local() if callable(to_local) else value
    if not isinstance(local, Tensor):
        raise TypeError("parameter shard is not a tensor")
    return local


def _clip_global_grad_norm(parameters: tuple[nn.Parameter, ...], max_norm: float) -> Tensor:
    local_sq = torch.zeros((), device=torch.cuda.current_device(), dtype=torch.float32)
    grads: list[Tensor] = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = _local_parameter_view(parameter.grad)
        local_sq.add_(grad.detach().float().square().sum())
        grads.append(parameter.grad)
    dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
    norm = local_sq.sqrt()
    scale = min(1.0, float(max_norm) / (float(norm) + 1e-6))
    if scale < 1.0:
        for grad in grads:
            grad.mul_(scale)
    return norm


def _distributed_max(value: float) -> float:
    tensor = torch.tensor(value, device=torch.cuda.current_device(), dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _distributed_sum(value: int) -> int:
    tensor = torch.tensor(value, device=torch.cuda.current_device(), dtype=torch.int64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def _parameter_moments(model: nn.Module) -> dict[str, float]:
    local = torch.zeros(3, device=torch.cuda.current_device(), dtype=torch.float64)
    for parameter in model.parameters():
        value = _local_parameter_view(parameter.detach()).double()
        local[0] += value.sum()
        local[1] += value.square().sum()
        local[2] += value.numel()
    dist.all_reduce(local, op=dist.ReduceOp.SUM)
    return {
        "sum": float(local[0].item()),
        "squared_l2": float(local[1].item()),
        "elements": int(local[2].item()),
    }


def _checkpoint_roundtrip(
    *,
    checkpoint: Path,
    model: nn.Module,
) -> tuple[float, float, int]:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
    )

    dist.barrier()
    start = time.perf_counter()
    dcp.save(_dcp_state(model), checkpoint_id=checkpoint)
    dist.barrier()
    save_seconds = _distributed_max(time.perf_counter() - start)

    loaded = _dcp_state(model)
    dist.barrier()
    start = time.perf_counter()
    dcp.load(loaded, checkpoint_id=checkpoint)
    set_model_state_dict(model, loaded["model"], options=StateDictOptions(full_state_dict=False, strict=True))
    dist.barrier()
    load_seconds = _distributed_max(time.perf_counter() - start)

    size = 0
    if dist.get_rank() == 0:
        size = sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())
    size_tensor = torch.tensor(size, device=torch.cuda.current_device(), dtype=torch.int64)
    dist.broadcast(size_tensor, src=0)
    dist.barrier()
    if dist.get_rank() == 0:
        shutil.rmtree(checkpoint)
    dist.barrier()
    return save_seconds, load_seconds, int(size_tensor.item())


def _dcp_state(model: nn.Module) -> dict[str, Any]:
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    options = StateDictOptions(full_state_dict=False, cpu_offload=False)
    return {
        "model": get_model_state_dict(model, options=options),
    }


def _save_training_checkpoint(
    *,
    root: Path,
    progress: SampleProgress,
    checkpoint_target_samples: int,
    model: nn.Module,
) -> Path:
    import torch.distributed.checkpoint as dcp

    started = time.perf_counter()
    rank = dist.get_rank()
    name = "final" if progress.done else f"samples-{progress.consumed_samples:012d}"
    target = root / name
    staging = root / f".{name}.staging"
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        if target.exists() or staging.exists():
            raise FileExistsError(f"refusing to overwrite SFT checkpoint {name}")
        staging.mkdir()
    dist.barrier()
    dcp.save(_dcp_state(model), checkpoint_id=staging / "dcp")
    dist.barrier()
    if rank == 0:
        metadata = {
            "schema": "sensenova.u15.forge.sft.checkpoint.v3",
            "trainer": "torch_fsdp2",
            **asdict(progress),
            "checkpoint_target_samples": checkpoint_target_samples,
            "world_size": dist.get_world_size(),
            "model_only": True,
            "batch_samples": int(gpc.config.data.batch_samples),
            "conversion_config": {
                "vit_cfg": {"num_hidden_layers": int(gpc.config.model.vit_cfg.num_hidden_layers)},
                "num_layers": int(gpc.config.model.num_layers),
                "moe_kwargs": {key: int(gpc.config.model.moe_kwargs.get(key, default))
                               for key, default in (("first_k_dense_replace", 0),
                                                    ("num_experts", 1), ("gen_num_experts", 1))},
            },
        }
        (staging / "checkpoint.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(staging, target)
    dist.barrier()
    seconds = _distributed_max(time.perf_counter() - started)
    if rank == 0:
        print(json.dumps({
            "component": "sensenova_u15.sft_fsdp2",
            "event": "checkpoint_saved",
            "consumed_samples": progress.consumed_samples,
            "checkpoint_target_samples": checkpoint_target_samples,
            "checkpoint": str(target),
            "seconds": seconds,
        }, sort_keys=True), flush=True)
    return target


def _prune_empty_hf_shards(target: Path) -> None:
    from safetensors import safe_open

    index_path = target / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("published HF checkpoint has no weight map")
    referenced = {str(value) for value in weight_map.values()}
    for shard in sorted(target.glob("*.safetensors")):
        if shard.name in referenced:
            continue
        with safe_open(shard, framework="pt", device="cpu") as handle:
            if list(handle.keys()):
                raise RuntimeError(f"unreferenced HF shard is not empty: {shard}")
        shard.unlink()


def _publish_hf_checkpoint(model: nn.Module, *, target: Path, base_model: Path) -> None:
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    marker = target.with_name(f"{target.name}.handoff.json")
    staging = target.with_name(f".{target.name}.staging")
    source = target.with_name(f".{target.name}.publish-source")
    error = None
    if dist.get_rank() == 0:
        occupied = [path for path in (target, marker, staging, source) if path.exists() or path.is_symlink()]
        if occupied:
            error = f"refusing to reuse SFT publication paths: {occupied}"
    errors = [error]
    dist.broadcast_object_list(errors, src=0)
    if errors[0] is not None:
        raise FileExistsError(str(errors[0]))
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    state = get_model_state_dict(model, options=options)
    dist.barrier()
    if dist.get_rank() == 0:
        source.mkdir(parents=True)
        try:
            torch.save(state, source / "model_wp0_pp0.pt")
            torch.save(dict(gpc.config.model), source / "model_config.pt")
            from tools.publish_hf import convert

            convert(
                src=str(source),
                tgt=str(staging),
                typ="neo++_mot",
                extras_from=str(base_model),
            )
            _prune_empty_hf_shards(staging)
            os.rename(staging, target)
            marker.write_text(
                json.dumps({"status": "ok", "target": str(target)}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except BaseException as exc:
            marker.write_text(
                json.dumps({"status": "failed", "detail": repr(exc)}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raise
        finally:
            shutil.rmtree(source, ignore_errors=True)
            if staging.exists():
                shutil.rmtree(staging)
    else:
        deadline = time.monotonic() + 21_600
        while True:
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                if payload.get("status") != "ok":
                    raise RuntimeError(f"rank-zero HF publication failed: {payload}")
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for HF publication marker: {marker}")
            time.sleep(1)


def _aggregate(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        raise ValueError("SFT benchmark has no measured updates after sample warmup")
    seconds = [float(record["seconds"]) for record in records]
    physical_tokens = sum(int(record["physical_tokens"]) for record in records)
    supervised_tokens = sum(int(record["supervised_tokens"]) for record in records)
    samples = sum(int(record["samples"]) for record in records)
    elapsed = sum(seconds)
    return {
        "optimizer_updates": len(records),
        "seconds": elapsed,
        "update_seconds_mean": statistics.fmean(seconds),
        "update_seconds_median": statistics.median(seconds),
        "update_seconds_p95": sorted(seconds)[math.ceil(0.95 * len(seconds)) - 1],
        "physical_tokens_per_second": physical_tokens / elapsed,
        "supervised_tokens_per_second": supervised_tokens / elapsed,
        "samples_per_second": samples / elapsed,
    }


def _forge_revision() -> str:
    root = Path(__file__).resolve().parent.parent
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main(args: Any, *, validation_callback=None, checkpoint_writer=_save_training_checkpoint) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if os.environ.get("SFT_RESUME_CHECKPOINT"):
        raise ValueError("SFT checkpoints contain model weights only; start a fresh run")
    gpu_name = _require_h200()
    if os.environ.get("SFT_SAMPLE_AUDIT_DIR"):
        hardware_root = Path(os.environ["SFT_SAMPLE_AUDIT_DIR"]).parent / "hardware"
        hardware_root.mkdir(parents=True, exist_ok=True)
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        with (hardware_root / f"rank-{rank:05d}.json").open("x", encoding="utf-8") as stream:
            json.dump({"hostname": socket.gethostname(), "rank": rank, "world_size": world_size,
                       "local_rank": torch.cuda.current_device(), "gpu": gpu_name,
                       "uuid": str(getattr(properties, "uuid", "")),
                       "memory_bytes": properties.total_memory,
                       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                       "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES")}, stream, indent=2)
    seed = int(args.seed)
    _seed_everything(seed)
    gpc.config.data.seed = seed
    obsolete = {"total_steps", "init_steps", "checkpoint_every", "metric_interval_steps",
                "grad_accm", "gradient_accumulation_steps", "packed_buffer_max_size", "packed_buffer_stale_threshold",
                "SFT_BENCHMARK_WARMUP_STEPS", "SFT_BENCHMARK_MEASURED_STEPS"} & os.environ.keys()
    if obsolete:
        raise ValueError(f"SFT accepts sample counts only; remove obsolete settings: {sorted(obsolete)}")
    progress = SampleProgress(int(gpc.config.data.samples_per_epoch), int(gpc.config.data.max_samples))
    train_dl, _dataset_types = build_train_loader_with_data_type()
    model = get_model(gpc.config.model, gpc.config.data).to(torch.cuda.current_device())
    # The checkpoint-specific model constructs parallel-aware linear modules
    # even when their process group has size one. Register the resulting no-op
    # MTP communicators before FSDP2 takes ownership of parameter sharding.
    initialize_unit_mtp_communicators(model)
    model, wrapped_modules = _shard_model(model)
    model.train()

    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(gpc.config.adam.lr),
        betas=(float(gpc.config.adam.adam_beta1), float(gpc.config.adam.adam_beta2)),
        eps=float(gpc.config.adam.adam_eps),
        weight_decay=float(gpc.config.adam.weight_decay),
        fused=_env_bool("FSDP2_FUSED_ADAMW", True),
    )
    criterion = FlashGPTLMLoss(
        parallel_output=gpc.config.model.parallel_output,
        label_smoothing=gpc.config.loss.label_smoothing,
        ce_loss_weight=float(gpc.config.get("ce_loss_weight", 1.0)),
    )
    benchmark_report = os.environ.get("SFT_BENCHMARK_REPORT")
    benchmark_only = _env_bool("SFT_BENCHMARK_ONLY", False)
    if benchmark_only and benchmark_report is None:
        raise ValueError("SFT_BENCHMARK_ONLY requires SFT_BENCHMARK_REPORT")
    if benchmark_only and os.environ.get("SFT_HF_OUTPUT"):
        raise ValueError("benchmark-only SFT cannot publish an HF checkpoint")
    benchmark_warmup_samples = _env_int("SFT_BENCHMARK_WARMUP_SAMPLES", 0)
    if benchmark_warmup_samples >= progress.max_samples:
        raise ValueError("benchmark warmup must leave measured samples")
    warmup_samples = int(gpc.config.lr_scheduler.warmup_samples)
    logging_samples = _env_int("logging_samples", 1, minimum=1)
    last_logged_samples = 0
    batch_samples_target = int(gpc.config.data.batch_samples)
    if batch_samples_target < world_size:
        raise ValueError("batch_samples must provide at least one sample per H200 rank")
    report_path = Path(benchmark_report).resolve() if benchmark_report else None
    if rank == 0 and report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if report_path.exists():
            raise FileExistsError(f"refusing to overwrite benchmark report: {report_path}")
    dist.barrier()

    initial_moments = _parameter_moments(model) if report_path is not None else None
    training_seconds = 0.0
    if validation_callback is not None:
        validation_callback(model, criterion, progress, training_seconds)
    torch.cuda.reset_peak_memory_stats()
    records: list[dict[str, Any]] = []
    checkpoint_root = Path(
        os.environ.get(
            "SFT_CHECKPOINT_ROOT",
            str(Path(os.environ.get("RUN_ROOT", "RUN")) / gpc.config.JOB_NAME / "checkpoints"),
        )
    ).expanduser().resolve()
    iterator = iter(train_dl)
    epoch = 0
    seen_sample_ids: set[str] = set()
    audit_root = os.environ.get("SFT_SAMPLE_AUDIT_DIR")
    audit_stream = None
    if audit_root:
        audit_path = Path(audit_root)
        audit_path.mkdir(parents=True, exist_ok=True)
        audit_stream = (audit_path / f"rank-{rank:05d}.jsonl").open("x", encoding="utf-8")
    launch_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    with training_profile(bool(args.profiling) or _env_bool("SFT_PROFILE_ENABLED", False), start_time=launch_time,
                          progress=progress, batch_samples=batch_samples_target) as profiler:
        while not progress.done:
            torch.cuda.synchronize()
            update_start = time.perf_counter()
            batch = next(iterator)
            if batch["sample_start"] != progress.consumed_samples:
                raise ValueError("SFT loader changed the global sample batch order")
            if batch["epoch"] != epoch:
                epoch = batch["epoch"]
                seen_sample_ids.clear()
            expected_samples = min(batch_samples_target,
                                   progress.max_samples - progress.consumed_samples,
                                   (epoch + 1) * progress.samples_per_epoch - progress.consumed_samples)
            if batch["sample_count"] != expected_samples:
                raise ValueError("SFT loader changed the global sample batch size")
            raw_microbatches = batch["microbatches"]
            sample_ids = [identity for data, _labels in raw_microbatches for identity in data["sample_ids"]]
            samples_local = sum(int(data["num_samples"]) for data, _labels in raw_microbatches)
            batch_samples = int(_distributed_sum(samples_local))
            if batch_samples != expected_samples or len(sample_ids) != samples_local:
                raise ValueError("SFT actual sample identities disagree with the global batch")
            if len(set(sample_ids)) != len(sample_ids) or seen_sample_ids.intersection(sample_ids):
                raise ValueError("SFT consumed a duplicate sample within one epoch")
            # All ranks execute the same number of FSDP forwards/backwards.
            # Short ranks pad with their cheapest physical sequence at zero loss.
            micro_steps = int(_distributed_max(len(raw_microbatches)))
            padding_batch = min(raw_microbatches, key=lambda item: item[0]["input_ids"].numel())
            data_seconds = time.perf_counter() - update_start
            progress.checkpoint_target(batch_samples)  # validate before mutating weights
            sample_denominator = batch_samples / world_size
            lr_ratio = progress.learning_rate_ratio(
                warmup_samples, gpc.config.lr_scheduler_type,
                float(gpc.config.lr_scheduler.eta_min),
            )
            for group in optimizer.param_groups:
                group["lr"] = float(gpc.config.adam.lr) * lr_ratio

            gpc.config.batch_count = progress.optimizer_updates
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            main_loss_value = 0.0
            auxiliary_loss_value = 0.0
            grad_norm = None
            physical_tokens_local = supervised_tokens_local = padding_tokens_local = 0
            for micro_step in range(micro_steps):
                is_last = micro_step + 1 == micro_steps
                sample_seed = int(np.random.SeedSequence([seed, progress.consumed_samples, rank, micro_step])
                                  .generate_state(1)[0])
                torch.manual_seed(sample_seed)
                torch.cuda.manual_seed_all(sample_seed)
                model.set_requires_gradient_sync(is_last, recurse=True)
                model.set_is_last_backward(is_last)
                real_microbatch = micro_step < len(raw_microbatches)
                raw = raw_microbatches[micro_step] if real_microbatch else padding_batch
                data, labels = move_to_device(raw)
                count = data.pop("num_samples") if real_microbatch else 0
                data.pop("num_samples", None)
                for name in ("sample_ids", "samples_per_microbatch", "worker_state_key_list",
                             "worker_state_dict_list", "worker_state_custom_infos_list"):
                    data.pop(name, None)
                if data.pop("is_empty_data_list", False):
                    raise ValueError("SFT received an invalid empty physical sequence")
                physical_tokens_local += data["input_ids"].numel()
                padding_tokens_local += data.pop("num_padding_tokens") if count else data["input_ids"].numel()
                data.pop("num_padding_tokens", None)
                if count:
                    supervised_tokens_local += int((labels != -100).sum().item())
                micro_data, micro_labels = _prepare_microbatch((data, labels), 0)
                micro_data["sample_loss_denominator"] = sample_denominator
                output, _mtp_outputs, *_extra = model(**micro_data)
                loss_weight = micro_data.pop("loss_weight", None)
                loss = criterion(
                    output,
                    micro_labels,
                    loss_weight=loss_weight,
                    sample_denominator=sample_denominator,
                )
                auxiliary = _numeric_extra_losses(tuple(_extra))
                # Both branches already divide by the entire update's sample
                # count. Padding microbatches participate in collectives only.
                if count == 0:
                    loss = loss * 0.0
                    auxiliary = auxiliary * 0.0
                total = loss + auxiliary
                total.backward()
                loss_value += float(total.detach().float().item())
                main_loss_value += float(loss.detach().float().item())
                if isinstance(auxiliary, Tensor):
                    auxiliary_loss_value += float(auxiliary.detach().float().item())
                else:
                    auxiliary_loss_value += float(auxiliary)
            grad_norm = _clip_global_grad_norm(parameters, float(gpc.config.hybrid_zero_optimizer.clip_grad_norm))
            optimizer.step()
            torch.cuda.synchronize()
            seconds = _distributed_max(time.perf_counter() - update_start)
            training_seconds += seconds
            loss_metrics = torch.tensor(
                [loss_value, main_loss_value, auxiliary_loss_value],
                device=torch.cuda.current_device(),
                dtype=torch.float64,
            )
            dist.all_reduce(loss_metrics, op=dist.ReduceOp.SUM)
            loss_metrics.div_(world_size)
            loss_value, main_loss_value, auxiliary_loss_value = (
                float(value) for value in loss_metrics.tolist()
            )
            before_samples = progress.consumed_samples
            checkpoint_target = progress.advance(batch_samples)
            profiler.step()
            seen_sample_ids.update(sample_ids)
            if audit_stream is not None:
                audit_stream.write(json.dumps({"epoch": epoch, "update": progress.optimizer_updates,
                                               "sample_ids": sample_ids}) + "\n")
                audit_stream.flush()
            record = {
                "consumed_samples": progress.consumed_samples,
                "max_samples": progress.max_samples,
                "epoch": progress.consumed_samples / progress.samples_per_epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": seconds,
                "loss": loss_value,
                "main_loss": main_loss_value,
                "auxiliary_loss": auxiliary_loss_value,
                "grad_norm": float(grad_norm.detach().float().item()),
                "physical_tokens": _distributed_sum(physical_tokens_local),
                "supervised_tokens": _distributed_sum(supervised_tokens_local),
                "samples": batch_samples,
                "batch_samples": batch_samples_target,
                "accumulation_microbatches": micro_steps,
                "padding_tokens": _distributed_sum(padding_tokens_local),
                "data_seconds": _distributed_max(data_seconds),
            }
            if report_path is not None and before_samples >= benchmark_warmup_samples:
                records.append(record)
            if rank == 0 and (progress.done or progress.consumed_samples - last_logged_samples >= logging_samples):
                last_logged_samples = progress.consumed_samples
                print(
                    json.dumps(
                        {
                            "component": "sensenova_u15.sft_fsdp2",
                            "event": "samples_complete",
                            "measured": before_samples >= benchmark_warmup_samples,
                            **record,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if not benchmark_only and checkpoint_target is not None:
                checkpoint_writer(
                    root=checkpoint_root,
                    progress=progress,
                    checkpoint_target_samples=checkpoint_target,
                    model=model,
                )
                if validation_callback is not None:
                    validation_callback(model, criterion, progress, training_seconds)

    peak_memory = _distributed_max(float(torch.cuda.max_memory_allocated()))
    final_moments = _parameter_moments(model) if report_path is not None else None
    checkpoint_result = None
    if audit_stream is not None:
        audit_stream.close()
    if _env_bool("SFT_BENCHMARK_CHECKPOINT", False):
        benchmark_parent = report_path.parent if report_path is not None else checkpoint_root
        checkpoint = benchmark_parent / "fsdp2-checkpoint.tmp"
        if rank == 0 and checkpoint.exists():
            raise FileExistsError(f"refusing to overwrite benchmark checkpoint: {checkpoint}")
        save_seconds, load_seconds, checkpoint_bytes = _checkpoint_roundtrip(
            checkpoint=checkpoint,
            model=model,
        )
        checkpoint_result = {
            "save_seconds": save_seconds,
            "load_seconds": load_seconds,
            "bytes": checkpoint_bytes,
        }

    hf_output = os.environ.get("SFT_HF_OUTPUT")
    if hf_output:
        _publish_hf_checkpoint(
            model,
            target=Path(hf_output).expanduser().resolve(),
            base_model=Path(gpc.config.model.model_name_or_path).expanduser().resolve(),
        )

    if rank == 0 and report_path is not None:
        report = {
            "schema": "sensenova.u15.forge.sft.run_report.v1",
            "trainer": "torch_fsdp2",
            "python": os.sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": gpu_name,
            "forge_revision": _forge_revision(),
            "world_size": world_size,
            "seed": seed,
            "job_name": gpc.config.JOB_NAME,
            "model_path": gpc.config.model.model_name_or_path,
            "data_meta": gpc.config.data.meta_path,
            "sequence_length": int(gpc.config.data.seq_len),
            "batch_samples": batch_samples_target,
            "activation_checkpoint_fraction": float(gpc.config.model.checkpoint),
            "bf16_compute": True,
            "bf16_gradient_reduction": True,
            "fp32_optimizer_master": True,
            "reshard_after_forward": _reshard_after_forward(),
            "prefetch_depth": _env_int("FSDP2_PREFETCH_DEPTH", 1),
            "fused_adamw": _env_bool("FSDP2_FUSED_ADAMW", True),
            "wrapped_modules": list(wrapped_modules),
            "warmup_samples": benchmark_warmup_samples,
            "progress": asdict(progress),
            "records": records,
            "aggregate": _aggregate(records),
            "peak_memory_bytes": int(peak_memory),
            "initial_parameter_moments": initial_moments,
            "final_parameter_moments": final_moments,
            "checkpoint": checkpoint_result,
            "benchmark_only": benchmark_only,
        }
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, report_path)
        print(json.dumps({"component": "sensenova_u15.sft_fsdp2", "event": "complete", **report["aggregate"]}, sort_keys=True), flush=True)
    dist.barrier()
    if rank == 0 and hf_output:
        output = Path(hf_output).expanduser().resolve()
        output.with_name(f"{output.name}.handoff.json").unlink()


if __name__ == "__main__":
    args = parse_args()
    initialize_distributed_env(config=args.config, launcher=args.launcher, master_port=args.port, seed=args.seed)
    init_pil()
    main(args)
