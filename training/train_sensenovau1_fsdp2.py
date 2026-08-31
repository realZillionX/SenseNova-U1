#!/usr/bin/env python
"""Production Torch 2.8 FSDP2 SFT runner for SenseNova-U1.5-8B-MoT.

The model, native-resolution packed loader, forward, and losses remain the
checkpoint-specific U1.5 implementations. FSDP2 is the sole trainer and owns
full-parameter sharding, optimizer state, EMA, checkpointing, and publication.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from sensenovalm.core.context import global_context as gpc
from sensenovalm.data.utils import packed_data_normalizer
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
    optimizer: torch.optim.Optimizer,
    ema: dict[str, Tensor] | None,
) -> tuple[float, float, int]:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
    )

    options = StateDictOptions(full_state_dict=False, cpu_offload=False)

    def state() -> dict[str, Any]:
        result: dict[str, Any] = {
            "model": get_model_state_dict(model, options=options),
            "optimizer": get_optimizer_state_dict(model, optimizer, options=options),
        }
        if ema is not None:
            result["ema"] = ema
        return result

    dist.barrier()
    start = time.perf_counter()
    dcp.save(state(), checkpoint_id=checkpoint)
    dist.barrier()
    save_seconds = _distributed_max(time.perf_counter() - start)

    loaded = state()
    dist.barrier()
    start = time.perf_counter()
    dcp.load(loaded, checkpoint_id=checkpoint)
    set_model_state_dict(model, loaded["model"], options=StateDictOptions(full_state_dict=False, strict=True))
    set_optimizer_state_dict(
        model,
        optimizer,
        optim_state_dict=loaded["optimizer"],
        options=StateDictOptions(full_state_dict=False, strict=True),
    )
    if ema is not None:
        for name, value in loaded["ema"].items():
            ema[name].copy_(value)
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


def _dcp_state(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
    )

    options = StateDictOptions(full_state_dict=False, cpu_offload=False)
    return {
        "model": get_model_state_dict(model, options=options),
        "optimizer": get_optimizer_state_dict(model, optimizer, options=options),
    }


def _rng_payload() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
    }


def _restore_rng(payload: dict[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    torch.cuda.set_rng_state(payload["torch_cuda"])


def _save_training_checkpoint(
    *,
    root: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: dict[str, Tensor] | None,
) -> Path:
    import torch.distributed.checkpoint as dcp

    rank = dist.get_rank()
    target = root / f"step-{step:08d}"
    staging = root / f".step-{step:08d}.staging"
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        if target.exists() or staging.exists():
            raise FileExistsError(f"refusing to overwrite SFT checkpoint step {step}")
        staging.mkdir()
    dist.barrier()
    dcp.save(_dcp_state(model, optimizer), checkpoint_id=staging / "dcp")
    torch.save(_rng_payload(), staging / f"rank-{rank:05d}-rng.pt")
    if ema is not None:
        torch.save(ema, staging / f"rank-{rank:05d}-ema.pt")
    dist.barrier()
    if rank == 0:
        metadata = {
            "schema": "sensenova.u15.forge.sft.checkpoint.v1",
            "trainer": "torch_fsdp2",
            "step": step,
            "world_size": dist.get_world_size(),
            "ema": ema is not None,
        }
        (staging / "checkpoint.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(staging, target)
    dist.barrier()
    return target


def _load_training_checkpoint(
    *,
    checkpoint: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: dict[str, Tensor] | None,
) -> tuple[int, dict[str, Any]]:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
        set_optimizer_state_dict,
    )

    metadata = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != "sensenova.u15.forge.sft.checkpoint.v1":
        raise ValueError("unsupported SFT checkpoint schema")
    if metadata.get("trainer") != "torch_fsdp2" or int(metadata.get("world_size", 0)) != dist.get_world_size():
        raise ValueError("SFT checkpoint topology differs from the current FSDP2 run")
    if bool(metadata.get("ema")) != (ema is not None):
        raise ValueError("SFT checkpoint EMA contract differs from the current run")
    state = _dcp_state(model, optimizer)
    dcp.load(state, checkpoint_id=checkpoint / "dcp")
    options = StateDictOptions(full_state_dict=False, cpu_offload=False, strict=True)
    set_model_state_dict(model, state["model"], options=options)
    set_optimizer_state_dict(
        model,
        optimizer,
        optim_state_dict=state["optimizer"],
        options=options,
    )
    rank = dist.get_rank()
    if ema is not None:
        loaded_ema = torch.load(
            checkpoint / f"rank-{rank:05d}-ema.pt",
            map_location=torch.device("cuda", torch.cuda.current_device()),
            weights_only=False,
        )
        if set(loaded_ema) != set(ema):
            raise ValueError("SFT checkpoint EMA parameter closure differs from the model")
        for name, value in loaded_ema.items():
            ema[name].copy_(value)
    rng = torch.load(
        checkpoint / f"rank-{rank:05d}-rng.pt", map_location="cpu", weights_only=False
    )
    return int(metadata["step"]), rng


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
            time.sleep(10)


def _aggregate(records: list[dict[str, Any]]) -> dict[str, float]:
    seconds = [float(record["seconds"]) for record in records]
    physical_tokens = sum(int(record["physical_tokens"]) for record in records)
    supervised_tokens = sum(int(record["supervised_tokens"]) for record in records)
    samples = sum(int(record["samples"]) for record in records)
    elapsed = sum(seconds)
    return {
        "steps": len(records),
        "seconds": elapsed,
        "step_seconds_mean": statistics.fmean(seconds),
        "step_seconds_median": statistics.median(seconds),
        "step_seconds_p95": sorted(seconds)[max(0, int(0.95 * len(seconds)) - 1)],
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


def main(args: Any) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    gpu_name = _require_h200()
    seed = int(args.seed)
    _seed_everything(seed)
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
    ema_decay = float(gpc.config.averaged_model.decay)
    ema = None
    if bool(gpc.config.averaged_model.enable):
        ema = {
            name: _local_parameter_view(parameter.detach()).to(dtype=torch.bfloat16).clone()
            for name, parameter in model.named_parameters()
        }

    total_steps = int(gpc.config.data.total_steps)
    benchmark_report = os.environ.get("SFT_BENCHMARK_REPORT")
    benchmark_only = _env_bool("SFT_BENCHMARK_ONLY", False)
    if benchmark_only and benchmark_report is None:
        raise ValueError("SFT_BENCHMARK_ONLY requires SFT_BENCHMARK_REPORT")
    if benchmark_only and os.environ.get("SFT_RESUME_CHECKPOINT"):
        raise ValueError("benchmark-only SFT cannot resume a production checkpoint")
    if benchmark_only and os.environ.get("SFT_HF_OUTPUT"):
        raise ValueError("benchmark-only SFT cannot publish an HF checkpoint")
    warmup_steps = _env_int("SFT_BENCHMARK_WARMUP_STEPS", 0 if benchmark_report is None else 3)
    measured_steps = _env_int(
        "SFT_BENCHMARK_MEASURED_STEPS",
        total_steps - warmup_steps,
        minimum=1,
    )
    if benchmark_report is not None and total_steps != warmup_steps + measured_steps:
        raise ValueError("benchmark total_steps must equal warmup plus measured steps")
    grad_accumulation = int(gpc.config.data.micro_num)
    if grad_accumulation < 1:
        raise ValueError("FSDP2 SFT gradient accumulation must be positive")
    report_path = Path(benchmark_report).resolve() if benchmark_report else None
    if rank == 0 and report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if report_path.exists():
            raise FileExistsError(f"refusing to overwrite benchmark report: {report_path}")
    dist.barrier()

    initial_moments = _parameter_moments(model)
    torch.cuda.reset_peak_memory_stats()
    records: list[dict[str, Any]] = []
    checkpoint_root = Path(
        os.environ.get(
            "SFT_CHECKPOINT_ROOT",
            str(Path(os.environ.get("RUN_ROOT", "RUN")) / gpc.config.JOB_NAME / "checkpoints"),
        )
    ).expanduser().resolve()
    resume_raw = os.environ.get("SFT_RESUME_CHECKPOINT")
    start_step = 0
    resume_rng = None
    if resume_raw:
        start_step, resume_rng = _load_training_checkpoint(
            checkpoint=Path(resume_raw).expanduser().resolve(),
            model=model,
            optimizer=optimizer,
            ema=ema,
        )
        if not 0 <= start_step < total_steps:
            raise ValueError("SFT resume step is outside the current plan")
    iterator = iter(train_dl)
    for _ in range(start_step):
        try:
            next(iterator)
        except StopIteration:
            iterator = iter(train_dl)
            next(iterator)
    if resume_rng is not None:
        _restore_rng(resume_rng)
    launch_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    with training_profile(bool(args.profiling), start_time=launch_time) as profiler:
        for step in range(start_step, total_steps):
            torch.cuda.synchronize()
            step_start = time.perf_counter()
            try:
                raw_batch = next(iterator)
            except StopIteration:
                iterator = iter(train_dl)
                raw_batch = next(iterator)
            raw_batch = move_to_device(raw_batch)
            data, labels = raw_batch
            data.pop("worker_state_key_list", None)
            data.pop("worker_state_dict_list", None)
            data.pop("worker_state_custom_infos_list", None)
            if data.pop("is_empty_data_list", False):
                raise RuntimeError("SFT input exhausted and produced an empty packed row")
            physical_tokens_local = int(data["input_ids"].numel())
            supervised_tokens_local = int((labels != -100).sum().item()) if isinstance(labels, Tensor) else 0
            samples_local = int(data.pop("num_samples", 0))
            data.pop("num_padding_tokens", None)
            if not samples_local:
                samples_local = sum(len(item) - 1 for item in data["cu_seqlens"])

            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            main_loss_value = 0.0
            auxiliary_loss_value = 0.0
            grad_norm = None
            for micro_step in range(grad_accumulation):
                is_last = micro_step + 1 == grad_accumulation
                global_position = step * world_size * grad_accumulation + rank * grad_accumulation + micro_step
                sample_seed = seed + global_position
                torch.manual_seed(sample_seed)
                torch.cuda.manual_seed_all(sample_seed)
                model.set_requires_gradient_sync(is_last, recurse=True)
                model.set_is_last_backward(is_last)
                micro_data, micro_labels = _prepare_microbatch((data, labels), micro_step)
                output, _mtp_outputs, *_extra = model(**micro_data)
                loss_weight = micro_data.pop("loss_weight", None)
                loss_reduction_all_gather = micro_data.pop("loss_reduction_all_gather", False)
                if _env_bool("SFT_PER_RANK_LOSS_REDUCTION", True):
                    # Each data rank owns one packed row, so it must keep its
                    # own denominator before FSDP averages gradients. A global
                    # denominator would change the objective to a token-weighted
                    # mean and let variable packing alter sample weights.
                    loss_reduction_all_gather = False
                loss = criterion(
                    output,
                    micro_labels,
                    loss_weight=loss_weight,
                    loss_reduction_all_gather=loss_reduction_all_gather,
                )
                auxiliary = _numeric_extra_losses(tuple(_extra))
                total = (loss + auxiliary) / grad_accumulation
                total.backward()
                loss_value += float(total.detach().float().item())
                main_loss_value += float((loss / grad_accumulation).detach().float().item())
                if isinstance(auxiliary, Tensor):
                    auxiliary_loss_value += float((auxiliary / grad_accumulation).detach().float().item())
                else:
                    auxiliary_loss_value += float(auxiliary) / grad_accumulation
            grad_norm = _clip_global_grad_norm(parameters, float(gpc.config.hybrid_zero_optimizer.clip_grad_norm))
            optimizer.step()
            if ema is not None:
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        current = _local_parameter_view(parameter.detach()).to(dtype=torch.bfloat16)
                        ema[name].lerp_(current, 1.0 - ema_decay)
            torch.cuda.synchronize()
            seconds = _distributed_max(time.perf_counter() - step_start)
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
            profiler.step()
            record = {
                "step": step + 1,
                "seconds": seconds,
                "loss": loss_value,
                "main_loss": main_loss_value,
                "auxiliary_loss": auxiliary_loss_value,
                "grad_norm": float(grad_norm.detach().float().item()),
                "physical_tokens": _distributed_sum(physical_tokens_local),
                "supervised_tokens": _distributed_sum(supervised_tokens_local),
                "samples": _distributed_sum(samples_local),
            }
            if report_path is not None and step >= warmup_steps:
                records.append(record)
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "component": "sensenova_u15.sft_fsdp2",
                            "event": "step_complete",
                            "measured": step >= warmup_steps,
                            **record,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            completed_step = step + 1
            checkpoint_every = _env_int("checkpoint_every", 100, minimum=1)
            if not benchmark_only and (
                completed_step % checkpoint_every == 0
                or completed_step == total_steps
            ):
                _save_training_checkpoint(
                    root=checkpoint_root,
                    step=completed_step,
                    model=model,
                    optimizer=optimizer,
                    ema=ema,
                )

    peak_memory = _distributed_max(float(torch.cuda.max_memory_allocated()))
    final_moments = _parameter_moments(model)
    checkpoint_result = None
    if _env_bool("SFT_BENCHMARK_CHECKPOINT", False):
        benchmark_parent = report_path.parent if report_path is not None else checkpoint_root
        checkpoint = benchmark_parent / "fsdp2-checkpoint.tmp"
        if rank == 0 and checkpoint.exists():
            raise FileExistsError(f"refusing to overwrite benchmark checkpoint: {checkpoint}")
        save_seconds, load_seconds, checkpoint_bytes = _checkpoint_roundtrip(
            checkpoint=checkpoint,
            model=model,
            optimizer=optimizer,
            ema=ema,
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
            "gradient_accumulation": grad_accumulation,
            "activation_checkpoint_fraction": float(gpc.config.model.checkpoint),
            "bf16_compute": True,
            "bf16_gradient_reduction": True,
            "fp32_optimizer_master": True,
            "reshard_after_forward": _reshard_after_forward(),
            "prefetch_depth": _env_int("FSDP2_PREFETCH_DEPTH", 1),
            "fused_adamw": _env_bool("FSDP2_FUSED_ADAMW", True),
            "wrapped_modules": list(wrapped_modules),
            "warmup_steps": warmup_steps,
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
