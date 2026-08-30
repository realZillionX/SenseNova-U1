#!/usr/bin/env python
"""Optimized Torch 2.8 FSDP2 SFT runner for controlled trainer ablations.

This entry reuses the exact InternEvo U1.5 model, native-resolution packed
loader, forward, losses, and checkpoint source weights.  Only the trainer and
parallel execution strategy change.  It intentionally lives beside the
InternEvo entry so performance comparisons cannot drift to a second model or
data implementation.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn

from sensenovalm.core.context import global_context as gpc
from sensenovalm.data.utils import packed_data_normalizer
from sensenovalm.initialize import initialize_distributed_env
from sensenovalm.model.losses.ce_loss import FlashGPTLMLoss
from sensenovalm.utils.common import move_to_device, parse_args
from sensenovavl.data import build_train_loader_with_data_type
from sensenovavl.train.pipeline import get_model, initialize_llm_profile
from sensenovavl.utils.utils import check_image_fn, init_pil


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
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.fsdp import FSDPModule
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
    reshard_after_forward = _env_bool("FSDP2_RESHARD_AFTER_FORWARD", True)
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


def main(args: Any) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    seed = int(args.seed)
    _seed_everything(seed)
    train_dl, _dataset_types = build_train_loader_with_data_type()
    model = get_model(gpc.config.model, gpc.config.data).to(torch.cuda.current_device())
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

    warmup_steps = _env_int("SFT_BENCHMARK_WARMUP_STEPS", 3)
    measured_steps = _env_int("SFT_BENCHMARK_MEASURED_STEPS", 10, minimum=1)
    total_steps = warmup_steps + measured_steps
    if int(gpc.config.data.total_steps) != total_steps:
        raise ValueError(
            "total_steps must equal SFT_BENCHMARK_WARMUP_STEPS + "
            "SFT_BENCHMARK_MEASURED_STEPS"
        )
    grad_accumulation = int(gpc.config.data.micro_num)
    if grad_accumulation < 1:
        raise ValueError("FSDP2 SFT gradient accumulation must be positive")
    report_path = Path(os.environ["SFT_BENCHMARK_REPORT"]).resolve()
    if rank == 0:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if report_path.exists():
            raise FileExistsError(f"refusing to overwrite benchmark report: {report_path}")
    dist.barrier()

    initial_moments = _parameter_moments(model)
    torch.cuda.reset_peak_memory_stats()
    records: list[dict[str, Any]] = []
    iterator = iter(train_dl)
    launch_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    with initialize_llm_profile(profiling=bool(args.profiling), start_time=launch_time) as profiler:
        for step in range(total_steps):
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
                raise RuntimeError("benchmark input exhausted and produced an empty packed row")
            physical_tokens_local = int(data["input_ids"].numel())
            supervised_tokens_local = int((labels != -100).sum().item()) if isinstance(labels, Tensor) else 0
            samples_local = int(data.pop("num_samples", 0))
            data.pop("num_padding_tokens", None)
            if not samples_local:
                samples_local = sum(len(item) - 1 for item in data["cu_seqlens"])

            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            grad_norm = None
            for micro_step in range(grad_accumulation):
                is_last = micro_step + 1 == grad_accumulation
                model.set_requires_gradient_sync(is_last, recurse=True)
                model.set_is_last_backward(is_last)
                micro_data, micro_labels = _prepare_microbatch((data, labels), micro_step)
                output, _mtp_outputs, *_extra = model(**micro_data)
                loss_weight = micro_data.pop("loss_weight", None)
                loss_reduction_all_gather = micro_data.pop("loss_reduction_all_gather", False)
                loss = criterion(
                    output,
                    micro_labels,
                    loss_weight=loss_weight,
                    loss_reduction_all_gather=loss_reduction_all_gather,
                )
                total = (loss + _numeric_extra_losses(tuple(_extra))) / grad_accumulation
                total.backward()
                loss_value += float(total.detach().float().item())
            grad_norm = _clip_global_grad_norm(parameters, float(gpc.config.hybrid_zero_optimizer.clip_grad_norm))
            optimizer.step()
            if ema is not None:
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        current = _local_parameter_view(parameter.detach()).to(dtype=torch.bfloat16)
                        ema[name].lerp_(current, 1.0 - ema_decay)
            torch.cuda.synchronize()
            seconds = _distributed_max(time.perf_counter() - step_start)
            profiler.step()
            record = {
                "step": step + 1,
                "seconds": seconds,
                "loss": loss_value,
                "grad_norm": float(grad_norm.detach().float().item()),
                "physical_tokens": _distributed_sum(physical_tokens_local),
                "supervised_tokens": _distributed_sum(supervised_tokens_local),
                "samples": _distributed_sum(samples_local),
            }
            if step >= warmup_steps:
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

    peak_memory = _distributed_max(float(torch.cuda.max_memory_allocated()))
    final_moments = _parameter_moments(model)
    checkpoint_result = None
    if _env_bool("SFT_BENCHMARK_CHECKPOINT", False):
        checkpoint = report_path.parent / "fsdp2-checkpoint.tmp"
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

    if rank == 0:
        report = {
            "schema": "sensenova_u15.sft_trainer_ablation.v1",
            "trainer": "torch_fsdp2",
            "python": os.sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
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
            "reshard_after_forward": _env_bool("FSDP2_RESHARD_AFTER_FORWARD", True),
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
        }
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, report_path)
        print(json.dumps({"component": "sensenova_u15.sft_fsdp2", "event": "complete", **report["aggregate"]}, sort_keys=True), flush=True)
    dist.barrier()


if __name__ == "__main__":
    args = parse_args()
    initialize_distributed_env(config=args.config, launcher=args.launcher, master_port=args.port, seed=args.seed)
    init_pil()
    main(args)
