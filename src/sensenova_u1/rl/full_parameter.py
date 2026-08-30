"""Full-parameter FSDP2 state for native SenseNova-U1.5 RLVR.

The U1.5 runtime calls the language model, vision model and flow
modules directly instead of routing every operation through
``NEOChatModel.forward``.  A single root FSDP wrapper would therefore be
incorrect: several trainable parameters would be used without entering the
root all-gather hooks.  This module shards every decoder/vision block and then
wraps each remaining parameter-owning execution root that the U1.5 runtime
actually calls.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn

BLOCK_CLASS_NAMES = frozenset({"NEOVisionModel", "Qwen3DecoderLayer", "Qwen3MoeDecoderLayer"})
MLP_CLASS_NAMES = frozenset({"Qwen3MLP", "Qwen3MoeMLP"})
PROCESS_GROUP_TIMEOUT_ENV = "SENSENOVA_FORGE_FSDP_TIMEOUT_SECONDS"
DEFAULT_PROCESS_GROUP_TIMEOUT_SECONDS = 600
_ACTIVATION_CHECKPOINT_MARKER = "_forge_activation_checkpointed"


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class _CpuOptimizerTensor:
    local: Tensor
    device_mesh: Any | None = None
    placements: tuple[Any, ...] | None = None
    global_shape: tuple[int, ...] | None = None
    global_stride: tuple[int, ...] | None = None


def offload_optimizer_state(optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    """Move only initialized Adam-style state tensors to rank-local CPU memory."""

    from torch.distributed.tensor import DTensor

    moved_bytes = 0
    moved_tensors = 0
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if isinstance(value, _CpuOptimizerTensor):
                raise RuntimeError("optimizer state is already CPU-offloaded")
            if isinstance(value, DTensor):
                local = value.to_local().detach().cpu()
                state[key] = _CpuOptimizerTensor(
                    local=local,
                    device_mesh=value.device_mesh,
                    placements=tuple(value.placements),
                    global_shape=tuple(value.shape),
                    global_stride=tuple(value.stride()),
                )
            elif isinstance(value, Tensor) and value.device.type == "cuda":
                local = value.detach().cpu()
                state[key] = _CpuOptimizerTensor(local=local)
            else:
                continue
            moved_bytes += local.numel() * local.element_size()
            moved_tensors += 1
    return moved_bytes, moved_tensors


def restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> tuple[int, int]:
    """Restore a rank-local CPU optimizer snapshot to its original Tensor ABI."""

    from torch.distributed.tensor import DTensor

    moved_bytes = 0
    moved_tensors = 0
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if not isinstance(value, _CpuOptimizerTensor):
                continue
            local = value.local.to(device=device)
            if value.device_mesh is None:
                state[key] = local
            else:
                if (
                    value.placements is None
                    or value.global_shape is None
                    or value.global_stride is None
                ):
                    raise RuntimeError("CPU-offloaded DTensor optimizer metadata is incomplete")
                state[key] = DTensor.from_local(
                    local,
                    device_mesh=value.device_mesh,
                    placements=value.placements,
                    run_check=False,
                    shape=value.global_shape,
                    stride=value.global_stride,
                )
            moved_bytes += local.numel() * local.element_size()
            moved_tensors += 1
    return moved_bytes, moved_tensors


def initialize_distributed() -> DistributedContext:
    """Initialize the one-rank-per-GPU NCCL process group used by FSDP."""

    if not torch.cuda.is_available():
        raise ValueError("SenseNova full-parameter RLVR requires CUDA")
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise ValueError(f"SenseNova full-parameter RLVR must be launched by torchrun; missing {missing}")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise ValueError("SenseNova full-parameter RLVR requires at least two ranks")
    if not 0 <= local_rank < torch.cuda.device_count():
        raise ValueError("LOCAL_RANK is outside the visible CUDA device set")
    raw_timeout = os.environ.get(PROCESS_GROUP_TIMEOUT_ENV, str(DEFAULT_PROCESS_GROUP_TIMEOUT_SECONDS))
    try:
        timeout_seconds = int(raw_timeout)
    except ValueError as exc:
        raise ValueError(f"{PROCESS_GROUP_TIMEOUT_ENV} must be an integer number of seconds") from exc
    if timeout_seconds < 1:
        raise ValueError(f"{PROCESS_GROUP_TIMEOUT_ENV} must be positive")
    torch.cuda.set_device(local_rank)
    gpu_name = torch.cuda.get_device_name(local_rank)
    if "H200" not in gpu_name.upper():
        raise ValueError(
            f"SenseNova full-parameter training supports only NVIDIA H200, found {gpu_name!r}"
        )
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout_seconds))
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise RuntimeError("torchrun environment disagrees with the process group")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
    )


def _floating_parameters(parameters: Iterable[nn.Parameter]) -> Iterator[nn.Parameter]:
    for parameter in parameters:
        if parameter.dtype.is_floating_point:
            yield parameter


def _cast_direct_parameters(module: nn.Module, dtype: torch.dtype) -> None:
    from torch.distributed.tensor import DTensor

    for parameter in _floating_parameters(module.parameters(recurse=False)):
        if not isinstance(parameter, DTensor) and parameter.dtype != dtype:
            parameter.data = parameter.data.to(dtype=dtype)


def _contains_unsharded_direct_parameter(module: nn.Module) -> bool:
    from torch.distributed.tensor import DTensor

    return any(not isinstance(parameter, DTensor) for parameter in module.parameters(recurse=False))


def shard_full_parameter_policy(
    model: nn.Module,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
    master_dtype: torch.dtype = torch.float32,
    activation_checkpointing: bool = False,
) -> tuple[str, ...]:
    """Shard the complete U1.5 parameter closure with FSDP2.

    Parameters remain FP32 optimizer masters.  FSDP casts gathered parameters
    to ``compute_dtype`` for forward and reduces gradients in FP32.  No adapter,
    frozen submodel or replicated trainable parameter is permitted.
    """

    if type(activation_checkpointing) is not bool:
        raise TypeError("activation_checkpointing must be a boolean")
    if not dist.is_initialized() or dist.get_world_size() < 2:
        raise ValueError("full-parameter sharding needs an initialized multi-rank group")
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import DTensor

    model.requires_grad_(True)
    if activation_checkpointing:
        from torch.utils.checkpoint import checkpoint

        checkpointed = 0
        for module in model.modules():
            if type(module).__name__ not in MLP_CLASS_NAMES:
                continue
            if getattr(module, _ACTIVATION_CHECKPOINT_MARKER, False):
                raise RuntimeError("U1.5 MLP activation checkpointing was applied twice")
            forward = module.forward

            def checkpointed_forward(
                *args: Any,
                _forward: Any = forward,
                **kwargs: Any,
            ) -> Any:
                if not torch.is_grad_enabled():
                    return _forward(*args, **kwargs)
                return checkpoint(
                    _forward,
                    *args,
                    use_reentrant=False,
                    preserve_rng_state=False,
                    **kwargs,
                )

            module.forward = checkpointed_forward  # type: ignore[method-assign]
            setattr(module, _ACTIVATION_CHECKPOINT_MARKER, True)
            checkpointed += 1
        if not checkpointed:
            raise ValueError("published U1.5 policy exposes no checkpointable MLPs")
    policy = MixedPrecisionPolicy(
        param_dtype=compute_dtype,
        reduce_dtype=torch.float32,
    )
    kwargs: dict[str, Any] = {
        "mp_policy": policy,
        "reshard_after_forward": True,
    }

    wrapped: list[str] = []
    named_modules = tuple(model.named_modules())
    blocks = [(name, module) for name, module in named_modules if type(module).__name__ in BLOCK_CLASS_NAMES]
    if not blocks:
        raise ValueError("published U1.5 policy exposes no recognized FSDP blocks")
    for name, module in blocks:
        for parameter in _floating_parameters(module.parameters()):
            if not isinstance(parameter, DTensor) and parameter.dtype != master_dtype:
                parameter.data = parameter.data.to(dtype=master_dtype)
        fully_shard(module, **kwargs)
        wrapped.append(name)

    # The image-velocity helper calls ``language_model.model``
    # directly, so that backbone is an execution root in its own right. Wrap it
    # after its decoder blocks and keep its remaining embedding/final-norm
    # parameters gathered through backward. This also keeps direct embedding
    # lookups valid until the following backbone forward registers its FSDP
    # backward hooks.
    language_model = getattr(model, "language_model", None)
    if not isinstance(language_model, nn.Module):
        raise TypeError("published U1.5 policy has no language_model module")
    language_backbone = getattr(language_model, "model", None)
    if not isinstance(language_backbone, nn.Module):
        raise TypeError("published U1.5 policy has no language-model backbone")
    for parameter in _floating_parameters(language_backbone.parameters()):
        if not isinstance(parameter, DTensor) and parameter.dtype != master_dtype:
            parameter.data = parameter.data.to(dtype=master_dtype)
    fully_shard(
        language_backbone,
        mp_policy=policy,
        reshard_after_forward=False,
    )
    wrapped.append("language_model.model")

    # The outer causal-LM root now owns only parameters such as lm_head that
    # its public forward actually executes.
    for parameter in _floating_parameters(language_model.parameters()):
        if not isinstance(parameter, DTensor) and parameter.dtype != master_dtype:
            parameter.data = parameter.data.to(dtype=master_dtype)
    fully_shard(language_model, mp_policy=policy, reshard_after_forward=True)
    wrapped.append("language_model")

    # Flow heads and any remaining U1.5 execution leaves are invoked
    # directly. Work bottom-up so every unclaimed direct parameter receives
    # hooks at the smallest callable module that owns it.
    for name, module in reversed(named_modules):
        if module is model or module is language_model:
            continue
        if not _contains_unsharded_direct_parameter(module):
            continue
        _cast_direct_parameters(module, master_dtype)
        fully_shard(module, **kwargs)
        wrapped.append(name)

    unsharded = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not isinstance(parameter, DTensor)
    ]
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    if unsharded or frozen:
        raise RuntimeError(
            f"SenseNova full-parameter closure is incomplete: unsharded={unsharded[:8]}, frozen={frozen[:8]}"
        )
    return tuple(wrapped)


def local_parameter_view(parameter: Tensor) -> Tensor:
    """Return the local tensor shard without materializing a full parameter."""

    to_local = getattr(parameter, "to_local", None)
    value = to_local() if callable(to_local) else parameter
    if not isinstance(value, Tensor):
        raise TypeError("parameter shard is not a tensor")
    return value


def reshard_full_parameter_policy(model: nn.Module) -> None:
    """Return the persistent language root to its local-shard representation.

    ``language_model.model`` is the sole root configured with
    ``reshard_after_forward=False``. It intentionally keeps gathered
    parameters live across the current-policy forward/backward boundary.
    Reference swapping, however, owns BF16 *local-shard* snapshots, so copying
    before explicitly resharding would compare a full gathered tensor with one
    local shard. Other nested roots already auto-reshard and must not be
    disturbed outside their own hook lifecycle.
    """

    from torch.distributed.fsdp import FSDPModule

    language_model = getattr(model, "language_model", None)
    language_backbone = getattr(language_model, "model", None)
    if not isinstance(language_backbone, FSDPModule):
        raise ValueError("SenseNova full-parameter policy has no FSDP2 language backbone")
    language_backbone.reshard()


def snapshot_reference_shards(model: nn.Module) -> dict[str, Tensor]:
    """Capture the fixed SFT reference as one BF16 local shard per parameter."""

    result: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        local = local_parameter_view(parameter)
        result[name] = local.detach().to(dtype=torch.bfloat16).clone()
    if len(result) != len(tuple(model.named_parameters())):
        raise RuntimeError("SenseNova reference snapshot lost parameter aliases")
    return result


def full_parameter_groups(
    model: nn.Module,
    *,
    text_learning_rate: float,
    visual_learning_rate: float,
    weight_decay: float,
) -> list[dict[str, object]]:
    """Build explicit text/shared and visual-generation optimizer groups."""

    visual_markers = (
        "fm_modules.",
        "_mot_gen.",
        ".mot_gen.",
        "norm_mot_gen.",
        "o_proj_mot_gen.",
        "q_proj_mot_gen.",
        "k_proj_mot_gen.",
        "v_proj_mot_gen.",
    )
    text: list[nn.Parameter] = []
    visual: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        (visual if any(marker in name for marker in visual_markers) else text).append(parameter)
    if not text or not visual:
        raise ValueError("SenseNova full-parameter optimizer groups are incomplete")
    return [
        {
            "params": text,
            "lr": float(text_learning_rate),
            "weight_decay": float(weight_decay),
            "name": "text_shared",
        },
        {
            "params": visual,
            "lr": float(visual_learning_rate),
            "weight_decay": float(weight_decay),
            "name": "visual_generation",
        },
    ]


def clip_global_grad_norm(parameters: Iterable[nn.Parameter], max_norm: float) -> Tensor:
    """Clip one global norm over every FSDP shard and both policy branches."""

    params = tuple(parameters)
    try:
        value = torch.nn.utils.clip_grad_norm_(params, max_norm)
        to_local = getattr(value, "to_local", None)
        return to_local() if callable(to_local) else value
    except RuntimeError as exc:
        if "DTensor" not in str(exc):
            raise
    local_sq = torch.zeros(
        (),
        device=torch.device("cuda", torch.cuda.current_device()),
        dtype=torch.float32,
    )
    grads: list[Tensor] = []
    for parameter in params:
        if parameter.grad is None:
            continue
        grad = local_parameter_view(parameter.grad)
        local_sq.add_(grad.detach().float().square().sum())
        grads.append(parameter.grad)
    dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
    norm = local_sq.sqrt()
    scale = min(1.0, float(max_norm) / (float(norm) + 1e-6))
    if scale < 1.0:
        for grad in grads:
            grad.mul_(scale)
    return norm


def dcp_state(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, object]:
    """Return per-rank sharded model and optimizer state for DCP."""

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


def save_dcp(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Collectively save sharded full-model and optimizer state."""

    import torch.distributed.checkpoint as dcp

    dcp.save(dcp_state(model, optimizer), checkpoint_id=path)


def load_dcp(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Collectively restore a sharded full-model and optimizer checkpoint."""

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
        set_optimizer_state_dict,
    )

    state = dcp_state(model, optimizer)
    dcp.load(state, checkpoint_id=path)
    options = StateDictOptions(full_state_dict=False, strict=True)
    set_model_state_dict(model, state["model"], options=options)
    set_optimizer_state_dict(
        model,
        optimizer,
        optim_state_dict=state["optimizer"],
        options=options,
    )


__all__ = [
    "BLOCK_CLASS_NAMES",
    "DistributedContext",
    "clip_global_grad_norm",
    "full_parameter_groups",
    "initialize_distributed",
    "load_dcp",
    "local_parameter_view",
    "offload_optimizer_state",
    "restore_optimizer_state",
    "save_dcp",
    "shard_full_parameter_policy",
    "snapshot_reference_shards",
]
