"""FSDP2-only model communication and profiling helpers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
from torch import nn

from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.core.parallel.comm.tensor import (
    EmbeddingTensorParallelCommunicator,
    HeadTensorParallelCommunicator,
    LinearRole,
    TensorParallelCommunicator,
)
from sensenovalm.model.modules.embedding import Embedding1D
from sensenovalm.model.modules.linear import (
    ColumnParallelLinear,
    RewardModelLinear,
    RowParallelLinear,
    ScaleColumnParallelLinear,
)
from sensenovalm.model.utils import _submodule_filter


def initialize_unit_mtp_communicators(model: nn.Module) -> None:
    """Attach size-one MTP communicators before FSDP2 shards the model."""

    if gpc.config.parallel.tensor.mode != "mtp" or int(gpc.config.parallel.tensor.size) != 1:
        raise ValueError("FSDP2 SFT requires tensor mode mtp with size one")
    if bool(gpc.config.model.use_moe):
        raise ValueError("the published U1.5 dense checkpoint must not enable MoE routing")
    retain_output = gpc.config.model.get("parallel_output", True)
    column = TensorParallelCommunicator(
        process_group=gpc.get_group(ParallelMode.TENSOR), role=LinearRole.COLUMN
    )
    row = TensorParallelCommunicator(
        process_group=gpc.get_group(ParallelMode.TENSOR), role=LinearRole.ROW
    )
    for module in _submodule_filter(model, ColumnParallelLinear):
        module.register_communicator(column)
    for module in _submodule_filter(model, RowParallelLinear):
        module.register_communicator(row)
    embedding = EmbeddingTensorParallelCommunicator(ParallelMode.TENSOR)
    for module in _submodule_filter(model, Embedding1D):
        embedding.register_module_hook(module)
    head = HeadTensorParallelCommunicator(ParallelMode.TENSOR, retain_output)
    for module in _submodule_filter(model, (ScaleColumnParallelLinear, RewardModelLinear)):
        module.register_communicator(head)


class _NoopProfiler:
    def step(self) -> None:
        pass


@contextmanager
def training_profile(enabled: bool, *, start_time: str) -> Iterator[object]:
    if not enabled:
        yield _NoopProfiler()
        return
    schedule = torch.profiler.schedule(
        wait=int(os.environ.get("SFT_PROFILE_WAIT", "1")),
        warmup=int(os.environ.get("SFT_PROFILE_WARMUP", "1")),
        active=int(os.environ.get("SFT_PROFILE_ACTIVE", "1")),
        repeat=1,
        skip_first=int(os.environ.get("SFT_PROFILE_SKIP_FIRST", "3")),
    )
    trace_root = Path(os.environ.get("SFT_PROFILE_ROOT", "RUN"))
    trace = trace_root / str(gpc.config.JOB_NAME) / start_time / f"rank-{gpc.get_global_rank():05d}"
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace)),
        with_stack=False,
        with_modules=False,
        record_shapes=False,
        profile_memory=True,
    ) as profiler:
        yield profiler


__all__ = ["initialize_unit_mtp_communicators", "training_profile"]
