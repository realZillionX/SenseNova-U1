"""Distributed full-parameter SenseNova-U1.5 GDPO/UniGDPO runner.

Every torchrun rank owns one shard of one policy. Rollout, frozen-old
anchoring, fixed-reference replay and current-policy replay are collective
FSDP executions. Only rank zero invokes the verifier and publishes artifacts.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from .api_rollout import SenseNovaRlApiClient
from .budget import BudgetLedger
from .flow import BranchWeights, RegularizationWeights
from .full_parameter import (
    DistributedContext,
    clip_global_grad_norm,
    full_parameter_groups,
    initialize_distributed,
    load_dcp,
    offload_optimizer_state,
    restore_optimizer_state,
    save_dcp,
)
from .objective import (
    compute_reward_advantages,
    compute_uni_gdpo_loss,
)
from .plan import RlPlan
from .policy_runtime import (
    ImageEvent,
    U15Policy,
    U15PolicyRollout,
    U15PolicyRuntime,
)
from .types import RewardBatch
from .weight_publisher import (
    ServingWeightPublisher,
    policy_version_for_initial,
    policy_version_for_step,
    publish_sharded_model_state,
)

CHECKPOINT_SCHEMA = "sensenova.u15.forge.rl.checkpoint.v1"
FINAL_STATE_SCHEMA = "sensenova.u15.forge.rl.final.v1"
RL_PROMPT_SCHEMA = "sensenova.u15.forge.prompt.v1"
_CHECKPOINT_NAME = re.compile(r"^checkpoint-([0-9]{8})$")

RewardClient = Callable[
    [RlPlan, str, str, str, Sequence[U15PolicyRollout], Path],
    tuple[RewardBatch, Mapping[str, object]],
]


def _install_scalar_all_reduce_trace() -> None:
    """Log the Python caller of scalar collectives during FSDP diagnosis."""

    if os.environ.get("SENSENOVA_FORGE_FSDP_REPLAY_TRACE") != "1":
        return
    original = dist.all_reduce

    def traced_all_reduce(tensor: Tensor, *args: object, **kwargs: object) -> object:
        if isinstance(tensor, Tensor) and tensor.numel() == 1:
            print(
                json.dumps(
                    {
                        "component": "sensenova_u1.rl",
                        "event": "scalar_all_reduce_trace",
                        "rank": dist.get_rank(),
                        "dtype": str(tensor.dtype),
                        "stack": traceback.format_stack(limit=16),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return original(tensor, *args, **kwargs)

    dist.all_reduce = traced_all_reduce  # type: ignore[method-assign]


@dataclass(frozen=True)
class PromptModality:
    prompt: str
    images: tuple[str, ...]
    rollout_group_key: str
    system_message: str | None = None


@dataclass(frozen=True)
class PromptRow:
    sample_id: str
    modality: str
    value: PromptModality

    def for_modality(self, modality: str) -> PromptModality:
        if modality != self.modality:
            raise ValueError(f"prompt row belongs to {self.modality!r}, not {modality!r}")
        return self.value


@dataclass(frozen=True)
class ReplayWorkItem:
    """One trajectory assigned to this FSDP rank for aligned replay."""

    row: PromptRow
    rollout: U15PolicyRollout
    advantage_index: int
    loss_weight: float
    signature: tuple[str, ...]


@dataclass(frozen=True)
class _ReplaySlot:
    prompt_index: int
    position: int
    advantage_index: int
    loss_weight: float
    signature: tuple[str, ...]


@dataclass(frozen=True)
class StepResult:
    step: int
    batch_index: int
    update_index: int
    sample_ids: tuple[str, ...]
    modality: str
    fsdp_world_size: int
    rollout_policy_version: str
    loss: float
    grad_norm: float
    ratio_max_error: float
    ratio_mean_error: float
    numeric_max_error: float
    rollout_seconds: float
    anchor_seconds: float
    verifier_seconds: float
    replay_backward_seconds: float
    optimizer_seconds: float
    total_seconds: float

    def performance_event(self) -> dict[str, object]:
        return {
            "component": "sensenova_u1.rl",
            "event": "step_complete",
            "step": self.step,
            "batch_index": self.batch_index,
            "update_index": self.update_index,
            "sample_ids": list(self.sample_ids),
            "modality": self.modality,
            "fsdp_world_size": self.fsdp_world_size,
            "rollout_policy_version": self.rollout_policy_version,
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "ratio_max_error": self.ratio_max_error,
            "ratio_mean_error": self.ratio_mean_error,
            "numeric_max_error": self.numeric_max_error,
            "rollout_seconds": self.rollout_seconds,
            "anchor_seconds": self.anchor_seconds,
            "verifier_seconds": self.verifier_seconds,
            "replay_backward_seconds": self.replay_backward_seconds,
            "optimizer_seconds": self.optimizer_seconds,
            "total_seconds": self.total_seconds,
        }


@dataclass(frozen=True)
class CheckpointState:
    path: Path
    step: int
    plan_digest: str
    budget: Mapping[str, int | float]
    world_size: int
    policy_version: str


class SenseNovaRlvrRows(Sequence[PromptRow]):
    """Verifier-neutral prompt rows consumed directly by Forge."""

    def __init__(self, path: Path) -> None:
        rows: list[PromptRow] = []
        seen: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"{path}:{line_number}: blank prompt row")
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: malformed JSON") from exc
                required = {
                    "schema",
                    "sample_id",
                    "modality",
                    "prompt",
                    "images",
                    "rollout_group_key",
                }
                if not isinstance(payload, Mapping) or not required <= set(payload):
                    raise ValueError(f"{path}:{line_number}: invalid RL prompt fields")
                if payload.get("schema") != RL_PROMPT_SCHEMA:
                    raise ValueError(f"{path}:{line_number}: wrong RL prompt schema")
                sample_id = payload["sample_id"]
                modality = payload["modality"]
                prompt = payload["prompt"]
                images = payload["images"]
                group = payload["rollout_group_key"]
                if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
                    raise ValueError(f"{path}:{line_number}: invalid/duplicate sample id")
                if modality not in {"ti2t", "ti2ti"}:
                    raise ValueError(f"{path}:{line_number}: invalid modality")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"{path}:{line_number}: prompt is empty")
                if not isinstance(group, str) or not group:
                    raise ValueError(f"{path}:{line_number}: rollout group is empty")
                if not isinstance(images, list) or not images:
                    raise ValueError(f"{path}:{line_number}: prompt images are empty")
                parsed: list[str] = []
                for position, value in enumerate(images):
                    if not isinstance(value, str):
                        raise ValueError(f"{path}:{line_number}: image {position} is invalid")
                    candidate = Path(value).expanduser()
                    image = (path.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
                    if not image.is_file() or image.is_symlink():
                        raise ValueError(f"{path}:{line_number}: image {position} does not exist")
                    parsed.append(str(image))
                system_message = payload.get("system_message")
                if system_message is not None and not isinstance(system_message, str):
                    raise ValueError(f"{path}:{line_number}: system_message must be a string or null")
                rows.append(
                    PromptRow(
                        sample_id=sample_id,
                        modality=modality,
                        value=PromptModality(
                            prompt=prompt,
                            images=tuple(parsed),
                            rollout_group_key=group,
                            system_message=system_message,
                        ),
                    )
                )
                seen.add(sample_id)
        if not rows:
            raise ValueError("SenseNova RLVR prompt asset is empty")
        self._rows = tuple(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> PromptRow:
        return self._rows[index]


def scheduled_prompt_batch(
    batch_index: int, rows: Sequence[PromptRow], *, prompts_per_batch: int
) -> tuple[PromptRow, ...]:
    if type(batch_index) is not int or batch_index < 0:
        raise ValueError("batch_index must be a non-negative integer")
    if not rows or prompts_per_batch < 2:
        raise ValueError("GDPO batches require rows and at least two prompts")
    start = batch_index * prompts_per_batch
    return tuple(rows[(start + offset) % len(rows)] for offset in range(prompts_per_batch))


def _combine_rewards(batches: Sequence[RewardBatch]) -> RewardBatch:
    if len(batches) < 2:
        raise ValueError("GDPO requires at least two prompt groups")
    names = batches[0].dimension_names
    if any(batch.dimension_names != names for batch in batches):
        raise ValueError("reward dimensions changed across prompt groups")
    return RewardBatch(
        matrix=np.concatenate([batch.matrix for batch in batches]),
        dimension_names=names,
        availability=np.concatenate([batch.availability for batch in batches]),
        group_ids=tuple(item for batch in batches for item in batch.group_ids),
        diagnostics=tuple(item for batch in batches for item in batch.diagnostics),
        errors=tuple(item for batch in batches for item in batch.errors),
    )


def _reward_request(
    plan: RlPlan,
    modality: str,
    sample_id: str,
    rollout_group_key: str,
    rollouts: Sequence[U15PolicyRollout],
    artifact_root: Path,
) -> dict[str, object]:
    return {
        "schema": "sensenova.u15.forge.reward.request.v1",
        "artifact_root": str(artifact_root.resolve()),
        "modality": modality,
        "sample_id": sample_id,
        "rollout_group_key": rollout_group_key,
        "reward_context": dict(plan.reward_context),
        "rollouts": [item.candidate.to_dict() for item in rollouts],
    }


def _decode_reward(payload: object) -> RewardBatch:
    if not isinstance(payload, Mapping):
        raise ValueError("reward payload must be an object")
    normalized = dict(payload)
    normalized["matrix"] = [
        [np.nan if value is None else float(value) for value in row] for row in normalized.get("matrix", ())
    ]
    return RewardBatch.from_dict(normalized)


def _decode_worker_response(plan: RlPlan, payload: object) -> tuple[RewardBatch, Mapping[str, object]]:
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "sensenova.u15.forge.reward.response.v1"
        or "reward" not in payload
    ):
        raise RuntimeError("reward provider returned the wrong response schema")
    budget = payload.get("budget") or {}
    if not isinstance(budget, Mapping):
        raise RuntimeError("reward provider budget must be an object")
    return _decode_reward(payload["reward"]), budget


def score_with_verifier_worker(
    plan: RlPlan,
    modality: str,
    sample_id: str,
    rollout_group_key: str,
    rollouts: Sequence[U15PolicyRollout],
    artifact_root: Path,
) -> tuple[RewardBatch, Mapping[str, object]]:
    process = subprocess.run(
        plan.reward_command,
        input=json.dumps(
            _reward_request(plan, modality, sample_id, rollout_group_key, rollouts, artifact_root),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode:
        raise RuntimeError("reward provider failed: " + (process.stderr or process.stdout).strip())
    return _decode_worker_response(plan, json.loads(process.stdout))


class PersistentRewardClient:
    """One downstream NDJSON reward process, owned only by rank zero."""

    def __init__(self, plan: RlPlan) -> None:
        self.plan = plan
        self.process: subprocess.Popen[str] | None = None

    def _start(self) -> subprocess.Popen[str]:
        if self.process is None:
            self.process = subprocess.Popen(
                self.plan.reward_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("reward provider pipes are unavailable")
        return self.process

    def __call__(
        self,
        plan: RlPlan,
        modality: str,
        sample_id: str,
        rollout_group_key: str,
        rollouts: Sequence[U15PolicyRollout],
        artifact_root: Path,
    ) -> tuple[RewardBatch, Mapping[str, object]]:
        if plan.digest != self.plan.digest:
            raise ValueError("reward client received a different plan")
        process = self._start()
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(
            json.dumps(
                _reward_request(
                    plan,
                    modality,
                    sample_id,
                    rollout_group_key,
                    rollouts,
                    artifact_root,
                ),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            detail = process.stderr.read().strip() if process.stderr else ""
            raise RuntimeError(f"reward provider exited without response: {detail}")
        return _decode_worker_response(plan, json.loads(line))

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        assert process.stdin is not None
        process.stdin.close()
        returncode = process.wait()
        detail = process.stderr.read().strip() if process.stderr else ""
        if returncode:
            raise RuntimeError(f"reward provider shutdown failed: {detail}")


def _set_ledger(ledger: BudgetLedger, payload: Mapping[str, object]) -> None:
    if not payload:
        return
    if set(payload) != set(ledger.as_dict()):
        raise ValueError("distributed budget payload has wrong fields")
    for name, value in payload.items():
        current = getattr(ledger, name)
        if isinstance(current, int) and (type(value) is not int or value < 0):
            raise ValueError(f"budget {name} is invalid")
        if isinstance(current, float) and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        ):
            raise ValueError(f"budget {name} is invalid")
        setattr(ledger, name, type(current)(value))


def _broadcast_primary(context: DistributedContext, producer: Callable[[], object]) -> object:
    envelope: list[object | None] = [None]
    if context.is_primary:
        try:
            envelope[0] = {"ok": True, "value": producer()}
        except BaseException as exc:  # noqa: BLE001
            envelope[0] = {"ok": False, "type": type(exc).__name__, "error": str(exc)}
    dist.broadcast_object_list(envelope, src=0)
    result = envelope[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result if isinstance(result, Mapping) else {}
        raise RuntimeError(f"rank-zero {detail.get('type', 'error')}: {detail.get('error', '')}")
    return result.get("value")


def _rollout_seed(plan: RlPlan, batch: int, prompt: int, position: int) -> int:
    payload = f"{plan.digest}:{plan.seed}:{batch}:{prompt}:{position}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def _wire_replay_signature(row: PromptRow, raw: object) -> tuple[str, ...]:
    """Describe every branch that changes an FSDP collective sequence."""

    if not isinstance(raw, Mapping):
        raise RuntimeError("distributed SenseNova RL rollout is malformed")
    events = raw.get("events")
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
        raise RuntimeError("distributed SenseNova RL events are malformed")
    signature = [f"prompt_images:{len(row.value.images)}"]
    for event in events:
        if not isinstance(event, Mapping):
            raise RuntimeError("distributed SenseNova RL event is malformed")
        event_type = event.get("type")
        if event_type == "text":
            signature.append("text")
            continue
        if event_type != "image":
            raise RuntimeError("distributed SenseNova RL event type is invalid")
        manifest = event.get("trace_manifest")
        if not isinstance(manifest, Mapping):
            raise RuntimeError("distributed SenseNova SDE trace manifest is invalid")
        geometry: list[str] = []
        for name in sorted(manifest):
            spec = manifest[name]
            if not isinstance(name, str) or not isinstance(spec, Mapping):
                raise RuntimeError("distributed SenseNova SDE trace geometry is invalid")
            shape = spec.get("shape")
            dtype = spec.get("dtype")
            if (
                isinstance(shape, (str, bytes))
                or not isinstance(shape, Sequence)
                or not all(type(value) is int and value >= 0 for value in shape)
                or not isinstance(dtype, str)
            ):
                raise RuntimeError("distributed SenseNova SDE trace geometry is invalid")
            geometry.append(f"{name}:{','.join(str(value) for value in shape)}:{dtype}")
        height = event.get("image_height")
        width = event.get("image_width")
        if type(height) is not int or type(width) is not int:
            raise RuntimeError("distributed SenseNova image geometry is invalid")
        signature.append(f"image:{height}x{width}:" + "|".join(geometry))
    if "text" not in signature:
        raise RuntimeError("distributed SenseNova rollout has no text event")
    return tuple(signature)


def _build_fsdp_replay_schedule(
    rows: Sequence[PromptRow],
    payload_groups: Sequence[Sequence[object]],
    *,
    world_size: int,
    group_size: int,
) -> tuple[tuple[_ReplaySlot, ...], ...]:
    """Build SPMD-safe waves without duplicating real training samples."""

    if world_size < 2 or group_size < 1:
        raise ValueError("FSDP replay schedule requires multiple ranks and a group")
    if len(rows) != world_size or len(payload_groups) != len(rows):
        raise ValueError("FSDP replay schedule requires one prompt group per rank")
    buckets: dict[tuple[str, ...], list[_ReplaySlot]] = {}
    for prompt_index, (row, payload) in enumerate(zip(rows, payload_groups, strict=True)):
        if len(payload) != group_size:
            raise RuntimeError("SenseNova RL API changed the requested group size")
        for position, raw in enumerate(payload):
            signature = _wire_replay_signature(row, raw)
            buckets.setdefault(signature, []).append(
                _ReplaySlot(
                    prompt_index=prompt_index,
                    position=position,
                    advantage_index=prompt_index * group_size + position,
                    loss_weight=1.0,
                    signature=signature,
                )
            )

    schedule: list[tuple[_ReplaySlot, ...]] = []
    rank_cursor = 0
    for signature, bucket in buckets.items():
        offset = 0
        while offset + world_size <= len(bucket):
            chunk = bucket[offset : offset + world_size]
            slots: list[_ReplaySlot | None] = [None] * world_size
            for position, item in enumerate(chunk):
                slots[(rank_cursor + position) % world_size] = item
            schedule.append(tuple(item for item in slots if item is not None))
            offset += world_size
        remainder = bucket[offset:]
        if remainder:
            source = bucket[0]
            slots = [
                _ReplaySlot(
                    prompt_index=source.prompt_index,
                    position=source.position,
                    advantage_index=source.advantage_index,
                    loss_weight=0.0,
                    signature=signature,
                )
                for _ in range(world_size)
            ]
            for position, item in enumerate(remainder):
                slots[(rank_cursor + position) % world_size] = item
            schedule.append(tuple(slots))
            rank_cursor = (rank_cursor + len(remainder)) % world_size

    expected = set(range(world_size * group_size))
    real_indices = [slot.advantage_index for wave in schedule for slot in wave if slot.loss_weight == 1.0]
    if len(real_indices) != len(expected) or set(real_indices) != expected:
        raise RuntimeError("FSDP replay schedule lost or duplicated a real rollout")
    real_per_rank = [sum(wave[rank].loss_weight == 1.0 for wave in schedule) for rank in range(world_size)]
    if real_per_rank != [group_size] * world_size:
        raise RuntimeError(f"FSDP replay schedule did not balance real trajectories: {real_per_rank}")
    if any(len(wave) != world_size or any(slot.signature != wave[0].signature for slot in wave) for wave in schedule):
        raise RuntimeError("FSDP replay schedule contains an unaligned wave")
    return tuple(schedule)


def _generate_and_anchor(
    plan: RlPlan,
    policy: U15Policy,
    rows: Sequence[PromptRow],
    batch_index: int,
    artifact_dir: Path,
    *,
    context: DistributedContext,
    rollout_clients: Sequence[SenseNovaRlApiClient] = (),
) -> tuple[tuple[ReplayWorkItem, ...], tuple[tuple[U15PolicyRollout, ...], ...]]:
    scoring_groups: list[tuple[U15PolicyRollout, ...]] = []
    work_items: list[ReplayWorkItem] = []
    device = next(policy.model.parameters()).device
    clients = tuple(rollout_clients)
    if clients and context.world_size != len(rows):
        raise ValueError("API rollout data parallelism requires one prompt group per FSDP rank")

    rollout_futures: list[Future[tuple[dict[str, object], ...]]] = []
    rollout_executor: ThreadPoolExecutor | None = None
    if clients and context.is_primary:
        # Submit the complete prompt batch at once.  Each request contributes
        # G members, so LightLLM sees prompts_per_batch * group_size requests
        # concurrently and its native continuous scheduler owns admission,
        # token budgeting, decode compaction and backfilling.  Do not impose a
        # second trainer-side batching policy here.
        rollout_executor = ThreadPoolExecutor(
            max_workers=len(rows),
            thread_name_prefix="sensenova-rollout-group",
        )
        for prompt_index, row in enumerate(rows):
            rollout_client = clients[prompt_index % len(clients)]
            prompt = row.for_modality(plan.modality)
            seeds = tuple(
                _rollout_seed(plan, batch_index, prompt_index, position) for position in range(plan.group_size)
            )
            rollout_futures.append(
                rollout_executor.submit(
                    rollout_client.generate_group_payload,
                    prompt=prompt.prompt,
                    prompt_images=prompt.images,
                    modality=plan.modality,
                    system_message=prompt.system_message,
                    seeds=seeds,
                    artifact_dir=artifact_dir,
                    rollout_key=f"{row.sample_id}-b{batch_index:08d}",
                    max_sequence_length=plan.max_sequence_length,
                    max_new_tokens=plan.max_new_tokens,
                    max_images=plan.max_images,
                    image_size=plan.image_size,
                    image_steps=plan.image_steps,
                    image_noise_level=plan.image_noise_level,
                    timestep_shift=plan.timestep_shift,
                    t_eps=plan.t_eps,
                    sde_window_start=plan.sde_window_start,
                    sde_window_end=plan.sde_window_end,
                    sde_window_steps=plan.sde_window_steps,
                )
            )

    payload_groups: list[tuple[dict[str, object], ...]] = []
    try:
        for prompt_index, row in enumerate(rows):
            prompt = row.for_modality(plan.modality)
            if clients:
                rollout_client = clients[prompt_index % len(clients)]
                payload = _broadcast_primary(
                    context,
                    lambda prompt_index=prompt_index: rollout_futures[prompt_index].result(),
                )
                if context.is_primary:
                    scoring_groups.append(rollout_client.scoring_group(payload, modality=plan.modality))
                if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
                    raise RuntimeError("distributed SenseNova RL group is malformed")
                payload_groups.append(tuple(payload))
                continue
            group: list[U15PolicyRollout] = []
            for position in range(plan.group_size):
                generator = torch.Generator(device=device)
                generator.manual_seed(_rollout_seed(plan, batch_index, prompt_index, position))
                rollout = policy.rollout(
                    prompt=prompt.prompt,
                    prompt_images=prompt.images,
                    modality=plan.modality,
                    system_message=prompt.system_message,
                    artifact_dir=artifact_dir,
                    rollout_key=f"{row.sample_id}-b{batch_index:08d}-r{position:03d}",
                    generator=generator,
                )
                group.append(
                    policy.anchor_rollout(
                        prompt=prompt.prompt,
                        prompt_images=prompt.images,
                        modality=plan.modality,
                        system_message=prompt.system_message,
                        rollout=rollout,
                    )
                )
            scoring_groups.append(tuple(group))
            for position, rollout in enumerate(group):
                work_items.append(
                    ReplayWorkItem(
                        row=row,
                        rollout=rollout,
                        advantage_index=prompt_index * plan.group_size + position,
                        loss_weight=1.0,
                        signature=tuple(
                            "image" if isinstance(event, ImageEvent) else "text" for event in rollout.events
                        ),
                    )
                )
    finally:
        if rollout_executor is not None:
            rollout_executor.shutdown(wait=True, cancel_futures=True)
    if clients:
        schedule = _build_fsdp_replay_schedule(
            rows,
            payload_groups,
            world_size=context.world_size,
            group_size=plan.group_size,
        )
        if context.is_primary:
            layouts: dict[str, int] = {}
            for wave in schedule:
                layout = "->".join(component.split(":", 1)[0] for component in wave[0].signature[1:])
                layouts[layout] = layouts.get(layout, 0) + sum(slot.loss_weight == 1.0 for slot in wave)
            print(
                json.dumps(
                    {
                        "component": "sensenova_u1.rl",
                        "event": "fsdp_replay_schedule",
                        "real_rollouts": context.world_size * plan.group_size,
                        "waves": len(schedule),
                        "padding_slots": sum(slot.loss_weight == 0.0 for wave in schedule for slot in wave),
                        "event_layouts": layouts,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        try:
            for wave in schedule:
                for owner_rank, slot in enumerate(wave):
                    rollout_client = clients[slot.prompt_index % len(clients)]
                    raw = payload_groups[slot.prompt_index][slot.position]
                    owned = rollout_client.materialize_group_for_rank(
                        (raw,),
                        modality=plan.modality,
                        device=device,
                        owner_rank=owner_rank,
                        consume_traces=False,
                    )
                    if context.rank == owner_rank:
                        if len(owned) != 1:
                            raise RuntimeError("FSDP replay owner lost its trajectory")
                        work_items.append(
                            ReplayWorkItem(
                                row=rows[slot.prompt_index],
                                rollout=owned[0],
                                advantage_index=slot.advantage_index,
                                loss_weight=slot.loss_weight,
                                signature=slot.signature,
                            )
                        )
        finally:
            for prompt_index, payload in enumerate(payload_groups):
                clients[prompt_index % len(clients)].release_payload_traces((payload,))
        if len(work_items) != len(schedule):
            raise RuntimeError("FSDP rank received the wrong replay wave count")
        return tuple(work_items), tuple(scoring_groups)
    return tuple(work_items), tuple(scoring_groups)


def _anchor_api_work_items(
    plan: RlPlan,
    policy: U15Policy,
    work_items: Sequence[ReplayWorkItem],
    *,
    context: DistributedContext,
) -> tuple[ReplayWorkItem, ...]:
    """Freeze API behavior likelihoods in FSDP replay geometry once."""

    device = next(policy.model.parameters()).device
    anchored_items: list[ReplayWorkItem] = []
    alignment_max = alignment_sum = alignment_numeric = 0.0
    alignment_actions = 0
    for item in work_items:
        prompt = item.row.for_modality(plan.modality)
        anchor = policy.anchor_rollout_with_metrics(
            prompt=prompt.prompt,
            prompt_images=prompt.images,
            modality=plan.modality,
            system_message=prompt.system_message,
            rollout=item.rollout,
        )
        anchored_items.append(replace(item, rollout=anchor.rollout))
        if item.loss_weight:
            alignment_max = max(alignment_max, anchor.behavior_ratio_max_error)
            alignment_sum += anchor.behavior_ratio_mean_error * anchor.behavior_ratio_action_count
            alignment_actions += anchor.behavior_ratio_action_count
            alignment_numeric = max(alignment_numeric, anchor.numeric_max_error)
    alignment_sums = torch.tensor(
        [alignment_sum, float(alignment_actions)],
        dtype=torch.float64,
        device=device,
    )
    alignment_maxima = torch.tensor(
        [alignment_max, alignment_numeric],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(alignment_sums, op=dist.ReduceOp.SUM)
    dist.all_reduce(alignment_maxima, op=dist.ReduceOp.MAX)
    if context.is_primary:
        print(
            json.dumps(
                {
                    "component": "sensenova_u1.rl",
                    "event": "behavior_replay_anchor",
                    "ratio_max_error": float(alignment_maxima[0]),
                    "ratio_mean_error": (
                        float(alignment_sums[0] / alignment_sums[1]) if float(alignment_sums[1]) else 0.0
                    ),
                    "numeric_max_error": float(alignment_maxima[1]),
                    "actions": int(alignment_sums[1]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return tuple(anchored_items)


def _primary_advantages(
    plan: RlPlan,
    rows: Sequence[PromptRow],
    groups: Sequence[Sequence[U15PolicyRollout]],
    artifact_dir: Path,
    ledger: BudgetLedger,
    reward_client: RewardClient,
) -> dict[str, object]:
    started = time.monotonic()
    rewards: list[RewardBatch] = []
    for row, group in zip(rows, groups, strict=True):
        reward, budget = reward_client(
            plan,
            plan.modality,
            row.sample_id,
            row.value.rollout_group_key,
            group,
            artifact_dir,
        )
        if tuple(reward.group_ids) != (row.value.rollout_group_key,) * plan.group_size:
            raise RuntimeError("reward-provider group ids differ from the prompt group")
        if tuple(reward.dimension_names) != plan.reward_dimension_names:
            raise RuntimeError("reward provider changed the planned reward dimensions")
        calls = budget.get("verifier_invocations")
        if calls is None:
            calls = 0
        if type(calls) is not int or calls < 0:
            raise ValueError("verifier budget invocation count is invalid")
        ledger.verifier_invocations += calls
        rewards.append(reward)
    elapsed = time.monotonic() - started
    ledger.verifier_seconds += elapsed
    advantages = compute_reward_advantages(
        _combine_rewards(rewards), modality=plan.modality, weights=plan.reward_weights
    ).advantages
    return {
        "advantages": advantages.tolist(),
        "budget": ledger.as_dict(),
        "verifier_seconds": elapsed,
    }


def _replay_backward(
    plan: RlPlan,
    policy: U15Policy,
    work_items: Sequence[ReplayWorkItem],
    advantages: Tensor,
    update_index: int,
    *,
    partitioned_replay: bool = False,
) -> tuple[float, float, float, float]:
    trace_replay = os.environ.get("SENSENOVA_FORGE_FSDP_REPLAY_TRACE") == "1"

    def emit_trace(
        phase: str,
        *,
        item_index: int | None = None,
        item: ReplayWorkItem | None = None,
        image_microbatches: int | None = None,
    ) -> None:
        if not trace_replay:
            return
        payload: dict[str, object] = {
            "component": "sensenova_u1.rl",
            "event": "fsdp_replay_trace",
            "rank": dist.get_rank(),
            "update_index": update_index,
            "phase": phase,
        }
        if item_index is not None:
            payload["item_index"] = item_index
        if item is not None:
            payload.update(
                {
                    "sample_id": item.row.sample_id,
                    "advantage_index": item.advantage_index,
                    "loss_weight": item.loss_weight,
                    "signature": list(item.signature),
                    "text_actions": [
                        int(event.trace.response_mask.sum().item())
                        for event in item.rollout.events
                        if not isinstance(event, ImageEvent)
                    ],
                    "image_actions": [
                        int(event.trace.steps * event.trace.batch_size)
                        for event in item.rollout.events
                        if isinstance(event, ImageEvent)
                    ],
                }
            )
        if image_microbatches is not None:
            payload["image_microbatches"] = image_microbatches
        print(json.dumps(payload, sort_keys=True), flush=True)

    total = plan.group_size if partitioned_replay else (plan.prompts_per_batch * plan.group_size)
    if partitioned_replay and sum(item.loss_weight == 1.0 for item in work_items) != plan.group_size:
        raise RuntimeError("FSDP rank does not own one balanced rollout group")
    local_signature = tuple(item.signature for item in work_items)
    gathered_signatures: list[object] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_signatures, local_signature)
    if any(signature != local_signature for signature in gathered_signatures):
        raise RuntimeError("FSDP replay event schedule differs across ranks; refusing to enter model collectives")
    # A frozen-reference forward is only required by the regularizers that
    # consume it.  Building it unconditionally snapshots every live FSDP
    # parameter shard and can nearly double peak memory even when both
    # reference-based coefficients are zero.
    has_image_actions = any(isinstance(event, ImageEvent) for item in work_items for event in item.rollout.events)
    policy.model.zero_grad(set_to_none=True)
    loss_total = ratio_error = numeric_error = 0.0
    ratio_error_sum = 0.0
    ratio_action_count = 0
    for item_index, item in enumerate(work_items):
        row, rollout = item.row, item.rollout
        prompt = row.for_modality(plan.modality)
        emit_trace("policy_replay_start", item_index=item_index, item=item)
        references = policy.reference_replay(
            prompt=prompt.prompt,
            prompt_images=prompt.images,
            modality=plan.modality,
            system_message=prompt.system_message,
            rollout=rollout,
            include_text=bool(plan.text_kl_beta),
            include_images=bool(has_image_actions and plan.velocity_mse_weight),
        )
        advantage = advantages[item.advantage_index : item.advantage_index + 1]
        total_text_actions = sum(
            int(event.trace.response_mask.sum().item()) for event in rollout.events if not isinstance(event, ImageEvent)
        )
        replayed_text_actions = 0
        text_spans = policy.iter_text_replay_spans(
            prompt=prompt.prompt,
            prompt_images=prompt.images,
            modality=plan.modality,
            system_message=prompt.system_message,
            rollout=rollout,
            reference_log_probs=references.text_log_probs,
        )
        for text_span in text_spans:
            current = text_span.log_probs
            old = text_span.trace.old_log_probs[text_span.trace.response_mask].reshape(1, -1)
            span_actions = int(text_span.trace.response_mask.sum().item())
            if item.loss_weight:
                ratio = torch.exp(current.detach().float() - old.detach().float())
                if not bool(torch.isfinite(ratio).all()):
                    raise RuntimeError("text policy ratio is non-finite")
                errors = (ratio - 1.0).abs()
                span_ratio_max = float(errors.max().cpu())
                ratio_error = max(ratio_error, span_ratio_max)
                ratio_error_sum += float(errors.sum().cpu())
                ratio_action_count += errors.numel()
                numeric_error = max(numeric_error, text_span.numeric_max_error)
                if update_index == 0 and ratio_error > 2e-4:
                    raise RuntimeError(f"pre-update old/current ratio is not one: {ratio_error:.6g}")
            text_result = compute_uni_gdpo_loss(
                advantages=advantage,
                image_replay=None,
                text_trace=text_span.trace,
                text_log_probs=current,
                text_ref_log_probs=text_span.ref_log_probs,
                weights=BranchWeights(image=0.0, text=1.0),
                regularization=RegularizationWeights(
                    image_velocity_mse=0.0,
                    text_kl=plan.text_kl_beta,
                ),
                image_clip_range=plan.image_clip_range,
                text_clip_range=plan.text_clip_range,
            )
            text_loss = text_result.value * (span_actions / total_text_actions) * item.loss_weight / total
            if not bool(torch.isfinite(text_loss)):
                raise RuntimeError("SenseNova text GDPO loss is non-finite")
            emit_trace("text_backward_start", item_index=item_index, item=item)
            text_loss.backward(retain_graph=not text_span.is_last)
            emit_trace("text_backward_end", item_index=item_index, item=item)
            loss_total += float(text_loss.detach())
            replayed_text_actions += span_actions
            del text_loss, text_result, current, old, text_span
        if replayed_text_actions != total_text_actions:
            raise RuntimeError(
                f"SenseNova text replay yielded {replayed_text_actions} actions, expected {total_text_actions}"
            )
        emit_trace("policy_replay_end", item_index=item_index, item=item)

        emit_trace("image_replay_start", item_index=item_index, item=item)
        image_microbatches_per_event = tuple(
            math.ceil(event.trace.steps / plan.image_replay_microbatch_size)
            for event in rollout.events
            if isinstance(event, ImageEvent)
        )
        image_microbatch_count = sum(image_microbatches_per_event)
        image_backward_boundaries: set[int] = set()
        cumulative_microbatches = 0
        for count in image_microbatches_per_event:
            cumulative_microbatches += count
            image_backward_boundaries.add(cumulative_microbatches)
        image_replays = policy.iter_image_action_microbatches(
            prompt=prompt.prompt,
            prompt_images=prompt.images,
            modality=plan.modality,
            system_message=prompt.system_message,
            rollout=rollout,
            microbatch_size=plan.image_replay_microbatch_size,
            reference_velocities=references.image_velocities,
        )
        emit_trace(
            "image_replay_stream_start",
            item_index=item_index,
            item=item,
            image_microbatches=image_microbatch_count,
        )
        total_image_actions = sum(
            event.trace.steps * event.trace.batch_size for event in rollout.events if isinstance(event, ImageEvent)
        )
        replayed_microbatches = 0
        pending_image_losses: list[Tensor] = []
        for microbatch_index, image_replay in enumerate(image_replays):
            current = image_replay.log_probs
            old = image_replay.trace.old_log_probs
            ratio = torch.exp(current.detach().float() - old.detach().float())
            if item.loss_weight:
                if not bool(torch.isfinite(ratio).all()):
                    raise RuntimeError("image policy ratio is non-finite")
                errors = (ratio - 1.0).abs()
                chunk_ratio_max = float(errors.max().cpu())
                ratio_error = max(ratio_error, chunk_ratio_max)
                ratio_error_sum += float(errors.sum().cpu())
                ratio_action_count += errors.numel()
                numeric_error = max(numeric_error, chunk_ratio_max)
                if update_index == 0 and ratio_error > 2e-4:
                    raise RuntimeError(f"pre-update old/current ratio is not one: {ratio_error:.6g}")
            image_result = compute_uni_gdpo_loss(
                advantages=advantage,
                image_replay=image_replay,
                text_trace=None,
                weights=BranchWeights(
                    image=plan.image_objective_weight,
                    text=0.0,
                ),
                regularization=RegularizationWeights(
                    image_velocity_mse=plan.velocity_mse_weight,
                    text_kl=0.0,
                ),
                image_clip_range=plan.image_clip_range,
                text_clip_range=plan.text_clip_range,
            )
            chunk_actions = image_replay.trace.steps * image_replay.trace.batch_size
            image_loss = image_result.value * (chunk_actions / total_image_actions) * item.loss_weight / total
            if not bool(torch.isfinite(image_loss)):
                raise RuntimeError("SenseNova image GDPO loss is non-finite")
            loss_total += float(image_loss.detach())
            pending_image_losses.append(image_loss)
            replayed_microbatches += 1
            del image_loss, image_result, current, old, ratio, image_replay
            if replayed_microbatches in image_backward_boundaries:
                emit_trace(
                    "image_backward_start",
                    item_index=item_index,
                    item=item,
                    image_microbatches=len(pending_image_losses),
                )
                image_backward_loss = torch.stack(pending_image_losses).sum()
                image_backward_loss.backward(retain_graph=replayed_microbatches < image_microbatch_count)
                emit_trace(
                    "image_backward_end",
                    item_index=item_index,
                    item=item,
                    image_microbatches=len(pending_image_losses),
                )
                del image_backward_loss
                pending_image_losses.clear()
        if replayed_microbatches != image_microbatch_count:
            raise RuntimeError(
                "SenseNova image replay yielded "
                f"{replayed_microbatches} microbatches, expected {image_microbatch_count}"
            )
        if pending_image_losses:
            raise RuntimeError("SenseNova image replay stopped inside an image event")
        emit_trace(
            "image_replay_stream_end",
            item_index=item_index,
            item=item,
            image_microbatches=image_microbatch_count,
        )
        del references, text_spans, image_replays
    emit_trace("metrics_all_reduce_start")
    sums = torch.tensor(
        [loss_total, ratio_error_sum, float(ratio_action_count)],
        dtype=torch.float64,
        device=next(policy.model.parameters()).device,
    )
    maxima = torch.tensor(
        [ratio_error, numeric_error],
        dtype=torch.float64,
        device=sums.device,
    )
    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
    emit_trace("metrics_all_reduce_end")
    sums[0] /= dist.get_world_size()
    ratio_mean_error = float(sums[1] / sums[2]) if float(sums[2]) else 0.0
    return (
        float(sums[0]),
        float(maxima[0]),
        ratio_mean_error,
        float(maxima[1]),
    )


def execute_on_policy_batch(
    *,
    batch_index: int,
    plan: RlPlan,
    rows: Sequence[PromptRow],
    policy: U15Policy,
    optimizer: Any,
    generator: torch.Generator,
    artifact_dir: Path,
    ledger: BudgetLedger,
    reward_client: RewardClient = score_with_verifier_worker,
    context: DistributedContext | None = None,
    rollout_clients: Sequence[SenseNovaRlApiClient] = (),
    **_unused: object,
) -> tuple[StepResult, ...]:
    """Execute one frozen-old multi-prompt batch on one sharded policy."""

    del generator
    context = context or _live_context()
    if len(rows) != plan.prompts_per_batch or any(row.modality != plan.modality for row in rows):
        raise ValueError("optimizer prompt batch differs from the immutable arm")
    batch_started = time.perf_counter()
    rollout_started = time.perf_counter()
    work_items, scoring_groups = _generate_and_anchor(
        plan,
        policy,
        rows,
        batch_index,
        artifact_dir,
        context=context,
        rollout_clients=rollout_clients,
    )
    dist.barrier()
    rollout_seconds = time.perf_counter() - rollout_started
    if context.is_primary:
        ledger.record_samples(len(rows))
        for group in scoring_groups:
            for rollout in group:
                ledger.record_rollouts(
                    text_tokens=rollout.text_tokens,
                    images=rollout.generated_images,
                    seconds=rollout.seconds,
                    truncated=rollout.finish_reason == "length",
                    image_limit_hit=(plan.modality == "ti2ti" and rollout.generated_images == plan.max_images),
                )
    # Reward evaluation is CPU/external work. Hide it under the mandatory
    # no-grad old-policy anchor instead of serializing verifier latency and an
    # extra full-model replay.
    reward_executor: ThreadPoolExecutor | None = None
    reward_future: Future[dict[str, object]] | None = None
    anchor_seconds = 0.0
    if context.is_primary:
        reward_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sensenova-reward")
        reward_future = reward_executor.submit(
            _primary_advantages,
            plan,
            rows,
            scoring_groups,
            artifact_dir,
            ledger,
            reward_client,
        )
    try:
        if rollout_clients:
            anchor_started = time.perf_counter()
            work_items = _anchor_api_work_items(
                plan,
                policy,
                work_items,
                context=context,
            )
            anchor_seconds = time.perf_counter() - anchor_started
        reward_payload = _broadcast_primary(
            context,
            lambda: (
                reward_future.result()
                if reward_future is not None
                else (_ for _ in ()).throw(RuntimeError("rank zero has no reward future"))
            ),
        )
    finally:
        if reward_executor is not None:
            reward_executor.shutdown(wait=True, cancel_futures=True)
    if not isinstance(reward_payload, Mapping):
        raise RuntimeError("rank-zero reward broadcast is malformed")
    _set_ledger(ledger, reward_payload["budget"])
    advantages = torch.tensor(reward_payload["advantages"], device=context.device, dtype=torch.float32)
    if advantages.shape != (plan.prompts_per_batch * plan.group_size,):
        raise RuntimeError("broadcast GDPO advantage has wrong shape")
    parameters = tuple(parameter for parameter in policy.model.parameters() if parameter.requires_grad)
    maximum_images = torch.tensor(
        max(
            (item.rollout.generated_images for item in work_items if item.loss_weight),
            default=0,
        ),
        device=context.device,
        dtype=torch.int64,
    )
    dist.all_reduce(maximum_images, op=dist.ReduceOp.MAX)
    use_optimizer_cpu_offload = int(maximum_images.item()) >= plan.optimizer_cpu_offload_min_images
    results: list[StepResult] = []
    for update_index in range(plan.policy_updates_per_batch):
        replay_started = time.perf_counter()
        offload_seconds = 0.0
        offloaded_bytes = offloaded_tensors = 0
        if use_optimizer_cpu_offload:
            offload_started = time.perf_counter()
            offloaded_bytes, offloaded_tensors = offload_optimizer_state(optimizer)
            torch.cuda.empty_cache()
            offload_seconds = time.perf_counter() - offload_started
        loss, ratio, ratio_mean, numeric = _replay_backward(
            plan,
            policy,
            work_items,
            advantages,
            update_index,
            partitioned_replay=bool(rollout_clients),
        )
        grad_norm = float(clip_global_grad_norm(parameters, plan.max_grad_norm))
        if not math.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError("SenseNova GDPO gradient norm is non-finite")
        replay_seconds = time.perf_counter() - replay_started
        optimizer_started = time.perf_counter()
        restored_bytes = restored_tensors = 0
        if use_optimizer_cpu_offload:
            restored_bytes, restored_tensors = restore_optimizer_state(
                optimizer,
                device=context.device,
            )
            if (restored_bytes, restored_tensors) != (
                offloaded_bytes,
                offloaded_tensors,
            ):
                raise RuntimeError("optimizer CPU offload/restore closure changed")
        optimizer.step()
        optimizer_seconds = time.perf_counter() - optimizer_started
        if context.is_primary and use_optimizer_cpu_offload:
            print(
                json.dumps(
                    {
                        "component": "sensenova_u1.rl",
                        "event": "optimizer_state_offload",
                        "update_index": update_index,
                        "bytes": offloaded_bytes,
                        "tensors": offloaded_tensors,
                        "offload_seconds": offload_seconds,
                        "restore_and_step_seconds": optimizer_seconds,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        step = batch_index * plan.policy_updates_per_batch + update_index + 1
        results.append(
            StepResult(
                step=step,
                batch_index=batch_index,
                update_index=update_index,
                sample_ids=tuple(row.sample_id for row in rows),
                modality=plan.modality,
                fsdp_world_size=context.world_size,
                rollout_policy_version=(
                    rollout_clients[0].expected_policy_version if rollout_clients else plan.rollout_policy_version
                ),
                loss=loss,
                grad_norm=grad_norm,
                ratio_max_error=ratio,
                ratio_mean_error=ratio_mean,
                numeric_max_error=numeric,
                rollout_seconds=rollout_seconds if update_index == 0 else 0.0,
                anchor_seconds=anchor_seconds if update_index == 0 else 0.0,
                verifier_seconds=float(reward_payload["verifier_seconds"]) if update_index == 0 else 0.0,
                replay_backward_seconds=replay_seconds,
                optimizer_seconds=optimizer_seconds,
                total_seconds=time.perf_counter() - batch_started,
            )
        )
    return tuple(results)


def _live_context() -> DistributedContext:
    if not dist.is_initialized():
        raise ValueError("full-parameter RLVR requires torch.distributed")
    local = torch.cuda.current_device()
    return DistributedContext(
        rank=dist.get_rank(),
        local_rank=local,
        world_size=dist.get_world_size(),
        device=torch.device("cuda", local),
    )


def _checkpoint_path(plan: RlPlan, step: int) -> Path:
    return plan.run_dir / "checkpoints" / f"checkpoint-{step:08d}"


def _checkpoint_metadata(path: Path) -> CheckpointState:
    payload = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    expected = {
        "schema",
        "step",
        "plan_digest",
        "budget",
        "world_size",
        "policy_version",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or not (path / "COMMIT").is_file()
    ):
        raise ValueError(f"incomplete FSDP checkpoint: {path}")
    if not isinstance(payload["policy_version"], str) or not payload["policy_version"].strip():
        raise ValueError(f"checkpoint has no active serving policy: {path}")
    return CheckpointState(
        path=path,
        step=int(payload["step"]),
        plan_digest=str(payload["plan_digest"]),
        budget=dict(payload["budget"]),
        world_size=int(payload["world_size"]),
        policy_version=str(payload["policy_version"]),
    )


def latest_checkpoint(
    plan: RlPlan, *, context: DistributedContext | None = None, **_unused: object
) -> CheckpointState | None:
    context = context or _live_context()

    def discover() -> object:
        root = plan.run_dir / "checkpoints"
        root.mkdir(parents=True, exist_ok=True)
        states = [
            _checkpoint_metadata(path)
            for path in root.iterdir()
            if _CHECKPOINT_NAME.fullmatch(path.name) and path.is_dir()
        ]
        leftovers = [path for path in root.iterdir() if path.name.startswith(".checkpoint-")]
        if leftovers:
            raise ValueError(f"unfinished FSDP checkpoint needs audit: {leftovers}")
        if not states:
            return None
        state = max(states, key=lambda item: item.step)
        return {
            "path": str(state.path),
            "step": state.step,
            "plan_digest": state.plan_digest,
            "budget": dict(state.budget),
            "world_size": state.world_size,
            "policy_version": state.policy_version,
        }

    payload = _broadcast_primary(context, discover)
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise RuntimeError("checkpoint discovery broadcast is malformed")
    return CheckpointState(
        path=Path(str(payload["path"])),
        step=int(payload["step"]),
        plan_digest=str(payload["plan_digest"]),
        budget=dict(payload["budget"]),
        world_size=int(payload["world_size"]),
        policy_version=str(payload["policy_version"]),
    )


def _rng_payload(generator: torch.Generator) -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
        "rollout": generator.get_state(),
    }


def save_checkpoint(
    *,
    plan: RlPlan,
    step: int,
    model: Any,
    optimizer: Any,
    generator: torch.Generator,
    ledger: BudgetLedger,
    policy_version: str,
    context: DistributedContext | None = None,
    **_unused: object,
) -> Path:
    context = context or _live_context()
    destination = _checkpoint_path(plan, step)
    temporary = destination.parent / f".{destination.name}.tmp"

    def prepare() -> str:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            state = _checkpoint_metadata(destination)
            if state.plan_digest != plan.digest or state.step != step:
                raise ValueError("existing checkpoint belongs to different state")
            if state.policy_version != policy_version:
                raise ValueError("existing checkpoint belongs to a different serving policy")
            return "exists"
        if temporary.exists():
            raise ValueError(f"unfinished FSDP checkpoint needs audit: {temporary}")
        temporary.mkdir()
        return "write"

    action = _broadcast_primary(context, prepare)
    if action == "exists":
        dist.barrier()
        return destination
    save_dcp(temporary / "dcp", model, optimizer)
    torch.save(_rng_payload(generator), temporary / f"rank-{context.rank:05d}-rng.pt")
    dist.barrier()
    if context.is_primary:
        (temporary / "trainer_state.json").write_text(
            json.dumps(
                {
                    "schema": CHECKPOINT_SCHEMA,
                    "step": step,
                    "plan_digest": plan.digest,
                    "budget": ledger.as_dict(),
                    "world_size": context.world_size,
                    "policy_version": policy_version,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (temporary / "COMMIT").write_text("committed\n", encoding="utf-8")
        os.replace(temporary, destination)
    dist.barrier()
    return destination


def restore_checkpoint(
    *,
    state: CheckpointState,
    plan: RlPlan,
    model: Any,
    optimizer: Any,
    generator: torch.Generator,
    ledger: BudgetLedger,
    context: DistributedContext | None = None,
    **_unused: object,
) -> int:
    context = context or _live_context()
    if state.plan_digest != plan.digest or state.world_size != context.world_size:
        raise ValueError("FSDP checkpoint is incompatible with this plan")
    load_dcp(state.path / "dcp", model, optimizer)
    payload = torch.load(
        state.path / f"rank-{context.rank:05d}-rng.pt",
        map_location="cpu",
        weights_only=False,
    )
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    torch.cuda.set_rng_state(payload["torch_cuda"])
    generator.set_state(payload["rollout"])
    _set_ledger(ledger, state.budget)
    return state.step


def _publish_live_policy(
    *,
    policy: U15Policy,
    publisher: ServingWeightPublisher | None,
    policy_version: str,
    context: DistributedContext,
) -> Mapping[str, object]:
    """Collectively export the current policy to every serving replica."""

    started = time.perf_counter()
    try:
        response = publish_sharded_model_state(
            policy.model,
            publisher=publisher,
            policy_version=policy_version,
            context=context,
        )
    finally:
        torch.cuda.empty_cache()
    if not isinstance(response, Mapping):
        raise RuntimeError("serving replica-set publication returned malformed data")
    dist.barrier()
    if context.is_primary:
        print(
            json.dumps(
                {
                    "component": "sensenova_u1.rl",
                    "event": "serving_policy_published",
                    "policy_version": policy_version,
                    "replicas": (publisher.replica_count if publisher is not None else 0),
                    "seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return response


def run_training_loop(
    *,
    plan: RlPlan,
    policy: U15Policy,
    optimizer: Any,
    rows: Sequence[PromptRow],
    reward_client: RewardClient = score_with_verifier_worker,
    resume: CheckpointState | None = None,
    context: DistributedContext | None = None,
    **_unused: object,
) -> tuple[int, BudgetLedger, str]:
    context = context or _live_context()
    generator = torch.Generator(device=context.device)
    generator.manual_seed(plan.seed ^ 0x15A8)
    ledger, completed = BudgetLedger(), 0
    if resume is not None:
        completed = restore_checkpoint(
            state=resume,
            plan=plan,
            model=policy.model,
            optimizer=optimizer,
            generator=generator,
            ledger=ledger,
            context=context,
        )
    if completed % plan.policy_updates_per_batch:
        raise ValueError("checkpoint splits a frozen-old update batch")
    scratch = plan.run_dir / ".tmp"
    reward_owner = (
        PersistentRewardClient(plan) if context.is_primary and reward_client is score_with_verifier_worker else None
    )
    active_reward = reward_owner or reward_client
    publisher = (
        ServingWeightPublisher(
            base_urls=plan.rollout_api_base_urls,
            master_address=plan.weight_update_master_address,
            base_port=plan.weight_update_base_port,
            backend=plan.weight_update_backend,
            bucket_bytes=plan.weight_update_bucket_bytes,
            default_dtype={
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[plan.dtype],
            device=context.device,
            group_name=f"forge-{plan.digest[:16]}",
        )
        if context.is_primary
        else None
    )
    active_policy_version = resume.policy_version if resume is not None else plan.rollout_policy_version
    serving_statuses = _broadcast_primary(
        context,
        lambda: (
            [dict(status) for status in publisher.statuses()]
            if publisher is not None
            else (_ for _ in ()).throw(RuntimeError("rank zero has no serving weight publisher"))
        ),
    )
    if not isinstance(serving_statuses, list) or len(serving_statuses) != len(plan.rollout_api_base_urls):
        raise RuntimeError("SenseNova serving replica status broadcast is malformed")
    observed_versions = set()
    for replica_index, serving_status in enumerate(serving_statuses):
        if not isinstance(serving_status, Mapping):
            raise RuntimeError(f"SenseNova serving replica {replica_index} returned malformed status")
        if serving_status.get("replica_id") != replica_index:
            raise RuntimeError(
                f"SenseNova serving URL {replica_index} reports replica_id={serving_status.get('replica_id')!r}"
            )
        observed_versions.add(serving_status.get("active_policy_version"))
        if serving_status.get("paused") or serving_status.get("pending_policy_version"):
            raise RuntimeError(f"SenseNova serving replica {replica_index} is not in a stable active state")
    if len(observed_versions) != 1:
        raise RuntimeError("SenseNova serving replicas expose different active policy versions")
    observed_version = next(iter(observed_versions))
    if resume is None:
        # Bootstrap every fresh arm from the live FSDP policy before admitting
        # its first rollout.  Besides preventing a stale serving copy, this
        # eagerly creates and warms the persistent NCCL update groups instead
        # of discovering transport failures after the first optimizer batch.
        active_policy_version = policy_version_for_initial(
            plan.rollout_policy_version,
            plan.digest,
        )
        _publish_live_policy(
            policy=policy,
            publisher=publisher,
            policy_version=active_policy_version,
            context=context,
        )
    elif resume is not None and observed_version != active_policy_version:
        _publish_live_policy(
            policy=policy,
            publisher=publisher,
            policy_version=active_policy_version,
            context=context,
        )
    rollout_clients = tuple(
        SenseNovaRlApiClient(
            base_url=base_url,
            expected_policy_version=active_policy_version,
        )
        for base_url in plan.rollout_api_base_urls
    )
    if context.is_primary:
        if scratch.exists():
            raise ValueError(f"scratch residue needs audit: {scratch}")
        scratch.mkdir()
    dist.barrier()
    try:
        first_batch = completed // plan.policy_updates_per_batch
        total_batches = plan.max_steps // plan.policy_updates_per_batch
        for batch_index in range(first_batch, total_batches):
            artifact = scratch / f"rollout-batch-{batch_index:08d}"
            if context.is_primary:
                artifact.mkdir()
            dist.barrier()
            batch_failed = False
            try:
                results = execute_on_policy_batch(
                    batch_index=batch_index,
                    plan=plan,
                    rows=scheduled_prompt_batch(batch_index, rows, prompts_per_batch=plan.prompts_per_batch),
                    policy=policy,
                    optimizer=optimizer,
                    generator=generator,
                    artifact_dir=artifact,
                    ledger=ledger,
                    reward_client=active_reward,
                    context=context,
                    rollout_clients=rollout_clients,
                )
                completed = results[-1].step
                if context.is_primary:
                    for result in results:
                        print(json.dumps(result.performance_event(), sort_keys=True), flush=True)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                active_policy_version = policy_version_for_step(
                    plan.rollout_policy_version,
                    plan.digest,
                    completed,
                )
                _publish_live_policy(
                    policy=policy,
                    publisher=publisher,
                    policy_version=active_policy_version,
                    context=context,
                )
                for rollout_client in rollout_clients:
                    rollout_client.expected_policy_version = active_policy_version
            except BaseException as exc:
                batch_failed = True
                print(
                    json.dumps(
                        {
                            "component": "sensenova_u1.rl",
                            "event": "batch_failed",
                            "rank": context.rank,
                            "batch_index": batch_index,
                            "exception": repr(exc),
                            "traceback": traceback.format_exc(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                raise
            finally:
                # A unilateral cleanup barrier masks the real exception and
                # mismatches peers that are still inside FSDP collectives.
                # Let torchrun abort the remaining ranks on failure; only a
                # successful batch may enter collective artifact cleanup.
                if not batch_failed:
                    dist.barrier()
                    if context.is_primary:
                        shutil.rmtree(artifact)
                    dist.barrier()
            if completed % plan.save_every_steps == 0 or completed == plan.max_steps:
                save_checkpoint(
                    plan=plan,
                    step=completed,
                    model=policy.model,
                    optimizer=optimizer,
                    generator=generator,
                    ledger=ledger,
                    policy_version=active_policy_version,
                    context=context,
                )
        _broadcast_primary(
            context,
            lambda: publisher.close() if publisher is not None else None,
        )
    finally:
        _broadcast_primary(
            context,
            lambda: reward_owner.close() if reward_owner is not None else None,
        )
        dist.barrier()
        if context.is_primary:
            scratch.rmdir()
        dist.barrier()
    return completed, ledger, active_policy_version


def _seed_process(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _run_plan(plan: RlPlan, context: DistributedContext) -> None:
    if plan.torchrun.world_size != context.world_size:
        raise ValueError("live FSDP world size differs from immutable plan")
    rows = SenseNovaRlvrRows(plan.prompts)
    if any(row.modality != plan.modality for row in rows):
        raise ValueError("RL prompt asset crossed independent arms")
    _seed_process(plan.seed)
    policy = U15PolicyRuntime.load(plan, device=context.device)
    optimizer = torch.optim.AdamW(
        full_parameter_groups(
            policy.model,
            text_learning_rate=plan.text_learning_rate,
            visual_learning_rate=plan.visual_learning_rate,
            weight_decay=plan.weight_decay,
        )
    )
    resume = latest_checkpoint(plan, context=context)
    completed, ledger, active_policy_version = run_training_loop(
        plan=plan,
        policy=policy,
        optimizer=optimizer,
        rows=rows,
        resume=resume,
        context=context,
    )
    if completed != plan.max_steps:
        raise RuntimeError("full-parameter run ended before max_steps")
    final = latest_checkpoint(plan, context=context)
    if final is None or final.step != completed:
        raise RuntimeError("final FSDP checkpoint does not close run")
    if context.is_primary:
        (plan.run_dir / "final_state.json").write_text(
            json.dumps(
                {
                    "schema": FINAL_STATE_SCHEMA,
                    "plan_digest": plan.digest,
                    "step": completed,
                    "budget": ledger.as_dict(),
                    "checkpoint": str(final.path),
                    "world_size": context.world_size,
                    "policy_init_kind": plan.policy_init_kind,
                    "initial_rollout_policy_version": plan.rollout_policy_version,
                    "rollout_policy_version": active_policy_version,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    dist.barrier()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m sensenova_u1.rl.trainer PLAN")
    context: DistributedContext | None = None
    lock_handle: Any = None
    try:
        context = initialize_distributed()
        _install_scalar_all_reduce_trace()
        requested = Path(os.path.abspath(os.path.expanduser(sys.argv[1])))
        plan = RlPlan.read(requested)

        def acquire_lock() -> bool:
            nonlocal lock_handle
            lock_path = plan.run_dir.parent / (
                ".sensenova-u15-forge-rl-" + hashlib.sha256(str(plan.run_dir).encode()).hexdigest() + ".lock"
            )
            lock_handle = lock_path.open("a+")
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True

        _broadcast_primary(context, acquire_lock)
        _run_plan(plan, context)
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        rank = dist.get_rank() if dist.is_initialized() else "?"
        raise SystemExit(f"SenseNova full-parameter RLVR rank {rank} refused: {exc}") from exc
    finally:
        if lock_handle is not None:
            lock_handle.close()
        if context is not None and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()


__all__ = [
    "CHECKPOINT_SCHEMA",
    "FINAL_STATE_SCHEMA",
    "CheckpointState",
    "PersistentRewardClient",
    "PromptModality",
    "PromptRow",
    "SenseNovaRlvrRows",
    "StepResult",
    "execute_on_policy_batch",
    "latest_checkpoint",
    "restore_checkpoint",
    "run_training_loop",
    "save_checkpoint",
    "scheduled_prompt_batch",
    "score_with_verifier_worker",
]
