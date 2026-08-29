"""Online publication of the live FSDP2 policy to SenseNova serving.

The trainer owns FP32 sharded optimizer masters, while LightLLM, the visual
encoder and LightX2V own independent inference copies.  Publication rebuilds
one serving bucket at a time directly from the live GPU shards and sends that
bucket GPU-to-GPU.  It never materializes a full CPU state dict and never
reloads a safetensors checkpoint from disk.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn

from .full_parameter import DistributedContext

_CONSUMERS = ("language", "vision", "x2v")
_DTYPES = {
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.int64": torch.int64,
    "torch.int32": torch.int32,
    "torch.int8": torch.int8,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}
_TRAINING_TRANSFER_GROUPS: dict[tuple[int, int], dict[str, object]] = {}


def _request(
    method: str,
    url: str,
    *,
    payload: Mapping[str, object] | None = None,
    timeout: float = 7200.0,
) -> object:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read()
            content_type = response.headers.get("content-type", "")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} returned HTTP {exc.code}: {detail[:2000]}") from exc
    if "application/json" in content_type:
        return json.loads(content)
    return content.decode("utf-8", errors="replace")


def _init_custom_process_group(
    *,
    backend: str,
    init_method: str,
    group_name: str,
    device: torch.device,
    warmup: bool,
) -> object:
    """Create the publisher half of a two-rank serving update group."""

    from torch.distributed.distributed_c10d import (
        Backend,
        PrefixStore,
        _new_process_group_helper,
        _world,
    )

    timeout = timedelta(hours=2)
    endpoint = urllib.parse.urlparse(init_method)
    if endpoint.scheme != "tcp" or endpoint.hostname is None or endpoint.port is None:
        raise ValueError(f"weight update group requires a tcp init method: {init_method}")
    rank, world_size = 0, 2
    # torchrun exports TORCHELASTIC_USE_AGENT_STORE=True.  The generic
    # rendezvous helper consequently assumes that an elastic agent already
    # owns every requested TCP endpoint, including our independent update
    # ports, and waits forever for a server that does not exist.  This
    # publisher is the endpoint owner, so create its TCPStore explicitly.
    store = dist.TCPStore(
        host_name=endpoint.hostname,
        port=endpoint.port,
        world_size=world_size,
        is_master=True,
        timeout=timeout,
        multi_tenant=True,
    )
    store.set_timeout(timeout)
    store = PrefixStore(group_name, store)
    torch_version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    option_name = "backend_options" if torch_version >= (2, 6) else "pg_options"
    group, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        Backend(backend),
        store,
        group_name=group_name,
        **{option_name: None},
        timeout=timeout,
        # Do not bind an externally joined NCCL group to a device here.
        # PyTorch's device-bound split path assumes every member belongs to
        # the existing default group; serving consumers are independent
        # processes, so that assumption deadlocks the first collective.
        device_id=None,
    )
    _world.pg_group_ranks[group] = {0: 0, 1: 1}
    if warmup:
        transport_device = device if backend == "nccl" else torch.device("cpu")
        probe = torch.tensor([20260827], dtype=torch.int64, device=transport_device)
        dist.broadcast(probe, src=0, group=group)
        if int(probe.item()) != 20260827:
            raise RuntimeError(f"weight group {group_name} failed its warmup probe")
    return group


def sharded_model_state(model: nn.Module, *, context: DistributedContext) -> dict[str, Tensor]:
    """Return the live FSDP2 DTensor state without gathering or CPU offload."""

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    state = get_model_state_dict(
        model,
        options=StateDictOptions(
            full_state_dict=False,
            cpu_offload=False,
            strict=True,
        ),
    )
    if not state:
        raise RuntimeError(f"rank {context.rank} received an empty FSDP2 state")
    result: dict[str, Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise TypeError("sharded FSDP2 state contains a non-tensor entry")
        result[name] = value.detach()
    return result


@dataclass(frozen=True)
class _Entry:
    name: str
    owner: str
    tensor: Tensor
    target_dtype: torch.dtype
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    shard_dim: int | None
    transpose: bool

    @property
    def dtype(self) -> str:
        return str(self.target_dtype)

    @property
    def shape(self) -> list[int]:
        return list(self.target_shape)

    @property
    def numel(self) -> int:
        return math.prod(self.target_shape)

    @property
    def nbytes(self) -> int:
        return self.numel * self.target_dtype.itemsize

    def local_tensor(self) -> Tensor:
        to_local = getattr(self.tensor, "to_local", None)
        value = to_local() if callable(to_local) else self.tensor
        if not isinstance(value, Tensor):
            raise TypeError(f"FSDP2 state {self.name} has no tensor shard")
        return value.detach()

    def padded_numel(self, world_size: int) -> int:
        if self.shard_dim is None:
            return math.prod(self.source_shape)
        dimension = self.source_shape[self.shard_dim]
        chunk = (dimension + world_size - 1) // world_size
        trailing = math.prod(size for index, size in enumerate(self.source_shape) if index != self.shard_dim)
        return chunk * trailing

    def rank_numel(self, world_size: int, rank: int) -> int:
        if self.shard_dim is None:
            return math.prod(self.source_shape)
        dimension = self.source_shape[self.shard_dim]
        chunk = (dimension + world_size - 1) // world_size
        local_size = max(0, min(chunk, dimension - chunk * rank))
        trailing = math.prod(size for index, size in enumerate(self.source_shape) if index != self.shard_dim)
        return local_size * trailing

    def rank_shape(self, world_size: int, rank: int) -> tuple[int, ...]:
        if self.shard_dim is None:
            return self.source_shape
        shape = list(self.source_shape)
        dimension = shape[self.shard_dim]
        chunk = (dimension + world_size - 1) // world_size
        shape[self.shard_dim] = max(0, min(chunk, dimension - chunk * rank))
        return tuple(shape)


class ServingWeightPublisher:
    """Rank-zero publisher for atomic language/vision/X2V policy updates."""

    def __init__(
        self,
        *,
        base_url: str,
        master_address: str,
        base_port: int,
        backend: str,
        bucket_bytes: int,
        default_dtype: torch.dtype,
        device: torch.device,
        group_name: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.master_address = master_address
        self.base_port = int(base_port)
        self.backend = backend
        self.bucket_bytes = int(bucket_bytes)
        self.default_dtype = default_dtype
        self.device = device
        self.group_name = group_name
        self.groups: dict[str, object] = {}
        self.closure_specs: dict[str, dict[str, dict[str, object]]] | None = None

    def status(self) -> Mapping[str, object]:
        response = _request("GET", f"{self.base_url}/v1/rl/status")
        if not isinstance(response, Mapping):
            raise RuntimeError("SenseNova serving returned malformed RL status")
        return response

    @property
    def initialized(self) -> bool:
        return bool(self.groups)

    def _initialize(self) -> None:
        if self.initialized:
            return
        ports = {
            "language": self.base_port,
            "vision": self.base_port + 1,
            "x2v": self.base_port + 2,
        }
        payload = {
            "master_address": self.master_address,
            "master_port": self.base_port,
            "master_ports": ports,
            "world_size": 4,
            "group_name": self.group_name,
            "backend": self.backend,
            "warmup": True,
        }
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(
                _request,
                "POST",
                f"{self.base_url}/init_weights_update_group",
                payload=payload,
            )
            for consumer, port in ports.items():
                self.groups[consumer] = _init_custom_process_group(
                    backend=self.backend,
                    init_method=f"tcp://{self.master_address}:{port}",
                    group_name=f"{self.group_name}:{consumer}",
                    device=self.device,
                    warmup=True,
                )
            response = pending.result()
        if not isinstance(response, Mapping):
            raise RuntimeError("weight-group initialization returned malformed data")
        self.closure_specs = self._parse_closures(response)

    @staticmethod
    def _parse_closures(
        response: Mapping[str, object],
    ) -> dict[str, dict[str, dict[str, object]]]:
        raw_receipts = response.get("receipts")
        if not isinstance(raw_receipts, Mapping):
            raise RuntimeError("weight-group receipt has no consumer receipts")
        selected: dict[str, object] = {}
        for consumer in _CONSUMERS:
            raw = raw_receipts.get(consumer)
            if not isinstance(raw, Mapping):
                raise RuntimeError(f"weight-group receipt has no {consumer} consumer")
            if consumer in {"language", "vision"}:
                ranks = raw.get("ranks")
                if not isinstance(ranks, list) or not ranks or not isinstance(ranks[0], Mapping):
                    raise RuntimeError(f"weight-group {consumer} rank receipt is malformed")
                selected[consumer] = ranks[0]
            else:
                selected[consumer] = raw
        result: dict[str, dict[str, dict[str, object]]] = {}
        owners: dict[str, str] = {}
        for consumer, raw in selected.items():
            assert isinstance(raw, Mapping)
            names = raw.get("closure_names")
            specs = raw.get("closure_specs", {})
            if not isinstance(names, list) or not names or not isinstance(specs, Mapping):
                raise RuntimeError(f"weight-group {consumer} closure is malformed")
            closure: dict[str, dict[str, object]] = {}
            for name in names:
                if not isinstance(name, str) or not name:
                    raise RuntimeError(f"weight-group {consumer} closure name is invalid")
                if name in owners:
                    raise RuntimeError(f"serving tensor {name} belongs to both {owners[name]} and {consumer}")
                owners[name] = consumer
                raw_spec = specs.get(name, {})
                if not isinstance(raw_spec, Mapping):
                    raise RuntimeError(f"serving tensor {name} has an invalid spec")
                closure[name] = dict(raw_spec)
            result[consumer] = closure
        return result

    def prepare(self) -> dict[str, dict[str, dict[str, object]]]:
        """Pause serving and return its immutable parameter closure."""

        if not self.initialized:
            self._initialize()
        else:
            _request("POST", f"{self.base_url}/pause_generation")
        if self.closure_specs is None:
            raise RuntimeError("serving closure is unavailable")
        return self.closure_specs

    def commit(
        self,
        response: object,
        *,
        entries: list[_Entry],
        policy_version: str,
    ) -> Mapping[str, object]:
        """Validate control-plane ACKs after GPU transfers finish."""

        if not isinstance(response, Mapping):
            raise RuntimeError("weight update returned malformed data")
        if response.get("policy_version") != policy_version:
            raise RuntimeError("serving committed a different policy version")
        self._verify_receipts(response, entries)
        status = self.status()
        if (
            status.get("active_policy_version") != policy_version
            or status.get("pending_policy_version") is not None
            or status.get("paused") is not False
        ):
            raise RuntimeError("serving did not expose the committed active policy")
        return response

    @staticmethod
    def _verify_receipts(response: Mapping[str, object], entries: list[_Entry]) -> None:
        raw_receipts = response.get("receipts")
        if not isinstance(raw_receipts, Mapping):
            raise RuntimeError("weight update has no consumer ACK receipts")
        expected = {
            consumer: sorted(entry.name for entry in entries if entry.owner == consumer) for consumer in _CONSUMERS
        }
        for consumer in _CONSUMERS:
            raw = raw_receipts.get(consumer)
            if not isinstance(raw, Mapping):
                raise RuntimeError(f"weight update has no {consumer} ACK")
            ranks = [raw]
            if consumer in {"language", "vision"}:
                raw_ranks = raw.get("ranks")
                if not isinstance(raw_ranks, list) or not raw_ranks:
                    raise RuntimeError(f"weight update has no {consumer} rank ACKs")
                ranks = raw_ranks
            for rank_receipt in ranks:
                if not isinstance(rank_receipt, Mapping):
                    raise RuntimeError(f"weight update {consumer} ACK is malformed")
                if rank_receipt.get("received_names") != expected[consumer]:
                    raise RuntimeError(f"weight update {consumer} received-name closure differs")

    def close(self) -> None:
        if not self.initialized:
            return
        _request(
            "POST",
            f"{self.base_url}/destroy_weights_update_group",
            payload={"group_name": self.group_name},
        )
        for group in self.groups.values():
            dist.destroy_process_group(group)
        self.groups.clear()


def _primary_control(
    publisher: ServingWeightPublisher | None,
    *,
    context: DistributedContext,
) -> dict[str, object]:
    envelope: list[object | None] = [None]
    if context.is_primary:
        try:
            if publisher is None:
                raise RuntimeError("rank zero has no serving weight publisher")
            envelope[0] = {
                "ok": True,
                "value": {
                    "closures": publisher.prepare(),
                    "bucket_bytes": publisher.bucket_bytes,
                    "default_dtype": str(publisher.default_dtype),
                    "backend": publisher.backend,
                },
            }
        except BaseException as exc:  # noqa: BLE001
            envelope[0] = {
                "ok": False,
                "type": type(exc).__name__,
                "error": str(exc),
            }
    dist.broadcast_object_list(envelope, src=0)
    result = envelope[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result if isinstance(result, Mapping) else {}
        raise RuntimeError(f"rank-zero {detail.get('type', 'error')}: {detail.get('error', '')}")
    control = result.get("value")
    if not isinstance(control, Mapping):
        raise RuntimeError("serving weight control is malformed")
    return dict(control)


def _entries_from_specs(
    state: Mapping[str, Tensor],
    closures: Mapping[str, object],
    *,
    default_dtype: torch.dtype,
) -> list[_Entry]:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Replicate, Shard

    entries: list[_Entry] = []
    for owner in _CONSUMERS:
        raw_closure = closures.get(owner)
        if not isinstance(raw_closure, Mapping):
            raise RuntimeError(f"serving closure has no {owner} consumer")
        for name, raw_spec in sorted(raw_closure.items()):
            if not isinstance(name, str) or not isinstance(raw_spec, Mapping):
                raise RuntimeError(f"serving {owner} closure is malformed")
            source = state.get(name)
            if not isinstance(source, Tensor):
                raise RuntimeError(f"live FSDP2 state is missing serving tensor {name}")
            tensor = source.detach()
            source_shape = tuple(int(dimension) for dimension in tensor.shape)
            target_shape_raw = raw_spec.get("shape")
            if target_shape_raw is None:
                target_shape = source_shape
            elif not isinstance(target_shape_raw, list) or not all(
                type(dimension) is int and dimension >= 0 for dimension in target_shape_raw
            ):
                raise RuntimeError(f"serving tensor {name} has invalid geometry")
            else:
                target_shape = tuple(target_shape_raw)
            transpose = False
            if source_shape != target_shape:
                if len(source_shape) == 2 and source_shape[::-1] == target_shape:
                    transpose = True
                else:
                    raise RuntimeError(
                        f"serving tensor {name} shape mismatch: {list(source_shape)} != {list(target_shape)}"
                    )
            dtype_name = raw_spec.get("dtype")
            if dtype_name is None:
                target_dtype = default_dtype if tensor.dtype.is_floating_point else tensor.dtype
            elif not isinstance(dtype_name, str) or dtype_name not in _DTYPES:
                raise RuntimeError(f"serving tensor {name} has invalid dtype")
            else:
                target_dtype = _DTYPES[dtype_name]
            shard_dim: int | None = None
            if isinstance(tensor, DTensor):
                placements = tensor.placements
                if len(placements) != 1:
                    raise RuntimeError(f"serving tensor {name} needs a one-dimensional FSDP mesh")
                placement = placements[0]
                if isinstance(placement, Shard):
                    shard_dim = int(placement.dim)
                    if shard_dim != 0:
                        raise RuntimeError(f"serving tensor {name} is sharded on unsupported dim {shard_dim}")
                elif not isinstance(placement, Replicate):
                    raise RuntimeError(f"serving tensor {name} has unsupported placement {placement}")
            entries.append(
                _Entry(
                    name=name,
                    owner=owner,
                    tensor=tensor,
                    target_dtype=target_dtype,
                    source_shape=source_shape,
                    target_shape=target_shape,
                    shard_dim=shard_dim,
                    transpose=transpose,
                )
            )
    return entries


def _buckets(entries: list[_Entry], *, bucket_bytes: int) -> list[dict[str, object]]:
    grouped: list[list[int]] = []
    current: list[int] = []
    current_owner = current_dtype = None
    current_bytes = 0
    for index, entry in enumerate(entries):
        split = bool(current) and (
            entry.owner != current_owner or entry.dtype != current_dtype or current_bytes + entry.nbytes > bucket_bytes
        )
        if split:
            grouped.append(current)
            current, current_bytes = [], 0
        current_owner, current_dtype = entry.owner, entry.dtype
        current.append(index)
        current_bytes += entry.nbytes
    if current:
        grouped.append(current)
    result: list[dict[str, object]] = []
    for bucket_index, indices in enumerate(grouped):
        owner = entries[indices[0]].owner
        result.append(
            {
                "id": f"bucket-{bucket_index:05d}",
                "dtype": entries[indices[0]].dtype,
                "numel": sum(entries[index].numel for index in indices),
                "entry_indices": indices,
                "consumers": [owner],
            }
        )
    return result


def _update_payload(
    entries: list[_Entry],
    buckets: list[dict[str, object]],
    *,
    group_name: str,
    policy_version: str,
) -> dict[str, object]:
    return {
        "names": [entry.name for entry in entries],
        "dtypes": [entry.dtype for entry in entries],
        "shapes": [entry.shape for entry in entries],
        "assignments": {entry.name: [entry.owner] for entry in entries},
        "required": {consumer: [entry.name for entry in entries if entry.owner == consumer] for consumer in _CONSUMERS},
        "policy_version": policy_version,
        "group_name": group_name,
        "buckets": buckets,
        "full_update": True,
    }


def _training_transfer_groups(context: DistributedContext) -> dict[str, object]:
    key = (context.world_size, context.local_rank)
    groups = _TRAINING_TRANSFER_GROUPS.get(key)
    if groups is not None:
        return groups
    ranks = list(range(context.world_size))
    groups = {consumer: dist.new_group(ranks=ranks, backend="nccl") for consumer in _CONSUMERS}
    _TRAINING_TRANSFER_GROUPS[key] = groups
    return groups


def _pack_local_bucket(
    bucket: Mapping[str, object],
    entries: list[_Entry],
    *,
    context: DistributedContext,
) -> tuple[Tensor, list[tuple[int, int]]]:
    raw_indices = bucket.get("entry_indices")
    if not isinstance(raw_indices, list) or not raw_indices:
        raise RuntimeError("serving bucket has no entries")
    pieces: list[Tensor] = []
    layout: list[tuple[int, int]] = []
    for raw_index in raw_indices:
        index = int(raw_index)
        entry = entries[index]
        padded_numel = entry.padded_numel(context.world_size)
        actual_numel = entry.rank_numel(context.world_size, context.rank)
        local = entry.local_tensor().to(dtype=entry.target_dtype).contiguous().view(-1)
        if entry.shard_dim is None:
            if local.numel() != actual_numel:
                raise RuntimeError(f"replicated serving tensor {entry.name} has wrong local size")
        elif local.numel() == padded_numel:
            local = local[:actual_numel]
        elif local.numel() != actual_numel:
            raise RuntimeError(
                f"serving tensor {entry.name} shard has {local.numel()} elements, "
                f"expected {actual_numel} or padded {padded_numel}"
            )
        if actual_numel < padded_numel:
            padded = torch.zeros(
                padded_numel,
                dtype=entry.target_dtype,
                device=context.device,
            )
            if actual_numel:
                padded[:actual_numel].copy_(local)
            local = padded
        pieces.append(local)
        layout.append((index, padded_numel))
    return torch.cat(pieces), layout


def _rebuild_full_bucket(
    gathered: Tensor,
    layout: list[tuple[int, int]],
    entries: list[_Entry],
    *,
    world_size: int,
) -> Tensor:
    per_rank_numel = sum(padded_numel for _, padded_numel in layout)
    shards = gathered.view(world_size, per_rank_numel)
    output: list[Tensor] = []
    offset = 0
    for index, padded_numel in layout:
        entry = entries[index]
        if entry.shard_dim is None:
            source = shards[0, offset : offset + entry.numel].view(entry.source_shape)
        else:
            parts: list[Tensor] = []
            for rank in range(world_size):
                actual_numel = entry.rank_numel(world_size, rank)
                part = shards[rank, offset : offset + actual_numel]
                parts.append(part.view(entry.rank_shape(world_size, rank)))
            source = torch.cat(parts, dim=entry.shard_dim)
        target = source.t() if entry.transpose else source
        if tuple(target.shape) != entry.target_shape:
            raise RuntimeError(f"rebuilt serving tensor {entry.name} has wrong shape")
        output.append(target.contiguous().view(-1))
        offset += padded_numel
    return torch.cat(output)


def _send_sharded_consumer(
    consumer: str,
    buckets: list[dict[str, object]],
    entries: list[_Entry],
    *,
    context: DistributedContext,
    training_group: object,
    serving_group: object | None,
) -> None:
    torch.cuda.set_device(context.device)
    stream = torch.cuda.Stream(device=context.device)
    with torch.cuda.stream(stream):
        for bucket in buckets:
            if consumer not in bucket["consumers"]:
                continue
            local, layout = _pack_local_bucket(bucket, entries, context=context)
            gathered = torch.empty(
                local.numel() * context.world_size,
                dtype=local.dtype,
                device=context.device,
            )
            work = dist.all_gather_into_tensor(
                gathered,
                local,
                group=training_group,
                async_op=True,
            )
            work.wait()
            if context.is_primary:
                if serving_group is None:
                    raise RuntimeError(f"rank zero has no {consumer} serving group")
                flat = _rebuild_full_bucket(
                    gathered,
                    layout,
                    entries,
                    world_size=context.world_size,
                )
                if flat.numel() != int(bucket["numel"]):
                    raise RuntimeError(f"serving bucket {bucket['id']} has wrong size")
                transfer = dist.broadcast(
                    flat,
                    src=0,
                    group=serving_group,
                    async_op=True,
                )
                transfer.wait()
                del flat
            del gathered, local
    stream.synchronize()


def publish_sharded_model_state(
    model: nn.Module,
    *,
    publisher: ServingWeightPublisher | None,
    policy_version: str,
    context: DistributedContext,
) -> Mapping[str, object]:
    """Collectively stream live FSDP2 GPU shards into serving consumers."""

    if not policy_version:
        raise ValueError("published policy version must be non-empty")
    control = _primary_control(publisher, context=context)
    closures = control.get("closures")
    dtype_name = control.get("default_dtype")
    bucket_bytes = control.get("bucket_bytes")
    backend = control.get("backend")
    if not isinstance(closures, Mapping):
        raise RuntimeError("serving closure control is malformed")
    if not isinstance(dtype_name, str) or dtype_name not in _DTYPES:
        raise RuntimeError("serving default dtype control is malformed")
    if type(bucket_bytes) is not int or bucket_bytes < 1:
        raise RuntimeError("serving bucket size control is malformed")
    if backend != "nccl":
        raise RuntimeError("direct sharded weight publication requires NCCL")

    state = sharded_model_state(model, context=context)
    try:
        entries = _entries_from_specs(
            state,
            closures,
            default_dtype=_DTYPES[dtype_name],
        )
        buckets = _buckets(entries, bucket_bytes=bucket_bytes)
        training_groups = _training_transfer_groups(context)
        worker_count = 4 if context.is_primary else 3
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            pending = None
            if context.is_primary:
                if publisher is None:
                    raise RuntimeError("rank zero has no serving weight publisher")
                payload = _update_payload(
                    entries,
                    buckets,
                    group_name=publisher.group_name,
                    policy_version=policy_version,
                )
                pending = executor.submit(
                    _request,
                    "POST",
                    f"{publisher.base_url}/update_weights_from_distributed",
                    payload=payload,
                )
            transfers = [
                executor.submit(
                    _send_sharded_consumer,
                    consumer,
                    buckets,
                    entries,
                    context=context,
                    training_group=training_groups[consumer],
                    serving_group=(
                        publisher.groups[consumer] if context.is_primary and publisher is not None else None
                    ),
                )
                for consumer in _CONSUMERS
            ]
            for transfer in transfers:
                transfer.result()
            response = pending.result() if pending is not None else None
        envelope: list[object | None] = [None]
        if context.is_primary:
            assert publisher is not None
            try:
                envelope[0] = {
                    "ok": True,
                    "value": publisher.commit(
                        response,
                        entries=entries,
                        policy_version=policy_version,
                    ),
                }
            except BaseException as exc:  # noqa: BLE001
                envelope[0] = {
                    "ok": False,
                    "type": type(exc).__name__,
                    "error": str(exc),
                }
        dist.broadcast_object_list(envelope, src=0)
        result = envelope[0]
        if not isinstance(result, Mapping) or result.get("ok") is not True:
            detail = result if isinstance(result, Mapping) else {}
            raise RuntimeError(f"rank-zero {detail.get('type', 'error')}: {detail.get('error', '')}")
        value = result.get("value")
        if not isinstance(value, Mapping):
            raise RuntimeError("serving weight publication returned malformed data")
        return value
    finally:
        state.clear()


def policy_version_for_step(initial: str, plan_digest: str, step: int) -> str:
    if step < 1:
        raise ValueError("published policy step must be positive")
    return f"{initial}-train-{plan_digest[:12]}-step-{step:08d}"


def policy_version_for_initial(initial: str, plan_digest: str) -> str:
    return f"{initial}-train-{plan_digest[:12]}-initial"


__all__ = [
    "ServingWeightPublisher",
    "policy_version_for_initial",
    "policy_version_for_step",
    "publish_sharded_model_state",
    "sharded_model_state",
]
