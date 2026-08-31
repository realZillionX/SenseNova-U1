"""LightLLM + LightX2V rollout transport for SenseNova RLVR.

The serving engine owns stochastic sampling and returns detached behavior-policy
likelihoods.  Forge owns differentiable replay and optimization while callers
own prompt and reward semantics.
Only rank zero talks to the service. Image SDE bundles arrive as raw
safetensors WebSocket frames, stay in rank-zero CPU memory, and are broadcast
as tensors over the existing training process group; the lightweight object
envelope contains geometry only.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import mimetypes
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import torch
import torch.distributed as dist
from PIL import Image
from safetensors.torch import load as load_safetensors

from .policy_runtime import (
    ImageEvent,
    TextEvent,
    U15PolicyRollout,
)
from .rollout import (
    ImageSdeTrace,
    TextRolloutTrace,
)
from .types import SYSTEM_MESSAGE_BY_MODALITY, CandidateResponse, ImageSegment, TextSegment

_BUNDLE_ID = re.compile(r"^[a-f0-9]{32}$")
_DATA_IMAGE = re.compile(
    r"\Adata:image/(?P<subtype>[A-Za-z0-9.+-]+);base64,(?P<data>.*)\Z",
    re.DOTALL,
)
_SAFE_KEY = re.compile(r"[^A-Za-z0-9_.-]+")
_TRACE_KEYS = frozenset(
    {
        "samples",
        "next_samples",
        "old_means",
        "old_log_probs",
        "timesteps",
        "next_timesteps",
        "scales",
        "indices",
        "final_latent",
        "sigma_max",
        "noise_level",
    }
)
_IMAGE_SUFFIX = {
    "jpeg": ".jpg",
    "jpg": ".jpg",
    "png": ".png",
    "webp": ".webp",
}


def _first_input_geometry(prompt_images: Sequence[str]) -> tuple[int, int]:
    """Return ``(height, width)`` from the first immutable prompt image."""

    if not prompt_images:
        raise ValueError("TI2TI first-input resolution requires a prompt image")
    path = Path(prompt_images[0]).expanduser().resolve()
    try:
        with Image.open(path) as image:
            width, height = image.size
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read first prompt image geometry: {path}") from exc
    if width < 1 or height < 1:
        raise ValueError(f"first prompt image has invalid geometry: {path}")
    return int(height), int(width)


class RlApiTransport(Protocol):
    def post_json(
        self,
        url: str,
        payload: Mapping[str, object],
        *,
        timeout: float,
    ) -> Mapping[str, object]: ...

    def stream_traces(
        self,
        url: str,
        bundle_ids: Sequence[str],
        *,
        timeout: float,
        maximum_bytes: int,
    ) -> Mapping[str, bytes]: ...


@dataclass(frozen=True)
class UrllibRlApiTransport:
    """Dependency-free bounded HTTP transport used inside the trainer."""

    maximum_json_bytes: int = 512 * 1024 * 1024

    @staticmethod
    def _request(request: Request, *, timeout: float):
        try:
            return urlopen(request, timeout=timeout)  # noqa: S310 - sealed URL
        except HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(f"SenseNova RL API returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"SenseNova RL API request failed: {exc}") from exc

    def post_json(
        self,
        url: str,
        payload: Mapping[str, object],
        *,
        timeout: float,
    ) -> Mapping[str, object]:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._request(request, timeout=timeout) as response:
            raw = response.read(self.maximum_json_bytes + 1)
        if len(raw) > self.maximum_json_bytes:
            raise RuntimeError("SenseNova RL API JSON response exceeds its bound")
        try:
            decoded = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("SenseNova RL API returned malformed JSON") from exc
        if not isinstance(decoded, Mapping):
            raise RuntimeError("SenseNova RL API response must be an object")
        return decoded

    def stream_traces(
        self,
        url: str,
        bundle_ids: Sequence[str],
        *,
        timeout: float,
        maximum_bytes: int,
    ) -> Mapping[str, bytes]:
        """Receive one rollout group's traces over a raw binary WebSocket."""

        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            raise RuntimeError("SenseNova direct SDE transport requires websockets") from exc
        with connect(
            url,
            open_timeout=timeout,
            close_timeout=min(timeout, 10.0),
            # This local side-channel continuously transfers multi-gigabyte
            # tensor streams.  Protocol keepalive can time out while the
            # server is busy writing binary frames, and per-message deflate
            # only burns CPU on safetensors payloads.
            ping_interval=None,
            compression=None,
            max_size=None,
        ) as connection:
            connection.send(
                json.dumps(
                    {"bundle_ids": list(bundle_ids)},
                    separators=(",", ":"),
                )
            )
            hello = self._websocket_json(connection.recv(), label="stream preamble")
            if hello.get("schema") != "mova.rl.sde_stream.v1" or hello.get("bundle_count") != len(bundle_ids):
                raise RuntimeError("SenseNova SDE stream preamble is invalid")
            result: dict[str, bytes] = {}
            total = 0
            for expected_id in bundle_ids:
                header = self._websocket_json(connection.recv(), label="trace header")
                if header.get("bundle_id") != expected_id:
                    raise RuntimeError("SenseNova SDE stream reordered trace bundles")
                size = header.get("size")
                if type(size) is not int or size <= 0:
                    raise RuntimeError("SenseNova SDE stream has invalid trace size")
                total += size
                if size > maximum_bytes or total > maximum_bytes * len(bundle_ids):
                    raise RuntimeError("SenseNova SDE stream exceeds its byte bound")
                received = bytearray()
                while len(received) < size:
                    frame = connection.recv()
                    if not isinstance(frame, bytes):
                        raise RuntimeError("SenseNova SDE stream expected a binary frame")
                    if len(received) + len(frame) > size:
                        raise RuntimeError("SenseNova SDE stream exceeded declared size")
                    received.extend(frame)
                result[expected_id] = bytes(received)
            complete = self._websocket_json(connection.recv(), label="stream completion")
            if complete != {"complete": True}:
                raise RuntimeError("SenseNova SDE stream did not complete cleanly")
            return result

    @staticmethod
    def _websocket_json(value: object, *, label: str) -> Mapping[str, object]:
        if not isinstance(value, str):
            raise RuntimeError(f"SenseNova SDE {label} is not JSON text")
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"SenseNova SDE {label} is malformed") from exc
        if not isinstance(decoded, Mapping):
            raise RuntimeError(f"SenseNova SDE {label} must be an object")
        if "error" in decoded:
            raise RuntimeError(f"SenseNova SDE stream failed: {decoded['error']}")
        return decoded


def _endpoint(base_url: str, suffix: str) -> str:
    parsed = urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("SenseNova RL API base URL must be absolute http(s)")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("SenseNova RL API URL must not embed credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("SenseNova RL API URL must not contain query or fragment")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, f"{path}{suffix}", "", ""))


def _websocket_endpoint(base_url: str, suffix: str) -> str:
    parsed = urlsplit(_endpoint(base_url, suffix))
    scheme = {"http": "ws", "https": "wss"}[parsed.scheme]
    return urlunsplit((scheme, parsed.netloc, parsed.path, "", ""))


def _image_data_url(path: str) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"SenseNova RL prompt image does not exist: {source}")
    mime = mimetypes.guess_type(source.name)[0]
    if mime is None or not mime.startswith("image/"):
        raise ValueError(f"unsupported SenseNova RL prompt image: {source}")
    return f"data:{mime};base64,{base64.b64encode(source.read_bytes()).decode('ascii')}"


def _messages(
    prompt: str,
    prompt_images: Sequence[str],
    modality: str,
    system_message: str | None = None,
) -> list[dict[str, object]]:
    if modality not in SYSTEM_MESSAGE_BY_MODALITY:
        raise ValueError(f"unsupported SenseNova RL modality {modality!r}")
    content: list[dict[str, object]] = [
        {"type": "image_url", "image_url": {"url": _image_data_url(path)}} for path in prompt_images
    ]
    content.append({"type": "text", "text": prompt})
    return [
        {
            "role": "system",
            "content": SYSTEM_MESSAGE_BY_MODALITY[modality] if system_message is None else system_message,
        },
        {"role": "user", "content": content},
    ]


def _decode_image(value: object) -> tuple[bytes, str]:
    if not isinstance(value, str):
        raise ValueError("SenseNova RL image event is not a data URL")
    match = _DATA_IMAGE.fullmatch(value)
    if match is None:
        raise ValueError("SenseNova RL image event is not an inline image")
    subtype = match.group("subtype").lower()
    if subtype not in _IMAGE_SUFFIX:
        raise ValueError(f"unsupported generated image subtype {subtype!r}")
    try:
        payload = base64.b64decode(match.group("data"), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("generated image has malformed base64") from exc
    if not payload:
        raise ValueError("generated image is empty")
    return payload, _IMAGE_SUFFIX[subtype]


def _safe_key(value: str) -> str:
    result = _SAFE_KEY.sub("-", value).strip("-._")
    return result[:96] or "rollout"


@dataclass
class SenseNovaRlApiClient:
    """Group rollout client whose wire payload stays cheap to broadcast."""

    base_url: str
    expected_policy_version: str
    timeout_seconds: float = 3600.0
    maximum_trace_bytes: int = 1024 * 1024 * 1024
    transport: RlApiTransport = UrllibRlApiTransport()

    def __post_init__(self) -> None:
        self.rollout_url = _endpoint(self.base_url, "/v1/rl/rollouts")
        self.trace_stream_url = _websocket_endpoint(self.base_url, "/v1/rl/traces/ws")
        if not self.expected_policy_version:
            raise ValueError("expected_policy_version must be non-empty")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        if self.maximum_trace_bytes < 1:
            raise ValueError("maximum_trace_bytes must be positive")
        self._pending_traces: dict[str, dict[str, torch.Tensor]] = {}

    def generate_group_payload(
        self,
        *,
        prompt: str,
        prompt_images: tuple[str, ...],
        modality: str,
        system_message: str | None = None,
        seeds: Sequence[int],
        artifact_dir: Path,
        rollout_key: str,
        max_sequence_length: int,
        max_new_tokens: int,
        max_images: int,
        image_resolution_mode: str,
        image_steps: int,
        image_noise_level: float,
        timestep_shift: float,
        t_eps: float,
        sde_window_start: int,
        sde_window_end: int,
        sde_window_steps: int,
    ) -> tuple[dict[str, object], ...]:
        """Request one prompt group and persist only batch-scoped artifacts."""

        normalized_seeds = [int(seed) for seed in seeds]
        if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
            raise ValueError("a rollout group requires distinct seeds")
        request: dict[str, object] = {
            "expected_policy_version": self.expected_policy_version,
            "modality": modality,
            "messages": _messages(prompt, prompt_images, modality, system_message),
            "seeds": normalized_seeds,
            "max_sequence_length": int(max_sequence_length),
            "max_new_tokens": int(max_new_tokens),
            "max_images": int(max_images if modality == "ti2ti" else 0),
            "temperature": 1.0,
            "top_p": 1.0,
        }
        if modality == "ti2ti":
            if image_resolution_mode != "first_input":
                raise ValueError("RL image resolution must follow the first prompt image")
            image_height, image_width = _first_input_geometry(prompt_images)
            request["image_policy"] = {
                "height": image_height,
                "width": image_width,
                "image_steps": int(image_steps),
                "timestep_shift": float(timestep_shift),
                "t_eps": float(t_eps),
                "image_noise_level": float(image_noise_level),
                "sde_window_start": int(sde_window_start),
                "sde_window_end": int(sde_window_end),
                "sde_selected_steps": int(sde_window_steps),
            }
        started = time.perf_counter()
        response = self.transport.post_json(
            self.rollout_url,
            request,
            timeout=self.timeout_seconds,
        )
        if response.get("policy_version") != self.expected_policy_version:
            raise RuntimeError("SenseNova RL API crossed the policy-version barrier")
        if response.get("modality") != modality:
            raise RuntimeError("SenseNova RL API returned a different modality")
        raw_rollouts = response.get("rollouts")
        if isinstance(raw_rollouts, (str, bytes)) or not isinstance(raw_rollouts, Sequence):
            raise RuntimeError("SenseNova RL API response has no rollout list")
        if len(raw_rollouts) != len(normalized_seeds):
            raise RuntimeError("SenseNova RL API changed the requested group size")
        bundle_ids: list[str] = []
        for raw in raw_rollouts:
            if not isinstance(raw, Mapping):
                raise RuntimeError("SenseNova RL API returned a malformed rollout")
            events = raw.get("events")
            if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
                raise RuntimeError("SenseNova RL API returned malformed events")
            for event in events:
                if isinstance(event, Mapping) and event.get("type") == "image":
                    bundle_id = event.get("trace_bundle_key")
                    if not isinstance(bundle_id, str) or _BUNDLE_ID.fullmatch(bundle_id) is None:
                        raise RuntimeError("SenseNova RL image event has invalid trace key")
                    bundle_ids.append(bundle_id)
        if len(set(bundle_ids)) != len(bundle_ids):
            raise RuntimeError("SenseNova RL API reused an SDE trace key")
        trace_payloads = (
            self.transport.stream_traces(
                self.trace_stream_url,
                bundle_ids,
                timeout=self.timeout_seconds,
                maximum_bytes=self.maximum_trace_bytes,
            )
            if bundle_ids
            else {}
        )
        if set(trace_payloads) != set(bundle_ids):
            raise RuntimeError("SenseNova SDE stream returned a different trace closure")
        elapsed = time.perf_counter() - started
        trace_wire_bytes = sum(len(payload) for payload in trace_payloads.values())
        print(
            json.dumps(
                {
                    "component": "sensenova_u1.rl",
                    "event": "rollout_group_transport",
                    "base_url": self.base_url,
                    "rollouts": len(normalized_seeds),
                    "generated_images": len(bundle_ids),
                    "trace_wire_bytes": trace_wire_bytes,
                    "seconds": elapsed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        trace_tensors: dict[str, dict[str, torch.Tensor]] = {}
        for bundle_id in bundle_ids:
            try:
                tensors = load_safetensors(trace_payloads[bundle_id])
            except Exception as exc:
                raise RuntimeError(f"SenseNova SDE stream returned invalid safetensors: {bundle_id}") from exc
            if set(tensors) != _TRACE_KEYS:
                raise RuntimeError(
                    "SenseNova RL trace keys differ: "
                    f"missing={sorted(_TRACE_KEYS - set(tensors))}, "
                    f"extra={sorted(set(tensors) - _TRACE_KEYS)}"
                )
            trace_tensors[bundle_id] = tensors
        artifact_dir.mkdir(parents=True, exist_ok=True)
        per_rollout_seconds = elapsed / len(normalized_seeds)
        decoded = []
        for position, (seed, raw) in enumerate(zip(normalized_seeds, raw_rollouts, strict=True)):
            if not isinstance(raw, Mapping) or raw.get("seed") != seed:
                raise RuntimeError("SenseNova RL API reordered or malformed a rollout")
            decoded.append(
                self._persist_rollout(
                    raw,
                    modality=modality,
                    artifact_dir=artifact_dir,
                    key=f"{_safe_key(rollout_key)}-r{position:03d}",
                    seconds=per_rollout_seconds,
                    trace_tensors=trace_tensors,
                    max_sequence_length=max_sequence_length,
                )
            )
        return tuple(decoded)

    def _persist_rollout(
        self,
        raw: Mapping[str, object],
        *,
        modality: str,
        artifact_dir: Path,
        key: str,
        seconds: float,
        trace_tensors: Mapping[str, Mapping[str, torch.Tensor]],
        max_sequence_length: int,
    ) -> dict[str, object]:
        events = raw.get("events")
        usage = raw.get("usage")
        if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
            raise RuntimeError("SenseNova RL rollout has no event list")
        if not isinstance(usage, Mapping):
            raise RuntimeError("SenseNova RL rollout has no usage object")
        usage_fields = {
            name: usage.get(name)
            for name in (
                "prompt_tokens",
                "completion_tokens",
                "image_count",
                "image_context_tokens",
                "sequence_tokens",
            )
        }
        if any(type(value) is not int or value < 0 for value in usage_fields.values()):
            raise RuntimeError("SenseNova RL rollout has invalid sequence usage")
        if usage_fields["prompt_tokens"] < 1:
            raise RuntimeError("SenseNova RL rollout has no prompt tokens")
        expected_sequence_tokens = (
            usage_fields["prompt_tokens"]
            + usage_fields["completion_tokens"]
            + usage_fields["image_context_tokens"]
        )
        if (
            usage_fields["sequence_tokens"] != expected_sequence_tokens
            or usage_fields["sequence_tokens"] > max_sequence_length
        ):
            raise RuntimeError("SenseNova RL rollout exceeded max_sequence_length")
        wire_events: list[dict[str, object]] = []
        text_tokens = 0
        image_count = 0
        for event_index, event in enumerate(events):
            if not isinstance(event, Mapping):
                raise RuntimeError("SenseNova RL event must be an object")
            event_type = event.get("type")
            if event_type == "text":
                token_ids = event.get("token_ids")
                log_probs = event.get("selected_token_logprobs")
                response_mask = event.get("response_mask")
                if not all(
                    isinstance(value, Sequence) and not isinstance(value, (str, bytes))
                    for value in (token_ids, log_probs, response_mask)
                ):
                    raise RuntimeError("SenseNova RL text trace columns are malformed")
                if not token_ids or not (len(token_ids) == len(log_probs) == len(response_mask)):
                    raise RuntimeError("SenseNova RL text trace lengths differ")
                if not all(type(value) is int for value in token_ids):
                    raise RuntimeError("SenseNova RL token IDs must be integers")
                if not all(type(value) is bool for value in response_mask):
                    raise RuntimeError("SenseNova RL response mask must be boolean")
                values = [float(value) for value in log_probs]
                if not all(math.isfinite(value) and value <= 0 for value in values):
                    raise RuntimeError("SenseNova RL token logprobs are invalid")
                stop = event.get("stop_token")
                if stop is not None and type(stop) is not int:
                    raise RuntimeError("SenseNova RL stop token must be int or null")
                text = event.get("text")
                if not isinstance(text, str):
                    raise RuntimeError("SenseNova RL text event has no text")
                wire_events.append(
                    {
                        "type": "text",
                        "text": text,
                        "token_ids": list(token_ids),
                        "old_log_probs": values,
                        "response_mask": list(response_mask),
                        "stopped": stop is not None,
                        "stop_token_id": stop,
                    }
                )
                text_tokens += sum(bool(value) for value in response_mask)
                continue
            if event_type != "image" or modality != "ti2ti":
                raise RuntimeError(f"unexpected SenseNova RL event type {event_type!r}")
            bundle_id = event.get("trace_bundle_key")
            if not isinstance(bundle_id, str) or _BUNDLE_ID.fullmatch(bundle_id) is None:
                raise RuntimeError("SenseNova RL image event has invalid trace key")
            payload, suffix = _decode_image(event.get("image"))
            image_path = artifact_dir / f"{key}-image-{image_count:02d}{suffix}"
            if image_path.exists():
                raise RuntimeError("SenseNova RL artifact would overwrite an existing file")
            image_path.write_bytes(payload)
            tensors = trace_tensors.get(bundle_id)
            if not isinstance(tensors, Mapping) or set(tensors) != _TRACE_KEYS:
                image_path.unlink(missing_ok=True)
                raise RuntimeError("SenseNova SDE stream omitted a trace payload")
            if bundle_id in self._pending_traces:
                image_path.unlink(missing_ok=True)
                raise RuntimeError("SenseNova SDE trace is already pending materialization")
            owned = {name: tensor.detach().cpu() for name, tensor in tensors.items()}
            geometry = event.get("sde_geometry")
            if not isinstance(geometry, Mapping):
                image_path.unlink(missing_ok=True)
                raise RuntimeError("SenseNova RL image event has no SDE geometry")
            image_height = geometry.get("height")
            image_width = geometry.get("width")
            if type(image_height) is not int or type(image_width) is not int or image_height <= 0 or image_width <= 0:
                image_path.unlink(missing_ok=True)
                raise RuntimeError("SenseNova RL image event has invalid dimensions")
            self._pending_traces[bundle_id] = owned
            wire_events.append(
                {
                    "type": "image",
                    "path": str(image_path.resolve()),
                    "trace_id": bundle_id,
                    "image_height": image_height,
                    "image_width": image_width,
                    "trace_manifest": {
                        name: {
                            "shape": list(tensor.shape),
                            "dtype": str(tensor.dtype),
                        }
                        for name, tensor in owned.items()
                    },
                }
            )
            image_count += 1
        if text_tokens != usage_fields["completion_tokens"] or image_count != usage_fields["image_count"]:
            raise RuntimeError("SenseNova RL rollout usage disagrees with its event trace")
        if not wire_events or not any(event["type"] == "text" for event in wire_events):
            raise RuntimeError("SenseNova RL rollout emitted no text policy action")
        finish_reason = raw.get("finish_reason")
        if finish_reason not in {"stop", "length"}:
            raise RuntimeError(f"SenseNova RL rollout has invalid finish_reason {finish_reason!r}")
        return {
            "events": wire_events,
            "text_tokens": text_tokens,
            "generated_images": image_count,
            "seconds": float(seconds),
            "finish_reason": finish_reason,
        }

    def materialize_group(
        self,
        payload: object,
        *,
        modality: str,
        device: torch.device,
    ) -> tuple[U15PolicyRollout, ...]:
        if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
            raise RuntimeError("distributed SenseNova RL group payload is malformed")
        return tuple(
            self._materialize_rollout(
                item,
                modality=modality,
                device=device,
                trace_owner_rank=None,
            )
            for item in payload
        )

    def materialize_group_for_rank(
        self,
        payload: object,
        *,
        modality: str,
        device: torch.device,
        owner_rank: int,
        consume_traces: bool = True,
    ) -> tuple[U15PolicyRollout, ...]:
        """Materialize one prompt group only on its data-parallel owner.

        Every rank enters this method in the same prompt/event order.  Text
        metadata remains an object broadcast, while each large SDE tensor is
        transferred point-to-point from rank zero to the one rank that will
        replay it.  Non-owners never allocate the image trace.
        """

        if not dist.is_initialized():
            raise RuntimeError("rank-owned rollout materialization requires distributed")
        if not 0 <= owner_rank < dist.get_world_size():
            raise ValueError("prompt-group owner rank is outside the FSDP world")
        if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
            raise RuntimeError("distributed SenseNova RL group payload is malformed")
        owned: list[U15PolicyRollout] = []
        for item in payload:
            rollout = self._materialize_rollout(
                item,
                modality=modality,
                device=device,
                trace_owner_rank=owner_rank,
                consume_traces=consume_traces,
            )
            if rollout is not None:
                owned.append(rollout)
        if dist.get_rank() == owner_rank and len(owned) != len(payload):
            raise RuntimeError("prompt-group owner lost a rollout during materialization")
        if dist.get_rank() != owner_rank and owned:
            raise RuntimeError("non-owner retained a prompt-group rollout")
        return tuple(owned)

    def release_payload_traces(self, payload_groups: Sequence[Sequence[object]]) -> None:
        """Release traces retained while one rollout serves padding owners."""

        if dist.is_initialized() and dist.get_rank() != 0:
            return
        trace_ids: list[str] = []
        for payload in payload_groups:
            for raw in payload:
                if not isinstance(raw, Mapping):
                    raise RuntimeError("distributed SenseNova RL rollout is malformed")
                events = raw.get("events")
                if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
                    raise RuntimeError("distributed SenseNova RL events are malformed")
                for event in events:
                    if isinstance(event, Mapping) and event.get("type") == "image":
                        trace_id = event.get("trace_id")
                        if not isinstance(trace_id, str):
                            raise RuntimeError("distributed SenseNova trace id is invalid")
                        trace_ids.append(trace_id)
        for trace_id in trace_ids:
            if self._pending_traces.pop(trace_id, None) is None:
                raise RuntimeError(f"rank zero has no retained SenseNova SDE trace {trace_id}")

    def scoring_group(
        self,
        payload: object,
        *,
        modality: str,
    ) -> tuple[U15PolicyRollout, ...]:
        """Build rank-zero verifier candidates without loading policy traces."""

        if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
            raise RuntimeError("distributed SenseNova RL group payload is malformed")
        rollouts: list[U15PolicyRollout] = []
        for raw in payload:
            if not isinstance(raw, Mapping):
                raise RuntimeError("distributed SenseNova RL rollout is malformed")
            raw_events = raw.get("events")
            if isinstance(raw_events, (str, bytes)) or not isinstance(raw_events, Sequence):
                raise RuntimeError("distributed SenseNova RL events are malformed")
            items: list[TextSegment | ImageSegment] = []
            for event in raw_events:
                if not isinstance(event, Mapping):
                    raise RuntimeError("distributed SenseNova RL event is malformed")
                if event.get("type") == "text":
                    value = event.get("text")
                    if isinstance(value, str) and value:
                        items.append(TextSegment(value))
                elif event.get("type") == "image":
                    items.append(ImageSegment(str(Path(str(event["path"])).resolve())))
                else:
                    raise RuntimeError("distributed SenseNova RL event type is invalid")
            rollouts.append(
                U15PolicyRollout(
                    candidate=CandidateResponse(modality=modality, items=tuple(items)),
                    events=(),
                    text_tokens=int(raw["text_tokens"]),
                    generated_images=int(raw["generated_images"]),
                    seconds=float(raw["seconds"]),
                    finish_reason=str(raw.get("finish_reason") or "stop"),
                )
            )
        return tuple(rollouts)

    def _materialize_rollout(
        self,
        raw: object,
        *,
        modality: str,
        device: torch.device,
        trace_owner_rank: int | None,
        consume_traces: bool = True,
    ) -> U15PolicyRollout | None:
        if not isinstance(raw, Mapping):
            raise RuntimeError("distributed SenseNova RL rollout is malformed")
        raw_events = raw.get("events")
        if isinstance(raw_events, (str, bytes)) or not isinstance(raw_events, Sequence):
            raise RuntimeError("distributed SenseNova RL events are malformed")
        events: list[TextEvent | ImageEvent] = []
        items: list[TextSegment | ImageSegment] = []
        active = trace_owner_rank is None or not dist.is_initialized() or dist.get_rank() == trace_owner_rank
        for raw_event in raw_events:
            if not isinstance(raw_event, Mapping):
                raise RuntimeError("distributed SenseNova RL event is malformed")
            if raw_event.get("type") == "text":
                if not active:
                    continue
                token_ids = torch.tensor([raw_event["token_ids"]], dtype=torch.long, device=device)
                old = torch.tensor([raw_event["old_log_probs"]], dtype=torch.float32, device=device)
                mask = torch.tensor([raw_event["response_mask"]], dtype=torch.bool, device=device)
                trace = TextRolloutTrace(
                    token_ids=token_ids,
                    old_log_probs=old,
                    response_mask=mask,
                    stopped=torch.tensor([bool(raw_event["stopped"])], dtype=torch.bool, device=device),
                )
                events.append(
                    TextEvent(
                        trace=trace,
                        stop_token_id=(
                            None if raw_event.get("stop_token_id") is None else int(raw_event["stop_token_id"])
                        ),
                    )
                )
                text = raw_event.get("text")
                if isinstance(text, str) and text:
                    items.append(TextSegment(text))
                continue
            if raw_event.get("type") != "image":
                raise RuntimeError("distributed SenseNova RL event type is invalid")
            trace_id = raw_event.get("trace_id")
            manifest = raw_event.get("trace_manifest")
            if not isinstance(trace_id, str) or not isinstance(manifest, Mapping):
                raise RuntimeError("distributed SenseNova SDE trace metadata is invalid")
            tensors = self._distribute_trace(
                trace_id,
                manifest,
                device=device,
                owner_rank=trace_owner_rank,
                consume_trace=consume_traces,
            )
            if not active:
                continue
            trace = ImageSdeTrace(
                samples=tensors["samples"].to(device),
                prev_samples=tensors["next_samples"].to(device),
                old_log_probs=tensors["old_log_probs"].float().to(device),
                old_means=tensors["old_means"].to(device),
                timesteps=tensors["timesteps"].float().to(device),
                next_timesteps=tensors["next_timesteps"].float().to(device),
                sde_indices=tensors["indices"].long().to(device),
                final_sample=tensors["final_latent"].to(device),
                sigma_max=float(tensors["sigma_max"].item()),
                noise_level=float(tensors["noise_level"].item()),
                image_height=int(raw_event["image_height"]),
                image_width=int(raw_event["image_width"]),
            )
            events.append(ImageEvent(trace=trace))
            items.append(ImageSegment(str(Path(str(raw_event["path"])).resolve())))
        if not active:
            return None
        return U15PolicyRollout(
            candidate=CandidateResponse(modality=modality, items=tuple(items)),
            events=tuple(events),
            text_tokens=int(raw["text_tokens"]),
            generated_images=int(raw["generated_images"]),
            seconds=float(raw["seconds"]),
            finish_reason=str(raw.get("finish_reason") or "stop"),
        )

    def _distribute_trace(
        self,
        trace_id: str,
        manifest: Mapping[str, object],
        *,
        device: torch.device,
        owner_rank: int | None = None,
        consume_trace: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Move a rank-zero WebSocket trace to its replay owner with NCCL."""

        if set(manifest) != _TRACE_KEYS:
            raise RuntimeError("distributed SenseNova SDE trace manifest is incomplete")
        distributed = dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        world_size = dist.get_world_size() if distributed else 1
        if owner_rank is not None and not 0 <= owner_rank < world_size:
            raise ValueError("SDE trace owner rank is outside the FSDP world")
        source = (
            self._pending_traces.pop(trace_id, None)
            if rank == 0 and consume_trace
            else self._pending_traces.get(trace_id)
            if rank == 0
            else None
        )
        if rank == 0 and source is None:
            raise RuntimeError(f"rank zero has no pending SenseNova SDE trace {trace_id}")
        result: dict[str, torch.Tensor] = {}
        for name in sorted(_TRACE_KEYS):
            spec = manifest[name]
            if not isinstance(spec, Mapping):
                raise RuntimeError("SenseNova SDE tensor manifest entry is invalid")
            shape = spec.get("shape")
            dtype_name = spec.get("dtype")
            if (
                not isinstance(shape, Sequence)
                or isinstance(shape, (str, bytes))
                or not all(type(dimension) is int and dimension >= 0 for dimension in shape)
                or not isinstance(dtype_name, str)
            ):
                raise RuntimeError("SenseNova SDE tensor geometry is invalid")
            dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
            if not isinstance(dtype, torch.dtype):
                raise RuntimeError(f"unsupported SenseNova SDE dtype {dtype_name!r}")
            if rank == 0:
                assert source is not None
                tensor = source[name]
                if list(tensor.shape) != list(shape) or tensor.dtype != dtype:
                    raise RuntimeError("rank-zero SenseNova SDE tensor changed geometry")
                tensor = tensor.to(device=device, non_blocking=True)
            elif owner_rank is None or rank == owner_rank:
                tensor = torch.empty(tuple(shape), dtype=dtype, device=device)
            else:
                tensor = None
            if distributed and owner_rank is None:
                assert tensor is not None
                dist.broadcast(tensor, src=0)
            elif distributed and owner_rank != 0:
                if rank == 0:
                    assert tensor is not None
                    dist.send(tensor, dst=owner_rank)
                elif rank == owner_rank:
                    assert tensor is not None
                    dist.recv(tensor, src=0)
            if owner_rank is None or rank == owner_rank:
                assert tensor is not None
                result[name] = tensor
        return result


__all__ = [
    "RlApiTransport",
    "SenseNovaRlApiClient",
    "UrllibRlApiTransport",
]
