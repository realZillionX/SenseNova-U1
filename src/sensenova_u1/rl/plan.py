"""Immutable, verifier-agnostic plan for full-parameter U1.5 RL."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class TorchrunSpec:
    nproc_per_node: int = 6
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29540

    @property
    def world_size(self) -> int:
        return self.nproc_per_node * self.nnodes

    def validate(self) -> None:
        if self.nproc_per_node < 1 or self.nnodes < 1:
            raise ValueError("torchrun sizes must be positive")
        if not 0 <= self.node_rank < self.nnodes:
            raise ValueError("node_rank is outside nnodes")
        if not self.master_addr or not 0 < self.master_port < 65536:
            raise ValueError("invalid torchrun rendezvous")


@dataclass(frozen=True)
class RlPlan:
    run_dir: Path
    prompts: Path
    policy_init: Path
    modality: str
    reward_command: tuple[str, ...]
    reward_dimension_names: tuple[str, ...]
    reward_weights: tuple[float, ...]
    max_steps: int
    reward_context: Mapping[str, Any] = field(default_factory=dict)
    policy_init_kind: str = "sft_checkpoint"
    group_size: int = 8
    prompts_per_batch: int = 6
    policy_updates_per_batch: int = 2
    text_learning_rate: float = 1e-6
    visual_learning_rate: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 42
    save_every_steps: int = 10
    rollout_api_base_url: str = "http://127.0.0.1:8000"
    rollout_policy_version: str = "startup"
    weight_update_master_address: str = "127.0.0.1"
    weight_update_base_port: int = 29680
    weight_update_backend: str = "nccl"
    weight_update_bucket_bytes: int = 256 * 1024 * 1024
    max_sequence_length: int = 8192
    max_new_tokens: int = 2048
    max_images: int = 7
    image_size: int = 512
    image_steps: int = 30
    image_replay_microbatch_size: int = 1
    image_noise_level: float = 0.7
    timestep_shift: float = 1.0
    t_eps: float = 0.02
    text_kl_beta: float = 0.0
    image_objective_weight: float = 0.0
    velocity_mse_weight: float = 0.0
    clip_ranges: tuple[float, float] = (1e-4, 0.2)
    sde_window_start: int = 0
    sde_window_end: int = 30
    sde_window_steps: int = 8
    device: str = "cuda"
    dtype: str = "bfloat16"
    attention_backend: str = "flash"
    activation_checkpointing: bool = False
    torchrun: TorchrunSpec = TorchrunSpec()

    def __post_init__(self) -> None:
        if self.modality not in {"ti2t", "ti2ti"}:
            raise ValueError("modality must be ti2t or ti2ti")
        if not self.reward_command:
            raise ValueError("reward_command must be non-empty")
        if self.policy_init_kind not in {"sft_checkpoint", "published_base"}:
            raise ValueError("unsupported policy_init_kind")
        try:
            json.dumps(self.reward_context, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("reward_context must be JSON-serializable") from exc
        if not self.reward_dimension_names or len(set(self.reward_dimension_names)) != len(self.reward_dimension_names):
            raise ValueError("reward dimensions must be non-empty and unique")
        if len(self.reward_weights) != len(self.reward_dimension_names):
            raise ValueError("reward weights must match reward dimensions")
        for name, value in (
            ("text_learning_rate", self.text_learning_rate),
            ("visual_learning_rate", self.visual_learning_rate),
            ("max_grad_norm", self.max_grad_norm),
            ("image_noise_level", self.image_noise_level),
            ("timestep_shift", self.timestep_shift),
            ("t_eps", self.t_eps),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for value in self.reward_weights:
            if not math.isfinite(value):
                raise ValueError("reward weights must be finite")
        for name in (
            "max_steps",
            "group_size",
            "prompts_per_batch",
            "policy_updates_per_batch",
            "save_every_steps",
            "max_sequence_length",
            "max_new_tokens",
            "image_size",
            "image_steps",
            "image_replay_microbatch_size",
            "weight_update_bucket_bytes",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if len(self.clip_ranges) != 2 or any(not 0 < value < 1 for value in self.clip_ranges):
            raise ValueError("clip_ranges must contain image/text values inside (0, 1)")
        if not 0 <= self.sde_window_start < self.sde_window_end <= self.image_steps:
            raise ValueError("SDE window is outside the image schedule")
        if not 1 <= self.sde_window_steps <= self.sde_window_end - self.sde_window_start:
            raise ValueError("invalid SDE window sample count")
        if self.modality == "ti2t" and (self.image_objective_weight != 0 or self.velocity_mse_weight != 0):
            raise ValueError("TI2T cannot enable image policy or velocity MSE")
        if self.device != "cuda":
            raise ValueError("full-parameter Forge RL requires CUDA")
        if self.dtype != "bfloat16" or self.attention_backend not in {"flash", "sdpa"}:
            raise ValueError("unsupported dtype or attention backend")
        self.torchrun.validate()

    @property
    def plan_path(self) -> Path:
        return self.run_dir / "plan.json"

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["run_dir"] = str(self.run_dir.resolve())
        payload["prompts"] = str(self.prompts.resolve())
        payload["policy_init"] = str(self.policy_init.resolve())
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RlPlan":
        values = dict(payload)
        for name in ("run_dir", "prompts", "policy_init"):
            values[name] = Path(values[name]).expanduser().resolve()
        for name in (
            "reward_command",
            "reward_dimension_names",
            "reward_weights",
            "clip_ranges",
        ):
            values[name] = tuple(values[name])
        values["torchrun"] = TorchrunSpec(**dict(values["torchrun"]))
        values["reward_context"] = dict(values.get("reward_context") or {})
        return cls(**values)

    @classmethod
    def read(cls, path: Path) -> "RlPlan":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("RL plan must be a JSON object")
        plan = cls.from_dict(payload)
        if plan.plan_path.resolve() != path.resolve():
            raise ValueError("RL plan path does not match its run directory")
        return plan

    def write(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.plan_path.write_text(self.to_json() + "\n", encoding="utf-8")


__all__ = ["RlPlan", "TorchrunSpec"]
