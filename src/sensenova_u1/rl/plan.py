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
    nproc_per_node: int = 8
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29540

    @property
    def world_size(self) -> int:
        return self.nproc_per_node * self.nnodes

    def validate(self) -> None:
        if type(self.nproc_per_node) is not int or type(self.nnodes) is not int:
            raise TypeError("torchrun sizes must be integers")
        if self.nproc_per_node < 1 or self.nnodes < 1:
            raise ValueError("torchrun sizes must be positive")
        if type(self.node_rank) is not int:
            raise TypeError("node_rank must be an integer")
        if not 0 <= self.node_rank < self.nnodes:
            raise ValueError("node_rank is outside nnodes")
        if not isinstance(self.master_addr, str) or not self.master_addr:
            raise ValueError("invalid torchrun rendezvous address")
        if type(self.master_port) is not int or not 0 < self.master_port < 65536:
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
    prompts_per_batch: int = 8
    policy_updates_per_batch: int = 2
    text_learning_rate: float = 1e-6
    visual_learning_rate: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 42
    save_every_steps: int = 10
    rollout_api_base_urls: tuple[str, ...] = ("http://127.0.0.1:8000",)
    rollout_policy_version: str = "startup"
    weight_update_master_address: str = "127.0.0.1"
    weight_update_base_port: int = 29680
    weight_update_backend: str = "nccl"
    weight_update_bucket_bytes: int = 256 * 1024 * 1024
    max_sequence_length: int = 8192
    max_new_tokens: int = 6144
    max_images: int = 10
    image_size: int = 512
    image_steps: int = 30
    image_replay_microbatch_size: int = 1
    image_noise_level: float = 0.7
    timestep_shift: float = 1.0
    t_eps: float = 0.02
    text_kl_beta: float = 0.0
    image_objective_weight: float = 0.0
    velocity_mse_weight: float = 0.0
    image_clip_range: float = 1e-4
    text_clip_range: float = 0.2
    sde_window_start: int = 0
    sde_window_end: int = 30
    sde_window_steps: int = 8
    device: str = "cuda"
    dtype: str = "bfloat16"
    attention_backend: str = "flash"
    activation_checkpointing: bool = False
    optimizer_cpu_offload_min_images: int = 6
    torchrun: TorchrunSpec = TorchrunSpec()

    def __post_init__(self) -> None:
        self.torchrun.validate()
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
        if not any(value != 0 for value in self.reward_weights):
            raise ValueError("at least one reward weight must be non-zero")
        for name, value in (
            ("weight_decay", self.weight_decay),
            ("text_kl_beta", self.text_kl_beta),
            ("image_objective_weight", self.image_objective_weight),
            ("velocity_mse_weight", self.velocity_mse_weight),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")
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
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be positive")
        if type(self.max_images) is not int or self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.group_size < 2:
            raise ValueError("GDPO requires at least two rollouts per prompt group")
        if (
            not self.rollout_api_base_urls
            or len(set(self.rollout_api_base_urls)) != len(self.rollout_api_base_urls)
            or any(not isinstance(url, str) or not url.startswith(("http://", "https://")) for url in self.rollout_api_base_urls)
        ):
            raise ValueError("rollout_api_base_urls must contain distinct absolute HTTP(S) URLs")
        if self.prompts_per_batch < 2:
            raise ValueError("GDPO batch normalization requires multiple prompt groups")
        if self.prompts_per_batch != self.torchrun.world_size:
            raise ValueError("production API rollout requires one prompt group per FSDP rank")
        if self.max_steps % self.policy_updates_per_batch:
            raise ValueError("max_steps must close a complete frozen-old PPO update batch")
        if self.max_new_tokens >= self.max_sequence_length:
            raise ValueError("max_new_tokens must leave room for the prompt")
        if self.image_size % 32:
            raise ValueError("image_size must be divisible by the U1.5 32-pixel generation grid")
        for name, value in (
            ("image_clip_range", self.image_clip_range),
            ("text_clip_range", self.text_clip_range),
        ):
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must lie inside (0, 1)")
        if not 0 <= self.sde_window_start < self.sde_window_end <= self.image_steps:
            raise ValueError("SDE window is outside the image schedule")
        if not 1 <= self.sde_window_steps <= self.sde_window_end - self.sde_window_start:
            raise ValueError("invalid SDE window sample count")
        if self.modality == "ti2t" and (self.image_objective_weight != 0 or self.velocity_mse_weight != 0):
            raise ValueError("TI2T cannot enable image policy or velocity MSE")
        if self.modality == "ti2ti" and self.image_objective_weight <= 0:
            raise ValueError("TI2TI must enable its image policy objective")
        if self.weight_decay != 0:
            raise ValueError("SenseNova full-parameter RL fixes weight_decay=0")
        if self.device != "cuda":
            raise ValueError("full-parameter Forge RL requires CUDA")
        if type(self.activation_checkpointing) is not bool:
            raise TypeError("activation_checkpointing must be a boolean")
        if (
            type(self.optimizer_cpu_offload_min_images) is not int
            or not 1
            <= self.optimizer_cpu_offload_min_images
            <= max(1, self.max_images)
        ):
            raise ValueError(
                "optimizer_cpu_offload_min_images must lie inside [1, max(1, max_images)]"
            )
        if self.dtype != "bfloat16" or self.attention_backend not in {"flash", "sdpa"}:
            raise ValueError("unsupported dtype or attention backend")
        if self.weight_update_backend != "nccl":
            raise ValueError("direct online policy publication requires NCCL")
        if type(self.weight_update_base_port) is not int or not 0 < self.weight_update_base_port <= 65533:
            raise ValueError("weight update base port must leave three valid ports")

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
            "rollout_api_base_urls",
        ):
            values[name] = tuple(values[name])
        values["torchrun"] = TorchrunSpec(**dict(values.get("torchrun") or {}))
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
