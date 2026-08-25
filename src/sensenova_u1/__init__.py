from __future__ import annotations

from importlib import metadata as _metadata
from typing import Any

from .models.neo_unify import (
    NEOChatConfig,
    NEOChatModel,
    NEOLLMConfig,
    NEOMoELLMConfig,
    NEOVisionConfig,
    NEOVisionModel,
    effective_attn_backend,
    fused_rms_norm_enabled,
    get_attn_backend,
    has_flash_attn,
    set_attn_backend,
    set_fused_rms_norm,
)
from .models.neo_unify import (
    register as _register,
)
try:
    __version__ = _metadata.version("sensenova-u1")
except _metadata.PackageNotFoundError:  # pragma: no cover - editable / not installed
    __version__ = "0.1.0"

__all__ = [
    "__version__",
    "NEOChatConfig",
    "NEOLLMConfig",
    "NEOMoELLMConfig",
    "NEOVisionConfig",
    "NEOChatModel",
    "NEOVisionModel",
    "ContiguousTextBatchSession",
    "ContinuousTextBatch",
    "ContinuousTextBatchEngine",
    "TextBatchRequest",
    "TextBatchResult",
    "batch_text_gen",
    "continuous_batch_text_gen",
    "check_checkpoint_compatibility",
    "set_attn_backend",
    "get_attn_backend",
    "effective_attn_backend",
    "has_flash_attn",
    "set_fused_rms_norm",
    "fused_rms_norm_enabled",
    "main",
]


def check_checkpoint_compatibility(config_or_dict: Any) -> None:
    """Raise ``RuntimeError`` if the installed ``sensenova_u1`` is too old for the checkpoint.

    The checkpoint can advertise a minimum package version by setting
    ``sensenova_u1_min_version`` in its ``config.json``. If the field is
    absent, no check is performed. This lets us evolve the modeling code
    in git while keeping old checkpoints loadable, and hard-fail with a
    clear message when a newer checkpoint requires a newer package.
    """
    try:
        from packaging.version import Version
    except ImportError:  # pragma: no cover
        return

    if hasattr(config_or_dict, "to_dict"):
        cfg: dict = config_or_dict.to_dict()
    elif isinstance(config_or_dict, dict):
        cfg = config_or_dict
    else:
        return

    required = cfg.get("sensenova_u1_min_version")
    if not required:
        return

    if Version(__version__) < Version(str(required)):
        raise RuntimeError(
            f"This checkpoint requires sensenova-u1 >= {required}, "
            f"but the installed version is {__version__}. "
            f"Please upgrade with `uv sync` or `pip install -U sensenova-u1`."
        )


from .batch_inference import (  # noqa: E402
    ContiguousTextBatchSession,
    ContinuousTextBatch,
    ContinuousTextBatchEngine,
    TextBatchRequest,
    TextBatchResult,
    batch_text_gen,
    continuous_batch_text_gen,
)


_register()


def main() -> None:
    print(f"SenseNova-U1 v{__version__}")
