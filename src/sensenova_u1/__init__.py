from __future__ import annotations

from importlib import import_module as _import_module
from importlib import metadata as _metadata
from importlib.util import find_spec as _find_spec
from typing import Any

_MODEL_EXPORTS = {
    "NEOChatConfig",
    "NEOChatModel",
    "NEOLLMConfig",
    "NEOMoELLMConfig",
    "NEOVisionConfig",
    "NEOVisionModel",
    "effective_attn_backend",
    "fused_rms_norm_enabled",
    "get_attn_backend",
    "has_flash_attn",
    "set_attn_backend",
    "set_fused_rms_norm",
}
_models_registered = False

try:
    __version__ = _metadata.version("sensenova-u15-forge")
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
    "check_checkpoint_compatibility",
    "set_attn_backend",
    "get_attn_backend",
    "effective_attn_backend",
    "has_flash_attn",
    "set_fused_rms_norm",
    "fused_rms_norm_enabled",
    "register_models",
]


def register_models() -> None:
    """Register NEO-Unify with Transformers when model dependencies are needed."""
    global _models_registered
    if _models_registered:
        return
    module = _import_module(".models.neo_unify", __name__)
    module.register()
    _models_registered = True


def __getattr__(name: str):
    if name not in _MODEL_EXPORTS:
        raise AttributeError(name)
    register_models()
    value = getattr(_import_module(".models.neo_unify", __name__), name)
    globals()[name] = value
    return value


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
            f"This checkpoint requires SenseNova-U1.5 Forge >= {required}, "
            f"but the installed version is {__version__}. "
            "Please upgrade the SenseNova-U1 Forge package."
        )


# Preserve the convenient registration side effect in model runtimes while
# keeping verifier/control-plane imports usable without the multi-gigabyte
# Torch dependency.
if _find_spec("torch") is not None:
    register_models()
