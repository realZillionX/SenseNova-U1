"""Model-ABI validation shared by Forge RL preflight and resume."""

from __future__ import annotations

from torch import nn


def validate_policy_model(model: nn.Module) -> None:
    required_attributes = (
        "language_model",
        "vision_model",
        "fm_modules",
        "patchify",
        "_build_it2i_inputs",
        "_t2i_predict_v",
    )
    missing = [name for name in required_attributes if not hasattr(model, name)]
    if missing:
        raise TypeError(f"U1.5 checkpoint model is missing replay primitives: {missing}")
    if not bool(getattr(model, "use_pixel_head", False)):
        raise ValueError("Forge RL requires the U1.5 pixel-head checkpoint ABI")
    parameters = tuple(model.named_parameters())
    if not parameters:
        raise ValueError("U1.5 policy exposes no parameters")
    duplicates = len(parameters) != len({name for name, _parameter in parameters})
    if duplicates:
        raise ValueError("U1.5 policy exposes duplicate parameter names")


__all__ = ["validate_policy_model"]
