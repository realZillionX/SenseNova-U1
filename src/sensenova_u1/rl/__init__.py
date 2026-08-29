"""Full-parameter GDPO/UniGDPO infrastructure for SenseNova-U1.5-8B-MoT."""

from importlib import import_module

_MODULE_EXPORTS = {
    "advantage": ("GdpoAdvantageResult", "compute_gdpo_advantage"),
    "flow": (
        "BranchWeights",
        "RegularizationWeights",
        "UniGdpoLoss",
        "uni_gdpo_loss",
    ),
    "plan": ("RlPlan", "TorchrunSpec"),
    "types": ("CandidateResponse", "ImageSegment", "RewardBatch", "TextSegment"),
}
_EXPORTS = {name: module for module, names in _MODULE_EXPORTS.items() for name in names}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
