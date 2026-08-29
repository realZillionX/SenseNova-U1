"""Full-parameter GDPO/UniGDPO infrastructure for SenseNova-U1.5-8B-MoT."""

from .advantage import GdpoAdvantageResult, compute_gdpo_advantage
from .flow import (
    BranchWeights,
    RegularizationWeights,
    UniGdpoLoss,
    uni_gdpo_loss,
)
from .plan import RlPlan, TorchrunSpec
from .types import CandidateResponse, ImageSegment, RewardBatch, TextSegment

__all__ = [
    "BranchWeights",
    "CandidateResponse",
    "GdpoAdvantageResult",
    "ImageSegment",
    "RegularizationWeights",
    "RewardBatch",
    "RlPlan",
    "TextSegment",
    "TorchrunSpec",
    "UniGdpoLoss",
    "compute_gdpo_advantage",
    "uni_gdpo_loss",
]
