"""Packed SFT reductions with equal total weight per original sample."""

import math

import torch


def sample_mean_sum(values, sample_ids, denominator):
    """Sum within-sample means, then divide by a shared sample denominator.

    For FSDP, the denominator is the entire optimizer batch's raw sample count
    divided by world size. Summing microbatch losses and averaging rank
    gradients then gives one global sample mean, including zero-image samples.
    """
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("sample denominator must be positive and finite")
    if values.ndim != 1 or sample_ids.shape != values.shape:
        raise ValueError("sample ids must align with one-dimensional losses")
    if values.numel() == 0:
        return values.sum() / denominator
    if bool((sample_ids < 0).any()):
        raise ValueError("padding must be excluded from sample losses")
    _, inverse, counts = torch.unique(sample_ids, return_inverse=True, return_counts=True)
    # Weight each action by its sample's own action count; no resolution,
    # image-count or text-length weighting is introduced between samples.
    return (values.float() / counts[inverse]).sum() / denominator
