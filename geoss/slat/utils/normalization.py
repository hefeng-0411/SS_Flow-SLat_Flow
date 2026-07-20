from __future__ import annotations

from typing import Mapping, Sequence

import torch


SLAT_TENSOR_CONTRACT_VERSION = "phase2_factorized_control_v2"


def normalize_slat(feats: torch.Tensor, normalization: Mapping[str, Sequence[float]]) -> torch.Tensor:
    """Map raw TRELLIS VAE SLAT features into the flow model's state space."""
    mean, std = slat_normalization_tensors(normalization, feats)
    return (feats - mean) / std


def denormalize_slat(feats: torch.Tensor, normalization: Mapping[str, Sequence[float]]) -> torch.Tensor:
    """Map TRELLIS flow-state features back to the VAE decoder's state space."""
    mean, std = slat_normalization_tensors(normalization, feats)
    return feats * std + mean


def slat_normalization_tensors(
    normalization: Mapping[str, Sequence[float]], reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(normalization, Mapping) or "mean" not in normalization or "std" not in normalization:
        raise KeyError("TRELLIS pipeline must expose slat_normalization with mean and std.")
    mean = torch.as_tensor(normalization["mean"], device=reference.device, dtype=reference.dtype)
    std = torch.as_tensor(normalization["std"], device=reference.device, dtype=reference.dtype)
    if mean.ndim != 1 or std.ndim != 1 or mean.numel() != reference.shape[-1] or std.numel() != reference.shape[-1]:
        raise ValueError(
            "SLAT normalization width must match the flow latent width: "
            f"mean={tuple(mean.shape)}, std={tuple(std.shape)}, latent={reference.shape[-1]}"
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("SLAT normalization contains non-finite values or non-positive standard deviations.")
    view_shape = (1,) * (reference.ndim - 1) + (reference.shape[-1],)
    return mean.view(view_shape), std.view(view_shape)
