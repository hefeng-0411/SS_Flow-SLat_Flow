from __future__ import annotations

import torch


def compute_visibility_score(
    in_bounds: torch.Tensor,
    sampled_mask: torch.Tensor,
    depth_residual: torch.Tensor,
    occlusion_score: torch.Tensor,
    *,
    surface_threshold: float = 0.06,
    has_depth: bool = True,
) -> torch.Tensor:
    """Continuous visibility score for active voxel/view evidence."""
    base = in_bounds.float() * sampled_mask.clamp(0, 1)
    if has_depth:
        depth_score = torch.exp(-depth_residual.clamp_min(0) / max(surface_threshold, 1e-6))
    else:
        depth_score = torch.ones_like(base)
    return (base * depth_score * (1.0 - occlusion_score.clamp(0, 1))).clamp(0, 1)


def appearance_conflict_from_samples(samples: torch.Tensor, visibility: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Visibility-weighted cross-view variance, normalized to [0,1]."""
    weights = visibility.clamp(0, 1)
    denom = weights.sum(dim=2, keepdim=True).clamp_min(eps)
    mean = (samples * weights).sum(dim=2, keepdim=True) / denom
    var = ((samples - mean).pow(2) * weights).sum(dim=2) / denom.squeeze(2)
    return (1.0 - torch.exp(-var.mean(dim=-1, keepdim=True))).clamp(0, 1)
