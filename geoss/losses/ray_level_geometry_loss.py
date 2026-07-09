from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def ray_level_geometry_loss(ray_debug: Dict[str, torch.Tensor], occ_score: torch.Tensor, free_score: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Approximate ray termination supervision over anchor samples.

    Uses the per-view anchor evidence already sorted by projected depth at loss
    time. This avoids treating every projected foreground anchor as occupied.
    """
    anchor_depth = ray_debug["evidence_debug"]["anchor_depth"]  # [B,M,N,1]
    surface_depth = ray_debug["evidence_debug"]["surface_depth"]
    ray_valid = ray_debug["ray_valid"]
    mask_samples = ray_debug["evidence_debug"]["mask_samples"]
    occupied = ray_debug["evidence_debug"]["occupied_geometry"]
    free = ray_debug["evidence_debug"]["free_geometry"]
    alpha = torch.sigmoid(occ_score - free_score).clamp(1e-4, 1.0 - 1e-4)
    order = anchor_depth.squeeze(-1).argsort(dim=1)
    alpha_sorted = alpha.gather(1, order[..., None])
    depth_sorted = anchor_depth.gather(1, order[..., None])
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha_sorted[:, :1]), 1.0 - alpha_sorted[:, :-1]], dim=1), dim=1)
    weights = alpha_sorted * trans
    hit_prob = weights.sum(dim=1).clamp(1e-4, 1.0 - 1e-4)
    target_hit = (mask_samples * ray_valid).amax(dim=1).clamp(0, 1)
    mask_loss = F.binary_cross_entropy(hit_prob, target_hit)
    expected_depth = (weights * depth_sorted).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)
    target_depth = surface_depth.masked_fill(ray_valid <= 0.5, 0).sum(dim=1) / ray_valid.sum(dim=1).clamp_min(1.0)
    depth_loss = (expected_depth - target_depth).abs().masked_select(target_hit > 0.5).mean() if (target_hit > 0.5).any() else expected_depth.new_zeros(())
    free_loss = F.binary_cross_entropy((1.0 - alpha).clamp(1e-4, 1.0 - 1e-4), free.clamp(0, 1))
    surface_loss = F.binary_cross_entropy(alpha.clamp(1e-4, 1.0 - 1e-4), occupied.clamp(0, 1))
    loss = mask_loss + depth_loss + 0.5 * free_loss + 0.5 * surface_loss
    return {
        "loss": loss,
        "mask_loss": mask_loss,
        "depth_loss": depth_loss,
        "free_space_loss": free_loss,
        "surface_loss": surface_loss,
        "hit_prob_mean": hit_prob.mean().detach(),
    }
