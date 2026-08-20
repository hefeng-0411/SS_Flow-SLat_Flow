from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def ray_free_space_loss(
    free_score: torch.Tensor,
    occ_score: torch.Tensor,
    ray_valid: torch.Tensor,
    depth_residual: torch.Tensor,
    *,
    signed_depth_residual: torch.Tensor | None = None,
    free_geometry: torch.Tensor | None = None,
    free_margin: float = -0.02,
) -> Dict[str, torch.Tensor]:
    if free_geometry is not None:
        free_target = ((free_geometry > 0.5) & (ray_valid > 0.5)).float()
    else:
        residual_for_free = signed_depth_residual if signed_depth_residual is not None else depth_residual
        free_target = ((residual_for_free < free_margin) & (ray_valid > 0.5)).float()
    valid = torch.nan_to_num(ray_valid.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    free_target = torch.nan_to_num(free_target.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    logits = torch.nan_to_num(free_score - occ_score, nan=0.0, posinf=30.0, neginf=-30.0)
    loss = F.binary_cross_entropy_with_logits(logits, free_target, reduction="none")
    denom = valid.sum().clamp_min(1.0)
    loss = (loss * valid).sum() / denom
    return {"ray_free_space": loss, "free_target_mean": free_target.mean().detach(), "loss": loss}
