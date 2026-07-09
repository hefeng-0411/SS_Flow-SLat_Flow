from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def visibility_confidence_loss(
    slat_confidence: torch.Tensor,
    visibility: torch.Tensor,
    depth_residual: torch.Tensor,
    appearance_conflict: torch.Tensor | None = None,
    occlusion_score: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    target = visibility.mean(dim=2).clamp(0, 1) * torch.exp(-depth_residual.mean(dim=2).clamp_min(0))
    if appearance_conflict is not None:
        target = target * (1.0 - appearance_conflict.clamp(0, 1))
    if occlusion_score is not None:
        target = target * (1.0 - occlusion_score.mean(dim=2).clamp(0, 1))
    target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    pred = torch.nan_to_num(slat_confidence, nan=0.0, posinf=1.0, neginf=0.0).clamp(1e-5, 1 - 1e-5)
    bce = F.binary_cross_entropy(pred, target.detach())
    mean = slat_confidence.mean()
    std = slat_confidence.std(unbiased=False)
    anti_collapse = (mean - 0.5).pow(2) * 0.02 + F.relu(0.02 - std)
    loss = bce + anti_collapse
    err = (slat_confidence.detach() - target.detach()).abs()
    corr = _corr(slat_confidence.detach().reshape(-1), err.reshape(-1))
    return {"loss": loss, "visibility_confidence": bce, "confidence_std": std, "confidence_error_corr": corr}


def _corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() < 2:
        return a.new_zeros(())
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).mean() / (a.std(unbiased=False) * b.std(unbiased=False)).clamp_min(1e-6)
