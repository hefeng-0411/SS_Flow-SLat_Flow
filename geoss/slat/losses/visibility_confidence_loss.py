from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from geoss.losses.stable_bce import probability_binary_cross_entropy


def visibility_confidence_loss(
    slat_confidence: torch.Tensor,
    visibility: torch.Tensor,
    depth_residual: torch.Tensor,
    appearance_conflict: torch.Tensor | None = None,
    occlusion_score: torch.Tensor | None = None,
    token_valid_mask: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    target = visibility.mean(dim=2).clamp(0, 1) * torch.exp(-depth_residual.mean(dim=2).clamp_min(0))
    if appearance_conflict is not None:
        target = target * (1.0 - appearance_conflict.clamp(0, 1))
    if occlusion_score is not None:
        target = target * (1.0 - occlusion_score.mean(dim=2).clamp(0, 1))
    target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    pred = torch.nan_to_num(slat_confidence, nan=0.0, posinf=1.0, neginf=0.0).clamp(1e-5, 1 - 1e-5)
    per_token_bce = probability_binary_cross_entropy(
        pred,
        target.detach(),
        reduction="none",
    )
    if token_valid_mask is None:
        bce = per_token_bce.mean()
    else:
        valid = token_valid_mask.to(device=pred.device, dtype=per_token_bce.dtype)
        bce = (per_token_bce * valid).sum() / valid.sum().clamp_min(1e-6)
    if token_valid_mask is None:
        valid_values = slat_confidence.reshape(-1)
    else:
        valid_values = slat_confidence[token_valid_mask.to(device=slat_confidence.device).expand_as(slat_confidence) > 0.5]
    if valid_values.numel() == 0:
        valid_values = slat_confidence.new_zeros(1)
    mean = valid_values.mean()
    std = valid_values.std(unbiased=False)
    anti_collapse = (mean - 0.5).pow(2) * 0.02 + F.relu(0.02 - std)
    loss = bce + anti_collapse
    err = (slat_confidence.detach() - target.detach()).abs()
    if token_valid_mask is None:
        corr = _corr(slat_confidence.detach().reshape(-1), err.reshape(-1))
    else:
        select = token_valid_mask.to(device=slat_confidence.device).expand_as(slat_confidence) > 0.5
        corr = _corr(slat_confidence.detach()[select], err.reshape_as(slat_confidence)[select])
    return {"loss": loss, "visibility_confidence": bce, "confidence_std": std, "confidence_error_corr": corr}


def _corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() < 2:
        return a.new_zeros(())
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).mean() / (a.std(unbiased=False) * b.std(unbiased=False)).clamp_min(1e-6)
