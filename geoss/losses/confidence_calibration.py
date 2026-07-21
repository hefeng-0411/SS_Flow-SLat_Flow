from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from geoss.losses.stable_bce import probability_binary_cross_entropy


def target_confidence_from_errors(
    *,
    depth_error: torch.Tensor,
    mask_error: torch.Tensor,
    reprojection_error: torch.Tensor | None = None,
    conflict_error: torch.Tensor | None = None,
    sigma: float = 0.25,
) -> torch.Tensor:
    err = depth_error.clamp_min(0) + mask_error.clamp_min(0)
    if reprojection_error is not None:
        err = err + reprojection_error.clamp_min(0)
    if conflict_error is not None:
        err = err + conflict_error.clamp_min(0)
    return torch.exp(-err / max(sigma, 1e-6)).clamp(0, 1).detach()


def confidence_calibration_loss(pred_confidence: torch.Tensor, target_confidence: torch.Tensor, *, mode: str = "brier") -> Dict[str, torch.Tensor]:
    pred = pred_confidence.clamp(1e-4, 1.0 - 1e-4)
    target = target_confidence.to(device=pred.device, dtype=pred.dtype).clamp(0, 1).detach()
    if mode == "bce":
        loss = probability_binary_cross_entropy(pred, target)
    else:
        loss = (pred - target).square().mean()
    ece = expected_calibration_error(pred.detach(), target.detach())
    brier = (pred - target).square().mean()
    return {"loss": loss, "brier": brier.detach(), "ece": ece, "mean_confidence": pred.mean().detach()}


def expected_calibration_error(pred: torch.Tensor, target: torch.Tensor, bins: int = 10) -> torch.Tensor:
    pred = pred.flatten().clamp(0, 1)
    target = target.flatten().clamp(0, 1)
    ece = pred.new_zeros(())
    edges = torch.linspace(0, 1, bins + 1, device=pred.device, dtype=pred.dtype)
    for i in range(bins):
        mask = (pred >= edges[i]) & (pred < edges[i + 1] if i + 1 < bins else pred <= edges[i + 1])
        if mask.any():
            ece = ece + mask.float().mean() * (pred[mask].mean() - target[mask].mean()).abs()
    return ece


def evidential_occupancy_stats(occ_evidence: torch.Tensor, free_evidence: torch.Tensor) -> Dict[str, torch.Tensor]:
    alpha_occ = F.softplus(occ_evidence) + 1.0
    alpha_free = F.softplus(free_evidence) + 1.0
    total = (alpha_occ + alpha_free).clamp_min(1e-6)
    return {
        "p_occ": alpha_occ / total,
        "uncertainty": 2.0 / total,
        "evidence_strength": total,
    }
