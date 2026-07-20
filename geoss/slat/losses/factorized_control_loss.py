from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def factorized_control_loss(
    correction_demand: torch.Tensor,
    residual_variance: torch.Tensor,
    predicted_residual: torch.Tensor,
    target_residual: torch.Tensor,
    *,
    token_mask: torch.Tensor | None = None,
    demand_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Calibrate correction demand separately from residual uncertainty.

    Demand is a bounded probability that the frozen prior needs a material
    correction. Variance is trained by a heteroscedastic Gaussian NLL and must
    not be reused as a synonym for evidence reliability.
    """
    if correction_demand.shape != (*target_residual.shape[:2], 1):
        raise ValueError("correction_demand must be [B,L,1] and align with target_residual")
    if residual_variance.shape != correction_demand.shape:
        raise ValueError("residual_variance must match correction_demand")
    if predicted_residual.shape != target_residual.shape:
        raise ValueError("predicted_residual and target_residual must have identical shapes")
    scale = max(float(demand_scale), 1e-6)
    target_magnitude = target_residual.detach().square().mean(dim=-1, keepdim=True).sqrt()
    demand_target = (1.0 - torch.exp(-target_magnitude / scale)).clamp(0, 1)
    demand_bce = F.binary_cross_entropy(correction_demand.clamp(1e-4, 1 - 1e-4), demand_target, reduction="none")
    variance = residual_variance.clamp(1e-4, 1e3)
    squared_error = (predicted_residual - target_residual.detach()).square().mean(dim=-1, keepdim=True)
    heteroscedastic_nll = squared_error / variance + variance.log()
    if token_mask is not None:
        mask = token_mask.to(device=demand_bce.device, dtype=demand_bce.dtype)
        if mask.shape != demand_bce.shape:
            raise ValueError(f"token_mask must be {tuple(demand_bce.shape)}, got {tuple(mask.shape)}")
        denom = mask.sum().clamp_min(1.0)
        demand_loss = (demand_bce * mask).sum() / denom
        uncertainty_loss = (heteroscedastic_nll * mask).sum() / denom
    else:
        demand_loss = demand_bce.mean()
        uncertainty_loss = heteroscedastic_nll.mean()
    loss = demand_loss + 0.25 * uncertainty_loss
    return {
        "loss": loss,
        "correction_demand_bce": demand_loss,
        "residual_uncertainty_nll": uncertainty_loss,
        "correction_demand_target_mean": demand_target.mean().detach(),
    }
