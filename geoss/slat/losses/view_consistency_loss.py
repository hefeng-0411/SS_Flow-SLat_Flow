from __future__ import annotations

from typing import Dict

import torch


def view_consistency_loss(
    sampled_features: torch.Tensor,
    visibility: torch.Tensor,
    token_valid_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    weights = visibility.clamp(0, 1)
    denom = weights.sum(dim=2, keepdim=True).clamp_min(eps)
    mean = (sampled_features * weights).sum(dim=2, keepdim=True) / denom
    var = ((sampled_features - mean).pow(2) * weights).sum(dim=2) / denom.squeeze(2)
    per_token = var.mean(dim=-1, keepdim=True)
    loss = _masked_mean(per_token, token_valid_mask, eps)
    return {"loss": loss, "view_consistency": loss}


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None, eps: float) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask = mask.to(device=values.device, dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(eps)
