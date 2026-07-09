from __future__ import annotations

from typing import Dict

import torch


def view_consistency_loss(sampled_features: torch.Tensor, visibility: torch.Tensor, eps: float = 1e-6) -> Dict[str, torch.Tensor]:
    weights = visibility.clamp(0, 1)
    denom = weights.sum(dim=2, keepdim=True).clamp_min(eps)
    mean = (sampled_features * weights).sum(dim=2, keepdim=True) / denom
    var = ((sampled_features - mean).pow(2) * weights).sum(dim=2) / denom.squeeze(2)
    loss = var.mean()
    return {"loss": loss, "view_consistency": loss}
