from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def appearance_feature_loss(
    slat_cond_tokens: torch.Tensor,
    sampled_features: torch.Tensor,
    visibility: torch.Tensor,
    view_weights: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    weights = (visibility.clamp(0, 1) * view_weights.clamp_min(0)).detach()
    target = (sampled_features * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1e-6)
    cond = slat_cond_tokens
    dim = min(cond.shape[-1], target.shape[-1])
    if dim == 0:
        loss = cond.sum() * 0.0
    else:
        loss = F.mse_loss(F.normalize(cond[..., :dim], dim=-1), F.normalize(target[..., :dim], dim=-1))
    return {"loss": loss, "appearance_feature": loss}
