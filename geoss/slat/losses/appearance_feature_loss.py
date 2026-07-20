from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def appearance_feature_loss(
    slat_cond_tokens: torch.Tensor,
    sampled_features: torch.Tensor,
    visibility: torch.Tensor,
    view_weights: torch.Tensor,
    token_valid_mask: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    weights = (visibility.clamp(0, 1) * view_weights.clamp_min(0)).detach()
    target = (sampled_features * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1e-6)
    cond = slat_cond_tokens
    dim = min(cond.shape[-1], target.shape[-1])
    if dim == 0:
        loss = cond.sum() * 0.0
    else:
        per_token = (
            F.normalize(cond[..., :dim], dim=-1) - F.normalize(target[..., :dim], dim=-1)
        ).square().mean(dim=-1, keepdim=True)
        if token_valid_mask is None:
            loss = per_token.mean()
        else:
            valid = token_valid_mask.to(device=per_token.device, dtype=per_token.dtype)
            loss = (per_token * valid).sum() / valid.sum().clamp_min(1e-6)
    return {"loss": loss, "appearance_feature": loss}
