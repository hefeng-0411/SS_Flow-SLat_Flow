from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _feats(x):
    return x.feats if hasattr(x, "feats") else x


def slat_flow_matching_loss(pred_velocity, target_velocity, token_mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
    """TRELLIS-compatible SLAT velocity MSE over SparseTensor.feats or dense tokens."""
    pred = _feats(pred_velocity)
    target = _feats(target_velocity)
    squared = (pred - target).square()
    if token_mask is None:
        loss = squared.mean()
    else:
        mask = token_mask.to(device=squared.device, dtype=squared.dtype)
        if mask.shape != (*squared.shape[:2], 1):
            raise ValueError(f"SLAT supervision mask must be [B,L,1], got {tuple(mask.shape)}")
        # Missing-GT samples contribute zero, rather than synthetic targets that
        # collapse the adapter's confidence distribution.
        loss = (squared * mask).sum() / (mask.sum() * squared.shape[-1]).clamp_min(1.0)
    return {"loss": loss, "slat_flow_mse": loss}
