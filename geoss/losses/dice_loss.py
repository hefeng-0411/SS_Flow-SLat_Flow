from __future__ import annotations

import torch


def dice_loss(pred_prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.nan_to_num(pred_prob.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0).reshape(pred_prob.shape[0], -1)
    tgt = torch.nan_to_num(target.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0).reshape(target.shape[0], -1)
    inter = (pred * tgt).sum(dim=1)
    denom = pred.sum(dim=1) + tgt.sum(dim=1)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()
