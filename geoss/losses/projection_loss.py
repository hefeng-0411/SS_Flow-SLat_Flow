from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from geoss.losses.stable_bce import probability_binary_cross_entropy
from geoss.utils.projection import project_points


def projection_consistency_loss(
    anchor_xyz: torch.Tensor,
    occ_prob: torch.Tensor,
    masks: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    B, M, _ = anchor_xyz.shape
    _, N, _, H, W = masks.shape
    points = anchor_xyz[:, None].expand(B, N, M, 3).reshape(B * N, M, 3)
    proj = project_points(points, K.reshape(B * N, 3, 3), w2c.reshape(B * N, 4, 4))
    uv = proj["uv"].reshape(B, N, M, 2)
    gx = uv[..., 0] / max(W - 1, 1) * 2.0 - 1.0
    gy = uv[..., 1] / max(H - 1, 1) * 2.0 - 1.0
    valid = ((gx >= -1) & (gx <= 1) & (gy >= -1) & (gy <= 1)).float().unsqueeze(-1)
    grid = torch.stack([gx, gy], dim=-1).reshape(B * N, M, 1, 2)
    safe_grid = torch.nan_to_num(grid, nan=2.0, posinf=2.0, neginf=-2.0)
    mask_samples = F.grid_sample(
        masks.reshape(B * N, 1, H, W).float(),
        safe_grid,
        align_corners=True,
        mode="bilinear",
        padding_mode="zeros",
    ).reshape(B, N, 1, M).permute(0, 1, 3, 2)
    mask_samples = torch.nan_to_num(mask_samples, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    pred = torch.nan_to_num(occ_prob[:, None].expand(B, N, M, 1), nan=0.0, posinf=1.0, neginf=0.0).clamp(1e-4, 1 - 1e-4)
    loss = probability_binary_cross_entropy(
        pred.clamp(1e-4, 1 - 1e-4),
        mask_samples,
        reduction="none",
    )
    loss = (loss * valid).sum() / valid.sum().clamp_min(1.0)
    return {"projection_consistency": loss, "loss": loss}
