from __future__ import annotations

from typing import Dict

import torch

from geoss.geometry.pointcloud_ops import voxel_downsample


def pointmap_surface_candidates(pointmap: torch.Tensor, mask: torch.Tensor | None = None, *, max_points: int = 20000) -> Dict[str, torch.Tensor]:
    B, N, _, H, W = pointmap.shape
    pts = pointmap.permute(0, 1, 3, 4, 2).reshape(B, N * H * W, 3)
    if mask is not None:
        valid = mask.reshape(B, -1) > 0.5
    else:
        valid = torch.ones(B, pts.shape[1], dtype=torch.bool, device=pts.device)
    out = []
    for b in range(B):
        cur = pts[b, valid[b]]
        out.append(cur[:max_points])
    return {"points": out}


def open3d_anchor_downsample(points: torch.Tensor, voxel_size: float = 0.02, *, real_mode: bool = False) -> torch.Tensor:
    return voxel_downsample(points, voxel_size, real_mode=real_mode)
