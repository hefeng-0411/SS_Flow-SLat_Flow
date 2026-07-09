from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from geoss.utils.optional_deps import require_dependency
from geoss.utils.projection import project_points as _manual_project_points, unproject_depth as _manual_unproject_depth


def project_points(points_world: torch.Tensor, K: torch.Tensor, w2c: torch.Tensor, *, require_kornia: bool = False) -> Dict[str, torch.Tensor]:
    if require_kornia:
        require_dependency("kornia", real_mode=True, feature="differentiable camera projection")
    return _manual_project_points(points_world, K, w2c)


def unproject_depth(depth: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor, *, require_kornia: bool = False) -> torch.Tensor:
    if require_kornia:
        require_dependency("kornia", real_mode=True, feature="differentiable depth unprojection")
    return _manual_unproject_depth(depth, K, c2w)


def relative_pose(c2w_a: torch.Tensor, c2w_b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(c2w_b) @ c2w_a


def warp_view(src: torch.Tensor, depth: torch.Tensor, K_src: torch.Tensor, c2w_src: torch.Tensor, K_tgt: torch.Tensor, w2c_tgt: torch.Tensor) -> Dict[str, torch.Tensor]:
    B, _, H, W = src.shape
    pts = unproject_depth(depth, K_src, c2w_src).reshape(B, H * W, 3)
    proj = project_points(pts, K_tgt, w2c_tgt)
    uv = proj["uv"].reshape(B, H, W, 2)
    grid = torch.stack([uv[..., 0] / max(W - 1, 1) * 2 - 1, uv[..., 1] / max(H - 1, 1) * 2 - 1], dim=-1)
    warped = F.grid_sample(src, grid, align_corners=True, padding_mode="zeros")
    return {"warped": warped, "grid": grid, "valid": proj["valid_z"].reshape(B, H, W, 1)}


def compute_reprojection_error(points_world: torch.Tensor, uv_gt: torch.Tensor, K: torch.Tensor, w2c: torch.Tensor) -> torch.Tensor:
    uv = project_points(points_world, K, w2c)["uv"]
    return (uv - uv_gt).norm(dim=-1)


def compute_epipolar_residual(points_a: torch.Tensor, points_b: torch.Tensor, Fmat: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(*points_a.shape[:-1], 1, device=points_a.device, dtype=points_a.dtype)
    xa = torch.cat([points_a, ones], dim=-1)
    xb = torch.cat([points_b, ones], dim=-1)
    lines = torch.einsum("bij,bmj->bmi", Fmat, xa)
    num = torch.einsum("bmi,bmi->bm", xb, lines).abs()
    den = lines[..., :2].norm(dim=-1).clamp_min(1e-6)
    return num / den
