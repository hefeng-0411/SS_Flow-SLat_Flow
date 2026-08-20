from __future__ import annotations

from typing import Dict, Tuple

import torch


def _as_batched_points(points: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if points.ndim == 2:
        return points.unsqueeze(0), True
    if points.ndim == 3:
        return points, False
    raise ValueError(f"points must be [M,3] or [B,M,3], got {tuple(points.shape)}")


def project_points(points_world: torch.Tensor, K: torch.Tensor, w2c: torch.Tensor, eps: float = 1e-6) -> Dict[str, torch.Tensor]:
    """Project world points with OpenCV K and w2c.

    Shapes:
        points_world: [B,M,3] or [M,3]
        K: [B,3,3] or [3,3]
        w2c: [B,4,4] or [4,4]
    """
    points_world, squeezed = _as_batched_points(points_world)
    B, M, _ = points_world.shape
    if K.ndim == 2:
        K = K.unsqueeze(0).expand(B, -1, -1)
    if w2c.ndim == 2:
        w2c = w2c.unsqueeze(0).expand(B, -1, -1)
    ones = torch.ones(B, M, 1, dtype=points_world.dtype, device=points_world.device)
    points_h = torch.cat([points_world, ones], dim=-1)
    cam = torch.einsum("bij,bmj->bmi", w2c[:, :3, :], points_h)
    z = cam[..., 2:3]
    xy_norm = cam[..., :2] / z.clamp_min(eps)
    uv_h = torch.einsum(
        "bij,bmj->bmi",
        K,
        torch.cat([xy_norm, torch.ones_like(z)], dim=-1),
    )
    uv = uv_h[..., :2]
    valid_z = z > eps
    if squeezed:
        cam, z, uv, valid_z = cam[0], z[0], uv[0], valid_z[0]
    return {"uv": uv, "depth": z, "cam_points": cam, "valid_z": valid_z}


def unproject_depth(depth: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """Unproject depth maps to world points in OpenCV convention.

    Shapes:
        depth: [B,1,H,W] or [B,H,W]
        K: [B,3,3]
        c2w: [B,4,4]
    Returns:
        world_points: [B,H,W,3]
    """
    if depth.ndim == 4:
        depth = depth[:, 0]
    if depth.ndim != 3:
        raise ValueError(f"depth must be [B,1,H,W] or [B,H,W], got {tuple(depth.shape)}")
    B, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    x = x[None].expand(B, -1, -1)
    y = y[None].expand(B, -1, -1)
    fx = K[:, 0, 0].view(B, 1, 1)
    fy = K[:, 1, 1].view(B, 1, 1)
    cx = K[:, 0, 2].view(B, 1, 1)
    cy = K[:, 1, 2].view(B, 1, 1)
    x_cam = (x - cx) * depth / fx
    y_cam = (y - cy) * depth / fy
    points_cam = torch.stack([x_cam, y_cam, depth], dim=-1)
    ones = torch.ones(B, H, W, 1, dtype=dtype, device=device)
    points_h = torch.cat([points_cam, ones], dim=-1)
    points_world = torch.einsum("bij,bhwj->bhwi", c2w, points_h)[..., :3]
    return points_world


def generate_camera_rays(H: int, W: int, K: torch.Tensor, c2w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate OpenCV camera rays in world coordinates.

    Returns:
        origins: [B,H,W,3]
        directions: [B,H,W,3], unit length
    """
    B = K.shape[0] if K.ndim == 3 else c2w.shape[0]
    if K.ndim == 2:
        K = K.unsqueeze(0).expand(B, -1, -1)
    if c2w.ndim == 2:
        c2w = c2w.unsqueeze(0).expand(B, -1, -1)
    depth = torch.ones(B, H, W, dtype=K.dtype, device=K.device)
    points_world = unproject_depth(depth, K, c2w)
    origins = c2w[:, :3, 3].view(B, 1, 1, 3).expand_as(points_world)
    directions = torch.nn.functional.normalize(points_world - origins, dim=-1)
    return origins, directions


def projection_roundtrip_check(
    points_world: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
    *,
    tolerance: float = 1e-4,
) -> Dict[str, torch.Tensor | bool]:
    """Project and unproject sparse points, returning the max reconstruction error."""
    w2c = torch.linalg.inv(c2w)
    proj = project_points(points_world, K, w2c)
    uv = proj["uv"]
    depth = proj["depth"]
    points, squeezed = _as_batched_points(points_world)
    if uv.ndim == 2:
        uv = uv.unsqueeze(0)
        depth = depth.unsqueeze(0)
    B, M, _ = points.shape
    fx = K[:, 0, 0].view(B, 1, 1)
    fy = K[:, 1, 1].view(B, 1, 1)
    cx = K[:, 0, 2].view(B, 1, 1)
    cy = K[:, 1, 2].view(B, 1, 1)
    x_cam = (uv[..., 0:1] - cx) * depth / fx
    y_cam = (uv[..., 1:2] - cy) * depth / fy
    cam = torch.cat([x_cam, y_cam, depth], dim=-1)
    cam_h = torch.cat([cam, torch.ones(B, M, 1, device=cam.device, dtype=cam.dtype)], dim=-1)
    recon = torch.einsum("bij,bmj->bmi", c2w, cam_h)[..., :3]
    err = (recon - points).norm(dim=-1)
    return {"max_error": err.max(), "ok": bool((err.max() < tolerance).item())}
