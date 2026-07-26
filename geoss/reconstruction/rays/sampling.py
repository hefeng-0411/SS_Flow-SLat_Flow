from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class CameraRayBundle:
    origins: torch.Tensor  # [B,V,P,3], world units
    directions: torch.Tensor  # [B,V,P,3], unit vectors
    pixels: torch.Tensor  # [B,V,P,2], image pixels (u,v)


def deterministic_pixel_grid(
    height: int,
    width: int,
    *,
    stride: int,
    batch_size: int,
    num_views: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Pixel-center grid shared by all views, returned as `[B,V,P,2]`."""

    if stride < 1:
        raise ValueError("stride must be positive")
    y = torch.arange(stride // 2, height, stride, device=device, dtype=dtype)
    x = torch.arange(stride // 2, width, stride, device=device, dtype=dtype)
    if y.numel() == 0:
        y = torch.tensor([(height - 1) * 0.5], device=device, dtype=dtype)
    if x.numel() == 0:
        x = torch.tensor([(width - 1) * 0.5], device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    pixels = torch.stack((xx, yy), dim=-1).reshape(1, 1, -1, 2)
    return pixels.expand(batch_size, num_views, -1, -1).contiguous()


def camera_rays_from_pixels(
    pixels: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
) -> CameraRayBundle:
    """Construct calibrated OpenCV rays for arbitrary pixels."""

    if pixels.ndim != 4 or pixels.shape[-1] != 2:
        raise ValueError(f"pixels must be [B,V,P,2], got {tuple(pixels.shape)}")
    if K.shape[:2] != pixels.shape[:2] or K.shape[-2:] != (3, 3):
        raise ValueError(f"K must be [B,V,3,3], got {tuple(K.shape)}")
    if c2w.shape[:2] != pixels.shape[:2] or c2w.shape[-2:] != (4, 4):
        raise ValueError(f"c2w must be [B,V,4,4], got {tuple(c2w.shape)}")
    b, v, p, _ = pixels.shape
    ones = torch.ones(b, v, p, 1, device=pixels.device, dtype=pixels.dtype)
    pixels_h = torch.cat((pixels, ones), dim=-1)
    camera_direction = torch.linalg.solve(
        K.float(),
        pixels_h.float().transpose(-1, -2),
    ).transpose(-1, -2)
    rotation = c2w[..., :3, :3].float()
    world_direction = torch.einsum("bvij,bvpj->bvpi", rotation, camera_direction)
    world_direction = F.normalize(world_direction, dim=-1, eps=1e-8).to(pixels.dtype)
    origins = c2w[..., :3, 3].to(pixels.dtype)[..., None, :].expand(b, v, p, 3)
    return CameraRayBundle(origins=origins, directions=world_direction, pixels=pixels)


def unproject_pixels(
    pixels: torch.Tensor,
    depth: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
) -> torch.Tensor:
    """Unproject camera-z depth at arbitrary pixels into world coordinates."""

    if depth.shape != pixels.shape[:-1] + (1,):
        raise ValueError(f"depth must be [B,V,P,1], got {tuple(depth.shape)}")
    ones = torch.ones_like(depth)
    pixels_h = torch.cat((pixels, ones), dim=-1)
    normalized = torch.linalg.solve(
        K.float(),
        pixels_h.float().transpose(-1, -2),
    ).transpose(-1, -2)
    camera = normalized * depth.float()
    world = torch.einsum(
        "bvij,bvpj->bvpi",
        c2w[..., :3, :3].float(),
        camera,
    ) + c2w[..., :3, 3].float()[..., None, :]
    return world.to(dtype=depth.dtype)


def camera_depth_to_ray_distance(
    pixels: torch.Tensor,
    camera_depth: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    """Convert camera-z depth to distance along a unit camera/world ray.

    VGGT and MeshFleet depth is parameterized by camera z, whereas the RAPC
    renderer integrates along unit-length world rays.  Comparing these values
    directly introduces a field-of-view-dependent depth bias.
    """

    if camera_depth.shape != pixels.shape[:-1] + (1,):
        raise ValueError("camera_depth must be [B,V,P,1] aligned with pixels")
    if K.shape != pixels.shape[:2] + (3, 3):
        raise ValueError("K must be [B,V,3,3] aligned with pixels")
    homogeneous = torch.cat((pixels, torch.ones_like(camera_depth)), dim=-1)
    camera_direction = torch.linalg.solve(
        K.float(),
        homogeneous.float().transpose(-1, -2),
    ).transpose(-1, -2)
    return (
        camera_depth.float() * camera_direction.norm(dim=-1, keepdim=True)
    ).to(camera_depth.dtype)


def sample_image_at_pixels(
    image: torch.Tensor,
    pixels: torch.Tensor,
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Sample `[B,V,C,H,W]` at `[B,V,P,2]`, returning `[B,V,P,C]`."""

    if image.ndim != 5 or pixels.ndim != 4:
        raise ValueError("image and pixels must be [B,V,C,H,W] and [B,V,P,2]")
    b, v, c, h, w = image.shape
    if pixels.shape[:2] != (b, v):
        raise ValueError("Image and pixel batch/view dimensions differ")
    grid = pixels.clone()
    grid[..., 0] = grid[..., 0] / max(w - 1, 1) * 2.0 - 1.0
    grid[..., 1] = grid[..., 1] / max(h - 1, 1) * 2.0 - 1.0
    sampled = F.grid_sample(
        image.reshape(b * v, c, h, w).float(),
        grid.reshape(b * v, -1, 1, 2).float(),
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.reshape(b, v, c, -1).permute(0, 1, 3, 2).to(image.dtype)


def sample_points_on_rays(
    origins: torch.Tensor,
    directions: torch.Tensor,
    depths: torch.Tensor,
) -> torch.Tensor:
    """Map ray depths to points.

    Args:
        origins/directions: `[B,R,3]`.
        depths: `[B,R,S]`.
    Returns:
        `[B,R,S,3]`.
    """

    if origins.shape != directions.shape or origins.ndim != 3:
        raise ValueError("origins and directions must both be [B,R,3]")
    if depths.shape[:2] != origins.shape[:2]:
        raise ValueError("depths must be [B,R,S]")
    return origins[..., None, :] + directions[..., None, :] * depths[..., None]


def ray_box_intersection(
    origins: torch.Tensor,
    directions: torch.Tensor,
    *,
    bounds: Tuple[float, float] = (-0.5, 0.5),
    epsilon: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stable slab intersection for any tensor ending in xyz."""

    if origins.shape != directions.shape or origins.shape[-1] != 3:
        raise ValueError("origins and directions must have equal shapes ending in xyz")
    lo, hi = bounds
    d = directions.float()
    o = origins.float()
    safe_d = torch.where(
        d.abs() >= epsilon,
        d,
        torch.where(d >= 0, torch.full_like(d, epsilon), torch.full_like(d, -epsilon)),
    )
    t0 = (lo - o) / safe_d
    t1 = (hi - o) / safe_d
    t_min = torch.minimum(t0, t1).amax(dim=-1)
    t_max = torch.maximum(t0, t1).amin(dim=-1)
    near = t_min.clamp_min(0.0)
    valid = t_max >= near
    return near.to(origins.dtype), t_max.to(origins.dtype), valid
