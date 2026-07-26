from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VisualHullConfig:
    """Canonical observation-only visual-hull reconstruction contract."""

    resolution: int = 160
    bounds_min: tuple[float, float, float] = (-0.5, -0.5, -0.5)
    bounds_max: tuple[float, float, float] = (0.5, 0.5, 0.5)
    mask_threshold: float = 0.5
    min_view_fraction: float = 1.0
    min_valid_views: int = 2
    mask_dilation_pixels: int = 2
    closing_iterations: int = 1
    chunk_size: int = 262144

    def __post_init__(self) -> None:
        if self.resolution < 16:
            raise ValueError("visual-hull resolution must be at least 16")
        if len(self.bounds_min) != 3 or len(self.bounds_max) != 3:
            raise ValueError("visual-hull bounds must be three-dimensional")
        if any(hi <= lo for lo, hi in zip(self.bounds_min, self.bounds_max)):
            raise ValueError("visual-hull upper bounds must exceed lower bounds")
        if not 0.0 < self.mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be in (0,1]")
        if not 0.0 < self.min_view_fraction <= 1.0:
            raise ValueError("min_view_fraction must be in (0,1]")
        if self.min_valid_views < 1 or self.chunk_size < 1:
            raise ValueError("min_valid_views and chunk_size must be positive")
        if self.mask_dilation_pixels < 0 or self.closing_iterations < 0:
            raise ValueError("morphology settings must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@torch.inference_mode()
def carve_visual_hull(
    masks: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
    config: VisualHullConfig,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Carve a canonical occupancy grid using only conditioning silhouettes.

    A voxel survives when its projection is inside the foreground silhouette in
    the configured fraction of cameras where it is in front of the image plane.
    No mesh, voxel, latent, depth target, or held-out render is accepted.
    """
    masks, K, w2c = _validate_inputs(masks, K, w2c)
    device = masks.device
    dtype = torch.float32
    masks = masks.to(dtype=dtype).clamp(0.0, 1.0)
    masks = _dilate_masks(masks, config.mask_dilation_pixels)
    resolution = int(config.resolution)
    bounds_min = torch.tensor(config.bounds_min, device=device, dtype=dtype)
    bounds_max = torch.tensor(config.bounds_max, device=device, dtype=dtype)
    axes = [
        torch.linspace(bounds_min[axis], bounds_max[axis], resolution, device=device, dtype=dtype)
        for axis in range(3)
    ]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    keep_parts = []
    support_sums = torch.zeros(masks.shape[0], device=device, dtype=torch.float64)
    valid_sums = torch.zeros(masks.shape[0], device=device, dtype=torch.float64)
    for start in range(0, grid.shape[0], int(config.chunk_size)):
        points = grid[start : start + int(config.chunk_size)]
        support, valid = _sample_silhouette_support(points, masks, K, w2c)
        support_sums += support.sum(dim=1, dtype=torch.float64)
        valid_sums += valid.sum(dim=1, dtype=torch.float64)
        valid_count = valid.sum(dim=0)
        inside_count = (support >= float(config.mask_threshold)).logical_and(valid).sum(dim=0)
        fraction = inside_count.float() / valid_count.clamp_min(1).float()
        keep_parts.append(
            (valid_count >= int(config.min_valid_views))
            & (fraction >= float(config.min_view_fraction))
        )
    occupancy = torch.cat(keep_parts).reshape(resolution, resolution, resolution)
    occupancy = _binary_close_3d(occupancy, config.closing_iterations)
    occupied = int(occupancy.sum().item())
    total = int(occupancy.numel())
    if occupied == 0:
        raise RuntimeError(
            "Visual-hull carving produced an empty volume. Check camera convention, "
            "mask alpha, framing, and min_view_fraction."
        )
    fraction = occupied / total
    if fraction >= 0.9:
        raise RuntimeError(
            f"Visual-hull occupancy fraction {fraction:.4f} is implausibly large; "
            "camera projections or masks are likely invalid."
        )
    report = {
        "protocol": "conditioning_visual_hull_v1",
        "config": config.as_dict(),
        "num_views": int(masks.shape[0]),
        "occupied_voxels": occupied,
        "total_voxels": total,
        "occupancy_fraction": fraction,
        "per_view_foreground_support_fraction": [
            float((support_sums[index] / valid_sums[index].clamp_min(1.0)).cpu())
            for index in range(masks.shape[0])
        ],
        "uses_heldout_views": False,
        "uses_gt_3d": False,
    }
    return occupancy, report


def extract_visual_hull_mesh(
    occupancy: torch.Tensor,
    bounds_min: Sequence[float] = (-0.5, -0.5, -0.5),
    bounds_max: Sequence[float] = (0.5, 0.5, 0.5),
):
    """Extract a closed triangle mesh while preserving canonical coordinates."""
    if occupancy.ndim != 3 or min(occupancy.shape) < 2:
        raise ValueError(f"occupancy must be a 3D grid, got {tuple(occupancy.shape)}")
    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:
        raise RuntimeError(
            "Visual-hull mesh extraction requires scikit-image in the active environment."
        ) from exc
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "Visual-hull mesh export requires trimesh in the active environment."
        ) from exc
    volume = occupancy.detach().bool().cpu().numpy()
    # Zero padding closes objects that legitimately touch the normalized cube.
    padded = np.pad(volume.astype(np.float32), 1, mode="constant", constant_values=0)
    lo = np.asarray(bounds_min, dtype=np.float32)
    hi = np.asarray(bounds_max, dtype=np.float32)
    shape = np.asarray(volume.shape, dtype=np.float32)
    spacing = (hi - lo) / np.maximum(shape - 1.0, 1.0)
    vertices, faces, normals, _ = marching_cubes(
        padded,
        level=0.5,
        spacing=tuple(float(value) for value in spacing),
        allow_degenerate=False,
    )
    vertices = vertices + lo[None] - spacing[None]
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,
        process=False,
    )
    mesh.remove_unreferenced_vertices()
    if hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    elif hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    elif hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError("Marching cubes returned an empty visual-hull mesh.")
    return mesh


def _validate_inputs(
    masks: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if masks.ndim == 3:
        masks = masks[:, None]
    if masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError(f"masks must be [V,1,H,W], got {tuple(masks.shape)}")
    views = masks.shape[0]
    if K.shape != (views, 3, 3):
        raise ValueError(f"K must be [V,3,3], got {tuple(K.shape)}")
    if w2c.shape != (views, 4, 4):
        raise ValueError(f"w2c must be [V,4,4], got {tuple(w2c.shape)}")
    if not torch.isfinite(masks).all() or not torch.isfinite(K).all() or not torch.isfinite(w2c).all():
        raise ValueError("visual-hull inputs contain NaN or Inf")
    return masks, K.to(device=masks.device, dtype=torch.float32), w2c.to(
        device=masks.device,
        dtype=torch.float32,
    )


def _sample_silhouette_support(
    points: torch.Tensor,
    masks: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    views, _, height, width = masks.shape
    homogeneous = torch.cat(
        [points, torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)],
        dim=-1,
    )
    camera = torch.einsum("vij,pj->vpi", w2c[:, :3, :], homogeneous)
    depth = camera[..., 2]
    projected = torch.einsum("vij,vpj->vpi", K, camera)
    u = projected[..., 0] / projected[..., 2].clamp_min(1e-8)
    v = projected[..., 1] / projected[..., 2].clamp_min(1e-8)
    x_grid = 2.0 * u / max(1, width - 1) - 1.0
    y_grid = 2.0 * v / max(1, height - 1) - 1.0
    sample_grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(2)
    support = F.grid_sample(
        masks,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0, :, 0]
    valid = (
        (depth > 1e-6)
        & (x_grid >= -1.0)
        & (x_grid <= 1.0)
        & (y_grid >= -1.0)
        & (y_grid <= 1.0)
    )
    return support, valid


def _dilate_masks(masks: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels <= 0:
        return masks
    kernel = 2 * int(pixels) + 1
    return F.max_pool2d(masks, kernel_size=kernel, stride=1, padding=int(pixels))


def _binary_close_3d(occupancy: torch.Tensor, iterations: int) -> torch.Tensor:
    value = occupancy.float()[None, None]
    for _ in range(int(iterations)):
        value = F.max_pool3d(value, kernel_size=3, stride=1, padding=1)
        value = 1.0 - F.max_pool3d(1.0 - value, kernel_size=3, stride=1, padding=1)
    return value[0, 0] >= 0.5
