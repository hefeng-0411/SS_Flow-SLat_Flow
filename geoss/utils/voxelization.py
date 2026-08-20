from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch

from .coordinates import anchor_to_occ_index, canonical_to_trellis_unit


def points_to_occupancy(
    points_xyz: torch.Tensor,
    resolution: int = 64,
    *,
    canonical_range: Tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """Voxelize points into a dense occupancy grid [R,R,R]."""
    if points_xyz.numel() == 0:
        return torch.zeros(resolution, resolution, resolution, dtype=torch.bool, device=points_xyz.device)
    if canonical_range == (-1.0, 1.0):
        unit = canonical_to_trellis_unit(points_xyz)
    else:
        lo, hi = canonical_range
        unit = (points_xyz - lo) / (hi - lo)
    coords = (unit.clamp(0, 1 - 1e-6) * resolution).long()
    occ = torch.zeros(resolution, resolution, resolution, dtype=torch.bool, device=points_xyz.device)
    occ[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return occ


def sample_occupancy_at_points(
    gt_occ: torch.Tensor,
    points_xyz: torch.Tensor,
    *,
    canonical_range: Tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """Nearest-neighbor sample occupancy grid at canonical points.

    Shapes:
        gt_occ: [B,R,R,R] or [R,R,R]
        points_xyz: [B,M,3]
    Returns:
        [B,M,1] float occupancy labels.
    """
    if gt_occ.ndim == 3:
        gt_occ = gt_occ.unsqueeze(0)
    B, R, _, _ = gt_occ.shape
    if points_xyz.shape[0] != B:
        if B == 1:
            gt_occ = gt_occ.expand(points_xyz.shape[0], -1, -1, -1)
            B = points_xyz.shape[0]
        else:
            raise ValueError(f"Batch mismatch gt_occ={B}, points={points_xyz.shape[0]}")
    if canonical_range == (-1.0, 1.0):
        idx = anchor_to_occ_index(points_xyz, R)
    else:
        lo, hi = canonical_range
        unit = (points_xyz - lo) / (hi - lo)
        idx = (unit.clamp(0, 1 - 1e-6) * R).long()
    batch = torch.arange(B, device=points_xyz.device)[:, None].expand(-1, idx.shape[1])
    labels = gt_occ.to(points_xyz.device)[batch, idx[..., 0], idx[..., 1], idx[..., 2]]
    return labels.float().unsqueeze(-1)


def voxelize_mesh_to_occ(
    mesh_path: str | Path,
    resolution: int = 64,
    *,
    normalize: bool = True,
) -> Optional[torch.Tensor]:
    """Best-effort mesh voxelization using trimesh if available."""
    try:
        import trimesh
    except Exception:
        return None
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        return None
    mesh = trimesh.load(mesh_path, force="mesh")
    if mesh.is_empty:
        return None
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    if normalize:
        center = (vertices.max(dim=0).values + vertices.min(dim=0).values) * 0.5
        scale = (vertices.max(dim=0).values - vertices.min(dim=0).values).max().clamp_min(1e-6)
        vertices = (vertices - center) / scale * 2.0
    return points_to_occupancy(vertices, resolution=resolution)
