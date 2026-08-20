from __future__ import annotations

from pathlib import Path

import torch

from geoss.utils.visualization import save_npz, write_point_cloud_ply


def write_active_voxels_ply(path: str | Path, active_xyz: torch.Tensor, values: torch.Tensor | None = None) -> None:
    write_point_cloud_ply(path, active_xyz.reshape(-1, 3), values.reshape(-1) if values is not None else None)


def save_slat_debug_npz(path: str | Path, **arrays) -> None:
    save_npz(path, **arrays)
