from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from geoss.utils.optional_deps import require_dependency


def read_reconstruction(path: str | Path, *, real_mode: bool = False) -> Dict[str, Any]:
    require_dependency("pycolmap", real_mode=real_mode, feature="COLMAP reconstruction read")
    import pycolmap

    rec = pycolmap.Reconstruction(str(path))
    return {
        "reconstruction": rec,
        "num_images": rec.num_reg_images(),
        "num_points3D": rec.num_points3D(),
        "mean_reprojection_error": rec.compute_mean_reprojection_error() if hasattr(rec, "compute_mean_reprojection_error") else None,
    }


def run_incremental_mapping(image_dir: str | Path, database_path: str | Path, output_dir: str | Path, *, real_mode: bool = False) -> Dict[str, Any]:
    require_dependency("pycolmap", real_mode=real_mode, feature="SfM sparse camera initialization")
    import pycolmap

    maps = pycolmap.incremental_mapping(str(database_path), str(image_dir), str(output_dir))
    return {"reconstructions": maps, "num_models": len(maps)}


def pose_consistency_summary(vggt_w2c, colmap_w2c) -> Dict[str, Any]:
    import torch

    trans = (vggt_w2c[..., :3, 3] - colmap_w2c[..., :3, 3]).norm(dim=-1)
    rot = torch.matmul(vggt_w2c[..., :3, :3], colmap_w2c[..., :3, :3].transpose(-1, -2))
    cos = ((torch.diagonal(rot, dim1=-1, dim2=-2).sum(dim=-1) - 1.0) * 0.5).clamp(-1, 1)
    angle = torch.rad2deg(torch.acos(cos))
    return {"translation_residual": trans, "rotation_residual_deg": angle}
