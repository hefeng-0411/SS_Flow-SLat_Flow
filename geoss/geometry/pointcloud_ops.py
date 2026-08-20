from __future__ import annotations

from typing import Dict, Tuple

import torch

from geoss.utils.optional_deps import require_dependency


def voxel_downsample(points: torch.Tensor, voxel_size: float, *, real_mode: bool = False) -> torch.Tensor:
    if require_dependency("open3d", real_mode=real_mode, feature="point cloud voxel downsample"):
        import open3d as o3d
        import numpy as np

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.detach().cpu().numpy()))
        pcd = pcd.voxel_down_sample(voxel_size)
        return torch.tensor(np.asarray(pcd.points), dtype=points.dtype, device=points.device)
    q = torch.round(points / voxel_size)
    return torch.unique(q, dim=0) * voxel_size


def estimate_normals(points: torch.Tensor, *, real_mode: bool = False) -> torch.Tensor:
    require_dependency("open3d", real_mode=real_mode, feature="normal estimation")
    import open3d as o3d
    import numpy as np

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.detach().cpu().numpy()))
    pcd.estimate_normals()
    return torch.tensor(np.asarray(pcd.normals), dtype=points.dtype, device=points.device)


def normalize_to_canonical_cube(points: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    lo = points.amin(dim=0)
    hi = points.amax(dim=0)
    center = (lo + hi) * 0.5
    scale = (hi - lo).amax().clamp_min(1e-6) * 0.5
    return ((points - center) / scale).clamp(-1, 1), {"center": center, "scale": scale}


def icp_align(source: torch.Tensor, target: torch.Tensor, *, threshold: float = 0.05, real_mode: bool = False) -> Dict[str, torch.Tensor]:
    require_dependency("open3d", real_mode=real_mode, feature="ICP alignment")
    import open3d as o3d
    import numpy as np

    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source.detach().cpu().numpy()))
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target.detach().cpu().numpy()))
    reg = o3d.pipelines.registration.registration_icp(src, tgt, threshold, np.eye(4))
    return {"transform": torch.tensor(reg.transformation, dtype=source.dtype, device=source.device), "fitness": torch.tensor(reg.fitness)}
