from __future__ import annotations

import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoss.slat.models.active_voxel_projector import ActiveVoxelProjector
from geoss.utils.coordinates import c2w_to_w2c


def test_active_voxel_projector_shapes():
    B, N, L, H, W = 1, 2, 16, 64, 64
    K = torch.eye(3).view(1, 1, 3, 3).repeat(B, N, 1, 1)
    K[..., 0, 0] = W
    K[..., 1, 1] = H
    K[..., 0, 2] = W / 2
    K[..., 1, 2] = H / 2
    c2w = torch.eye(4).view(1, 1, 4, 4).repeat(B, N, 1, 1)
    c2w[..., 2, 3] = -2.0
    w2c = c2w_to_w2c(c2w)
    active_indices = torch.randint(24, 40, (B, L, 3))
    out = ActiveVoxelProjector(resolution=64)(active_indices, K, w2c, (H, W), c2w=c2w)
    assert out["active_xyz"].shape == (B, L, 3)
    assert out["uv"].shape == (B, L, N, 2)
    assert out["z_active"].shape == (B, L, N, 1)
    assert out["in_bounds"].shape == (B, L, N, 1)
    assert out["ss_confidence"].shape == (B, L, 1)


if __name__ == "__main__":
    test_active_voxel_projector_shapes()
