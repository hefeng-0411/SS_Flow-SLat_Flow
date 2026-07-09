from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from geoss.slat.utils.active_voxel_utils import indices_to_active_xyz
from geoss.geometry.differentiable_camera import project_points


class ActiveVoxelProjector(nn.Module):
    """Project TRELLIS/SS active voxels into every input view."""

    def __init__(self, resolution: int = 64) -> None:
        super().__init__()
        self.resolution = resolution

    def forward(
        self,
        active_indices: Optional[torch.Tensor],
        K: torch.Tensor,
        w2c: torch.Tensor,
        image_size: Tuple[int, int],
        *,
        active_xyz: Optional[torch.Tensor] = None,
        ss_confidence: Optional[torch.Tensor] = None,
        c2w: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if K.ndim != 4 or w2c.ndim != 4:
            raise ValueError(f"K and w2c must be [B,N,3,3]/[B,N,4,4], got {tuple(K.shape)} {tuple(w2c.shape)}")
        B, N = K.shape[:2]
        H, W = int(image_size[0]), int(image_size[1])
        if active_xyz is None:
            if active_indices is None:
                raise ValueError("active_indices or active_xyz is required")
            if active_indices.ndim != 3:
                raise ValueError(f"active_indices must be [B,L,3/4], got {tuple(active_indices.shape)}")
            active_xyz = indices_to_active_xyz(active_indices, self.resolution).to(device=K.device, dtype=K.dtype)
        else:
            active_xyz = active_xyz.to(device=K.device, dtype=K.dtype)
        if active_xyz.ndim != 3 or active_xyz.shape[0] != B or active_xyz.shape[-1] != 3:
            raise ValueError(f"active_xyz must be [B,L,3] with B={B}, got {tuple(active_xyz.shape)}")
        L = active_xyz.shape[1]

        points = active_xyz[:, None].expand(B, N, L, 3).reshape(B * N, L, 3)
        proj = project_points(points, K.reshape(B * N, 3, 3), w2c.reshape(B * N, 4, 4))
        uv = proj["uv"].reshape(B, N, L, 2).permute(0, 2, 1, 3).contiguous()
        z_active = proj["depth"].reshape(B, N, L, 1).permute(0, 2, 1, 3).contiguous()
        valid_z = proj["valid_z"].reshape(B, N, L, 1).permute(0, 2, 1, 3).contiguous()
        in_bounds = (
            valid_z
            & (uv[..., 0:1] >= 0)
            & (uv[..., 0:1] <= max(W - 1, 1))
            & (uv[..., 1:2] >= 0)
            & (uv[..., 1:2] <= max(H - 1, 1))
        )

        if ss_confidence is None:
            ss_confidence = torch.ones(B, L, 1, device=K.device, dtype=K.dtype)
        else:
            ss_confidence = ss_confidence.to(device=K.device, dtype=K.dtype)
            if ss_confidence.ndim == 2:
                ss_confidence = ss_confidence[..., None]
            if ss_confidence.shape[1] != L:
                ss_confidence = _resize_confidence(ss_confidence, L)
            if ss_confidence.shape != (B, L, 1):
                raise ValueError(f"ss_confidence must be [B,L,1], got {tuple(ss_confidence.shape)}")

        return {
            "active_xyz": active_xyz,
            "uv": uv,
            "z_active": z_active,
            "in_bounds": in_bounds.to(K.dtype),
            "ss_confidence": ss_confidence.clamp(0, 1),
            "projection_debug": {
                "image_size": torch.tensor([H, W], device=K.device),
                "valid_ratio": in_bounds.float().mean(),
                "has_c2w": torch.tensor(c2w is not None, device=K.device),
            },
        }


def _resize_confidence(confidence: torch.Tensor, target_len: int) -> torch.Tensor:
    if confidence.shape[1] > target_len:
        return confidence[:, :target_len]
    pad = confidence[:, -1:].expand(-1, target_len - confidence.shape[1], -1)
    return torch.cat([confidence, pad], dim=1)
