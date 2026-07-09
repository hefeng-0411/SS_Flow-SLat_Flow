from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoss.geometry.differentiable_camera import project_points


class RayEvidenceSampler(nn.Module):
    """Sample occupied/free-space evidence for each anchor along each camera ray."""

    def __init__(
        self,
        evidence_dim: int = 128,
        depth_near_threshold: float = 0.035,
        depth_free_margin: float = 0.02,
    ) -> None:
        super().__init__()
        self.evidence_dim = evidence_dim
        self.depth_near_threshold = depth_near_threshold
        self.depth_free_margin = depth_free_margin
        self.evidence_mlp = nn.Sequential(
            nn.Linear(12, evidence_dim),
            nn.SiLU(),
            nn.Linear(evidence_dim, evidence_dim),
        )
        self.occ_head = nn.Linear(evidence_dim, 1)
        self.free_head = nn.Linear(evidence_dim, 1)
        nn.init.zeros_(self.occ_head.weight)
        nn.init.zeros_(self.occ_head.bias)
        nn.init.zeros_(self.free_head.weight)
        nn.init.zeros_(self.free_head.bias)

    def forward(
        self,
        anchor_xyz: torch.Tensor,
        K: torch.Tensor,
        c2w: torch.Tensor,
        w2c: torch.Tensor,
        masks: torch.Tensor,
        depths: Optional[torch.Tensor] = None,
        vggt_depth: Optional[torch.Tensor] = None,
        vggt_pointmap: Optional[torch.Tensor] = None,
        vggt_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if anchor_xyz.ndim != 3 or anchor_xyz.shape[-1] != 3:
            raise ValueError(f"anchor_xyz must be [B,M,3], got {tuple(anchor_xyz.shape)}")
        if masks.ndim != 5:
            raise ValueError(f"masks must be [B,N,1,H,W], got {tuple(masks.shape)}")
        B, M, _ = anchor_xyz.shape
        Bm, N, _, H, W = masks.shape
        if B != Bm:
            raise ValueError(f"Batch mismatch anchors={B}, masks={Bm}")
        depth_source = depths if depths is not None else vggt_depth
        if depth_source is not None:
            depth_source = _depth_to_b_n_1_h_w(depth_source, B, N)

        flat_points = anchor_xyz[:, None].expand(B, N, M, 3).reshape(B * N, M, 3)
        flat_K = K.reshape(B * N, 3, 3)
        flat_w2c = w2c.reshape(B * N, 4, 4)
        proj = project_points(flat_points, flat_K, flat_w2c)
        uv = proj["uv"].reshape(B, N, M, 2)
        anchor_depth = proj["depth"].reshape(B, N, M, 1)
        valid_z = proj["valid_z"].reshape(B, N, M, 1)

        grid_x = uv[..., 0] / max(W - 1, 1) * 2.0 - 1.0
        grid_y = uv[..., 1] / max(H - 1, 1) * 2.0 - 1.0
        in_bounds = ((grid_x >= -1.0) & (grid_x <= 1.0) & (grid_y >= -1.0) & (grid_y <= 1.0)).unsqueeze(-1)
        sample_grid = torch.stack([grid_x, grid_y], dim=-1).reshape(B * N, M, 1, 2)
        feature_samples = _sample_spatial_features(vggt_features, sample_grid, B, N, M) if vggt_features is not None else None

        mask_samples = F.grid_sample(
            masks.reshape(B * N, 1, H, W).float(),
            sample_grid,
            align_corners=True,
            mode="bilinear",
            padding_mode="zeros",
        ).reshape(B, N, 1, M).permute(0, 1, 3, 2)

        if depth_source is not None:
            depth_samples = F.grid_sample(
                depth_source.reshape(B * N, 1, H, W).float(),
                sample_grid,
                align_corners=True,
                mode="bilinear",
                padding_mode="zeros",
            ).reshape(B, N, 1, M).permute(0, 1, 3, 2)
            valid_depth = depth_samples > 1e-6
        else:
            depth_samples = torch.zeros_like(anchor_depth)
            valid_depth = torch.zeros_like(anchor_depth, dtype=torch.bool)

        depth_residual = anchor_depth - depth_samples
        abs_residual = depth_residual.abs()
        near_surface = valid_depth & (abs_residual <= self.depth_near_threshold)
        in_front_of_surface = valid_depth & (anchor_depth < (depth_samples - self.depth_free_margin))
        mask_inside = mask_samples > 0.5
        mask_outside = ~mask_inside
        ray_valid = valid_z & in_bounds
        occ_geom = ray_valid & mask_inside & near_surface
        free_geom = ray_valid & (mask_outside | in_front_of_surface)
        valid_count = ray_valid.float().sum(dim=1).clamp_min(1.0)
        occ_count = occ_geom.float().sum(dim=1)
        free_count = free_geom.float().sum(dim=1)
        conflict_score = torch.minimum(occ_count, free_count) / valid_count
        view_conflict = conflict_score[:, None].expand(B, N, M, 1)

        uv_norm = torch.stack([grid_x, grid_y], dim=-1)
        max_depth = anchor_depth.detach().abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
        evidence = torch.cat(
            [
                uv_norm,
                in_bounds.float(),
                mask_samples,
                (anchor_depth / max_depth),
                (depth_samples / max_depth),
                (depth_residual / max_depth),
                (abs_residual / max_depth),
                near_surface.float(),
                in_front_of_surface.float(),
                mask_outside.float(),
                ray_valid.float(),
            ],
            dim=-1,
        )
        view_tokens = self.evidence_mlp(evidence).permute(0, 2, 1, 3).contiguous()
        occ_score = self.occ_head(view_tokens) + 2.0 * occ_geom.permute(0, 2, 1, 3).float()
        free_score = self.free_head(view_tokens) + 2.0 * free_geom.permute(0, 2, 1, 3).float()
        visibility = (ray_valid & mask_inside).permute(0, 2, 1, 3).float()
        out = {
            "view_tokens": view_tokens,
            "occ_score": occ_score,
            "free_score": free_score,
            "visibility": visibility,
            "depth_residual": abs_residual.permute(0, 2, 1, 3).contiguous(),
            "signed_depth_residual": depth_residual.permute(0, 2, 1, 3).contiguous(),
            "ray_valid": ray_valid.permute(0, 2, 1, 3).float().contiguous(),
            "conflict_score": conflict_score.contiguous(),
            "view_conflict": view_conflict.permute(0, 2, 1, 3).contiguous(),
            "evidence_debug": {
                "uv": uv.permute(0, 2, 1, 3).contiguous(),
                "in_bounds": in_bounds.permute(0, 2, 1, 3).float().contiguous(),
                "mask_samples": mask_samples.permute(0, 2, 1, 3).contiguous(),
                "anchor_depth": anchor_depth.permute(0, 2, 1, 3).contiguous(),
                "surface_depth": depth_samples.permute(0, 2, 1, 3).contiguous(),
                "valid_depth": valid_depth.permute(0, 2, 1, 3).float().contiguous(),
                "occupied_geometry": occ_geom.permute(0, 2, 1, 3).float().contiguous(),
                "free_geometry": free_geom.permute(0, 2, 1, 3).float().contiguous(),
                "feature_samples": feature_samples,
            },
        }
        assert out["view_tokens"].shape == (B, M, N, self.evidence_dim)
        return out


def _depth_to_b_n_1_h_w(depth: torch.Tensor, B: int, N: int) -> torch.Tensor:
    if depth.ndim != 5:
        raise ValueError(f"depth must be 5D, got {tuple(depth.shape)}")
    if depth.shape[:3] == (B, N, 1):
        return depth
    if depth.shape[0] == B and depth.shape[1] == N and depth.shape[-1] == 1:
        return depth.permute(0, 1, 4, 2, 3).contiguous()
    raise ValueError(f"Unsupported depth shape {tuple(depth.shape)} for B={B}, N={N}")


def _sample_spatial_features(
    features: torch.Tensor,
    sample_grid: torch.Tensor,
    B: int,
    N: int,
    M: int,
) -> Optional[torch.Tensor]:
    """Best-effort feature sampler for `[B,N,C,Hf,Wf]` feature maps.

    Token-form features are left untouched by returning `None`; the evidence
    geometry remains valid without dense feature sampling.
    """
    if features.ndim != 5 or features.shape[0] != B or features.shape[1] != N:
        if features.ndim == 4 and features.shape[0] == B and features.shape[1] == N:
            side = int(features.shape[2] ** 0.5)
            if side * side != features.shape[2]:
                return None
            features = features.permute(0, 1, 3, 2).reshape(B, N, features.shape[-1], side, side).contiguous()
        else:
            return None
    _, _, C, Hf, Wf = features.shape
    sampled = F.grid_sample(
        features.reshape(B * N, C, Hf, Wf).float(),
        sample_grid,
        align_corners=True,
        mode="bilinear",
        padding_mode="zeros",
    ).reshape(B, N, C, M).permute(0, 3, 1, 2).contiguous()
    return sampled.to(features.dtype)
