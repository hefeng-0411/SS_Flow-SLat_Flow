from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoss.slat.utils.visibility import appearance_conflict_from_samples, compute_visibility_score


class VisibilityEvidenceSampler(nn.Module):
    """Sample per-active-voxel multi-view appearance, depth, and visibility evidence."""

    def __init__(
        self,
        evidence_dim: int = 128,
        feature_dim: int = 16,
        surface_threshold: float = 0.06,
        occlusion_margin: float = 0.08,
    ) -> None:
        super().__init__()
        self.evidence_dim = evidence_dim
        self.feature_dim = feature_dim
        self.surface_threshold = surface_threshold
        self.occlusion_margin = occlusion_margin
        # rgb(3), feature summary(feature_dim), mask, depth, residual, visibility, occlusion, in_bounds
        in_dim = 3 + feature_dim + 6
        self.token_mlp = nn.Sequential(
            nn.Linear(in_dim, evidence_dim),
            nn.SiLU(),
            nn.Linear(evidence_dim, evidence_dim),
        )

    def forward(
        self,
        active_xyz: torch.Tensor,
        uv: torch.Tensor,
        z_active: torch.Tensor,
        images: torch.Tensor,
        masks: torch.Tensor,
        in_bounds: torch.Tensor,
        *,
        depths: Optional[torch.Tensor] = None,
        vggt_depth: Optional[torch.Tensor] = None,
        vggt_pointmap: Optional[torch.Tensor] = None,
        vggt_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if images.ndim != 5 or masks.ndim != 5:
            raise ValueError(f"images/masks must be [B,N,C,H,W], got {tuple(images.shape)} {tuple(masks.shape)}")
        B, N, _, H, W = images.shape
        if uv.shape[:3] != (B, active_xyz.shape[1], N):
            raise ValueError(f"uv shape must be [B,L,N,2], got {tuple(uv.shape)}")
        L = active_xyz.shape[1]

        sampled_rgb = _grid_sample_views(images, uv, H, W)
        sampled_mask = _grid_sample_views(masks, uv, H, W).clamp(0, 1)
        depth_source = depths if depths is not None else vggt_depth
        if depth_source is not None:
            sampled_depth = _grid_sample_views(depth_source, uv, H, W)
            depth_valid = (sampled_depth > 1e-6).float()
            signed = z_active - sampled_depth
            depth_residual = signed.abs() * depth_valid
            occlusion_score = ((signed > self.occlusion_margin) & (sampled_mask > 0.5) & (depth_valid > 0.5)).float()
            has_depth = True
        else:
            sampled_depth = torch.zeros(B, L, N, 1, device=images.device, dtype=images.dtype)
            depth_valid = torch.zeros_like(sampled_depth)
            depth_residual = torch.zeros_like(sampled_depth)
            occlusion_score = torch.zeros_like(sampled_depth)
            has_depth = False

        sampled_features = _sample_vggt_features(vggt_features, uv, H, W, B, L, N, self.feature_dim, images.dtype, images.device)
        view_valid = (in_bounds > 0.5).float()
        mask_inside = (sampled_mask > 0.5).float()
        visibility = compute_visibility_score(
            view_valid,
            mask_inside,
            depth_residual,
            occlusion_score,
            surface_threshold=self.surface_threshold,
            has_depth=has_depth,
        )
        appearance_signal = sampled_features if sampled_features.shape[-1] > 0 else sampled_rgb
        appearance_conflict = appearance_conflict_from_samples(appearance_signal, visibility)

        token_input = torch.cat(
            [
                sampled_rgb,
                sampled_features,
                sampled_mask,
                sampled_depth,
                depth_residual,
                visibility,
                occlusion_score,
                view_valid,
            ],
            dim=-1,
        )
        view_slat_tokens = self.token_mlp(torch.nan_to_num(token_input))
        view_slat_tokens = view_slat_tokens * view_valid
        if not torch.isfinite(view_slat_tokens).all():
            raise FloatingPointError("VisibilityEvidenceSampler produced NaN/Inf tokens")

        return {
            "view_slat_tokens": view_slat_tokens,
            "sampled_rgb": sampled_rgb,
            "sampled_features": sampled_features,
            "visibility": visibility,
            "occlusion_score": occlusion_score,
            "depth_residual": depth_residual,
            "view_valid": view_valid,
            "appearance_conflict": appearance_conflict,
            "debug": {
                "sampled_mask": sampled_mask,
                "sampled_depth": sampled_depth,
                "depth_valid": depth_valid,
                "has_depth": torch.tensor(has_depth, device=images.device),
                "visibility_mean": visibility.mean(),
                "occlusion_mean": occlusion_score.mean(),
                "appearance_conflict_mean": appearance_conflict.mean(),
                "vggt_pointmap_used": torch.tensor(vggt_pointmap is not None, device=images.device),
            },
        }


def _grid_sample_views(maps: torch.Tensor, uv: torch.Tensor, H: int, W: int) -> torch.Tensor:
    B, N, C = maps.shape[:3]
    L = uv.shape[1]
    maps_flat = maps.reshape(B * N, C, maps.shape[-2], maps.shape[-1])
    grid = uv.permute(0, 2, 1, 3).reshape(B * N, L, 1, 2).to(maps.dtype)
    grid_x = grid[..., 0] / max(W - 1, 1) * 2.0 - 1.0
    grid_y = grid[..., 1] / max(H - 1, 1) * 2.0 - 1.0
    norm_grid = torch.stack([grid_x, grid_y], dim=-1)
    sampled = F.grid_sample(maps_flat, norm_grid, align_corners=True, padding_mode="zeros")
    return sampled[..., 0].permute(0, 2, 1).reshape(B, N, L, C).permute(0, 2, 1, 3).contiguous()


def _sample_vggt_features(
    features: Optional[torch.Tensor],
    uv: torch.Tensor,
    H: int,
    W: int,
    B: int,
    L: int,
    N: int,
    feature_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if features is None:
        return torch.zeros(B, L, N, feature_dim, dtype=dtype, device=device)
    features = features.to(device=device, dtype=dtype)
    if features.ndim == 5:
        sampled = _grid_sample_views(features, uv, H, W)
    elif features.ndim == 4:
        dense = _tokens_to_grid(features)
        if dense is None:
            raise ValueError("VGGT token features must include a square/patch-grid token count; per-view mean fallback is disabled.")
        sampled = _grid_sample_views(dense, uv, H, W)
    elif features.ndim == 3:
        sampled = features[:, None].expand(B, L, N, features.shape[-1])
    else:
        raise ValueError(f"Unsupported vggt_features shape {tuple(features.shape)}")
    if sampled.shape[-1] > feature_dim:
        sampled = sampled[..., :feature_dim]
    elif sampled.shape[-1] < feature_dim:
        pad = sampled.new_zeros(*sampled.shape[:-1], feature_dim - sampled.shape[-1])
        sampled = torch.cat([sampled, pad], dim=-1)
    return sampled


def _tokens_to_grid(features: torch.Tensor) -> Optional[torch.Tensor]:
    B, N, T, C = features.shape
    side = int(T ** 0.5)
    if side * side != T:
        return None
    return features.permute(0, 1, 3, 2).reshape(B, N, C, side, side).contiguous()
