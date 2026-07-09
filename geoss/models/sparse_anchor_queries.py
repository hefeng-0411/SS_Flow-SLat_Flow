from __future__ import annotations

import math
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAnchorQueries(nn.Module):
    """Generate sparse 3D anchor queries in canonical [-1, 1]^3 space."""

    def __init__(
        self,
        num_anchors: int = 4096,
        feature_dim: int = 256,
        init: Literal["grid", "random"] = "grid",
        learnable_xyz: bool = True,
        learnable_feat: bool = True,
        xyz_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        if num_anchors not in {2048, 4096, 8192}:
            raise ValueError("SparseAnchorQueries supports M in {2048, 4096, 8192}")
        self.num_anchors = num_anchors
        self.feature_dim = feature_dim
        self.xyz_range = xyz_range
        anchors = self._make_initial_xyz(num_anchors, init, xyz_range)
        self.anchor_xyz = nn.Parameter(anchors, requires_grad=learnable_xyz)
        self.anchor_feat = nn.Parameter(torch.randn(num_anchors, feature_dim) * 0.02, requires_grad=learnable_feat)

    @staticmethod
    def _make_initial_xyz(num_anchors: int, init: str, xyz_range: Tuple[float, float]) -> torch.Tensor:
        lo, hi = xyz_range
        if init == "random":
            return torch.empty(num_anchors, 3).uniform_(lo, hi)
        if init != "grid":
            raise ValueError(f"Unknown anchor init {init}")
        side = math.ceil(num_anchors ** (1.0 / 3.0))
        lin = torch.linspace(lo, hi, side)
        grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), dim=-1).reshape(-1, 3)
        return grid[:num_anchors].contiguous()

    def forward(self, batch_size: int, device: torch.device | str | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or self.anchor_xyz.device
        lo, hi = self.xyz_range
        xyz = self.anchor_xyz.to(device).clamp(lo, hi)
        feat = self.anchor_feat.to(device)
        anchor_xyz = xyz.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        anchor_feat = feat.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        assert anchor_xyz.shape == (batch_size, self.num_anchors, 3)
        assert anchor_feat.shape == (batch_size, self.num_anchors, self.feature_dim)
        return anchor_xyz, anchor_feat

    def forward_dynamic(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        aligned_pointmap: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        num_surface: Optional[int] = None,
        num_boundary: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build learnable + VGGT-surface + uncertainty/boundary anchors.

        Returns fixed-size `[B,M,*]` tensors so existing samplers remain simple.
        Metadata fields:
          source_type: 0 learnable, 1 surface, 2 boundary/uncertainty
          scale_level: 0 coarse, 1 fine
        """
        global_xyz, global_feat = self.forward(batch_size, device=device)
        B, M, _ = global_xyz.shape
        device = global_xyz.device
        dtype = global_xyz.dtype
        n_surface = int(num_surface if num_surface is not None else max(0, M // 2))
        n_boundary = int(num_boundary if num_boundary is not None else max(0, M // 8))
        n_global = max(1, M - n_surface - n_boundary)
        global_xyz = global_xyz[:, :n_global]
        global_feat = global_feat[:, :n_global]
        meta = _metadata_template(B, n_global, device, dtype, source_type=0.0, scale_level=0.0)
        xyz_parts = [global_xyz]
        feat_parts = [global_feat]
        meta_parts = [meta]

        if aligned_pointmap is not None and n_surface + n_boundary > 0:
            surface = _sample_pointmap_anchors(aligned_pointmap, masks, confidence, n_surface, boundary=False)
            boundary = _sample_pointmap_anchors(aligned_pointmap, masks, confidence, n_boundary, boundary=True)
            if n_surface > 0:
                xyz_parts.append(surface["xyz"].to(device=device, dtype=dtype))
                feat_parts.append(_feature_from_metadata(surface["metadata"].to(device=device, dtype=dtype), self.feature_dim))
                meta_parts.append(surface["metadata"].to(device=device, dtype=dtype))
            if n_boundary > 0:
                xyz_parts.append(boundary["xyz"].to(device=device, dtype=dtype))
                feat_parts.append(_feature_from_metadata(boundary["metadata"].to(device=device, dtype=dtype), self.feature_dim))
                meta_parts.append(boundary["metadata"].to(device=device, dtype=dtype))

        anchor_xyz = torch.cat(xyz_parts, dim=1)
        anchor_feat = torch.cat(feat_parts, dim=1)
        metadata = torch.cat(meta_parts, dim=1)
        if anchor_xyz.shape[1] < M:
            pad = M - anchor_xyz.shape[1]
            anchor_xyz = torch.cat([anchor_xyz, global_xyz[:, :1].expand(-1, pad, -1)], dim=1)
            anchor_feat = torch.cat([anchor_feat, global_feat[:, :1].expand(-1, pad, -1)], dim=1)
            metadata = torch.cat([metadata, meta[:, :1].expand(-1, pad, -1)], dim=1)
        anchor_xyz = anchor_xyz[:, :M].clamp(*self.xyz_range).contiguous()
        anchor_feat = anchor_feat[:, :M].contiguous()
        metadata = metadata[:, :M].contiguous()
        return {
            "anchor_xyz": anchor_xyz,
            "anchor_feat": anchor_feat,
            "anchor_metadata": metadata,
            "source_type": metadata[..., 0:1],
            "scale_level": metadata[..., 1:2],
            "surface_confidence": metadata[..., 2:3],
            "free_space_confidence": metadata[..., 3:4],
            "view_support_count": metadata[..., 4:5],
            "conflict_score": metadata[..., 5:6],
            "uncertainty_score": metadata[..., 6:7],
        }


def _metadata_template(B: int, M: int, device, dtype, *, source_type: float, scale_level: float) -> torch.Tensor:
    meta = torch.zeros(B, M, 7, device=device, dtype=dtype)
    meta[..., 0] = source_type
    meta[..., 1] = scale_level
    meta[..., 2] = 0.5
    meta[..., 3] = 0.5
    meta[..., 4] = 1.0
    meta[..., 6] = 0.5
    return meta


def _sample_pointmap_anchors(
    pointmap: torch.Tensor,
    masks: Optional[torch.Tensor],
    confidence: Optional[torch.Tensor],
    count: int,
    *,
    boundary: bool,
) -> Dict[str, torch.Tensor]:
    B, N, _, H, W = pointmap.shape
    if count <= 0:
        return {"xyz": pointmap.new_zeros(B, 0, 3), "metadata": pointmap.new_zeros(B, 0, 7)}
    requested = count
    pts = pointmap.permute(0, 1, 3, 4, 2).reshape(B, N * H * W, 3)
    if masks is None:
        score = torch.ones(B, N, 1, H, W, device=pointmap.device, dtype=pointmap.dtype)
    else:
        score = masks.to(device=pointmap.device, dtype=pointmap.dtype)
        if score.shape[-2:] != (H, W):
            score = F.interpolate(score.reshape(B * N, 1, score.shape[-2], score.shape[-1]), size=(H, W), mode="bilinear", align_corners=False).reshape(B, N, 1, H, W)
        if boundary:
            pooled = F.avg_pool2d(score.reshape(B * N, 1, H, W), 3, stride=1, padding=1).reshape(B, N, 1, H, W)
            score = (score - pooled).abs()
    if confidence is not None:
        conf = confidence if confidence.ndim == 5 else confidence.unsqueeze(2)
        if conf.shape[-2:] != (H, W):
            conf = F.interpolate(conf.reshape(B * N, 1, conf.shape[-2], conf.shape[-1]).float(), size=(H, W), mode="bilinear", align_corners=False).reshape(B, N, 1, H, W).to(pointmap.dtype)
        score = score * conf.clamp(0, 1)
    flat_score = score.reshape(B, N * H * W).clamp_min(0)
    topk = min(count * 4, flat_score.shape[1])
    idx_pool = flat_score.topk(topk, dim=1).indices
    # Deterministic farthest-point-style thinning over a high-confidence pool.
    gathered = pts.gather(1, idx_pool[..., None].expand(-1, -1, 3))
    chosen = _fps(gathered, requested)
    xyz = gathered.gather(1, chosen[..., None].expand(-1, -1, 3))
    chosen_score = flat_score.gather(1, idx_pool.gather(1, chosen))
    meta = _metadata_template(B, xyz.shape[1], pointmap.device, pointmap.dtype, source_type=2.0 if boundary else 1.0, scale_level=1.0 if boundary else 0.0)
    meta[..., 2] = chosen_score.clamp(0, 1)
    meta[..., 4] = (chosen_score > 0.05).float()
    meta[..., 6] = (1.0 - chosen_score).clamp(0, 1)
    if xyz.shape[1] < requested:
        pad = requested - xyz.shape[1]
        xyz = torch.cat([xyz, xyz[:, -1:].expand(-1, pad, -1)], dim=1)
        meta = torch.cat([meta, meta[:, -1:].expand(-1, pad, -1)], dim=1)
    return {"xyz": xyz, "metadata": meta}


def _fps(points: torch.Tensor, count: int) -> torch.Tensor:
    B, P, _ = points.shape
    count = min(count, P)
    idx = torch.zeros(B, count, dtype=torch.long, device=points.device)
    farthest = torch.zeros(B, dtype=torch.long, device=points.device)
    dist = torch.full((B, P), float("inf"), device=points.device, dtype=points.dtype)
    batch = torch.arange(B, device=points.device)
    for i in range(count):
        idx[:, i] = farthest
        centroid = points[batch, farthest].unsqueeze(1)
        dist = torch.minimum(dist, (points - centroid).square().sum(dim=-1))
        farthest = dist.argmax(dim=1)
    if count < idx.shape[1]:
        idx[:, count:] = idx[:, count - 1 : count]
    return idx


def _feature_from_metadata(metadata: torch.Tensor, feature_dim: int) -> torch.Tensor:
    if feature_dim <= metadata.shape[-1]:
        return metadata[..., :feature_dim]
    return torch.cat([metadata, metadata.new_zeros(*metadata.shape[:-1], feature_dim - metadata.shape[-1])], dim=-1)
