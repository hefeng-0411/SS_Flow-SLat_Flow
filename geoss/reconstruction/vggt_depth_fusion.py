from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VGGTDepthFusionConfig:
    """Contract for conservative, confidence-filtered VGGT free-space carving."""

    bounds_min: tuple[float, float, float] = (-0.5, -0.5, -0.5)
    bounds_max: tuple[float, float, float] = (0.5, 0.5, 0.5)
    mask_threshold: float = 0.5
    confidence_threshold: float = 0.15
    free_space_margin: float = 0.0125
    min_depth_views: int = 2
    min_not_free_fraction: float = 0.75
    reprojection_sigma_pixels: float = 4.0
    max_reprojection_error_pixels: float = 12.0
    chunk_size: int = 262144

    def __post_init__(self) -> None:
        if len(self.bounds_min) != 3 or len(self.bounds_max) != 3:
            raise ValueError("depth-fusion bounds must be three-dimensional")
        if any(hi <= lo for lo, hi in zip(self.bounds_min, self.bounds_max)):
            raise ValueError("depth-fusion upper bounds must exceed lower bounds")
        if not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be in [0,1]")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0,1]")
        if self.free_space_margin < 0.0:
            raise ValueError("free_space_margin must be non-negative")
        if self.min_depth_views < 1:
            raise ValueError("min_depth_views must be positive")
        if not 0.0 < self.min_not_free_fraction <= 1.0:
            raise ValueError("min_not_free_fraction must be in (0,1]")
        if self.reprojection_sigma_pixels <= 0.0:
            raise ValueError("reprojection_sigma_pixels must be positive")
        if self.max_reprojection_error_pixels <= 0.0:
            raise ValueError("max_reprojection_error_pixels must be positive")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@torch.inference_mode()
def aligned_pointmap_depth_evidence(
    aligned_pointmap: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
    masks: torch.Tensor,
    confidence: Optional[torch.Tensor],
    *,
    reprojection_sigma_pixels: float,
    max_reprojection_error_pixels: float,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Convert aligned VGGT world points into known-ray-consistent depth.

    An aligned point is admitted only when it projects back near its source
    pixel through the calibrated MeshFleet camera. This rejects Sim(3)
    alignments that fit camera centers but leave camera rotations inconsistent.
    """
    if aligned_pointmap.ndim == 5:
        if aligned_pointmap.shape[0] != 1:
            raise ValueError("Only one object may be fused per reconstruction process.")
        aligned_pointmap = aligned_pointmap[0]
    if aligned_pointmap.ndim != 4 or aligned_pointmap.shape[1] != 3:
        raise ValueError(
            f"aligned_pointmap must be [V,3,H,W] or [1,V,3,H,W], got {tuple(aligned_pointmap.shape)}"
        )
    views, _, height, width = aligned_pointmap.shape
    K, w2c, masks = _normalize_camera_mask_inputs(K, w2c, masks, views, height, width)
    device = aligned_pointmap.device
    dtype = torch.float32
    points = aligned_pointmap.to(dtype=dtype).permute(0, 2, 3, 1)
    homogeneous = torch.cat(
        [points, torch.ones(views, height, width, 1, device=device, dtype=dtype)],
        dim=-1,
    )
    camera = torch.einsum("vij,vhwj->vhwi", w2c[:, :3, :], homogeneous)
    depth = camera[..., 2]
    projected = torch.einsum("vij,vhwj->vhwi", K, camera)
    safe_depth = projected[..., 2].clamp_min(1e-8)
    u = projected[..., 0] / safe_depth
    v = projected[..., 1] / safe_depth
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    reprojection_error = torch.sqrt((u - xx[None]).square() + (v - yy[None]).square())
    finite = torch.isfinite(points).all(dim=-1) & torch.isfinite(reprojection_error)
    foreground = masks[:, 0] >= 0.5
    consistent = (
        finite
        & foreground
        & (depth > 1e-6)
        & (reprojection_error <= float(max_reprojection_error_pixels))
    )

    if confidence is None:
        confidence_map = torch.ones(views, height, width, device=device, dtype=dtype)
    else:
        confidence_map = _normalize_confidence(confidence, views, height, width, device)
    reprojection_weight = torch.exp(
        -0.5 * (reprojection_error / float(reprojection_sigma_pixels)).square()
    )
    evidence_confidence = confidence_map * reprojection_weight * consistent.to(dtype)
    depth_map = torch.where(consistent, depth, torch.zeros_like(depth))[:, None]
    evidence_confidence = evidence_confidence[:, None]

    median_error = []
    for view in range(views):
        selected = reprojection_error[view][finite[view] & foreground[view]]
        median_error.append(float(selected.median().cpu()) if selected.numel() else None)
    report = {
        "protocol": "known_camera_reprojection_filter_v1",
        "per_view_valid_fraction": [
            float(value.cpu()) for value in consistent.float().mean(dim=(-2, -1))
        ],
        "per_view_median_reprojection_error_pixels": median_error,
        "mean_evidence_confidence": float(evidence_confidence.mean().cpu()),
        "max_reprojection_error_pixels": float(max_reprojection_error_pixels),
        "reprojection_sigma_pixels": float(reprojection_sigma_pixels),
    }
    return depth_map, evidence_confidence, report


@torch.inference_mode()
def carve_visual_hull_with_depth(
    visual_hull: torch.Tensor,
    depth: torch.Tensor,
    confidence: torch.Tensor,
    masks: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
    config: VGGTDepthFusionConfig,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Remove only repeatedly observed free space from a silhouette hull."""
    if visual_hull.ndim != 3 or len(set(visual_hull.shape)) != 1:
        raise ValueError(
            f"visual_hull must be a cubic [R,R,R] grid, got {tuple(visual_hull.shape)}"
        )
    resolution = int(visual_hull.shape[0])
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError(f"depth must be [V,1,H,W], got {tuple(depth.shape)}")
    views = int(depth.shape[0])
    height, width = depth.shape[-2:]
    K, w2c, masks = _normalize_camera_mask_inputs(
        K, w2c, masks, views, height, width
    )
    confidence = _normalize_confidence(
        confidence, views, height, width, depth.device
    )[:, None]
    depth = depth.float()
    masks = masks.float()
    device = depth.device
    axes = [
        torch.linspace(
            float(config.bounds_min[axis]),
            float(config.bounds_max[axis]),
            resolution,
            device=device,
            dtype=torch.float32,
        )
        for axis in range(3)
    ]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    source_hull = visual_hull.to(device=device, dtype=torch.bool).reshape(-1)
    result_parts = []
    supported_voxels = 0
    carved_voxels = 0
    evidence_counts = torch.zeros(views, device=device, dtype=torch.float64)

    for start in range(0, grid.shape[0], int(config.chunk_size)):
        points = grid[start : start + int(config.chunk_size)]
        hull_chunk = source_hull[start : start + points.shape[0]]
        sampled_depth, sampled_conf, sampled_mask, projected_depth, valid = (
            _sample_depth_evidence(points, depth, confidence, masks, K, w2c)
        )
        evidence = (
            valid
            & (sampled_mask >= float(config.mask_threshold))
            & (sampled_depth > 1e-6)
            & (sampled_conf >= float(config.confidence_threshold))
        )
        evidence_counts += evidence.sum(dim=1, dtype=torch.float64)
        not_observed_free = projected_depth >= (
            sampled_depth - float(config.free_space_margin)
        )
        evidence_count = evidence.sum(dim=0)
        not_free_count = (not_observed_free & evidence).sum(dim=0)
        not_free_fraction = not_free_count.float() / evidence_count.clamp_min(1).float()
        sufficiently_observed = evidence_count >= int(config.min_depth_views)
        keep_from_depth = not_free_fraction >= float(config.min_not_free_fraction)
        keep = hull_chunk & (~sufficiently_observed | keep_from_depth)
        supported_voxels += int((hull_chunk & sufficiently_observed).sum().item())
        carved_voxels += int(
            (hull_chunk & sufficiently_observed & ~keep_from_depth).sum().item()
        )
        result_parts.append(keep)

    fused = torch.cat(result_parts).reshape_as(visual_hull)
    occupied = int(fused.sum().item())
    source_occupied = int(visual_hull.sum().item())
    if occupied == 0:
        raise RuntimeError(
            "VGGT depth fusion removed the complete visual hull; alignment or "
            "depth scale is invalid."
        )
    if occupied < max(8, int(0.01 * source_occupied)):
        raise RuntimeError(
            "VGGT depth fusion retained less than 1% of the visual hull; "
            "rejecting a likely camera/depth alignment failure."
        )
    report = {
        "protocol": "conservative_vggt_free_space_carving_v1",
        "config": config.as_dict(),
        "source_occupied_voxels": source_occupied,
        "occupied_voxels": occupied,
        "retained_fraction": occupied / max(1, source_occupied),
        "depth_supported_hull_voxels": supported_voxels,
        "carved_voxels": carved_voxels,
        "per_view_evidence_voxels": [int(value.item()) for value in evidence_counts],
        "uses_heldout_views": False,
        "uses_gt_3d": False,
    }
    return fused, report


def _normalize_camera_mask_inputs(
    K: torch.Tensor,
    w2c: torch.Tensor,
    masks: torch.Tensor,
    views: int,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if K.ndim == 4:
        if K.shape[0] != 1:
            raise ValueError("Only one object may be fused per reconstruction process.")
        K = K[0]
    if w2c.ndim == 4:
        if w2c.shape[0] != 1:
            raise ValueError("Only one object may be fused per reconstruction process.")
        w2c = w2c[0]
    if masks.ndim == 5:
        if masks.shape[0] != 1:
            raise ValueError("Only one object may be fused per reconstruction process.")
        masks = masks[0]
    if masks.ndim == 3:
        masks = masks[:, None]
    if K.shape != (views, 3, 3):
        raise ValueError(f"K must be [V,3,3], got {tuple(K.shape)}")
    if w2c.shape != (views, 4, 4):
        raise ValueError(f"w2c must be [V,4,4], got {tuple(w2c.shape)}")
    if masks.shape != (views, 1, height, width):
        masks = F.interpolate(
            masks.reshape(views, 1, *masks.shape[-2:]).float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    device = masks.device
    return (
        K.to(device=device, dtype=torch.float32),
        w2c.to(device=device, dtype=torch.float32),
        masks.to(device=device, dtype=torch.float32),
    )


def _normalize_confidence(
    confidence: torch.Tensor,
    views: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    if confidence.ndim == 5:
        if confidence.shape[0] != 1:
            raise ValueError("Only one object may be fused per reconstruction process.")
        confidence = confidence[0]
    if confidence.ndim == 4 and confidence.shape[1] == 1:
        confidence = confidence[:, 0]
    if confidence.ndim != 3 or confidence.shape[0] != views:
        raise ValueError(
            f"confidence must be [V,H,W], [V,1,H,W], or batched, got {tuple(confidence.shape)}"
        )
    if confidence.shape[-2:] != (height, width):
        confidence = F.interpolate(
            confidence[:, None].float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    return confidence.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)


def _sample_depth_evidence(
    points: torch.Tensor,
    depth: torch.Tensor,
    confidence: torch.Tensor,
    masks: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    views, _, height, width = depth.shape
    homogeneous = torch.cat(
        [points, torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)],
        dim=-1,
    )
    camera = torch.einsum("vij,pj->vpi", w2c[:, :3, :], homogeneous)
    projected_depth = camera[..., 2]
    projected = torch.einsum("vij,vpj->vpi", K, camera)
    safe = projected[..., 2].clamp_min(1e-8)
    u = projected[..., 0] / safe
    v = projected[..., 1] / safe
    x_grid = 2.0 * u / max(1, width - 1) - 1.0
    y_grid = 2.0 * v / max(1, height - 1) - 1.0
    grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(2)

    def sample(value: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(
            value,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0, :, 0]

    sampled_depth = sample(depth)
    sampled_confidence = sample(confidence)
    sampled_mask = sample(masks)
    valid = (
        (projected_depth > 1e-6)
        & (x_grid >= -1.0)
        & (x_grid <= 1.0)
        & (y_grid >= -1.0)
        & (y_grid <= 1.0)
    )
    return sampled_depth, sampled_confidence, sampled_mask, projected_depth, valid
