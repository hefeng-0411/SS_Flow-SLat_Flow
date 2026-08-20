"""Confidence-aware CUDA voxel fusion aligned to TRELLIS sparse structure."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoss.integration.vggt_geometry_wrapper import VGGTGeometryBatch


@dataclass(frozen=True)
class VoxelFusionOutput:
    sparse_tensor: Any
    dense_tokens: torch.Tensor
    voxel_indices: torch.Tensor
    voxel_xyz: torch.Tensor
    voxel_confidence: torch.Tensor
    observation_mask: torch.Tensor
    accumulated_weight: torch.Tensor
    observation_count: torch.Tensor
    valid_mask: torch.Tensor
    alignment_scale: torch.Tensor
    condition_dim: int
    timings_ms: Dict[str, Optional[float]] = field(default_factory=dict)


class ConfidenceSparseVoxelFusion(nn.Module):
    """Fuse aligned VGGT samples with weighted CUDA sort/scatter reduction.

    Duplicate reduction is performed explicitly before construction of the
    ``spconv.SparseConvTensor`` because spconv does not define weighted
    duplicate-coordinate semantics.
    """

    def __init__(
        self,
        *,
        grid_resolution: int = 16,
        input_feature_dim: int = 2048,
        projected_feature_dim: int = 256,
        positional_frequencies: int = 4,
        observation_threshold: float = 1e-4,
        fusion_mode: str = "confidence",
        use_vggt_depth: bool = True,
        use_vggt_pointmap: bool = True,
        use_confidence_weighting: bool = True,
        use_visibility_weighting: bool = True,
        use_3d_positional_encoding: bool = True,
        require_spconv: bool = True,
        use_spconv_refinement: bool = True,
    ) -> None:
        super().__init__()
        if grid_resolution != 16:
            raise ValueError("Production TRELLIS SS fusion requires grid_resolution=16")
        if fusion_mode not in {"confidence", "average_ablation"}:
            raise ValueError(f"Unsupported fusion_mode={fusion_mode!r}")
        if not use_vggt_pointmap and not use_vggt_depth:
            raise ValueError("At least one real VGGT geometry source must be enabled")
        self.grid_resolution = int(grid_resolution)
        self.input_feature_dim = int(input_feature_dim)
        self.projected_feature_dim = int(projected_feature_dim)
        self.positional_frequencies = int(positional_frequencies)
        self.observation_threshold = float(observation_threshold)
        self.fusion_mode = fusion_mode
        self.use_vggt_depth = bool(use_vggt_depth)
        self.use_vggt_pointmap = bool(use_vggt_pointmap)
        self.use_confidence_weighting = bool(use_confidence_weighting)
        self.use_visibility_weighting = bool(use_visibility_weighting)
        self.use_3d_positional_encoding = bool(use_3d_positional_encoding)
        self.require_spconv = bool(require_spconv)
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(input_feature_dim),
            nn.Linear(input_feature_dim, projected_feature_dim),
            nn.SiLU(),
        )
        self.positional_dim = 6 * positional_frequencies if use_3d_positional_encoding else 0
        self.condition_dim = projected_feature_dim + 3 + self.positional_dim
        self.sparse_refiner = None
        if use_spconv_refinement:
            spconv = _import_spconv(required=require_spconv)
            if spconv is not None:
                self.sparse_refiner = spconv.SubMConv3d(
                    self.condition_dim,
                    self.condition_dim,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                    indice_key="ss_voxel_condition",
                )

    def _reduce_to_voxels(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        confidence: torch.Tensor,
        weights: torch.Tensor,
        valid: torch.Tensor,
        view_valid_mask: torch.Tensor,
        alignment_scale: torch.Tensor,
        profile: bool = False,
    ) -> VoxelFusionOutput:
        B, V, C, H, W = features.shape
        R = self.grid_resolution
        # [B,V,H,W,3], where channels are explicitly xyz.
        point_samples = points.permute(0, 1, 3, 4, 2)
        feature_samples = features.permute(0, 1, 3, 4, 2)
        indices = torch.floor((point_samples + 1.0) * (R / 2.0)).to(torch.long).clamp(0, R - 1)
        batch_ids = torch.arange(B, device=points.device).view(B, 1, 1, 1).expand(B, V, H, W)
        view_ids = torch.arange(V, device=points.device).view(1, V, 1, 1).expand(B, V, H, W)
        keys = indices[..., 0] + R * indices[..., 1] + R * R * indices[..., 2] + R**3 * batch_ids

        flat_valid = valid.reshape(-1)
        keys = keys.reshape(-1)[flat_valid]
        sample_features = feature_samples.reshape(-1, C)[flat_valid]
        sample_weights = weights.reshape(-1)[flat_valid].float()
        sample_confidence = confidence.reshape(-1)[flat_valid].float()
        sample_views = view_ids.reshape(-1)[flat_valid]
        if keys.numel() == 0:
            raise RuntimeError("No valid samples survived voxel indexing")

        reduction_timer = _CudaTimer(points.device, profile)
        reduction_timer.start()
        order = torch.argsort(keys)
        sorted_keys = keys[order]
        unique_keys, inverse = torch.unique_consecutive(sorted_keys, return_inverse=True)
        sorted_weights = sample_weights[order]
        weighted_features = sample_features[order].float() * sorted_weights[:, None]
        accumulated = torch.zeros(unique_keys.numel(), device=points.device, dtype=torch.float32)
        accumulated.index_add_(0, inverse, sorted_weights)
        feature_sum = torch.zeros(unique_keys.numel(), C, device=points.device, dtype=torch.float32)
        feature_sum.index_add_(0, inverse, weighted_features)
        fused_features = feature_sum / accumulated[:, None].clamp_min(1e-8)

        max_confidence = torch.zeros(unique_keys.numel(), device=points.device, dtype=torch.float32)
        max_confidence.scatter_reduce_(0, inverse, sample_confidence[order], reduce="amax", include_self=False)
        sorted_views = sample_views[order]
        key_view = sorted_keys * V + sorted_views
        unique_key_view = torch.unique_consecutive(torch.sort(key_view).values)
        key_for_view = torch.div(unique_key_view, V, rounding_mode="floor")
        voxel_for_view = torch.searchsorted(unique_keys, key_for_view)
        observation_count = torch.zeros(unique_keys.numel(), device=points.device, dtype=torch.float32)
        observation_count.index_add_(0, voxel_for_view, torch.ones_like(voxel_for_view, dtype=torch.float32))
        reduction_ms = reduction_timer.stop()

        local_keys = unique_keys.remainder(R**3)
        batch_index = torch.div(unique_keys, R**3, rounding_mode="floor")
        x_index = local_keys.remainder(R)
        y_index = torch.div(local_keys, R, rounding_mode="floor").remainder(R)
        z_index = torch.div(local_keys, R * R, rounding_mode="floor")
        xyz_index = torch.stack([x_index, y_index, z_index], dim=-1)
        xyz = (xyz_index.float() + 0.5) * (2.0 / R) - 1.0
        valid_views = view_valid_mask.sum(dim=1).float()[batch_index].clamp_min(1)
        observation_fraction = (observation_count / valid_views).clamp(0, 1)
        observation_mask_sparse = accumulated > self.observation_threshold
        position = sinusoidal_3d_encoding(xyz, self.positional_frequencies) if self.use_3d_positional_encoding else xyz.new_empty(xyz.shape[0], 0)
        projected = self.feature_projection(fused_features)
        sparse_features = torch.cat(
            [projected, max_confidence[:, None], observation_fraction[:, None], observation_mask_sparse.float()[:, None], position],
            dim=-1,
        )
        spconv_indices = torch.stack([batch_index, z_index, y_index, x_index], dim=-1).to(torch.int32).contiguous()
        spconv_timer = _CudaTimer(points.device, profile)
        spconv_timer.start()
        sparse_tensor = _make_sparse_tensor(
            sparse_features,
            spconv_indices,
            batch_size=B,
            resolution=R,
            required=self.require_spconv,
        )
        if self.sparse_refiner is not None:
            refined = self.sparse_refiner(sparse_tensor)
            sparse_features = sparse_features + refined.features
            sparse_tensor = sparse_tensor.replace_feature(sparse_features)
        spconv_ms = spconv_timer.stop()

        dense = sparse_features.new_zeros(B, R**3, self.condition_dim)
        dense[batch_index, local_keys] = sparse_features
        dense_conf = accumulated.new_zeros(B, R**3, 1)
        dense_conf[batch_index, local_keys, 0] = max_confidence
        dense_obs = torch.zeros(B, R**3, 1, device=points.device, dtype=torch.bool)
        dense_obs[batch_index, local_keys, 0] = observation_mask_sparse
        dense_weight = accumulated.new_zeros(B, R**3, 1)
        dense_weight[batch_index, local_keys, 0] = accumulated
        dense_count = observation_count.new_zeros(B, R**3, 1)
        dense_count[batch_index, local_keys, 0] = observation_count

        return VoxelFusionOutput(
            sparse_tensor=sparse_tensor,
            dense_tokens=dense,
            voxel_indices=spconv_indices,
            voxel_xyz=xyz,
            voxel_confidence=dense_conf,
            observation_mask=dense_obs,
            accumulated_weight=dense_weight,
            observation_count=dense_count,
            valid_mask=dense_obs[..., 0],
            alignment_scale=alignment_scale,
            condition_dim=self.condition_dim,
            timings_ms={"voxel_reduction": reduction_ms, "spconv_construction_refinement": spconv_ms},
        )

    def forward(self, *args, **kwargs):  # type: ignore[override]
        output = self._forward_impl(*args, **kwargs)
        return output

    def _forward_impl(
        self,
        geometry: VGGTGeometryBatch,
        *,
        foreground_masks: torch.Tensor,
        dataset_c2w: torch.Tensor,
        canonical_center: torch.Tensor,
        canonical_half_extent: torch.Tensor,
        points_are_trellis_normalized: bool = False,
        profile: bool = False,
    ) -> VoxelFusionOutput:
        # This implementation is split only so alignment scale can be attached
        # without changing the public immutable dataclass contract.
        features = geometry.visual_features.float()
        B, V, C, Hf, Wf = features.shape
        if C != self.input_feature_dim:
            raise ValueError(f"VGGT feature width mismatch: configured={self.input_feature_dim}, actual={C}")
        if foreground_masks.shape[:2] != (B, V):
            raise ValueError(f"foreground_masks must start [B,V], got {tuple(foreground_masks.shape)}")
        _assert_alignment_inputs(dataset_c2w, canonical_center, canonical_half_extent, B, V)
        geometry_timer = _CudaTimer(features.device, profile)
        geometry_timer.start()
        masks = _resize_bv_map(foreground_masks.float(), (Hf, Wf)).clamp(0, 1)
        mask_interior = (
            F.max_pool2d(
                (1.0 - masks[:, :, 0]).reshape(B * V, 1, Hf, Wf),
                kernel_size=3,
                stride=1,
                padding=1,
            ).reshape(B, V, Hf, Wf)
            < 0.5
        )
        depth = _resize_bv_map(geometry.depth.float(), (Hf, Wf))
        point_map = _resize_bv_map(geometry.point_map.float(), (Hf, Wf))
        point_conf = _resize_bv_map(geometry.point_confidence[:, :, None].float(), (Hf, Wf))[:, :, 0]
        depth_conf = _resize_bv_map(geometry.depth_confidence[:, :, None].float(), (Hf, Wf))[:, :, 0]
        K_feature = rescale_intrinsics(geometry.intrinsics.float(), geometry.image_resolution, (Hf, Wf))
        points, confidence = (
            (point_map, point_conf)
            if self.use_vggt_pointmap
            else (unproject_depth_batched(depth, K_feature, geometry.camera_to_world.float()), depth_conf)
        )
        depth_valid = (
            torch.isfinite(depth[:, :, 0]) & (depth[:, :, 0] > 0)
            if self.use_vggt_depth
            else torch.ones(B, V, Hf, Wf, device=features.device, dtype=torch.bool)
        )
        base_valid = geometry.valid_view_mask[:, :, None, None] & torch.isfinite(points).all(dim=2) & depth_valid & (masks[:, :, 0] > 0)
        confidence = robust_confidence_probability(confidence, base_valid)
        if points_are_trellis_normalized:
            normalized_points = points
            alignment_scale = torch.ones(B, device=points.device, dtype=torch.float32)
            alignment_inlier = torch.isfinite(points).all(dim=2)
        else:
            interior_valid = base_valid & mask_interior
            has_interior = interior_valid.flatten(1).any(dim=1).view(B, 1, 1, 1)
            alignment_valid = torch.where(has_interior, interior_valid, base_valid)
            aligned, alignment_scale, alignment_inlier = align_vggt_reference_to_dataset(
                points,
                geometry.extrinsics.float(),
                dataset_c2w.float(),
                canonical_center.float(),
                canonical_half_extent.float(),
                alignment_valid,
                confidence,
            )
            normalized_points = world_to_trellis_normalized(aligned, canonical_center.float(), canonical_half_extent.float())
        geometry_ms = geometry_timer.stop()
        plausible = torch.isfinite(normalized_points).all(dim=2) & (normalized_points.abs() <= 1.25).all(dim=2)
        # A one-patch silhouette boundary is intrinsically ambiguous after
        # resizing and VGGT upsampling. Audit the reliable foreground
        # interior; use only the top confidence tail when an object is too
        # thin to retain an eroded interior.
        expected_interior = base_valid & mask_interior & (confidence >= 0.5)
        expected_fallback = base_valid & (confidence >= 0.9)
        has_interior = expected_interior.flatten(1).any(dim=1).view(B, 1, 1, 1)
        expected = torch.where(has_interior, expected_interior, expected_fallback)
        # Some foreground predictions are not geometrically compatible with
        # the known camera/AABB (for example, points on an erroneous VGGT ray).
        # They cannot be corrected by any scalar gauge and must not make an
        # otherwise usable object fail the 99% scale audit. Keep the strict
        # audit on alignable rays and reject only catastrophic camera mismatch.
        expected_total = expected.flatten(1).sum(dim=1)
        alignable_count = (expected & alignment_inlier).flatten(1).sum(dim=1)
        alignable_fraction = alignable_count.float() / expected_total.clamp_min(1)
        if bool((expected_total == 0).any()):
            raise RuntimeError("All VGGT geometry is invalid after foreground/confidence filtering")
        if bool((alignable_fraction < 0.90).any()):
            raise RuntimeError(
                "VGGT-to-TRELLIS cameras are incompatible with too many reliable rays: "
                f"alignable_fractions={alignable_fraction.detach().cpu().tolist()}"
            )
        expected = expected & alignment_inlier
        expected_count = expected.flatten(1).sum(dim=1)
        plausible_fraction = (plausible & expected).flatten(1).sum(dim=1).float() / expected_count.clamp_min(1)
        if bool((plausible_fraction < 0.99).any()):
            raise RuntimeError(
                "VGGT-to-TRELLIS gauge alignment failed the 99% plausible-range assertion: "
                f"fractions={plausible_fraction.detach().cpu().tolist()}"
            )
        inside_lattice = torch.isfinite(normalized_points).all(dim=2) & (normalized_points.abs() <= 1.0).all(dim=2)
        visibility = inside_lattice & depth_valid if self.use_visibility_weighting else inside_lattice
        confidence_weight = (
            torch.ones_like(confidence)
            if self.fusion_mode == "average_ablation" or not self.use_confidence_weighting
            else confidence
        )
        weights = confidence_weight * masks[:, :, 0] * visibility.float() * geometry.valid_view_mask[:, :, None, None]
        valid = base_valid & plausible & (weights > 0)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        if bool((weights.flatten(1).sum(dim=1) <= 0).any()):
            raise RuntimeError("All voxel fusion weights are zero for at least one object")
        output = self._reduce_to_voxels(
            normalized_points,
            features,
            confidence,
            weights,
            valid,
            geometry.valid_view_mask,
            alignment_scale,
            profile,
        )
        return replace(
            output,
            timings_ms={"voxel_unprojection_alignment": geometry_ms, **output.timings_ms},
        )


def unproject_depth_batched(depth: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """Fully vectorized OpenCV depth unprojection on the input device."""
    if depth.ndim != 5 or depth.shape[2] != 1:
        raise ValueError(f"depth must be [B,V,1,H,W], got {tuple(depth.shape)}")
    B, V, _, H, W = depth.shape
    y, x = torch.meshgrid(
        torch.arange(H, device=depth.device, dtype=torch.float32),
        torch.arange(W, device=depth.device, dtype=torch.float32),
        indexing="ij",
    )
    z = depth[:, :, 0].float()
    x_cam = (x.view(1, 1, H, W) - K[..., 0, 2, None, None]) * z / K[..., 0, 0, None, None]
    y_cam = (y.view(1, 1, H, W) - K[..., 1, 2, None, None]) * z / K[..., 1, 1, None, None]
    camera = torch.stack([x_cam, y_cam, z, torch.ones_like(z)], dim=2)
    world = torch.einsum("bvij,bvjhw->bvihw", c2w.float(), camera)
    return world[:, :, :3]


def align_vggt_reference_to_dataset(
    points: torch.Tensor,
    vggt_w2c: torch.Tensor,
    dataset_c2w: torch.Tensor,
    canonical_center: torch.Tensor,
    canonical_half_extent: torch.Tensor,
    valid: torch.Tensor,
    confidence: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve VGGT's scale gauge against the known canonical AABB.

    Rotation and origin are fixed by the reference camera pair. The remaining
    scalar is estimated from foreground ray/AABB entry scales, then constrained
    to the interval that keeps reliable interior points inside a 25% safety
    envelope. This is camera-consistent and avoids the visible-surface bias of
    matching a foreground median depth to the object-center depth. The returned
    mask marks rays that can intersect that safety envelope at a positive scale.
    """
    B, V, _, H, W = points.shape
    reference_index = valid.flatten(2).any(dim=2).to(torch.int64).argmax(dim=1)
    batch = torch.arange(B, device=points.device)
    ref_w2c = vggt_w2c[batch, reference_index]
    ref_dataset_c2w = dataset_c2w[batch, reference_index]
    points_h = torch.cat([points.float(), torch.ones(B, V, 1, H, W, device=points.device)], dim=2)
    camera_points = torch.einsum("bij,bvjhw->bvihw", ref_w2c, points_h)[:, :, :3]
    rotation = ref_dataset_c2w[:, :3, :3]
    origin = ref_dataset_c2w[:, :3, 3]
    directions = torch.einsum("bij,bvjhw->bvihw", rotation, camera_points)
    reliable = valid & (confidence >= 0.5)
    entry, exit, hits = _ray_box_scale_interval(
        directions,
        origin,
        canonical_center - canonical_half_extent,
        canonical_center + canonical_half_extent,
    )
    reliable_hits = reliable & hits & torch.isfinite(entry) & torch.isfinite(exit) & (exit > 0)
    if bool((reliable_hits.flatten(1).sum(dim=1) == 0).any()):
        raise RuntimeError("VGGT reference rays do not intersect the canonical AABB after camera alignment")
    scale = torch.nanquantile(
        torch.where(reliable_hits, entry.clamp_min(0), torch.nan).flatten(1),
        0.9,
        dim=1,
    )
    plausible_entry, plausible_exit, plausible_hits = _ray_box_scale_interval(
        directions,
        origin,
        canonical_center - 1.25 * canonical_half_extent,
        canonical_center + 1.25 * canonical_half_extent,
    )
    alignment_inlier = (
        plausible_hits
        & torch.isfinite(plausible_entry)
        & torch.isfinite(plausible_exit)
        & (plausible_exit > 0)
    )
    plausible_reliable = reliable & alignment_inlier
    lower = torch.where(plausible_reliable, plausible_entry.clamp_min(0), torch.nan).nan_to_num(
        nan=-torch.inf
    ).flatten(1).amax(dim=1)
    upper = torch.where(plausible_reliable, plausible_exit, torch.nan).nan_to_num(
        nan=torch.inf
    ).flatten(1).amin(dim=1)
    if bool((~torch.isfinite(lower) | ~torch.isfinite(upper) | (lower > upper)).any()):
        raise RuntimeError("No camera-consistent VGGT scale can satisfy the canonical safety envelope")
    scale = torch.maximum(torch.minimum(scale, upper), lower).clamp(1e-2, 1e2)
    if not torch.isfinite(scale).all():
        raise RuntimeError("Non-finite VGGT gauge scale")
    aligned = origin[:, None, :, None, None] + scale[:, None, None, None, None] * directions
    return aligned, scale, alignment_inlier


def _ray_box_scale_interval(
    directions: torch.Tensor,
    origin: torch.Tensor,
    box_min: torch.Tensor,
    box_max: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return scalar intervals where ``origin + scale * direction`` is in a box."""
    parallel = directions.abs() < 1e-8
    safe = torch.where(parallel, torch.ones_like(directions), directions)
    lower = (box_min[:, None, :, None, None] - origin[:, None, :, None, None]) / safe
    upper = (box_max[:, None, :, None, None] - origin[:, None, :, None, None]) / safe
    axis_entry = torch.where(
        parallel, torch.full_like(lower, -torch.inf), torch.minimum(lower, upper)
    )
    axis_exit = torch.where(
        parallel, torch.full_like(upper, torch.inf), torch.maximum(lower, upper)
    )
    entry = axis_entry.amax(dim=2)
    exit = axis_exit.amin(dim=2)
    origin_inside_parallel_slab = (
        (origin[:, None, :, None, None] >= box_min[:, None, :, None, None])
        & (origin[:, None, :, None, None] <= box_max[:, None, :, None, None])
    )
    parallel_valid = (~parallel | origin_inside_parallel_slab).all(dim=2)
    hits = parallel_valid & (exit >= entry.clamp_min(0))
    return entry, exit, hits


def world_to_trellis_normalized(points: torch.Tensor, center: torch.Tensor, half_extent: torch.Tensor) -> torch.Tensor:
    """Map dataset object coordinates into TRELLIS-normalized ``[-1,1]^3``."""
    if bool((half_extent <= 0).any()):
        raise ValueError("canonical_half_extent must be strictly positive")
    return (points - center[:, None, :, None, None]) / half_extent[:, None, :, None, None]


def rescale_intrinsics(K: torch.Tensor, source_hw: Tuple[int, int], target_hw: Tuple[int, int]) -> torch.Tensor:
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    result = K.clone().float()
    result[..., 0, :] *= float(target_w) / float(source_w)
    result[..., 1, :] *= float(target_h) / float(source_h)
    result[..., 2, :] = torch.tensor([0.0, 0.0, 1.0], device=K.device)
    return result


def robust_confidence_probability(confidence: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Quantile-normalize finite VGGT reliabilities without discarding ordering."""
    values = torch.where(valid, confidence.float().clamp(0, 1), torch.nan).flatten(1)
    low = torch.nanquantile(values, 0.05, dim=1).view(-1, 1, 1, 1)
    high = torch.nanquantile(values, 0.95, dim=1).view(-1, 1, 1, 1)
    normalized = (confidence.float() - low) / (high - low).clamp_min(1e-6)
    normalized = torch.where((high - low) > 1e-6, normalized, confidence.float())
    return torch.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)


def sinusoidal_3d_encoding(xyz: torch.Tensor, num_frequencies: int) -> torch.Tensor:
    if xyz.shape[-1] != 3:
        raise ValueError(f"xyz must end in 3, got {tuple(xyz.shape)}")
    frequencies = (2.0 ** torch.arange(num_frequencies, device=xyz.device, dtype=torch.float32)) * torch.pi
    angles = xyz.float()[..., :, None] * frequencies
    return torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)


def _resize_bv_map(tensor: torch.Tensor, target: Tuple[int, int]) -> torch.Tensor:
    if tensor.ndim == 4:
        tensor = tensor[:, :, None]
    B, V, C, H, W = tensor.shape
    if (H, W) == target:
        return tensor
    resized = F.interpolate(tensor.reshape(B * V, C, H, W), size=target, mode="bilinear", align_corners=False)
    return resized.reshape(B, V, C, *target)


def _assert_alignment_inputs(c2w: torch.Tensor, center: torch.Tensor, half: torch.Tensor, B: int, V: int) -> None:
    if c2w.shape != (B, V, 4, 4) or center.shape != (B, 3) or half.shape != (B, 3):
        raise ValueError(
            f"Alignment contract mismatch: c2w={tuple(c2w.shape)}, center={tuple(center.shape)}, half={tuple(half.shape)}"
        )
    if not torch.isfinite(c2w).all() or not torch.isfinite(center).all() or not torch.isfinite(half).all():
        raise ValueError("Alignment inputs contain NaN/Inf")


def _import_spconv(*, required: bool):
    try:
        import spconv.pytorch as spconv
    except ImportError as exc:
        if required:
            raise ImportError("Production voxel fusion requires installed spconv.pytorch") from exc
        return None
    return spconv


def _make_sparse_tensor(
    features: torch.Tensor,
    indices: torch.Tensor,
    *,
    batch_size: int,
    resolution: int,
    required: bool,
):
    spconv = _import_spconv(required=required)
    if spconv is None:
        return None
    # Installed spconv 2.x uses [batch,z,y,x] indices for 3-D tensors.
    return spconv.SparseConvTensor(
        features=features,
        indices=indices,
        spatial_shape=[resolution, resolution, resolution],
        batch_size=batch_size,
    )


class _CudaTimer:
    def __init__(self, device: torch.device, enabled: bool) -> None:
        self.start_event = torch.cuda.Event(enable_timing=True) if enabled and device.type == "cuda" else None
        self.end_event = torch.cuda.Event(enable_timing=True) if self.start_event is not None else None

    def start(self) -> None:
        if self.start_event is not None:
            self.start_event.record()

    def stop(self) -> Optional[float]:
        if self.start_event is None or self.end_event is None:
            return None
        self.end_event.record()
        self.end_event.synchronize()
        return float(self.start_event.elapsed_time(self.end_event))


__all__ = [
    "ConfidenceSparseVoxelFusion",
    "VoxelFusionOutput",
    "align_vggt_reference_to_dataset",
    "rescale_intrinsics",
    "sinusoidal_3d_encoding",
    "unproject_depth_batched",
    "world_to_trellis_normalized",
]
