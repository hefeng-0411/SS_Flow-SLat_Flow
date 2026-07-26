from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoss.geometry.differentiable_camera import project_points
from geoss.reconstruction.rays import sample_image_at_pixels, unproject_pixels


FUSED_FEATURE_DIM = 16
FUSED_EVIDENCE_DIM = 39


@dataclass
class RayEvidence:
    raw: torch.Tensor  # [B,N,V,C_E]
    state_probability: torch.Tensor  # [B,N,V,5]
    sdf_observation: torch.Tensor  # [B,N,V,1]
    sdf_precision: torch.Tensor  # [B,N,V,1]
    sampled_rgb: torch.Tensor  # [B,N,V,3]
    sampled_normal: torch.Tensor  # [B,N,V,3]
    surface_points: torch.Tensor  # [B,N,V,3]
    physical_visibility: torch.Tensor  # [B,N,V,1]
    valid: torch.Tensor  # [B,N,V,1]
    fused: torch.Tensor  # [B,N,39]
    fused_state_probability: torch.Tensor  # [B,N,5]
    calibration: Dict[str, torch.Tensor]


class VGGTDepthCalibrator(nn.Module):
    """Calibrate raw VGGT confidence into bias, scale, and outlier mass.

    Raw VGGT confidence is intentionally not interpreted as a probability.
    Calibration conditions on log confidence, incidence, image texture,
    reprojection residual, and camera/alignment reliability.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        base_scale: float = 0.02,
        base_outlier_probability: float = 0.05,
        minimum_scale: float = 1e-4,
    ) -> None:
        super().__init__()
        self.base_scale = float(base_scale)
        self.minimum_scale = float(minimum_scale)
        self.base_outlier_logit = math.log(base_outlier_probability / (1.0 - base_outlier_probability))
        self.network = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        *,
        raw_confidence: torch.Tensor,
        incidence: torch.Tensor,
        texture: torch.Tensor,
        reprojection_residual: torch.Tensor,
        camera_reliability: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        features = torch.cat(
            (
                torch.log1p(raw_confidence.float().clamp_min(0.0)).clamp_max(10.0),
                incidence.float().clamp(0.0, 1.0),
                texture.float().clamp(0.0, 10.0),
                reprojection_residual.float().clamp(0.0, 1.0),
                camera_reliability.float().clamp(0.0, 1.0),
            ),
            dim=-1,
        )
        delta = self.network(features)
        # A weak monotone initialization uses confidence only as evidence for
        # precision; the learned correction is calibrated by a proper score.
        confidence_precision = torch.log1p(raw_confidence.float().clamp_min(0.0)).clamp_min(0.25)
        log_scale = (
            math.log(self.base_scale)
            - 0.5 * confidence_precision.log()
            + delta[..., 0:1].clamp(-5.0, 5.0)
        )
        scale = log_scale.exp().clamp_min(self.minimum_scale)
        bias = self.base_scale * delta[..., 1:2].clamp(-5.0, 5.0)
        outlier_probability = torch.sigmoid(self.base_outlier_logit + delta[..., 2:3])
        return {
            "scale": scale,
            "log_scale": scale.log(),
            "bias": bias,
            "outlier_probability": outlier_probability,
        }


class RayEvidenceBuilder(nn.Module):
    """Structured calibrated evidence at spatial points across all views."""

    def __init__(
        self,
        *,
        depth_calibrator: Optional[VGGTDepthCalibrator] = None,
        student_degrees_of_freedom: float = 4.0,
        feature_dim: int = FUSED_FEATURE_DIM,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.depth_calibrator = depth_calibrator or VGGTDepthCalibrator()
        self.student_degrees_of_freedom = float(student_degrees_of_freedom)
        self.feature_dim = int(feature_dim)
        self.epsilon = float(epsilon)

    def forward(
        self,
        *,
        points: torch.Tensor,
        images: torch.Tensor,
        masks: torch.Tensor,
        K: torch.Tensor,
        c2w: torch.Tensor,
        w2c: torch.Tensor,
        aligned_depth: torch.Tensor,
        raw_vggt_confidence: Optional[torch.Tensor] = None,
        aligned_pointmap: Optional[torch.Tensor] = None,
        vggt_features: Optional[torch.Tensor] = None,
        camera_reliability: Optional[torch.Tensor] = None,
        physical_visibility: Optional[torch.Tensor] = None,
    ) -> RayEvidence:
        _validate_inputs(points, images, masks, K, c2w, w2c, aligned_depth)
        b, n, _ = points.shape
        _, v, _, h, w = images.shape
        pixels, camera_depth, projection_valid = _project_all_views(points, K, w2c, h, w)
        rgb = sample_image_at_pixels(images, pixels)
        mask = sample_image_at_pixels(masks, pixels)
        depth = sample_image_at_pixels(aligned_depth, pixels)
        depth_valid = depth > self.epsilon
        valid = projection_valid & depth_valid

        if aligned_pointmap is not None:
            surface_points = sample_image_at_pixels(aligned_pointmap, pixels)
            world_normal_map = _normal_map_from_world_points(aligned_pointmap, c2w)
        else:
            surface_points = unproject_pixels(pixels, depth, K, c2w)
            world_pointmap = _world_pointmap_from_depth(aligned_depth, K, c2w)
            world_normal_map = _normal_map_from_world_points(world_pointmap, c2w)
        surface_normal = sample_image_at_pixels(world_normal_map, pixels)
        surface_to_camera = c2w[..., :3, 3][..., None, :] - surface_points
        surface_to_camera = F.normalize(surface_to_camera.float(), dim=-1, eps=1e-8).to(surface_points.dtype)
        orientation = torch.sign((surface_normal * surface_to_camera).sum(dim=-1, keepdim=True))
        orientation = torch.where(orientation == 0, torch.ones_like(orientation), orientation)
        surface_normal = F.normalize(surface_normal * orientation, dim=-1, eps=1e-8)
        incidence = (surface_normal * surface_to_camera).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)

        texture_map = _texture_magnitude(images)
        texture = sample_image_at_pixels(texture_map, pixels)
        if raw_vggt_confidence is None:
            raw_confidence = torch.ones_like(depth)
        else:
            raw_confidence = sample_image_at_pixels(
                _as_channel_map(raw_vggt_confidence, b, v),
                pixels,
            )
        if camera_reliability is None:
            reliability = torch.ones_like(depth)
        elif camera_reliability.ndim == 2:
            reliability = camera_reliability[..., None, None].expand(b, v, n, 1)
        elif camera_reliability.ndim == 5:
            reliability = sample_image_at_pixels(camera_reliability, pixels)
        else:
            raise ValueError("camera_reliability must be [B,V] or [B,V,1,H,W]")

        sampled_features, feature_consistency = _sample_feature_sketch(
            vggt_features,
            pixels,
            b,
            v,
            self.feature_dim,
            depth,
            source_height=h,
            source_width=w,
        )
        reprojection_residual = torch.zeros_like(depth)
        calibration = self.depth_calibrator(
            raw_confidence=raw_confidence,
            incidence=incidence,
            texture=texture,
            reprojection_residual=reprojection_residual,
            camera_reliability=reliability,
        )
        corrected_depth = depth + calibration["bias"]
        ray_residual = camera_depth - corrected_depth
        sigma = calibration["scale"]
        standardized = ray_residual / sigma
        student_kernel = (1.0 + standardized.square() / self.student_degrees_of_freedom).pow(
            -0.5 * (self.student_degrees_of_freedom + 1.0)
        )
        if physical_visibility is None:
            visibility = (
                torch.exp(-ray_residual.abs() / sigma.clamp_min(self.epsilon))
                * valid.to(depth.dtype)
            )
        else:
            if physical_visibility.shape != (b, n, v, 1):
                raise ValueError(
                    f"physical_visibility must be [B,N,V,1], got {tuple(physical_visibility.shape)}"
                )
            visibility = physical_visibility.permute(0, 2, 1, 3).to(depth.dtype)

        foreground = mask.clamp(0.0, 1.0)
        before_surface = torch.sigmoid(-standardized)
        behind_surface = torch.sigmoid(standardized)
        inlier = 1.0 - calibration["outlier_probability"]
        p_free = projection_valid.to(depth.dtype) * (
            (1.0 - foreground) + foreground * depth_valid.to(depth.dtype) * before_surface
        )
        p_surface = (
            projection_valid.to(depth.dtype)
            * foreground
            * depth_valid.to(depth.dtype)
            * student_kernel
            * visibility
            * inlier
        )
        p_occluded = (
            projection_valid.to(depth.dtype)
            * foreground
            * depth_valid.to(depth.dtype)
            * behind_surface
            * (1.0 - visibility * student_kernel)
        )
        p_unobserved = 1.0 - projection_valid.to(depth.dtype) * (
            (1.0 - foreground) + foreground * depth_valid.to(depth.dtype)
        )
        p_contradictory = projection_valid.to(depth.dtype) * calibration["outlier_probability"]
        state = torch.cat((p_free, p_surface, p_occluded, p_unobserved, p_contradictory), dim=-1)
        state = state.clamp_min(self.epsilon)
        state = state / state.sum(dim=-1, keepdim=True)

        # Local tangent signed distance: normals face the observing camera, so
        # pre-surface free space is positive as required by the SDF convention.
        point_view = points[:, None, :, :].expand(b, v, n, 3)
        sdf_observation = ((point_view - surface_points) * surface_normal).sum(dim=-1, keepdim=True)
        precision = (
            p_surface
            * inlier
            * incidence.clamp_min(0.05).square()
            * reliability
            / sigma.square().clamp_min(self.epsilon)
        )
        camera_center = c2w[..., :3, 3][..., None, :]
        view_direction = F.normalize(camera_center - point_view, dim=-1, eps=1e-8)

        raw = torch.cat(
            (
                mask,
                projection_valid.to(depth.dtype),
                camera_depth,
                corrected_depth,
                ray_residual,
                sigma,
                calibration["outlier_probability"],
                state,
                rgb,
                surface_normal,
                incidence,
                view_direction,
                visibility,
                feature_consistency,
            ),
            dim=-1,
        )
        fused, fused_state = _information_fuse(
            state=state,
            sdf_observation=sdf_observation,
            precision=precision,
            rgb=rgb,
            normal=surface_normal,
            view_direction=view_direction,
            features=sampled_features,
            visibility=visibility,
            incidence=incidence,
            outlier=calibration["outlier_probability"],
            valid=projection_valid.to(depth.dtype),
            epsilon=self.epsilon,
        )
        return RayEvidence(
            raw=raw.permute(0, 2, 1, 3).contiguous(),
            state_probability=state.permute(0, 2, 1, 3).contiguous(),
            sdf_observation=sdf_observation.permute(0, 2, 1, 3).contiguous(),
            sdf_precision=precision.permute(0, 2, 1, 3).contiguous(),
            sampled_rgb=rgb.permute(0, 2, 1, 3).contiguous(),
            sampled_normal=surface_normal.permute(0, 2, 1, 3).contiguous(),
            surface_points=surface_points.permute(0, 2, 1, 3).contiguous(),
            physical_visibility=visibility.permute(0, 2, 1, 3).contiguous(),
            valid=valid.permute(0, 2, 1, 3).contiguous(),
            fused=fused,
            fused_state_probability=fused_state,
            calibration={key: value.permute(0, 2, 1, 3).contiguous() for key, value in calibration.items()},
        )


def robust_student_t_nll(
    residual: torch.Tensor,
    scale: torch.Tensor,
    outlier_probability: torch.Tensor,
    *,
    degrees_of_freedom: float = 4.0,
    broad_scale: float = 0.5,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Negative log likelihood of a Student-t plus broad outlier mixture."""

    r = residual.float()
    sigma = scale.float().clamp_min(epsilon)
    nu = float(degrees_of_freedom)
    log_norm = (
        torch.lgamma(r.new_tensor((nu + 1.0) * 0.5))
        - torch.lgamma(r.new_tensor(nu * 0.5))
        - 0.5 * math.log(nu * math.pi)
        - sigma.log()
    )
    log_inlier = log_norm - 0.5 * (nu + 1.0) * torch.log1p(r.square() / (nu * sigma.square()))
    broad = r.new_tensor(float(broad_scale))
    log_broad = -0.5 * (r / broad).square() - broad.log() - 0.5 * math.log(2.0 * math.pi)
    pi = outlier_probability.float().clamp(epsilon, 1.0 - epsilon)
    log_probability = torch.logaddexp(torch.log1p(-pi) + log_inlier, pi.log() + log_broad)
    return -log_probability


def _information_fuse(
    *,
    state: torch.Tensor,
    sdf_observation: torch.Tensor,
    precision: torch.Tensor,
    rgb: torch.Tensor,
    normal: torch.Tensor,
    view_direction: torch.Tensor,
    features: torch.Tensor,
    visibility: torch.Tensor,
    incidence: torch.Tensor,
    outlier: torch.Tensor,
    valid: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Inputs use [B,V,N,C].  The contractions are explicit information-form
    # sufficient statistics; no generic attention substitutes for geometry.
    information = precision.sum(dim=1).clamp_min(epsilon)
    sdf_mean = (precision * sdf_observation).sum(dim=1) / information
    surface_weight = precision * state[..., 1:2]
    weight_sum = surface_weight.sum(dim=1).clamp_min(epsilon)
    rgb_mean = (surface_weight * rgb).sum(dim=1) / weight_sum
    rgb_variance = (surface_weight * (rgb - rgb_mean[:, None]).square()).sum(dim=1) / weight_sum
    normal_mean = F.normalize((surface_weight * normal).sum(dim=1), dim=-1, eps=epsilon)
    direction_mean = F.normalize((surface_weight * view_direction).sum(dim=1), dim=-1, eps=epsilon)
    feature_mean = (surface_weight * features).sum(dim=1) / weight_sum

    free_support = 1.0 - (1.0 - state[..., 0:1]).prod(dim=1)
    surface_support = 1.0 - (1.0 - state[..., 1:2]).prod(dim=1)
    occluded_support = (state[..., 2:3] * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(epsilon)
    unobserved_support = (1.0 - valid).prod(dim=1)
    cross_view_conflict = free_support * surface_support
    explicit_conflict = state[..., 4:5].amax(dim=1)
    contradiction_support = torch.maximum(cross_view_conflict, explicit_conflict)
    state_support = torch.cat(
        (free_support, surface_support, occluded_support, unobserved_support, contradiction_support),
        dim=-1,
    ).clamp_min(epsilon)
    fused_state = state_support / state_support.sum(dim=-1, keepdim=True)
    view_count = valid.sum(dim=1) / max(valid.shape[1], 1)
    visibility_mean = (visibility * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(epsilon)
    incidence_mean = (incidence * surface_weight).sum(dim=1) / weight_sum
    outlier_mean = (outlier * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(epsilon)
    fused = torch.cat(
        (
            sdf_mean,
            information.log(),
            fused_state,
            rgb_mean,
            rgb_variance,
            normal_mean,
            direction_mean,
            feature_mean,
            view_count,
            visibility_mean,
            incidence_mean,
            outlier_mean,
        ),
        dim=-1,
    )
    if fused.shape[-1] != FUSED_EVIDENCE_DIM:
        raise RuntimeError(f"Fused evidence width is {fused.shape[-1]}, expected {FUSED_EVIDENCE_DIM}")
    return fused, fused_state


def _project_all_views(points, K, w2c, height: int, width: int):
    b, n, _ = points.shape
    v = K.shape[1]
    flat_points = points[:, None].expand(b, v, n, 3).reshape(b * v, n, 3)
    projection = project_points(flat_points, K.reshape(b * v, 3, 3), w2c.reshape(b * v, 4, 4))
    pixels = projection["uv"].reshape(b, v, n, 2)
    camera_depth = projection["depth"].reshape(b, v, n, 1)
    in_bounds = (
        (pixels[..., 0:1] >= 0.0)
        & (pixels[..., 0:1] <= width - 1)
        & (pixels[..., 1:2] >= 0.0)
        & (pixels[..., 1:2] <= height - 1)
    )
    valid = projection["valid_z"].reshape(b, v, n, 1) & in_bounds
    return pixels, camera_depth, valid


def _world_pointmap_from_depth(depth: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    b, v, _, h, w = depth.shape
    y, x = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    pixels = torch.stack((x, y), dim=-1).reshape(1, 1, h * w, 2).expand(b, v, -1, -1)
    world = unproject_pixels(pixels, depth.permute(0, 1, 3, 4, 2).reshape(b, v, h * w, 1), K, c2w)
    return world.reshape(b, v, h, w, 3).permute(0, 1, 4, 2, 3).contiguous()


def _normal_map_from_world_points(points: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    if points.ndim != 5 or points.shape[2] != 3:
        raise ValueError("world pointmap must be [B,V,3,H,W]")
    dx, dy = _spatial_central_differences(points)
    normal = torch.cross(dx, dy, dim=2)
    normal = F.normalize(normal.float(), dim=2, eps=1e-8).to(points.dtype)
    camera = c2w[..., :3, 3][..., None, None]
    toward_camera = camera - points
    orientation = torch.sign((normal * toward_camera).sum(dim=2, keepdim=True))
    orientation = torch.where(orientation == 0, torch.ones_like(orientation), orientation)
    return normal * orientation


def _texture_magnitude(images: torch.Tensor) -> torch.Tensor:
    gray = (images.float() * images.new_tensor((0.2126, 0.7152, 0.0722))[None, None, :, None, None]).sum(
        dim=2,
        keepdim=True,
    )
    dx, dy = _spatial_central_differences(gray)
    return (dx.square() + dy.square() + 1e-8).sqrt()


def _spatial_central_differences(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Central image derivatives for `[B,V,C,H,W]` without backend-specific padding."""

    if value.ndim != 5:
        raise ValueError("spatial derivatives require [B,V,C,H,W]")
    dx = torch.zeros_like(value)
    dy = torch.zeros_like(value)
    if value.shape[-1] > 1:
        dx[..., 1:-1] = 0.5 * (value[..., 2:] - value[..., :-2])
        dx[..., 0] = value[..., 1] - value[..., 0]
        dx[..., -1] = value[..., -1] - value[..., -2]
    if value.shape[-2] > 1:
        dy[..., 1:-1, :] = 0.5 * (value[..., 2:, :] - value[..., :-2, :])
        dy[..., 0, :] = value[..., 1, :] - value[..., 0, :]
        dy[..., -1, :] = value[..., -1, :] - value[..., -2, :]
    return dx, dy


def _sample_feature_sketch(
    features,
    pixels,
    b,
    v,
    feature_dim,
    reference,
    *,
    source_height: int,
    source_width: int,
):
    if features is None or features.ndim != 5 or features.shape[:2] != (b, v):
        sketch = reference.new_zeros((b, v, reference.shape[2], feature_dim))
        consistency = reference.new_zeros(reference.shape)
        return sketch, consistency
    c = features.shape[2]
    if c >= feature_dim:
        usable = (c // feature_dim) * feature_dim
        reduced = features[:, :, :usable].reshape(
            b,
            v,
            feature_dim,
            usable // feature_dim,
            *features.shape[-2:],
        ).mean(dim=3)
    else:
        reduced = F.pad(features, (0, 0, 0, 0, 0, feature_dim - c))
    feature_pixels = pixels.clone()
    feature_pixels[..., 0] = feature_pixels[..., 0] * (
        max(reduced.shape[-1] - 1, 1) / max(source_width - 1, 1)
    )
    feature_pixels[..., 1] = feature_pixels[..., 1] * (
        max(reduced.shape[-2] - 1, 1) / max(source_height - 1, 1)
    )
    sampled = sample_image_at_pixels(reduced, feature_pixels)
    normalized = F.normalize(sampled.float(), dim=-1, eps=1e-8)
    consensus = F.normalize(normalized.mean(dim=1, keepdim=True), dim=-1, eps=1e-8)
    consistency = (normalized * consensus).sum(dim=-1, keepdim=True).to(reference.dtype)
    return sampled.to(reference.dtype), consistency


def _as_channel_map(value: torch.Tensor, b: int, v: int) -> torch.Tensor:
    if value.ndim == 4 and value.shape[:2] == (b, v):
        return value.unsqueeze(2)
    if value.ndim == 5 and value.shape[:3] == (b, v, 1):
        return value
    raise ValueError(f"Expected confidence [B,V,H,W] or [B,V,1,H,W], got {tuple(value.shape)}")


def _validate_inputs(points, images, masks, K, c2w, w2c, depth) -> None:
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must be [B,N,3], got {tuple(points.shape)}")
    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError(f"images must be [B,V,3,H,W], got {tuple(images.shape)}")
    b, v = images.shape[:2]
    if masks.shape[:3] != (b, v, 1) or depth.shape[:3] != (b, v, 1):
        raise ValueError("masks and aligned_depth must be [B,V,1,H,W]")
    if K.shape != (b, v, 3, 3) or c2w.shape != (b, v, 4, 4) or w2c.shape != (b, v, 4, 4):
        raise ValueError("Camera tensors do not match image batch/view dimensions")
