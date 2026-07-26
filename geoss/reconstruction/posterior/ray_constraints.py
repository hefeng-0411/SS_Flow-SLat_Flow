from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoss.reconstruction.evidence import VGGTDepthCalibrator
from geoss.reconstruction.posterior.authority import (
    SparseLinearConstraints,
    make_trilinear_constraints,
)
from geoss.reconstruction.rays import (
    camera_rays_from_pixels,
    deterministic_pixel_grid,
    ray_box_intersection,
    sample_image_at_pixels,
    unproject_pixels,
)


@dataclass
class ObservationConstraintBundle:
    surface: SparseLinearConstraints
    free: SparseLinearConstraints
    surface_points: torch.Tensor
    free_points: torch.Tensor
    surface_reliable: torch.Tensor
    free_reliable: torch.Tensor
    calibrated_scale: torch.Tensor
    outlier_probability: torch.Tensor


class RayConstraintBuilder(nn.Module):
    """Build equality and free-space inequality constraints from calibrated rays."""

    def __init__(
        self,
        *,
        depth_calibrator: Optional[VGGTDepthCalibrator] = None,
        pixel_stride: int = 8,
        surface_tolerance: float = 0.01,
        reliability_probability: float = 0.68,
        free_samples: int = 3,
        foreground_alpha: float = 0.5,
        background_alpha: float = 0.05,
    ) -> None:
        super().__init__()
        self.depth_calibrator = depth_calibrator or VGGTDepthCalibrator()
        self.pixel_stride = int(pixel_stride)
        self.surface_tolerance = float(surface_tolerance)
        self.reliability_probability = float(reliability_probability)
        self.free_samples = int(free_samples)
        self.foreground_alpha = float(foreground_alpha)
        self.background_alpha = float(background_alpha)

    def forward(
        self,
        *,
        aligned_depth: torch.Tensor,
        masks: torch.Tensor,
        K: torch.Tensor,
        c2w: torch.Tensor,
        resolution: int | Tuple[int, int, int],
        raw_vggt_confidence: Optional[torch.Tensor] = None,
        camera_reliability: Optional[torch.Tensor] = None,
        bounds: Tuple[float, float] = (-0.5, 0.5),
    ) -> ObservationConstraintBundle:
        if aligned_depth.ndim != 5 or aligned_depth.shape[2] != 1:
            raise ValueError("aligned_depth must be [B,V,1,H,W]")
        b, v, _, h, w = aligned_depth.shape
        pixels = deterministic_pixel_grid(
            h,
            w,
            stride=self.pixel_stride,
            batch_size=b,
            num_views=v,
            device=aligned_depth.device,
            dtype=aligned_depth.dtype,
        )
        depth = sample_image_at_pixels(aligned_depth, pixels)
        alpha = sample_image_at_pixels(masks, pixels)
        if raw_vggt_confidence is None:
            confidence = torch.ones_like(depth)
        else:
            conf_map = raw_vggt_confidence.unsqueeze(2) if raw_vggt_confidence.ndim == 4 else raw_vggt_confidence
            confidence = sample_image_at_pixels(conf_map, pixels)
        if camera_reliability is None:
            reliability = torch.ones_like(depth)
        elif camera_reliability.ndim == 2:
            reliability = camera_reliability[..., None, None].expand_as(depth)
        elif camera_reliability.ndim == 5:
            reliability = sample_image_at_pixels(camera_reliability, pixels)
        else:
            raise ValueError("camera_reliability must be [B,V] or [B,V,1,H,W]")
        calibration = self.depth_calibrator(
            raw_confidence=confidence,
            incidence=torch.ones_like(depth),
            texture=torch.zeros_like(depth),
            reprojection_residual=torch.zeros_like(depth),
            camera_reliability=reliability,
        )
        corrected_depth = depth + calibration["bias"]
        surface_points = unproject_pixels(pixels, corrected_depth, K, c2w)
        ray_bundle = camera_rays_from_pixels(pixels, K, c2w)
        surface_distance = (surface_points - ray_bundle.origins).norm(dim=-1)
        box_near, box_far, intersects = ray_box_intersection(
            ray_bundle.origins,
            ray_bundle.directions,
            bounds=bounds,
        )

        # Reliability is a calibrated credible-event probability, rather than
        # a raw-confidence threshold.  The Gaussian CDF approximation is used
        # only to decide which constraints receive hard authority; all other
        # observations still contribute through posterior precision.
        inlier_interval_probability = torch.erf(
            self.surface_tolerance
            / (calibration["scale"].clamp_min(1e-6) * (2.0**0.5))
        )
        reliable_probability = (
            (1.0 - calibration["outlier_probability"])
            * inlier_interval_probability
            * reliability
        )
        foreground = alpha[..., 0] >= self.foreground_alpha
        depth_valid = corrected_depth[..., 0] > 1e-6
        surface_reliable = (
            foreground
            & depth_valid
            & intersects
            & (surface_distance > box_near)
            & (surface_distance < box_far + self.surface_tolerance)
            & (reliable_probability[..., 0] >= self.reliability_probability)
        )

        if self.free_samples < 1:
            raise ValueError("free_samples must be positive")
        fractions = torch.linspace(
            1.0 / (self.free_samples + 1),
            self.free_samples / (self.free_samples + 1),
            self.free_samples,
            device=aligned_depth.device,
            dtype=aligned_depth.dtype,
        )
        background = alpha[..., 0] <= self.background_alpha
        foreground_far = (surface_distance - self.surface_tolerance).clamp_min(box_near)
        free_far = torch.where(foreground, foreground_far, box_far)
        free_depths = box_near[..., None] * (1.0 - fractions) + free_far[..., None] * fractions
        free_points = (
            ray_bundle.origins[..., None, :]
            + ray_bundle.directions[..., None, :] * free_depths[..., None]
        )
        free_reliable_base = intersects & (
            background | (foreground & depth_valid & (reliable_probability[..., 0] >= self.reliability_probability))
        )
        free_reliable = free_reliable_base[..., None].expand_as(free_depths)

        surface_flat = surface_points.reshape(b, -1, 3)
        surface_valid_flat = surface_reliable.reshape(b, -1)
        free_flat = free_points.reshape(b, -1, 3)
        free_valid_flat = free_reliable.reshape(b, -1)
        return ObservationConstraintBundle(
            surface=make_trilinear_constraints(
                surface_flat,
                resolution=resolution,
                valid=surface_valid_flat,
                bounds=bounds,
            ),
            free=make_trilinear_constraints(
                free_flat,
                resolution=resolution,
                valid=free_valid_flat,
                bounds=bounds,
            ),
            surface_points=surface_flat,
            free_points=free_flat,
            surface_reliable=surface_valid_flat,
            free_reliable=free_valid_flat,
            calibrated_scale=calibration["scale"],
            outlier_probability=calibration["outlier_probability"],
        )
