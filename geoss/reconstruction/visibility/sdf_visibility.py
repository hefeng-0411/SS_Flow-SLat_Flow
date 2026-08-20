from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from geoss.reconstruction.fields import PosteriorFieldState, query_posterior_field
from geoss.reconstruction.rays import ray_box_intersection, sample_points_on_rays


def shell_density(
    sdf: torch.Tensor,
    gradient_norm: torch.Tensor,
    *,
    beta: float,
    eta: float,
) -> torch.Tensor:
    """Numerically stable SDF-shell density in FP32.

    `sech²(x)` is evaluated as `4 exp(-2|x|)/(1+exp(-2|x|))²`, avoiding
    `cosh` overflow under mixed precision.
    """

    if beta <= 0.0 or eta <= 0.0:
        raise ValueError("beta and eta must be positive")
    x = (sdf.float() / float(beta)).abs()
    exp_term = torch.exp((-2.0 * x).clamp_min(-80.0))
    sech2 = 4.0 * exp_term / (1.0 + exp_term).square()
    density = (float(eta) / (2.0 * float(beta))) * sech2 * gradient_norm.float().clamp(0.0, 10.0)
    return density


def volume_render_samples(
    *,
    density: torch.Tensor,
    depths: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    variances: Optional[torch.Tensor] = None,
    background: Optional[torch.Tensor] = None,
    valid: Optional[torch.Tensor] = None,
    transmittance_floor: float = 1e-10,
) -> Dict[str, torch.Tensor]:
    """Log-domain first-hit integration over ordered ray samples."""

    if density.shape != depths.shape:
        raise ValueError("density and depths must share [B,R,S]")
    density32 = density.float().clamp_min(0.0)
    depth32 = depths.float()
    delta = depth32[..., 1:] - depth32[..., :-1]
    final_delta = delta[..., -1:].clamp_min(1e-5) if delta.shape[-1] else torch.ones_like(depth32)
    delta = torch.cat((delta, final_delta), dim=-1).clamp_min(1e-6)
    optical = density32 * delta
    if valid is not None:
        optical = optical * valid.to(optical.dtype)
    alpha = -torch.expm1(-optical.clamp_max(80.0))
    exclusive_optical = torch.cat(
        (torch.zeros_like(optical[..., :1]), optical[..., :-1].cumsum(dim=-1)),
        dim=-1,
    )
    log_transmittance = -exclusive_optical
    weights = torch.exp(log_transmittance.clamp_min(-80.0)) * alpha
    terminal = torch.exp((-optical.sum(dim=-1, keepdim=True)).clamp_min(-80.0))
    opacity = weights.sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
    expected_depth = (weights * depth32).sum(dim=-1, keepdim=True)
    expected_depth = expected_depth / opacity.clamp_min(transmittance_floor)
    depth_variance = (
        weights * (depth32 - expected_depth).square()
    ).sum(dim=-1, keepdim=True) / opacity.clamp_min(transmittance_floor)
    out: Dict[str, torch.Tensor] = {
        "weights": weights,
        "alpha_samples": alpha,
        "log_transmittance": log_transmittance,
        "terminal_transmittance": terminal,
        "opacity": opacity,
        "depth": expected_depth,
        "depth_variance": depth_variance,
        "free_probability": terminal,
        "surface_probability": opacity,
    }
    if colors is not None:
        if colors.shape != density.shape + (3,):
            raise ValueError("colors must be [B,R,S,3]")
        rendered = (weights[..., None] * colors.float()).sum(dim=-2)
        if background is not None:
            if background.ndim == 2:
                background = background[:, None, :]
            rendered = rendered + terminal * background.float()
        out["rgb"] = rendered
    if variances is not None:
        if variances.shape != density.shape:
            raise ValueError("variances must be [B,R,S]")
        aleatoric = (weights * variances.float()).sum(dim=-1, keepdim=True)
        out["uncertainty"] = aleatoric + depth_variance
    return out


class SDFRayRenderer(nn.Module):
    """Differentiable renderer for the shared RAPC-3D posterior field."""

    def __init__(
        self,
        *,
        num_samples: int = 96,
        beta: float = 0.008,
        eta: float = 12.0,
        point_chunk_size: int = 262_144,
    ) -> None:
        super().__init__()
        if num_samples < 2:
            raise ValueError("num_samples must be at least two")
        self.num_samples = int(num_samples)
        self.beta = float(beta)
        self.eta = float(eta)
        self.point_chunk_size = int(point_chunk_size)

    def forward(
        self,
        field: PosteriorFieldState,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        *,
        near: Optional[torch.Tensor] = None,
        far: Optional[torch.Tensor] = None,
        background: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if ray_origins.shape != ray_directions.shape or ray_origins.ndim != 3:
            raise ValueError("ray origins/directions must both be [B,R,3]")
        ray_directions = torch.nn.functional.normalize(ray_directions, dim=-1, eps=1e-8)
        box_near, box_far, box_valid = ray_box_intersection(
            ray_origins,
            ray_directions,
            bounds=field.bounds,
        )
        near = box_near if near is None else torch.maximum(near, box_near)
        far = box_far if far is None else torch.minimum(far, box_far)
        valid_ray = box_valid & (far > near)
        t = torch.linspace(
            0.0,
            1.0,
            self.num_samples,
            device=ray_origins.device,
            dtype=ray_origins.dtype,
        )
        depths = near[..., None] * (1.0 - t) + far[..., None] * t
        points = sample_points_on_rays(ray_origins, ray_directions, depths)
        b, r, s, _ = points.shape
        flat_points = points.reshape(b, r * s, 3)
        view_direction = (-ray_directions)[..., None, :].expand(b, r, s, 3).reshape(b, r * s, 3)
        chunks = []
        for start in range(0, flat_points.shape[1], self.point_chunk_size):
            stop = min(start + self.point_chunk_size, flat_points.shape[1])
            chunks.append(
                query_posterior_field(
                    field,
                    flat_points[:, start:stop],
                    directions=view_direction[:, start:stop],
                    require_gradient=True,
                )
            )
        sdf = torch.cat([chunk.sdf_mean for chunk in chunks], dim=1).reshape(b, r, s)
        variance = torch.cat([chunk.sdf_variance for chunk in chunks], dim=1).reshape(b, r, s)
        gradient = torch.cat([chunk.spatial_gradient for chunk in chunks if chunk.spatial_gradient is not None], dim=1)
        color = torch.cat([chunk.color for chunk in chunks if chunk.color is not None], dim=1).reshape(b, r, s, 3)
        gradient_norm = gradient.norm(dim=-1).reshape(b, r, s)
        density = shell_density(sdf, gradient_norm, beta=self.beta, eta=self.eta)
        valid_samples = valid_ray[..., None].expand_as(density)
        out = volume_render_samples(
            density=density,
            depths=depths,
            colors=color,
            variances=variance,
            background=background,
            valid=valid_samples,
        )
        out.update(
            {
                "sdf_samples": sdf,
                "density_samples": density,
                "sample_points": points,
                "valid_ray": valid_ray,
            }
        )
        return out


def field_transmittance_to_points(
    field: PosteriorFieldState,
    points: torch.Tensor,
    camera_centers: torch.Tensor,
    *,
    num_samples: int = 24,
    beta: float = 0.008,
    eta: float = 12.0,
    point_chunk_size: int = 2048,
) -> torch.Tensor:
    """Physical visibility `T` from each camera to each spatial point.

    Returns `[B,N,V,1]`.  Integration starts at the canonical-box entry and
    stops just before the query point so surface support is not self-occluded.
    """

    if points.ndim != 3 or camera_centers.ndim != 3:
        raise ValueError("points and camera_centers must be [B,N,3] and [B,V,3]")
    b, n, _ = points.shape
    if camera_centers.shape[0] != b:
        raise ValueError("Point and camera batches differ")
    v = camera_centers.shape[1]
    output = []
    fractions = torch.linspace(
        0.0,
        1.0 - 1.0 / max(num_samples, 2),
        num_samples,
        device=points.device,
        dtype=points.dtype,
    )
    for start in range(0, n, point_chunk_size):
        stop = min(start + point_chunk_size, n)
        query = points[:, start:stop, None, :].expand(b, stop - start, v, 3)
        origin = camera_centers[:, None, :, :].expand_as(query)
        vector = query - origin
        distance = vector.norm(dim=-1).clamp_min(1e-8)
        direction = vector / distance[..., None]
        entry, _, intersects = ray_box_intersection(origin, direction, bounds=field.bounds)
        segment = (distance - entry).clamp_min(0.0)
        sample_depth = entry[..., None] + segment[..., None] * fractions
        sample_points = origin[..., None, :] + direction[..., None, :] * sample_depth[..., None]
        flat = sample_points.reshape(b, -1, 3)
        samples = query_posterior_field(field, flat, require_gradient=True)
        sdf = samples.sdf_mean.reshape(b, stop - start, v, num_samples)
        if samples.spatial_gradient is None:
            raise RuntimeError("Visibility requires SDF spatial gradients")
        gradient_norm = samples.spatial_gradient.norm(dim=-1).reshape(b, stop - start, v, num_samples)
        density = shell_density(sdf, gradient_norm, beta=beta, eta=eta)
        delta = segment[..., None] / max(num_samples, 1)
        optical = (density.float() * delta.float()).sum(dim=-1)
        transmittance = torch.exp((-optical).clamp_min(-80.0))
        valid = intersects & (segment > 0.0)
        output.append(torch.where(valid, transmittance, torch.zeros_like(transmittance))[..., None])
    return torch.cat(output, dim=1)
