from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoss.reconstruction.evidence import robust_student_t_nll
from geoss.reconstruction.fields import (
    FIELD_CHANNELS,
    EVIDENCE_STATES,
    make_canonical_grid,
    query_posterior_field,
)
from geoss.reconstruction.models.rapc3d import RAPC3DOutput


@dataclass(frozen=True)
class RAPCLossWeights:
    ray_surface: float = 1.0
    ray_free: float = 1.0
    uncertainty: float = 0.25
    evidence_state: float = 0.25
    occupancy: float = 1.0
    thin_structure: float = 0.5
    point_surface: float = 1.0
    chamfer: float = 0.25
    normal: float = 0.2
    eikonal: float = 0.1
    curvature: float = 0.02
    rgb: float = 1.0
    foreground_rgb: float = 0.5
    mask: float = 1.0
    boundary: float = 0.25
    ssim: float = 0.2
    perceptual: float = 0.1
    depth: float = 0.25
    prior_hidden: float = 0.1
    prior_observed_regression: float = 1.0
    mesh_gaussian_tether: float = 1.0
    gaussian_covariance: float = 0.02
    directional_regularization: float = 0.01
    vehicle_symmetry: float = 0.02


class RAPC3DObjective(nn.Module):
    """Probabilistic objective over the final RAPC field and decoded assets."""

    def __init__(
        self,
        weights: Optional[RAPCLossWeights] = None,
        *,
        occupancy_temperature: float = 0.01,
        free_margin: float = 0.002,
        symmetry_axis: int = 0,
        perceptual_extractor: Optional[nn.Module] = None,
        perceptual_metric: Optional[nn.Module] = None,
        chamfer_points: int = 8192,
    ) -> None:
        super().__init__()
        self.weights = weights or RAPCLossWeights()
        self.occupancy_temperature = float(occupancy_temperature)
        self.free_margin = float(free_margin)
        self.symmetry_axis = int(symmetry_axis)
        self.perceptual_extractor = perceptual_extractor
        self.perceptual_metric = perceptual_metric
        self.chamfer_points = int(chamfer_points)

    def forward(
        self,
        output: RAPC3DOutput,
        batch: Dict[str, torch.Tensor],
        *,
        render_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        field = output.field
        b, _, d, h, w = field.volume.shape
        points = make_canonical_grid(
            (d, h, w),
            batch_size=b,
            bounds=field.bounds,
            device=field.volume.device,
            dtype=field.volume.dtype,
        )
        samples = query_posterior_field(field, points, require_gradient=True)
        if samples.spatial_gradient is None or samples.normal is None:
            raise RuntimeError("RAPC objective requires differentiable SDF normals")
        n = points.shape[1]
        evidence = output.evidence
        if evidence.sdf_observation.shape[1] != n:
            # Adaptive evidence describes fine nodes, while the main evidence
            # retained by RAPC3DOutput describes the coarse lattice.
            raise ValueError("Output evidence and coarse field node counts differ")
        prediction = samples.sdf_mean[:, :, None, :]
        residual = prediction - evidence.sdf_observation
        precision = evidence.sdf_precision.clamp_min(0.0)
        surface_weight = evidence.state_probability[..., 1:2] * evidence.valid.to(precision.dtype)
        free_weight = evidence.state_probability[..., 0:1] * evidence.valid.to(precision.dtype)
        ray_surface = _weighted_mean(0.5 * precision * residual.square(), surface_weight)
        ray_free = _weighted_mean(
            F.softplus((self.free_margin - prediction) / self.occupancy_temperature),
            free_weight,
        )
        field_scale = samples.sdf_variance.sqrt()[:, :, None, :]
        calibration_scale = (
            output.evidence.calibration["scale"].square() + field_scale.square()
        ).sqrt()
        outlier_probability = output.evidence.calibration["outlier_probability"]
        uncertainty = _weighted_mean(
            robust_student_t_nll(
                residual,
                calibration_scale,
                outlier_probability,
            ),
            evidence.valid.to(residual.dtype),
        )
        target_outlier = _weighted_view_mean(
            outlier_probability,
            evidence.valid.to(outlier_probability.dtype),
        ).detach()
        uncertainty = uncertainty + F.binary_cross_entropy(
            samples.outlier_probability.clamp(1e-6, 1.0 - 1e-6),
            target_outlier.clamp(0.0, 1.0),
        )
        field_state = samples.state_probability
        evidence_state = -(output.evidence.fused_state_probability.clamp_min(1e-8).detach() * field_state.clamp_min(1e-8).log()).sum(
            dim=-1
        ).mean()
        losses: Dict[str, torch.Tensor] = {
            "ray_surface": ray_surface,
            "ray_free": ray_free,
            "uncertainty": uncertainty,
            "evidence_state": evidence_state,
        }
        evidence_normal = output.evidence.sampled_normal
        predicted_normal = samples.normal[:, :, None, :]
        normal_weight = surface_weight * evidence.physical_visibility
        losses["normal"] = _weighted_mean(
            1.0 - (predicted_normal * evidence_normal).sum(dim=-1, keepdim=True).abs(),
            normal_weight,
        )

        gt_occ = batch.get("gt_occ")
        if gt_occ is not None:
            gt_occ = _occupancy_to_field(gt_occ, (d, h, w), field.volume.dtype)
            geometry_valid = batch.get(
                "geometry_supervision_valid",
                torch.ones(b, device=field.volume.device, dtype=field.volume.dtype),
            ).to(field.volume.dtype).reshape(b, 1, 1, 1, 1)
            # MeshFleet/TRELLIS voxel PLYs contain active surface voxels, not a
            # watertight filled occupancy volume.  Missing voxels are therefore
            # unknown, not exterior.  Supervise a zero-level set at positives;
            # ray evidence supplies the legal free-space sign constraints.
            surface_element = torch.sqrt(
                field.volume[:, 0:1].square() + 1e-8
            )
            surface_valid = geometry_valid * gt_occ
            losses["occupancy"] = _weighted_mean(surface_element, surface_valid)
            local_support = F.avg_pool3d(gt_occ, 3, stride=1, padding=1)
            thin_weight = (local_support + 1.0 / 27.0).rsqrt().detach()
            losses["thin_structure"] = _weighted_mean(
                surface_element,
                surface_valid * thin_weight,
            )
        else:
            zero = field.volume.sum() * 0.0
            losses["occupancy"] = zero
            losses["thin_structure"] = zero

        gt_points = batch.get("gt_sparse_xyz")
        if gt_points is not None and gt_points.ndim == 3:
            coordinate_scale = batch.get("gt_sparse_to_field_scale")
            if coordinate_scale is not None:
                gt_points = gt_points * coordinate_scale.to(gt_points.dtype)[:, None, None]
            else:
                gt_points = _to_field_bounds_if_needed(gt_points, field.bounds)
            gt_point_valid = batch.get(
                "gt_sparse_valid",
                torch.ones(gt_points.shape[:2], device=gt_points.device, dtype=torch.bool),
            ).bool()
            gt_surface = query_posterior_field(field, gt_points, require_gradient=True)
            losses["point_surface"] = _weighted_mean(
                gt_surface.sdf_mean.abs(),
                gt_point_valid[..., None],
            )
            if output.gaussians is not None:
                losses["chamfer"] = _masked_batched_chamfer(
                    output.gaussians.means,
                    output.gaussians.valid,
                    gt_points,
                    gt_point_valid,
                    count=self.chamfer_points,
                )
            else:
                losses["chamfer"] = losses["point_surface"] * 0.0
            gt_normals = batch.get("gt_sparse_normals")
            if gt_normals is not None and gt_surface.normal is not None:
                losses["normal"] = losses["normal"] + (
                    _weighted_mean(
                        1.0 - (gt_surface.normal * gt_normals).sum(dim=-1).abs(),
                        gt_point_valid,
                    )
                )
        else:
            zero = field.volume.sum() * 0.0
            losses.update({"point_surface": zero, "chamfer": zero})

        gradient_norm = samples.spatial_gradient.norm(dim=-1)
        observed_or_near_surface = (
            samples.state_probability[..., 1]
            + samples.state_probability[..., 2]
            + torch.exp(-samples.sdf_mean[..., 0].abs() / self.occupancy_temperature)
        ).detach()
        losses["eikonal"] = _weighted_mean(
            (gradient_norm - 1.0).square(),
            observed_or_near_surface,
        )
        losses["curvature"] = _curvature_loss(field.volume[:, 0:1], samples.state_probability[..., 1], (d, h, w))

        if output.render is not None and render_targets is not None:
            losses.update(self._render_losses(output, render_targets))
        else:
            zero = field.volume.sum() * 0.0
            losses.update(
                {
                    "rgb": zero,
                    "foreground_rgb": zero,
                    "mask": zero,
                    "boundary": zero,
                    "ssim": zero,
                    "perceptual": zero,
                    "depth": zero,
                }
            )

        hidden = (
            samples.state_probability[..., 2:3]
            + samples.state_probability[..., 3:4]
        ).detach()
        prior_sdf = output.prior.sdf_mean.reshape(b, 1, -1).transpose(1, 2)
        prior_precision = F.softplus(output.prior.log_precision).reshape(b, 1, -1).transpose(1, 2)
        losses["prior_hidden"] = _weighted_mean(
            (samples.sdf_mean - prior_sdf).abs(),
            hidden * prior_precision,
        )
        authority_terms = []
        for diagnostic in output.diagnostics.get("authority_iterations", []):
            if "prior_constraint_after" in diagnostic:
                authority_terms.append(diagnostic["prior_constraint_after"].mean())
        losses["prior_observed_regression"] = (
            torch.stack(authority_terms).mean() if authority_terms else field.volume.sum() * 0.0
        )

        if output.gaussians is not None:
            gaussian = output.gaussians
            losses["mesh_gaussian_tether"] = _weighted_mean(
                gaussian.sdf_at_means.abs(),
                gaussian.valid[..., None].to(gaussian.sdf_at_means.dtype),
            )
            eigenvalues = torch.linalg.eigvalsh(gaussian.covariance.float()).clamp_min(1e-12)
            condition = eigenvalues[..., -1] / eigenvalues[..., 0]
            covariance_loss = F.softplus((condition.log() - torch.log(condition.new_tensor(400.0))))
            losses["gaussian_covariance"] = _weighted_mean(
                covariance_loss,
                gaussian.valid.to(covariance_loss.dtype),
            )
        else:
            zero = field.volume.sum() * 0.0
            losses["mesh_gaussian_tether"] = zero
            losses["gaussian_covariance"] = zero
        directional = field.volume[:, FIELD_CHANNELS["directional_sh"]]
        losses["directional_regularization"] = directional.square().mean()
        losses["vehicle_symmetry"] = _uncertainty_gated_symmetry(
            field.volume,
            symmetry_axis=self.symmetry_axis,
        )

        total = field.volume.sum() * 0.0
        for name, value in losses.items():
            total = total + float(getattr(self.weights, name)) * value
        losses["total"] = total
        return losses

    def _render_losses(
        self,
        output: RAPC3DOutput,
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        assert output.render is not None
        rendered_rgb = output.render["rgb"]
        target_rgb = targets["rgb"].reshape_as(rendered_rgb).to(rendered_rgb.dtype)
        rendered_alpha = output.render["opacity"]
        target_mask = targets["mask"].reshape_as(rendered_alpha).to(rendered_alpha.dtype)
        foreground_weight = 0.25 + 0.75 * target_mask
        rgb = _weighted_mean(
            torch.sqrt((rendered_rgb - target_rgb).square() + 1e-6),
            foreground_weight,
        )
        mask = F.binary_cross_entropy(
            rendered_alpha.clamp(1e-6, 1.0 - 1e-6),
            target_mask,
        )
        if "image_shape" in targets:
            height, width = (int(value) for value in targets["image_shape"])
            pred_image = rendered_rgb.reshape(-1, height, width, 3).permute(0, 3, 1, 2)
            target_image = target_rgb.reshape(-1, height, width, 3).permute(0, 3, 1, 2)
            pred_mask = rendered_alpha.reshape(-1, height, width, 1).permute(0, 3, 1, 2)
            target_mask_image = target_mask.reshape(-1, height, width, 1).permute(
                0, 3, 1, 2
            )
            ssim = _ssim_loss(pred_image, target_image)
            foreground_rgb = _foreground_crop_distance(
                pred_image,
                target_image,
                target_mask_image,
            )
            boundary = _mask_boundary_loss(pred_mask, target_mask_image)
            if self.perceptual_metric is not None:
                perceptual = self.perceptual_metric(
                    pred_image * 2.0 - 1.0,
                    target_image * 2.0 - 1.0,
                ).mean()
            elif self.perceptual_extractor is not None:
                pred_feature = self.perceptual_extractor(pred_image)
                target_feature = self.perceptual_extractor(target_image)
                perceptual = _feature_distance(pred_feature, target_feature)
            else:
                perceptual = pred_image.sum() * 0.0
        else:
            ssim = rendered_rgb.sum() * 0.0
            foreground_rgb = rendered_rgb.sum() * 0.0
            boundary = rendered_rgb.sum() * 0.0
            perceptual = rendered_rgb.sum() * 0.0
        if "depth" in targets:
            target_depth = targets["depth"].reshape_as(output.render["depth"]).to(rendered_rgb.dtype)
            depth = _weighted_mean(
                (output.render["depth"] - target_depth).abs(),
                target_mask,
            )
        else:
            depth = rendered_rgb.sum() * 0.0
        return {
            "rgb": rgb,
            "foreground_rgb": foreground_rgb,
            "mask": mask,
            "boundary": boundary,
            "ssim": ssim,
            "perceptual": perceptual,
            "depth": depth,
        }


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    weight = weight.to(value.dtype)
    value, weight = torch.broadcast_tensors(value, weight)
    return (value * weight).sum() / weight.sum().clamp_min(epsilon)


def _weighted_view_mean(value: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    numerator = (value * weight).sum(dim=2)
    denominator = weight.sum(dim=2).clamp_min(epsilon)
    return numerator / denominator


def _occupancy_to_field(occupancy: torch.Tensor, shape, dtype) -> torch.Tensor:
    if occupancy.ndim == 4:
        occupancy = occupancy[:, None]
    if occupancy.ndim != 5:
        raise ValueError("gt_occ must be [B,D,H,W] or [B,1,D,H,W]")
    source_shape = occupancy.shape[-3:]
    if all(source >= target for source, target in zip(source_shape, shape)):
        occupancy = F.adaptive_max_pool3d(occupancy.float(), shape)
    else:
        occupancy = F.interpolate(occupancy.float(), size=shape, mode="nearest")
    return occupancy.to(dtype)


def _occupancy_boundary(occupancy: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool3d(occupancy, 3, stride=1, padding=1)
    eroded = -F.max_pool3d(-occupancy, 3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def _curvature_loss(sdf_volume: torch.Tensor, surface_probability: torch.Tensor, shape) -> torch.Tensor:
    laplacian = (
        -6.0 * sdf_volume
        + torch.roll(sdf_volume, 1, -1)
        + torch.roll(sdf_volume, -1, -1)
        + torch.roll(sdf_volume, 1, -2)
        + torch.roll(sdf_volume, -1, -2)
        + torch.roll(sdf_volume, 1, -3)
        + torch.roll(sdf_volume, -1, -3)
    )
    weight = surface_probability.reshape(sdf_volume.shape[0], 1, *shape).detach()
    return _weighted_mean(laplacian.abs(), weight)


def _uncertainty_gated_symmetry(volume: torch.Tensor, *, symmetry_axis: int) -> torch.Tensor:
    spatial_dimension = {-1: -1, 0: -1, 1: -2, 2: -3}.get(symmetry_axis)
    if spatial_dimension is None:
        raise ValueError("symmetry_axis must be 0, 1, or 2")
    sdf = volume[:, 0:1]
    probability = volume[:, FIELD_CHANNELS["evidence_logits"]].softmax(dim=1)
    unobserved = probability[:, 3:4].detach()
    reflected_sdf = torch.flip(sdf, dims=(spatial_dimension,))
    reflected_unobserved = torch.flip(unobserved, dims=(spatial_dimension,))
    return _weighted_mean(
        F.smooth_l1_loss(sdf, reflected_sdf, reduction="none"),
        torch.minimum(unobserved, reflected_unobserved),
    )


def _to_field_bounds_if_needed(points: torch.Tensor, bounds) -> torch.Tensor:
    # Existing MeshFleet `gt_sparse_xyz` is historically normalized to [-1,1].
    if float(points.detach().abs().amax()) > max(abs(bounds[0]), abs(bounds[1])) + 1e-4:
        return points * 0.5
    return points


def _masked_batched_chamfer(
    predicted: torch.Tensor,
    predicted_valid: torch.Tensor,
    target: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    count: int,
) -> torch.Tensor:
    values = []
    for batch_index in range(predicted.shape[0]):
        pred = predicted[batch_index, predicted_valid[batch_index]]
        tgt = target[batch_index, target_valid[batch_index]]
        if pred.numel() == 0 or tgt.numel() == 0:
            continue
        if pred.shape[0] > count:
            index = torch.linspace(0, pred.shape[0] - 1, count, device=pred.device).round().long()
            pred = pred[index]
        if tgt.shape[0] > count:
            index = torch.linspace(0, tgt.shape[0] - 1, count, device=tgt.device).round().long()
            tgt = tgt[index]
        values.append(_symmetric_chamfer(pred[None], tgt[None]))
    if not values:
        return predicted.sum() * 0.0
    return torch.stack(values).mean()


def _symmetric_chamfer(predicted: torch.Tensor, target: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    def nearest(source, destination):
        minima = []
        for start in range(0, source.shape[1], chunk):
            distance = torch.cdist(source[:, start : start + chunk].float(), destination.float())
            minima.append(distance.amin(dim=-1))
        return torch.cat(minima, dim=1).mean()

    return 0.5 * (nearest(predicted, target) + nearest(target, predicted))


def _ssim_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    channels = prediction.shape[1]
    window = torch.ones(channels, 1, 7, 7, device=prediction.device, dtype=prediction.dtype) / 49.0
    mu_x = F.conv2d(prediction, window, padding=3, groups=channels)
    mu_y = F.conv2d(target, window, padding=3, groups=channels)
    sigma_x = F.conv2d(prediction.square(), window, padding=3, groups=channels) - mu_x.square()
    sigma_y = F.conv2d(target.square(), window, padding=3, groups=channels) - mu_y.square()
    sigma_xy = F.conv2d(prediction * target, window, padding=3, groups=channels) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(1e-8)
    return 1.0 - ssim.mean()


def _feature_distance(prediction, target) -> torch.Tensor:
    if isinstance(prediction, (tuple, list)):
        return torch.stack(
            [(left - right.detach()).abs().mean() for left, right in zip(prediction, target)]
        ).mean()
    return (prediction - target.detach()).abs().mean()


def _foreground_crop_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Object-balanced robust RGB loss over per-view foreground boxes."""

    values = []
    for index in range(prediction.shape[0]):
        foreground = torch.nonzero(target_mask[index, 0] > 0.25, as_tuple=False)
        if foreground.numel() == 0:
            continue
        lower = foreground.amin(dim=0)
        upper = foreground.amax(dim=0) + 1
        height = int(upper[0] - lower[0])
        width = int(upper[1] - lower[1])
        padding = max(1, int(round(0.05 * max(height, width))))
        y0 = max(0, int(lower[0]) - padding)
        x0 = max(0, int(lower[1]) - padding)
        y1 = min(prediction.shape[-2], int(upper[0]) + padding)
        x1 = min(prediction.shape[-1], int(upper[1]) + padding)
        difference = prediction[index, :, y0:y1, x0:x1] - target[
            index, :, y0:y1, x0:x1
        ]
        values.append(torch.sqrt(difference.square() + 1e-6).mean())
    if not values:
        return prediction.sum() * 0.0
    return torch.stack(values).mean()


def _mask_boundary_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Match differentiable silhouette gradients, emphasizing thin parts."""

    def gradient_magnitude(value: torch.Tensor) -> torch.Tensor:
        dx = value[..., :, 1:] - value[..., :, :-1]
        dy = value[..., 1:, :] - value[..., :-1, :]
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        return torch.sqrt(dx.square() + dy.square() + 1e-8)

    target_boundary = gradient_magnitude(target).detach()
    predicted_boundary = gradient_magnitude(prediction)
    weight = 0.25 + 3.0 * target_boundary
    return _weighted_mean(
        F.smooth_l1_loss(
            predicted_boundary,
            target_boundary,
            reduction="none",
        ),
        weight,
    )
