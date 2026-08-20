"""Structured SS-flow and differentiable coarse-geometry objectives."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowMatchingLossBuilder(nn.Module):
    """Build CFM, decoder occupancy, volume reprojection and prior losses."""

    def __init__(
        self,
        *,
        lambda_cfm: float = 1.0,
        lambda_depth: float = 0.1,
        lambda_silhouette: float = 0.1,
        lambda_occupancy: float = 0.5,
        lambda_surface: float = 0.05,
        lambda_prior: float = 0.01,
        use_depth_loss: bool = True,
        use_silhouette_loss: bool = True,
        use_surface_loss: bool = False,
        use_prior_preservation: bool = True,
        surface_backend: str = "geomloss",
        surface_blur: float = 0.05,
        render_resolution: int = 64,
        ray_samples: int = 48,
        extensions_root: Optional[str] = None,
    ) -> None:
        super().__init__()
        if surface_backend not in {"geomloss", "frnn"}:
            raise ValueError(f"Unsupported surface_backend={surface_backend!r}")
        self.weights = {
            "cfm": float(lambda_cfm),
            "depth": float(lambda_depth),
            "silhouette": float(lambda_silhouette),
            "occupancy": float(lambda_occupancy),
            "surface": float(lambda_surface),
            "prior": float(lambda_prior),
        }
        self.use_depth_loss = bool(use_depth_loss)
        self.use_silhouette_loss = bool(use_silhouette_loss)
        self.use_surface_loss = bool(use_surface_loss)
        self.use_prior_preservation = bool(use_prior_preservation)
        self.surface_backend = surface_backend
        self.surface_blur = float(surface_blur)
        self.render_resolution = int(render_resolution)
        self.ray_samples = int(ray_samples)
        self.extensions_root = Path(extensions_root) if extensions_root else None

    def forward(
        self,
        *,
        v_final: torch.Tensor,
        v_target: torch.Tensor,
        v_base: torch.Tensor,
        gate: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        predicted_clean_grid: Optional[torch.Tensor] = None,
        ss_decoder: Optional[nn.Module] = None,
        gt_occ: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        K_dataset: Optional[torch.Tensor] = None,
        c2w_dataset: Optional[torch.Tensor] = None,
        canonical_center: Optional[torch.Tensor] = None,
        canonical_half_extent: Optional[torch.Tensor] = None,
        vggt_depth: Optional[torch.Tensor] = None,
        vggt_depth_confidence: Optional[torch.Tensor] = None,
        alignment_scale: Optional[torch.Tensor] = None,
        target_surface_points: Optional[torch.Tensor] = None,
        target_surface_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if v_final.shape != v_target.shape or v_final.shape != v_base.shape:
            raise ValueError("v_final, v_target, and v_base must have identical shapes")
        zero = v_final.new_zeros((), dtype=torch.float32)
        diagnostics: Dict[str, Any] = {}
        loss_cfm = _masked_velocity_mse(v_final, v_target, valid_mask)
        loss_base = _masked_velocity_mse(v_base, v_target, valid_mask)
        diagnostics.update(
            {
                "cfm_available": True,
                "loss_base_cfm": loss_base.detach(),
                "adapter_cfm_gain": (loss_base - loss_cfm).detach(),
                "residual_target_norm": torch.linalg.vector_norm(
                    (v_target.float() - v_base.float()), dim=-1
                ).mean().detach(),
            }
        )

        occupancy_logits = None
        if predicted_clean_grid is not None and ss_decoder is not None:
            occupancy_logits = ss_decoder(predicted_clean_grid)
            if occupancy_logits.ndim != 5 or occupancy_logits.shape[1] != 1:
                raise RuntimeError(f"TRELLIS SS decoder must produce [B,1,D,H,W], got {tuple(occupancy_logits.shape)}")
            if occupancy_logits.numel() == 0 or not torch.isfinite(occupancy_logits).all():
                raise RuntimeError("TRELLIS SS decoder produced empty/non-finite support logits")
            if bool((occupancy_logits <= 0).flatten(1).all(dim=1).any()):
                raise RuntimeError("TRELLIS SS decoder produced empty active support for at least one object")
        occupancy_probability = occupancy_logits.sigmoid() if occupancy_logits is not None else None

        if occupancy_probability is not None and gt_occ is not None:
            target_occ = _resize_occupancy(gt_occ.float(), occupancy_probability.shape[-3:])
            loss_occ_bce = F.binary_cross_entropy_with_logits(occupancy_logits.float(), target_occ)
            loss_occ_dice = _dice_loss(occupancy_probability.float(), target_occ)
            loss_occupancy = loss_occ_bce + loss_occ_dice
            diagnostics.update(
                {
                    "occupancy_available": True,
                    "loss_occupancy_bce": loss_occ_bce.detach(),
                    "loss_occupancy_dice": loss_occ_dice.detach(),
                }
            )
        else:
            loss_occupancy = zero
            diagnostics["occupancy_available"] = False
            diagnostics["occupancy_unavailable_reason"] = "missing decoded logits or gt_occ"

        render_inputs = (occupancy_logits, masks, K_dataset, c2w_dataset, canonical_center, canonical_half_extent)
        rendered = None
        if (self.use_depth_loss or self.use_silhouette_loss) and all(value is not None for value in render_inputs):
            rendered = render_occupancy_volume(
                occupancy_logits,
                K_dataset,
                c2w_dataset,
                canonical_center,
                canonical_half_extent,
                output_resolution=self.render_resolution,
                ray_samples=self.ray_samples,
            )

        if self.use_silhouette_loss and rendered is not None and masks is not None:
            target_mask = _resize_view_maps(masks.float(), self.render_resolution)[:, :, 0]
            predicted_silhouette = rendered["silhouette"].clamp(1e-6, 1 - 1e-6)
            valid_ray = rendered["ray_box_valid"].float()
            sil_bce = _probability_bce(predicted_silhouette.float(), target_mask.float())
            sil_bce = (sil_bce * valid_ray).sum() / valid_ray.sum().clamp_min(1)
            sil_dice = _dice_loss(predicted_silhouette, target_mask)
            loss_silhouette = sil_bce + sil_dice
            diagnostics.update(
                {
                    "silhouette_available": True,
                    "silhouette_path": "differentiable_volume_projection",
                    "nvdiffrast_used": False,
                }
            )
        else:
            loss_silhouette = zero
            diagnostics["silhouette_available"] = False
            diagnostics["silhouette_unavailable_reason"] = (
                "disabled" if not self.use_silhouette_loss else "missing occupancy/camera/mask modality"
            )

        depth_ready = rendered is not None and vggt_depth is not None and vggt_depth_confidence is not None
        if self.use_depth_loss and depth_ready:
            target_depth = _resize_view_maps(vggt_depth.float(), self.render_resolution)[:, :, 0]
            target_confidence = _resize_view_maps(vggt_depth_confidence[:, :, None].float(), self.render_resolution)[:, :, 0]
            if alignment_scale is None:
                raise ValueError("Depth reprojection requires explicit VGGT-to-dataset alignment_scale")
            target_depth = target_depth * alignment_scale[:, None, None, None]
            target_mask = _resize_view_maps(masks.float(), self.render_resolution)[:, :, 0]
            depth_mask = (
                target_mask
                * target_confidence.clamp(0, 1)
                * (target_depth > 0).float()
                * rendered["depth_valid"].float()
            )
            residual = rendered["depth"] - target_depth
            robust = torch.sqrt(residual.square() + 1e-6)
            loss_depth = (robust * depth_mask).sum() / depth_mask.sum().clamp_min(1e-8)
            diagnostics["depth_available"] = bool((depth_mask.sum() > 0).item())
        else:
            loss_depth = zero
            diagnostics["depth_available"] = False
            diagnostics["depth_unavailable_reason"] = (
                "disabled" if not self.use_depth_loss else "missing occupancy/camera/VGGT depth modality"
            )

        if self.use_surface_loss and occupancy_probability is not None and target_surface_points is not None:
            loss_surface = self._surface_loss(occupancy_probability, target_surface_points, target_surface_weights)
            diagnostics.update({"surface_available": True, "surface_backend": self.surface_backend})
        else:
            loss_surface = zero
            diagnostics["surface_available"] = False
            diagnostics["surface_unavailable_reason"] = (
                "disabled" if not self.use_surface_loss else "missing occupancy or target point cloud"
            )

        if self.use_prior_preservation:
            gate_tensor = gate.float()
            if gate_tensor.ndim == v_final.ndim - 1:
                gate_tensor = gate_tensor.unsqueeze(-1)
            loss_prior = ((1.0 - gate_tensor) * (v_final.float() - v_base.float()).square()).mean()
            diagnostics["prior_available"] = True
        else:
            loss_prior = zero
            diagnostics["prior_available"] = False

        total = (
            self.weights["cfm"] * loss_cfm
            + self.weights["depth"] * loss_depth
            + self.weights["silhouette"] * loss_silhouette
            + self.weights["occupancy"] * loss_occupancy
            + self.weights["surface"] * loss_surface
            + self.weights["prior"] * loss_prior
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite geometric objective: {diagnostics}")
        return {
            "loss_total": total,
            "loss_cfm": loss_cfm,
            "loss_depth": loss_depth,
            "loss_silhouette": loss_silhouette,
            "loss_occupancy": loss_occupancy,
            "loss_surface": loss_surface,
            "loss_prior": loss_prior,
            "diagnostics": diagnostics,
        }

    def _surface_loss(
        self,
        occupancy_probability: torch.Tensor,
        target_points: torch.Tensor,
        target_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.surface_backend == "geomloss":
            SamplesLoss = self._load_geomloss()
            probability = F.adaptive_avg_pool3d(occupancy_probability.float(), 16).flatten(2)[:, 0]
            centers = _voxel_centers(16, probability.device).expand(probability.shape[0], -1, -1)
            source_weights = probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-8)
            if target_weights is None:
                target_weights = torch.ones(target_points.shape[:2], device=target_points.device)
            target_weights = target_weights.float().clamp_min(0)
            target_weights = target_weights / target_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=self.surface_blur, backend="online")
            # GeomLoss/KeOps performs the large pairwise reduction without an NxM tensor.
            value = sinkhorn(
                source_weights[..., None],
                centers,
                target_weights[..., None],
                target_points.float(),
            )
            return value.mean()
        frnn = self._load_frnn()
        centers = _voxel_centers(16, occupancy_probability.device).expand(occupancy_probability.shape[0], -1, -1)
        lengths_source = torch.full((centers.shape[0],), centers.shape[1], device=centers.device, dtype=torch.long)
        lengths_target = torch.full((target_points.shape[0],), target_points.shape[1], device=centers.device, dtype=torch.long)
        distances, _, _, _ = frnn.frnn_grid_points(
            target_points.float(), centers.float(), lengths_target, lengths_source, K=1, r=2.0, return_nn=False
        )
        valid = distances[..., 0] >= 0
        if not bool(valid.any()):
            raise RuntimeError("FRNN surface query found no neighbors")
        return distances[..., 0][valid].mean()

    def _load_geomloss(self):
        if self.extensions_root is not None:
            for path in (
                self.extensions_root / "geomloss" / "src",
                self.extensions_root / "keops" / "keopscore",
                self.extensions_root / "keops" / "pykeops",
            ):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
        try:
            from geomloss import SamplesLoss
        except ImportError as exc:
            raise ImportError("Enabled surface loss requires geomloss (and pykeops for the online backend)") from exc
        return SamplesLoss

    def _load_frnn(self):
        if self.extensions_root is not None and str(self.extensions_root / "FRNN") not in sys.path:
            sys.path.insert(0, str(self.extensions_root / "FRNN"))
        try:
            import frnn
        except ImportError as exc:
            raise ImportError("surface_backend=frnn requires the installed FRNN extension") from exc
        return frnn


def render_occupancy_volume(
    occupancy_logits: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
    center: torch.Tensor,
    half_extent: torch.Tensor,
    *,
    output_resolution: int,
    ray_samples: int,
) -> Dict[str, torch.Tensor]:
    """Vectorized differentiable alpha projection of a canonical occupancy grid."""
    B, V = K.shape[:2]
    K_render = K.float().clone()
    # Dataset views are square; scale against their principal point convention.
    source_w = (K_render[..., 0, 2] * 2.0).clamp_min(1.0)
    source_h = (K_render[..., 1, 2] * 2.0).clamp_min(1.0)
    K_render[..., 0, :] *= output_resolution / source_w[..., None]
    K_render[..., 1, :] *= output_resolution / source_h[..., None]
    y, x = torch.meshgrid(
        torch.arange(output_resolution, device=K.device, dtype=torch.float32) + 0.5,
        torch.arange(output_resolution, device=K.device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    x_cam = (x[None, None] - K_render[..., 0, 2, None, None]) / K_render[..., 0, 0, None, None]
    y_cam = (y[None, None] - K_render[..., 1, 2, None, None]) / K_render[..., 1, 1, None, None]
    camera_direction = torch.stack([x_cam, y_cam, torch.ones_like(x_cam)], dim=-1)
    world_direction = torch.einsum("bvij,bvhwj->bvhwi", c2w[..., :3, :3].float(), camera_direction)
    origin = c2w[..., :3, 3].float()[:, :, None, None]
    box_min = (center - half_extent)[:, None, None, None]
    box_max = (center + half_extent)[:, None, None, None]
    safe_direction = torch.where(world_direction.abs() < 1e-8, world_direction.sign() * 1e-8 + 1e-8, world_direction)
    t0 = (box_min - origin) / safe_direction
    t1 = (box_max - origin) / safe_direction
    near = torch.minimum(t0, t1).amax(dim=-1).clamp_min(0)
    far = torch.maximum(t0, t1).amin(dim=-1)
    ray_valid = far > near + 1e-6
    steps = torch.linspace(0, 1, ray_samples, device=K.device, dtype=torch.float32)
    depth_samples = near[..., None] + (far - near).clamp_min(0)[..., None] * steps
    world_samples = origin[..., None, :] + world_direction[..., None, :] * depth_samples[..., None]
    normalized = (world_samples - center[:, None, None, None, None]) / half_extent[:, None, None, None, None]
    grid = normalized.reshape(B * V, output_resolution, output_resolution, ray_samples, 3)
    density_grid = F.softplus(occupancy_logits.float())
    density_views = density_grid[:, None].expand(B, V, *density_grid.shape[1:]).reshape(B * V, *density_grid.shape[1:])
    density = F.grid_sample(
        density_views,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[:, 0].reshape(B, V, output_resolution, output_resolution, ray_samples)
    step_size = ((far - near).clamp_min(0) / max(ray_samples - 1, 1))[..., None]
    alpha = 1.0 - torch.exp(-density * step_size)
    alpha = alpha * ray_valid[..., None]
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), (1.0 - alpha + 1e-7)], dim=-1), dim=-1
    )[..., :-1]
    weights = alpha * transmittance
    silhouette = weights.sum(dim=-1).clamp(0, 1)
    depth = (weights * depth_samples).sum(dim=-1) / silhouette.clamp_min(1e-8)
    return {
        "silhouette": silhouette,
        "depth": depth,
        "ray_box_valid": ray_valid,
        "depth_valid": ray_valid & (silhouette > 1e-5),
    }


def _masked_velocity_mse(prediction: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    error = (prediction.float() - target.float()).square()
    if mask is None:
        return error.mean()
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(-1)
    weighted = error * mask.float()
    return weighted.sum() / (mask.float().sum() * error.shape[-1]).clamp_min(1)


def _dice_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.flatten(1)
    target = target.flatten(1)
    return (1.0 - (2.0 * (prediction * target).sum(dim=1) + 1e-6) / (
        prediction.sum(dim=1) + target.sum(dim=1) + 1e-6
    )).mean()


def _probability_bce(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Elementwise FP32 BCE for non-logit renderer probabilities under AMP."""
    prediction = prediction.float().clamp(1e-6, 1.0 - 1e-6)
    target = target.float()
    return -(target * prediction.log() + (1.0 - target) * torch.log1p(-prediction))


def _resize_occupancy(occupancy: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
    if occupancy.ndim == 4:
        occupancy = occupancy[:, None]
    if occupancy.shape[-3:] == target_shape:
        return occupancy
    return F.interpolate(occupancy, size=target_shape, mode="nearest")


def _resize_view_maps(tensor: torch.Tensor, resolution: int) -> torch.Tensor:
    if tensor.ndim == 4:
        tensor = tensor[:, :, None]
    B, V, C, H, W = tensor.shape
    resized = F.interpolate(tensor.reshape(B * V, C, H, W), (resolution, resolution), mode="bilinear", align_corners=False)
    return resized.reshape(B, V, C, resolution, resolution)


def _voxel_centers(resolution: int, device: torch.device) -> torch.Tensor:
    axis = (torch.arange(resolution, device=device, dtype=torch.float32) + 0.5) * (2.0 / resolution) - 1.0
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack([x, y, z], dim=-1).reshape(1, -1, 3)


__all__ = ["FlowMatchingLossBuilder", "render_occupancy_volume"]
