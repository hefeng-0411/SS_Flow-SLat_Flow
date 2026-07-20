from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import torch

from geoss.losses.render_losses import build_lpips, render_level_losses
from geoss.renderers.gsplat_renderer import render_gaussians
from geoss.slat.utils.normalization import denormalize_slat


def flow_x0_from_velocity(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    timestep: torch.Tensor,
    sigma_min: float,
) -> torch.Tensor:
    """Invert TRELLIS' linear flow path: x0=(1-sigma_min)x_t-s(t)v."""
    if x_t.shape != velocity.shape:
        raise ValueError("x_t and velocity must have the same shape")
    t = timestep.to(device=x_t.device, dtype=x_t.dtype).reshape(x_t.shape[0], *([1] * (x_t.ndim - 1)))
    sigma = float(sigma_min) + (1.0 - float(sigma_min)) * t
    return (1.0 - float(sigma_min)) * x_t - sigma * velocity


@dataclass
class DecodedAssetLossConfig:
    enabled: bool = False
    every: int = 4
    views_per_object: int = 2
    rgb_weight: float = 1.0
    foreground_rgb_weight: float = 1.0
    ssim_weight: float = 0.2
    lpips_weight: float = 0.1
    mask_weight: float = 0.5
    depth_weight: float = 0.0
    geometry_weight: float = 0.1
    geometry_points: int = 2048
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.every < 1 or self.views_per_object < 1 or self.geometry_points < 1:
            raise ValueError("decoded supervision cadence, view count, and geometry point count must be positive")
        weights = (
            self.rgb_weight,
            self.foreground_rgb_weight,
            self.ssim_weight,
            self.lpips_weight,
            self.mask_weight,
            self.depth_weight,
            self.geometry_weight,
        )
        if any(float(value) < 0 for value in weights):
            raise ValueError("decoded supervision weights must be non-negative")
        if len(self.background) != 3 or any(not 0.0 <= float(value) <= 1.0 for value in self.background):
            raise ValueError("decoded supervision background must contain three values in [0,1]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "DecodedAssetLossConfig":
        value = dict(value or {})
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise KeyError(f"Unknown decoded-supervision options: {sorted(unknown)}")
        if "background" in value:
            value["background"] = tuple(float(v) for v in value["background"])
        return cls(**value)


class DecodedAssetSupervisor:
    """Backpropagate final Gaussian RGB/silhouette/geometry errors to SLAT control."""

    def __init__(self, pipeline, config: Mapping[str, Any] | None = None) -> None:
        self.pipeline = pipeline
        self.config = DecodedAssetLossConfig.from_mapping(config)
        if self.config.enabled and "slat_decoder_gs" not in pipeline.models:
            raise RuntimeError("Decoded asset supervision requires TRELLIS slat_decoder_gs.")
        self.lpips_model = None
        if self.config.enabled and self.config.lpips_weight > 0:
            self.lpips_model = build_lpips(real_mode=True)
            decoder = pipeline.models["slat_decoder_gs"]
            device = next(decoder.parameters()).device
            self.lpips_model = self.lpips_model.to(device).eval()
            for parameter in self.lpips_model.parameters():
                parameter.requires_grad_(False)

    def __call__(self, out: dict, batch: dict, step: int) -> Dict[str, torch.Tensor]:
        anchor = out["v_slat_geo"].sum() * 0.0
        if not self.config.enabled or step % max(1, self.config.every) != 0:
            return {"loss": anchor, "enabled": torch.tensor(False, device=anchor.device), "applied": torch.tensor(False, device=anchor.device)}
        x0_pred = flow_x0_from_velocity(
            batch["slat_latent_tokens"], out["v_slat_geo"], batch["timestep"], float(batch["flow_sigma_min"])
        )
        raw_pred = denormalize_slat(x0_pred, self.pipeline.slat_normalization)
        sparse = _padded_to_sparse(raw_pred, batch["slat_indices"], batch.get("slat_token_valid_mask"))
        gaussians = self.pipeline.models["slat_decoder_gs"](sparse)
        render_terms = []
        geometry_terms = []
        for object_index, gaussian in enumerate(gaussians):
            view_count = int(batch["images"].shape[1])
            count = min(max(1, self.config.views_per_object), view_count)
            # Torch RNG is checkpointed with the training process, so this
            # remains reproducible while exposing different cameras over time.
            selected = torch.randperm(view_count, device=batch["images"].device)[:count]
            cameras = {"K": batch["K"][object_index, selected], "w2c": batch["w2c"][object_index, selected]}
            backgrounds = batch["images"].new_tensor(self.config.background).view(1, 3).expand(count, -1)
            rendered = render_gaussians(
                gaussian,
                cameras,
                tuple(batch["images"].shape[-2:]),
                backgrounds=backgrounds,
                return_depth=self.config.depth_weight > 0,
            )
            targets = batch["images"][object_index, selected]
            target_masks = batch["masks"][object_index, selected]
            target_depth = None
            if self.config.depth_weight > 0:
                depth_source = batch.get("aligned_depth", batch.get("depths"))
                if isinstance(depth_source, torch.Tensor):
                    target_depth = depth_source[object_index, selected].detach()
            losses = render_level_losses(
                rendered["rendered_rgb"],
                targets,
                rendered.get("rendered_alpha"),
                target_masks,
                rendered.get("rendered_depth"),
                target_depth,
                lpips_model=self.lpips_model,
            )
            render_terms.append(
                self.config.rgb_weight * losses["L_rgb"]
                + self.config.foreground_rgb_weight * losses.get("L_rgb_foreground", anchor)
                + self.config.ssim_weight * losses["L_ssim"]
                + self.config.lpips_weight * losses.get("L_lpips", anchor)
                + self.config.mask_weight * losses.get("L_mask", anchor)
                + self.config.depth_weight * losses.get("L_depth", anchor)
            )
            if self.config.geometry_weight > 0 and isinstance(batch.get("gt_occ"), torch.Tensor):
                geometry_terms.append(
                    _decoded_geometry_loss(gaussian, batch["gt_occ"][object_index], self.config.geometry_points)
                )
        render_loss = torch.stack(render_terms).mean() if render_terms else anchor
        geometry_loss = torch.stack(geometry_terms).mean() if geometry_terms else anchor
        total = render_loss + self.config.geometry_weight * geometry_loss
        return {
            "loss": total,
            "render_loss": render_loss,
            "geometry_loss": geometry_loss,
            "enabled": torch.tensor(True, device=total.device),
            "applied": torch.tensor(True, device=total.device),
        }


def _padded_to_sparse(feats: torch.Tensor, indices: torch.Tensor, valid_mask: torch.Tensor | None):
    from trellis.modules import sparse as sp

    if feats.ndim != 3 or indices.shape != (*feats.shape[:2], 3):
        raise ValueError("Decoded SLAT tensors must be feats [B,L,C] and indices [B,L,3].")
    if valid_mask is None:
        valid_mask = torch.ones(*feats.shape[:2], 1, device=feats.device, dtype=torch.bool)
    valid = valid_mask.squeeze(-1).bool()
    flat_feats, flat_coords = [], []
    for batch_index in range(feats.shape[0]):
        selected = valid[batch_index]
        flat_feats.append(feats[batch_index, selected])
        batch_column = torch.full(
            (int(selected.sum()), 1), batch_index, device=indices.device, dtype=indices.dtype
        )
        flat_coords.append(torch.cat([batch_column, indices[batch_index, selected]], dim=-1))
    if not flat_feats or sum(item.shape[0] for item in flat_feats) == 0:
        raise RuntimeError("Decoded supervision received no valid SLAT tokens.")
    return sp.SparseTensor(feats=torch.cat(flat_feats, dim=0), coords=torch.cat(flat_coords, dim=0).int().contiguous())


def _decoded_geometry_loss(gaussian, occupancy: torch.Tensor, max_points: int) -> torch.Tensor:
    means = gaussian.get_xyz
    opacity = gaussian.get_opacity.reshape(-1)
    keep_pred = min(max(1, int(max_points)), means.shape[0])
    pred_indices = opacity.topk(keep_pred, largest=True).indices
    pred = means[pred_indices]
    occ = occupancy.squeeze().bool()
    gt_indices = torch.nonzero(occ, as_tuple=False)
    if gt_indices.numel() == 0:
        return means.sum() * 0.0
    if gt_indices.shape[0] > max_points:
        stride = max(1, gt_indices.shape[0] // max_points)
        gt_indices = gt_indices[::stride][:max_points]
    resolution = float(occ.shape[0])
    gt = (gt_indices.to(device=means.device, dtype=means.dtype) + 0.5) / resolution - 0.5
    distances = torch.cdist(pred.float(), gt.float()).to(means.dtype)
    return 0.5 * (distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean())
