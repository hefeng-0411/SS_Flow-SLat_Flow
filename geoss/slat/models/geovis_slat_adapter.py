from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .active_voxel_projector import ActiveVoxelProjector
from .geovis_slat_aggregator import GeoVisSLATAggregator
from .slat_velocity_adapter import SLATVelocityAdapter
from .visibility_evidence_sampler import VisibilityEvidenceSampler


class GeoVisSLATAdapter(nn.Module):
    """End-to-end GeoVis-SLAT adapter for active-voxel SLAT velocity control."""

    def __init__(
        self,
        resolution: int = 64,
        slat_dim: int = 8,
        evidence_dim: int = 128,
        hidden_dim: int = 128,
        feature_dim: int = 16,
        num_heads: int = 4,
        trust_region: float = 0.15,
        beta_mode: str = "cosine",
        beta_strength: float = 1.0,
        fusion_mode: str = "aligned",
        confidence_floor: float = 0.05,
        factorized_control: bool = True,
        use_geovis_slat: bool = True,
    ) -> None:
        super().__init__()
        self.projector = ActiveVoxelProjector(resolution=resolution)
        self.evidence_sampler = VisibilityEvidenceSampler(
            evidence_dim=evidence_dim,
            feature_dim=feature_dim,
        )
        self.aggregator = GeoVisSLATAggregator(
            evidence_dim=evidence_dim,
            slat_dim=slat_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )
        self.velocity_adapter = SLATVelocityAdapter(
            slat_dim=slat_dim,
            cond_dim=slat_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            beta_mode=beta_mode,
            beta_strength=beta_strength,
            trust_region=trust_region,
            fusion_mode=fusion_mode,
            confidence_floor=confidence_floor,
            enabled=use_geovis_slat,
        )
        self.gradient_checkpointing = False
        self.factorized_control = bool(factorized_control)

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Checkpoint the attention-heavy aggregation path during joint training."""
        self.gradient_checkpointing = bool(enabled)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        images = batch["images"]
        if images.ndim != 5:
            raise ValueError(f"images must be [B,N,3,H,W], got {tuple(images.shape)}")
        H, W = images.shape[-2:]
        camera = batch.get("aligned_camera", {}) if isinstance(batch.get("aligned_camera", {}), dict) else {}
        projection = self.projector(
            batch.get("ss_active_indices"),
            camera.get("K", batch["K"]),
            camera.get("w2c", batch["w2c"]),
            (H, W),
            active_xyz=batch.get("ss_active_xyz"),
            ss_confidence=batch.get("ss_confidence"),
            c2w=camera.get("c2w", batch.get("c2w")),
        )
        evidence = self.evidence_sampler(
            projection["active_xyz"],
            projection["uv"],
            projection["z_active"],
            images,
            batch["masks"],
            projection["in_bounds"],
            depths=batch.get("aligned_depth", batch.get("depths")),
            vggt_depth=batch.get("aligned_depth", batch.get("vggt_depth")),
            vggt_pointmap=batch.get("aligned_pointmap", batch.get("vggt_pointmap")),
            vggt_features=batch.get("vggt_features"),
        )
        def aggregate(view_tokens, visibility, occlusion, depth_residual, confidence, conflict):
            return self.aggregator(
                view_tokens, visibility, occlusion, depth_residual, confidence,
                appearance_conflict=conflict,
                ss_geo_tokens=batch.get("ss_geo_tokens"),
                # Evidence aggregation uses a clean/predicted object reference,
                # while the velocity branch sees the noisy flow state. This
                # matches inference, where Stage 2 provides a fixed prior and
                # TRELLIS evolves a separate stochastic state over time.
                slat_latent_tokens=batch.get("slat_reference_tokens", batch["slat_latent_tokens"]),
            )
        # The non-reentrant implementation preserves the nested dict output and
        # recomputes attention in backward, substantially reducing joint VRAM.
        aggregated = checkpoint(
            aggregate,
            evidence["view_slat_tokens"], evidence["visibility"], evidence["occlusion_score"],
            evidence["depth_residual"], projection["ss_confidence"], evidence["appearance_conflict"],
            use_reentrant=False,
        ) if self.training and self.gradient_checkpointing else aggregate(
            evidence["view_slat_tokens"], evidence["visibility"], evidence["occlusion_score"],
            evidence["depth_residual"], projection["ss_confidence"], evidence["appearance_conflict"],
        )
        token_valid_mask = batch.get("slat_token_valid_mask")
        if token_valid_mask is not None:
            if token_valid_mask.shape != (*batch["slat_latent_tokens"].shape[:2], 1):
                raise ValueError(
                    "slat_token_valid_mask must be [B,L,1], got "
                    f"{tuple(token_valid_mask.shape)} for tokens {tuple(batch['slat_latent_tokens'].shape)}"
                )
            token_valid_mask = token_valid_mask.to(
                device=batch["slat_latent_tokens"].device,
                dtype=batch["slat_latent_tokens"].dtype,
            )
            aggregated["slat_cond_tokens"] = aggregated["slat_cond_tokens"] * token_valid_mask
            aggregated["slat_confidence"] = aggregated["slat_confidence"] * token_valid_mask
            aggregated["evidence_reliability"] = aggregated["evidence_reliability"] * token_valid_mask
            aggregated["correction_demand"] = aggregated["correction_demand"] * token_valid_mask
        velocity = self.velocity_adapter(
            batch["slat_latent_tokens"],
            aggregated["slat_cond_tokens"],
            aggregated["slat_confidence"],
            projection["ss_confidence"],
            batch["timestep"],
            batch["v_slat_base"],
            use_geovis_slat=batch.get("use_geovis_slat", None),
            token_valid_mask=token_valid_mask,
            correction_demand=aggregated["correction_demand"] if self.factorized_control else None,
            residual_variance=aggregated["residual_variance"] if self.factorized_control else None,
        )
        return {
            "slat_cond_tokens": aggregated["slat_cond_tokens"],
            "slat_confidence": aggregated["slat_confidence"],
            "evidence_reliability": aggregated["evidence_reliability"],
            "correction_demand": aggregated["correction_demand"],
            "residual_variance": aggregated["residual_variance"],
            "delta_v_slat": velocity["delta_v_slat"],
            "v_slat_geo": velocity["v_slat_geo"],
            "view_weights": aggregated["view_weights"],
            "visibility": evidence["visibility"],
            "occlusion_score": evidence["occlusion_score"],
            "depth_residual": evidence["depth_residual"],
            "appearance_conflict": evidence["appearance_conflict"],
            "active_xyz": projection["active_xyz"],
            "ss_confidence": projection["ss_confidence"],
            "sampled_rgb": evidence["sampled_rgb"],
            "sampled_features": evidence["sampled_features"],
            "debug": {
                "projection": projection["projection_debug"],
                "evidence": evidence["debug"],
                "aggregation": aggregated["debug"],
                "velocity": velocity["debug"],
                "beta_t": velocity["beta_t"],
                "joint_confidence": velocity["joint_confidence"],
                "correction_gate": velocity["correction_gate"],
                "clipping_ratio": velocity["clipping_ratio"],
            },
        }
