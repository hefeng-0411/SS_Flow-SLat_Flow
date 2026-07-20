from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .cross_view_evidence_aggregator import CrossViewEvidenceAggregator
from .ray_evidence_sampler import RayEvidenceSampler
from .sparse_anchor_queries import SparseAnchorQueries
from .ss_velocity_adapter import SSVelocityAdapter
from geoss.geometry.alignment import GeometryAlignment


class SparseRayGeoSSAdapter(nn.Module):
    """Sparse Ray-Evidence Geometry Adapter for TRELLIS Sparse Structure Flow."""

    def __init__(
        self,
        num_anchors: int = 4096,
        anchor_dim: int = 256,
        evidence_dim: int = 128,
        geo_dim: int = 256,
        latent_dim: int = 8,
        alignment_enabled: bool = True,
        dynamic_anchor_enabled: bool = True,
        local_attention_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.alignment_enabled = alignment_enabled
        self.dynamic_anchor_enabled = dynamic_anchor_enabled
        self.anchor_queries = SparseAnchorQueries(num_anchors=num_anchors, feature_dim=anchor_dim)
        self.geometry_alignment = GeometryAlignment()
        self.ray_sampler = RayEvidenceSampler(evidence_dim=evidence_dim)
        self.aggregator = CrossViewEvidenceAggregator(anchor_dim=anchor_dim, evidence_dim=evidence_dim, hidden_dim=geo_dim)
        self.velocity_adapter = SSVelocityAdapter(latent_dim=latent_dim, geo_dim=geo_dim, local_attention=local_attention_enabled)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        images = batch["images"]
        B = images.shape[0]
        device = images.device
        if self.alignment_enabled and "vggt_pointmap" in batch and batch.get("vggt_pointmap") is not None:
            alignment = self.geometry_alignment(
                vggt_depth=batch.get("vggt_depth"),
                vggt_pointmap=batch.get("vggt_pointmap"),
                K=batch["K"],
                c2w=batch["c2w"],
                w2c=batch["w2c"],
                masks=batch.get("masks"),
                vggt_confidence=batch.get("vggt_confidence"),
                dataset_depth=batch.get("depths"),
                vggt_camera=batch.get("vggt_camera"),
            )
            batch = dict(batch)
            batch["aligned_pointmap"] = alignment["aligned_pointmap"]
            batch["aligned_depth"] = alignment["aligned_depth"]
            batch["alignment_confidence"] = alignment["alignment_confidence"]
        else:
            alignment = {}
        if self.dynamic_anchor_enabled:
            anchors = self.anchor_queries.forward_dynamic(
                B,
                device=device,
                # Camera-aligned VGGT points share TRELLIS' physical
                # [-0.5,0.5] world frame; anchor queries/occupancy use [-1,1].
                aligned_pointmap=(batch["aligned_pointmap"] * 2.0) if batch.get("aligned_pointmap") is not None else None,
                masks=batch.get("masks"),
                confidence=batch.get("alignment_confidence", batch.get("vggt_confidence")),
            )
            anchor_xyz = anchors["anchor_xyz"]
            anchor_feat = anchors["anchor_feat"]
            anchor_metadata = anchors["anchor_metadata"]
        else:
            anchor_xyz, anchor_feat = self.anchor_queries(B, device=device)
            anchor_metadata = None
        ray = self.ray_sampler(
            anchor_xyz=anchor_xyz,
            K=batch["K"],
            c2w=batch["c2w"],
            w2c=batch["w2c"],
            masks=batch["masks"],
            depths=batch.get("aligned_depth", batch.get("depths")),
            vggt_depth=batch.get("aligned_depth", batch.get("vggt_depth")),
            vggt_pointmap=batch.get("aligned_pointmap", batch.get("vggt_pointmap")),
            vggt_features=batch.get("vggt_features"),
        )
        agg = self.aggregator(ray["view_tokens"], anchor_feat, ray["ray_valid"], conflict_score=ray.get("conflict_score"))
        if "ss_latent_tokens" not in batch:
            return {
                "anchor_xyz": anchor_xyz,
                "anchor_metadata": anchor_metadata if anchor_metadata is not None else torch.zeros(B, anchor_xyz.shape[1], 7, device=device, dtype=anchor_xyz.dtype),
                "geo_tokens": agg["geo_tokens"],
                "occ_evidence": agg["occ_evidence"],
                "free_evidence": agg["free_evidence"],
                "p_occ": agg["p_occ"],
                "uncertainty": agg["uncertainty"],
                "geo_confidence": agg["geo_confidence"],
                "debug": {"ray": ray, "confidence": agg["confidence_stats"], "alignment": alignment, "context_only": True},
            }
        ss_latent_tokens = batch["ss_latent_tokens"]
        v_base = batch.get("v_base")
        if v_base is None:
            raise KeyError("SparseRayGeoSSAdapter requires v_base when ss_latent_tokens are provided; zero base velocity is not allowed.")
        timestep = batch.get("timestep", torch.zeros(B, device=device))
        vel = self.velocity_adapter(
            ss_latent_tokens=ss_latent_tokens,
            geo_tokens=agg["geo_tokens"],
            geo_confidence=agg["geo_confidence"],
            timestep=timestep,
            v_base=v_base,
            voxel_xyz=batch.get("ss_voxel_xyz"),
            anchor_xyz=anchor_xyz,
            anchor_metadata=anchor_metadata,
        )
        return {
            "anchor_xyz": anchor_xyz,
            "anchor_metadata": anchor_metadata if anchor_metadata is not None else torch.zeros(B, anchor_xyz.shape[1], 7, device=device, dtype=anchor_xyz.dtype),
            "geo_tokens": agg["geo_tokens"],
            "occ_evidence": agg["occ_evidence"],
            "free_evidence": agg["free_evidence"],
            "p_occ": agg["p_occ"],
            "uncertainty": agg["uncertainty"],
            "geo_confidence": agg["geo_confidence"],
            "v_geo": vel["v_geo"],
            "delta_v_geo": vel["delta_v_geo"],
            "token_confidence": vel["token_confidence"],
            "debug": {"ray": ray, "velocity": vel["debug"], "alpha_t": vel["alpha_t"], "confidence": agg["confidence_stats"], "alignment": alignment},
        }
