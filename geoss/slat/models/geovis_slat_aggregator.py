from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class GeoVisSLATAggregator(nn.Module):
    """Aggregate per-voxel multi-view evidence into SLAT conditioning tokens."""

    def __init__(
        self,
        evidence_dim: int = 128,
        slat_dim: int = 8,
        hidden_dim: int = 128,
        num_heads: int = 4,
        view_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.slat_dim = slat_dim
        self.view_dropout = view_dropout
        self.query_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.evidence_proj = nn.Linear(evidence_dim, hidden_dim)
        self.slat_proj = nn.Linear(slat_dim, hidden_dim)
        self.geo_proj = nn.Linear(slat_dim, hidden_dim)
        self.appearance_branch = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, slat_dim))
        # Reliability answers "can the observed views support a correction?".
        # Correction demand answers "does the current prior need correction?".
        # Uncertainty describes the residual error distribution. These are
        # deliberately separate statistical quantities.
        self.confidence_head = nn.Sequential(nn.Linear(hidden_dim + 4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.correction_demand_head = nn.Sequential(nn.Linear(hidden_dim + 4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.uncertainty_head = nn.Sequential(nn.Linear(hidden_dim + 4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(
        self,
        view_slat_tokens: torch.Tensor,
        visibility: torch.Tensor,
        occlusion_score: torch.Tensor,
        depth_residual: torch.Tensor,
        ss_confidence: torch.Tensor,
        *,
        appearance_conflict: Optional[torch.Tensor] = None,
        ss_geo_tokens: Optional[torch.Tensor] = None,
        slat_latent_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, L, N, Ce = view_slat_tokens.shape
        if ss_confidence.shape != (B, L, 1):
            raise ValueError(f"ss_confidence must be [B,L,1], got {tuple(ss_confidence.shape)}")
        if appearance_conflict is None:
            appearance_conflict = torch.zeros(B, L, 1, device=view_slat_tokens.device, dtype=view_slat_tokens.dtype)

        evidence = self.evidence_proj(view_slat_tokens).reshape(B * L, N, -1)
        query = self.query_token.expand(B * L, 1, -1)
        if slat_latent_tokens is not None:
            query = query + self.slat_proj(_fit_dim(slat_latent_tokens, self.slat_dim)).reshape(B * L, 1, -1)
        if ss_geo_tokens is not None:
            query = query + self.geo_proj(ss_geo_tokens).reshape(B * L, 1, -1)

        reliability = visibility.clamp(0, 1) * (1.0 - occlusion_score.clamp(0, 1)) * torch.exp(-depth_residual.clamp_min(0))
        if self.training and self.view_dropout > 0:
            keep = (torch.rand_like(reliability) > self.view_dropout).float()
            reliability = reliability * keep
        flat_rel = reliability.reshape(B * L, N, 1)
        key_padding_mask = flat_rel.squeeze(-1) <= 1e-6
        all_invalid = key_padding_mask.all(dim=1)
        if all_invalid.any():
            key_padding_mask[all_invalid, 0] = False
            flat_rel[all_invalid, 0] = 1.0

        weighted_evidence = evidence * flat_rel
        attn_out, attn_weights = self.attn(query, weighted_evidence, weighted_evidence, key_padding_mask=key_padding_mask, need_weights=True)
        h = self.norm(query + attn_out + self.appearance_branch(attn_out)).reshape(B, L, -1)
        slat_cond_tokens = self.out(h)

        view_weights = attn_weights.reshape(B, L, N, 1) * reliability
        view_weights = view_weights / view_weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
        visible_support = visibility.mean(dim=2)
        occlusion_mean = occlusion_score.mean(dim=2)
        residual_score = torch.exp(-depth_residual.mean(dim=2).clamp_min(0))
        appearance_consistency = (1.0 - appearance_conflict.clamp(0, 1)).clamp(0, 1)
        conf_input = torch.cat([h, visible_support, 1.0 - occlusion_mean, residual_score, appearance_consistency], dim=-1)
        learned_conf = torch.sigmoid(self.confidence_head(conf_input))
        evidence_reliability = (
            learned_conf
            * ss_confidence.clamp(0, 1).sqrt()
            * visible_support.clamp(0, 1)
            * appearance_consistency
        ).clamp(0, 1)
        correction_demand_logits = self.correction_demand_head(conf_input)
        correction_demand = torch.sigmoid(correction_demand_logits)
        residual_variance = torch.nn.functional.softplus(self.uncertainty_head(conf_input)) + 1e-4

        return {
            "slat_cond_tokens": slat_cond_tokens,
            "slat_confidence": evidence_reliability,
            "evidence_reliability": evidence_reliability,
            "correction_demand": correction_demand,
            "correction_demand_logits": correction_demand_logits,
            "residual_variance": residual_variance,
            "view_weights": view_weights,
            "appearance_consistency": appearance_consistency,
            "debug": {
                "visible_support": visible_support,
                "occlusion_mean": occlusion_mean,
                "confidence_mean": evidence_reliability.mean(),
                "confidence_std": evidence_reliability.std(unbiased=False),
                "confidence_all_zero": (evidence_reliability <= 1e-6).all(),
                "confidence_all_one": (evidence_reliability >= 1.0 - 1e-6).all(),
                "correction_demand_mean": correction_demand.mean(),
                "residual_variance_mean": residual_variance.mean(),
            },
        }


def _fit_dim(tokens: torch.Tensor, target_dim: int) -> torch.Tensor:
    if tokens.shape[-1] == target_dim:
        return tokens
    if tokens.shape[-1] > target_dim:
        raise ValueError(f"Refusing to truncate SLAT latent tokens from {tokens.shape[-1]} to {target_dim}.")
    return torch.cat([tokens, tokens.new_zeros(*tokens.shape[:-1], target_dim - tokens.shape[-1])], dim=-1)
