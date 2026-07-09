from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossViewEvidenceAggregator(nn.Module):
    """Aggregate per-view ray evidence for each anchor with lightweight cross-view attention."""

    def __init__(
        self,
        anchor_dim: int = 256,
        evidence_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        view_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.view_dropout = view_dropout
        self.anchor_proj = nn.Linear(anchor_dim, hidden_dim)
        self.evidence_proj = nn.Linear(evidence_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim))
        self.occ_head = nn.Linear(hidden_dim, 1)
        self.free_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        view_tokens: torch.Tensor,
        anchor_feat: torch.Tensor,
        ray_valid: torch.Tensor,
        conflict_score: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if view_tokens.ndim != 4:
            raise ValueError(f"view_tokens must be [B,M,N,C_e], got {tuple(view_tokens.shape)}")
        B, M, N, _ = view_tokens.shape
        if anchor_feat.shape[:2] != (B, M):
            raise ValueError(f"anchor_feat batch/anchor mismatch: {tuple(anchor_feat.shape)} vs {(B, M)}")
        valid = ray_valid > 0.5
        if valid.shape != (B, M, N, 1):
            raise ValueError(f"ray_valid must be [B,M,N,1], got {tuple(ray_valid.shape)}")
        if self.training and self.view_dropout > 0:
            keep = torch.rand(B, M, N, 1, device=view_tokens.device) >= self.view_dropout
            valid = valid & keep

        q = self.anchor_proj(anchor_feat).reshape(B * M, 1, self.hidden_dim)
        kv = self.evidence_proj(view_tokens).reshape(B * M, N, self.hidden_dim)
        valid_flat = valid.reshape(B * M, N)
        all_invalid = ~valid_flat.any(dim=1)
        key_padding_mask = ~valid_flat
        key_padding_mask[all_invalid] = False
        kv = kv.masked_fill((~valid_flat).unsqueeze(-1), 0.0)
        attn, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask)
        token = self.norm(q + attn)
        token = self.norm(token + self.ffn(token))
        geo_tokens = token.reshape(B, M, self.hidden_dim)
        occ_evidence = self.occ_head(geo_tokens)
        free_evidence = self.free_head(geo_tokens)
        alpha_occ = F.softplus(occ_evidence) + 1.0
        alpha_free = F.softplus(free_evidence) + 1.0
        p_occ = alpha_occ / (alpha_occ + alpha_free).clamp_min(1e-6)
        uncertainty = 2.0 / (alpha_occ + alpha_free).clamp_min(1e-6)
        geo_confidence = torch.exp(-uncertainty)
        if conflict_score is not None:
            if conflict_score.shape != (B, M, 1):
                raise ValueError(f"conflict_score must be [B,M,1], got {tuple(conflict_score.shape)}")
            geo_confidence = geo_confidence * torch.exp(-conflict_score.clamp_min(0.0))
        geo_confidence = geo_confidence * valid.any(dim=2).float()
        return {
            "geo_tokens": geo_tokens,
            "occ_evidence": occ_evidence,
            "free_evidence": free_evidence,
            "p_occ": p_occ,
            "uncertainty": uncertainty,
            "geo_confidence": geo_confidence,
            "confidence_stats": {
                "mean": geo_confidence.mean().detach(),
                "std": geo_confidence.std(unbiased=False).detach(),
                "all_zero": (geo_confidence <= 1e-6).all().detach(),
                "all_one": (geo_confidence >= 1.0 - 1e-6).all().detach(),
            },
        }
