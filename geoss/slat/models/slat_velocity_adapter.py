from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from geoss.slat.models.slat_guidance_gate import GuidanceGate, normalize_timestep


class SLATVelocityAdapter(nn.Module):
    """Confidence-gated velocity residual for TRELLIS SLAT Flow."""

    def __init__(
        self,
        slat_dim: int = 8,
        cond_dim: int = 8,
        hidden_dim: int = 128,
        num_heads: int = 4,
        beta_mode: str = "cosine",
        beta_strength: float = 1.0,
        trust_region: float = 0.15,
        enabled: bool = True,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        self.slat_dim = slat_dim
        self.cond_dim = cond_dim
        self.trust_region = trust_region
        self.enabled = enabled
        self.latent_proj = nn.Linear(slat_dim, hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.delta_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, slat_dim))
        if zero_init:
            nn.init.zeros_(self.delta_head[-1].weight)
            nn.init.zeros_(self.delta_head[-1].bias)
        self.gate = GuidanceGate(mode=beta_mode, strength=beta_strength)

    def forward(
        self,
        slat_latent_tokens: torch.Tensor,
        slat_cond_tokens: torch.Tensor,
        slat_confidence: torch.Tensor,
        ss_confidence: torch.Tensor,
        timestep: torch.Tensor | float,
        v_slat_base: torch.Tensor,
        *,
        use_geovis_slat: bool | None = None,
    ) -> Dict[str, torch.Tensor]:
        enabled = self.enabled if use_geovis_slat is None else bool(use_geovis_slat)
        if not enabled:
            return self._identity(v_slat_base)

        B, L, C = slat_latent_tokens.shape
        if C != self.slat_dim:
            raise ValueError(f"slat dim mismatch: adapter={self.slat_dim}, input={C}")
        if v_slat_base.shape != (B, L, C):
            raise ValueError(f"v_slat_base must match slat tokens, got {tuple(v_slat_base.shape)} vs {(B, L, C)}")
        if slat_cond_tokens.shape[:2] != (B, L):
            raise ValueError(f"slat_cond_tokens must align [B,L,*], got {tuple(slat_cond_tokens.shape)}")
        if slat_confidence.shape != (B, L, 1) or ss_confidence.shape != (B, L, 1):
            raise ValueError("slat_confidence and ss_confidence must be [B,L,1]")

        q = self.latent_proj(slat_latent_tokens)
        k = self.cond_proj(_fit_dim(slat_cond_tokens, self.cond_dim))
        joint_confidence = (slat_confidence.clamp(0, 1) * ss_confidence.clamp(0, 1)).clamp(0, 1)
        kv = k * joint_confidence
        attn_out, attn_weights = self.cross_attn(q, kv, kv, need_weights=True)
        h = self.norm(q + attn_out)
        delta_raw = self.delta_head(h)

        t_norm = normalize_timestep(timestep, B).to(v_slat_base.device)
        tau = self.trust_region * (0.25 + 0.75 * t_norm).view(B, 1, 1)
        delta_v_slat = delta_raw.clamp(-tau, tau)
        clipping_ratio = (delta_raw.abs() > tau).float().mean()
        beta_t = self.gate(timestep, B).to(v_slat_base.device, v_slat_base.dtype)
        v_slat_geo = v_slat_base + beta_t * joint_confidence * delta_v_slat

        return {
            "v_slat_geo": v_slat_geo,
            "delta_v_slat": delta_v_slat,
            "beta_t": beta_t,
            "joint_confidence": joint_confidence,
            "clipping_ratio": clipping_ratio,
            "debug": {
                "delta_raw": delta_raw,
                "attn_weights": attn_weights,
                "delta_norm": delta_v_slat.norm(dim=-1).mean(),
                "base_norm": v_slat_base.norm(dim=-1).mean(),
                "geo_norm": v_slat_geo.norm(dim=-1).mean(),
                "confidence_mean": joint_confidence.mean(),
                "confidence_std": joint_confidence.std(unbiased=False),
                "clipping_ratio": clipping_ratio,
            },
        }

    def _identity(self, v_slat_base: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = torch.zeros_like(v_slat_base)
        B, L = v_slat_base.shape[:2]
        scalar_zero = torch.zeros((), device=v_slat_base.device, dtype=v_slat_base.dtype)
        return {
            "v_slat_geo": v_slat_base,
            "delta_v_slat": zero,
            "beta_t": torch.zeros(B, 1, 1, device=v_slat_base.device, dtype=v_slat_base.dtype),
            "joint_confidence": torch.zeros(B, L, 1, device=v_slat_base.device, dtype=v_slat_base.dtype),
            "clipping_ratio": scalar_zero,
            "debug": {
                "delta_norm": scalar_zero,
                "base_norm": v_slat_base.norm(dim=-1).mean(),
                "geo_norm": v_slat_base.norm(dim=-1).mean(),
                "confidence_mean": scalar_zero,
                "confidence_std": scalar_zero,
                "clipping_ratio": scalar_zero,
            },
        }


def _fit_dim(tokens: torch.Tensor, target_dim: int) -> torch.Tensor:
    if tokens.shape[-1] == target_dim:
        return tokens
    if tokens.shape[-1] > target_dim:
        raise ValueError(
            f"Refusing to truncate SLAT condition tokens from {tokens.shape[-1]} to {target_dim}. "
            "Use GeoVisSLATAggregator learned projection before SLATVelocityAdapter."
        )
    return torch.cat([tokens, tokens.new_zeros(*tokens.shape[:-1], target_dim - tokens.shape[-1])], dim=-1)
