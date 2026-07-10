from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .guidance_gate import GuidanceGate, normalize_timestep


class SSVelocityAdapter(nn.Module):
    """Confidence-gated velocity residual for TRELLIS SS Flow."""

    def __init__(
        self,
        latent_dim: int = 8,
        geo_dim: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 4,
        alpha_mode: str = "cosine",
        alpha_strength: float = 1.0,
        trust_region: float = 0.25,
        enabled: bool = True,
        local_attention: bool = True,
        knn_k: int = 32,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.geo_dim = geo_dim
        self.hidden_dim = hidden_dim
        self.trust_region = trust_region
        self.enabled = enabled
        self.local_attention = local_attention
        self.knn_k = knn_k
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.geo_proj = nn.Linear(geo_dim, hidden_dim)
        self.cross_attn = None if local_attention else nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.rel_mlp = nn.Sequential(nn.Linear(4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)) if local_attention else None
        self.norm = nn.LayerNorm(hidden_dim)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        self.gate = GuidanceGate(mode=alpha_mode, strength=alpha_strength)

    def forward(
        self,
        ss_latent_tokens: torch.Tensor,
        geo_tokens: torch.Tensor,
        geo_confidence: torch.Tensor,
        timestep: torch.Tensor,
        v_base: torch.Tensor,
        voxel_xyz: torch.Tensor | None = None,
        anchor_xyz: torch.Tensor | None = None,
        anchor_metadata: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.enabled:
            zero = torch.zeros_like(v_base)
            return {
                "v_geo": v_base,
                "delta_v_geo": zero,
                "token_confidence": torch.zeros(*v_base.shape[:2], 1, device=v_base.device, dtype=v_base.dtype),
                "alpha_t": torch.zeros(v_base.shape[0], 1, 1, device=v_base.device, dtype=v_base.dtype),
                "debug": {"clipping_ratio": torch.tensor(0.0, device=v_base.device), "confidence_mean": torch.tensor(0.0, device=v_base.device)},
            }
        B, L, C = ss_latent_tokens.shape
        if v_base.shape != (B, L, C):
            raise ValueError(f"v_base must match ss_latent_tokens, got {tuple(v_base.shape)} vs {(B, L, C)}")
        if C != self.latent_dim:
            raise ValueError(f"latent dim mismatch: adapter={self.latent_dim}, input={C}")
        if geo_confidence.shape[:2] != geo_tokens.shape[:2]:
            raise ValueError("geo_confidence must align with geo_tokens")

        q = self.latent_proj(ss_latent_tokens)
        k = self.geo_proj(geo_tokens)
        geo_conf = geo_confidence.clamp(0, 1)
        if self.local_attention and voxel_xyz is not None and anchor_xyz is not None:
            attn_out, attn_weights, token_confidence, local_debug = self._local_anchor_attention(q, k, geo_conf, voxel_xyz, anchor_xyz, anchor_metadata)
        else:
            if self.cross_attn is None:
                raise ValueError("SSVelocityAdapter was constructed for local attention but voxel_xyz/anchor_xyz were not provided.")
            k_weighted = k * geo_conf
            attn_out, attn_weights = self.cross_attn(q, k_weighted, k_weighted, need_weights=True)
            weights = attn_weights.clamp_min(0.0)
            token_confidence = torch.bmm(weights, geo_conf).clamp(0, 1)
            local_debug = {"local_attention": torch.tensor(False, device=v_base.device)}
        h = self.norm(q + attn_out)
        delta_raw = self.delta_head(h)

        t_norm = normalize_timestep(timestep, B).to(v_base.device)
        tau = self.trust_region * (0.25 + 0.75 * t_norm).view(B, 1, 1)
        delta_clipped = delta_raw.clamp(-tau, tau)
        clipping_ratio = (delta_raw.abs() > tau).float().mean()

        alpha_t = self.gate(timestep, B).to(v_base.device, v_base.dtype)
        v_geo = v_base + alpha_t * token_confidence * delta_clipped
        return {
            "v_geo": v_geo,
            "delta_v_geo": delta_clipped,
            "token_confidence": token_confidence,
            "alpha_t": alpha_t,
            "debug": {
                "delta_raw": delta_raw,
                "clipping_ratio": clipping_ratio,
                "velocity_norm": v_geo.norm(dim=-1).mean(),
                "velocity_base_norm": v_base.norm(dim=-1).mean(),
                "velocity_delta_norm": delta_clipped.norm(dim=-1).mean(),
                "confidence_mean": token_confidence.mean(),
                "confidence_std": token_confidence.std(unbiased=False),
                "confidence_all_zero": (token_confidence <= 1e-6).all(),
                "confidence_all_one": (token_confidence >= 1.0 - 1e-6).all(),
                **local_debug,
            },
        }

    def _local_anchor_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        geo_conf: torch.Tensor,
        voxel_xyz: torch.Tensor,
        anchor_xyz: torch.Tensor,
        anchor_metadata: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        B, L, H = q.shape
        if self.rel_mlp is None:
            raise RuntimeError("Local anchor attention requires rel_mlp, but this adapter was constructed with local_attention=False.")
        M = anchor_xyz.shape[1]
        k_nn = min(self.knn_k, M)
        dist = torch.cdist(voxel_xyz.to(anchor_xyz.dtype), anchor_xyz.to(anchor_xyz.dtype)).clamp_min(1e-6)
        knn_dist, knn_idx = dist.topk(k_nn, dim=-1, largest=False)
        gather_h = knn_idx[..., None].expand(-1, -1, -1, H)
        local_k = k[:, None].expand(B, L, M, H).gather(2, gather_h)
        local_conf = geo_conf[:, None].expand(B, L, M, 1).gather(2, knn_idx[..., None])
        rel = voxel_xyz[:, :, None, :] - anchor_xyz[:, None].expand(B, L, M, 3).gather(2, knn_idx[..., None].expand(-1, -1, -1, 3))
        rel_feat = self.rel_mlp(torch.cat([rel, knn_dist[..., None]], dim=-1))
        local_k = local_k + rel_feat
        score = (q[:, :, None, :] * local_k).sum(dim=-1) / (H ** 0.5)
        score = score - knn_dist
        if anchor_metadata is not None:
            meta = anchor_metadata[:, None].expand(B, L, M, anchor_metadata.shape[-1]).gather(2, knn_idx[..., None].expand(-1, -1, -1, anchor_metadata.shape[-1]))
            source_bonus = torch.where(meta[..., 0] > 0.5, 0.15, 0.0)
            uncertainty_penalty = meta[..., 6].clamp(0, 1)
            support_bonus = torch.log1p(meta[..., 4].clamp_min(0))
            conflict_penalty = meta[..., 5].clamp_min(0)
            score = score + source_bonus + support_bonus - uncertainty_penalty - conflict_penalty
            local_conf = (local_conf * meta[..., 2:3].clamp(0, 1) * torch.exp(-conflict_penalty[..., None])).clamp(0, 1)
        score = score + local_conf.squeeze(-1).clamp_min(1e-4).log()
        weights = torch.softmax(score, dim=-1)
        attn_out = (weights[..., None] * local_k).sum(dim=2)
        token_confidence = (weights[..., None] * local_conf).sum(dim=2).clamp(0, 1)
        dense_weights = q.new_zeros(B, L, M)
        dense_weights.scatter_add_(2, knn_idx, weights)
        return attn_out, dense_weights, token_confidence, {
            "local_attention": torch.tensor(True, device=q.device),
            "knn_distance_mean": knn_dist.mean(),
            "knn_k": torch.tensor(k_nn, device=q.device),
        }
