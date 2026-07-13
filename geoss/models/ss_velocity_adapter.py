from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

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
        attention_chunk_size: int = 8192,
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.geo_dim = geo_dim
        self.hidden_dim = hidden_dim
        self.trust_region = trust_region
        self.enabled = enabled
        self.local_attention = local_attention
        self.knn_k = knn_k
        self.attention_chunk_size = max(1, int(attention_chunk_size))
        self.activation_checkpointing = bool(activation_checkpointing)
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
        # Do not initialize the terminal projection exactly to zero. With DDP,
        # an exactly-zero final layer blocks first-step gradients into
        # latent_proj/geo_proj, so their reducer buckets are never marked ready.
        # This tiny init preserves a near-zero residual while keeping the full
        # adapter graph active from step 1.
        nn.init.normal_(self.delta_head[-1].weight, mean=0.0, std=1e-5)
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
        voxel_valid_mask: torch.Tensor | None = None,
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
            attn_out, attn_weights, token_confidence, local_debug = self._local_attention_with_optional_pruning(
                q, k, geo_conf, voxel_xyz, anchor_xyz, anchor_metadata, voxel_valid_mask,
            )
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
        # Backward-compatibility for old checkpoints whose final delta head was
        # saved as exact zeros: keep a zero-valued autograd edge from delta_raw
        # to h so DDP marks upstream projection parameters as used even before
        # the terminal head has learned nonzero weights.
        graph_anchor = h.sum(dim=-1, keepdim=True).expand_as(delta_raw) * 0.0
        delta_raw = delta_raw + graph_anchor

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

    def _local_attention_with_optional_pruning(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        geo_conf: torch.Tensor,
        voxel_xyz: torch.Tensor,
        anchor_xyz: torch.Tensor,
        anchor_metadata: torch.Tensor | None,
        voxel_valid_mask: torch.Tensor | None,
    ):
        if voxel_valid_mask is None or q.shape[0] != 1:
            return self._local_anchor_attention(q, k, geo_conf, voxel_xyz, anchor_xyz, anchor_metadata)
        mask = voxel_valid_mask.to(device=q.device, dtype=torch.bool)
        if mask.shape != q.shape[:2]:
            raise ValueError(f"voxel_valid_mask must be [B,L], got {tuple(mask.shape)}")
        active = torch.nonzero(mask[0], as_tuple=False).flatten()
        if active.numel() == q.shape[1]:
            return self._local_anchor_attention(q, k, geo_conf, voxel_xyz, anchor_xyz, anchor_metadata)
        # MeshFleet encodes padded/inactive SS sites as exact zeros. Skipping only
        # those sites leaves learned latent values untouched and returns v_base
        # there, avoiding needless geometry-attention activations.
        if active.numel() == 0:
            zero_attn = torch.zeros_like(q)
            zero_conf = q.new_zeros(q.shape[0], q.shape[1], 1)
            return zero_attn, q.new_empty(0), zero_conf, {"local_attention": torch.tensor(True, device=q.device), "voxel_prune_ratio": torch.tensor(1.0, device=q.device)}
        active_attn, _, active_conf, debug = self._local_anchor_attention(
            q[:, active], k, geo_conf, voxel_xyz[:, active], anchor_xyz, anchor_metadata,
        )
        attn_out = torch.zeros_like(q).index_copy(1, active, active_attn)
        token_confidence = q.new_zeros(q.shape[0], q.shape[1], 1).index_copy(1, active, active_conf)
        debug["voxel_prune_ratio"] = q.new_tensor(1.0 - active.numel() / q.shape[1])
        return attn_out, q.new_empty(0), token_confidence, debug

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
        attn_chunks, confidence_chunks, distance_chunks = [], [], []
        # Never materialize [B,L,M] distances or [B,L,M,H] expanded keys. At
        # 64^3 voxels and 4096 anchors those temporary tensors dominate VRAM.
        for start in range(0, L, self.attention_chunk_size):
            end = min(L, start + self.attention_chunk_size)
            voxel_chunk = voxel_xyz[:, start:end]
            with torch.no_grad():
                distances = torch.cdist(voxel_chunk.float(), anchor_xyz.float()).clamp_min(1e-6)
                knn_dist, knn_idx = distances.topk(k_nn, dim=-1, largest=False)
            if self.training and self.activation_checkpointing:
                # Keep optional metadata out of checkpoint's tensor argument
                # list: older PyTorch releases only guarantee tensor inputs.
                def run_chunk(q_chunk, keys, confidence, voxels, anchors, indices, distances):
                    return self._local_attention_chunk(
                        q_chunk, keys, confidence, voxels, anchors, indices, distances, anchor_metadata,
                    )

                attn_chunk, confidence_chunk = checkpoint(
                    run_chunk, q[:, start:end], k, geo_conf, voxel_chunk, anchor_xyz, knn_idx, knn_dist,
                    use_reentrant=False,
                )
            else:
                attn_chunk, confidence_chunk = self._local_attention_chunk(
                    q[:, start:end], k, geo_conf, voxel_chunk, anchor_xyz, knn_idx, knn_dist, anchor_metadata,
                )
            attn_chunks.append(attn_chunk)
            confidence_chunks.append(confidence_chunk)
            distance_chunks.append(knn_dist)
        attn_out = torch.cat(attn_chunks, dim=1)
        token_confidence = torch.cat(confidence_chunks, dim=1)
        # Attention weights were not consumed by the SS flow. Returning an empty
        # tensor avoids retaining a 64^3 x 4096 diagnostic allocation.
        return attn_out, q.new_empty(0), token_confidence, {
            "local_attention": torch.tensor(True, device=q.device),
            "knn_distance_mean": torch.cat(distance_chunks, dim=1).mean(),
            "knn_k": torch.tensor(k_nn, device=q.device),
            "attention_chunk_size": torch.tensor(self.attention_chunk_size, device=q.device),
        }

    def _local_attention_chunk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        geo_conf: torch.Tensor,
        voxel_xyz: torch.Tensor,
        anchor_xyz: torch.Tensor,
        knn_idx: torch.Tensor,
        knn_dist: torch.Tensor,
        anchor_metadata: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact local-attention math for one bounded voxel chunk."""
        B, _, H = q.shape
        batch_index = torch.arange(B, device=q.device).view(B, 1, 1)
        local_k = k[batch_index, knn_idx]
        local_conf = geo_conf[batch_index, knn_idx]
        local_anchor = anchor_xyz[batch_index, knn_idx]
        rel = voxel_xyz[:, :, None, :] - local_anchor
        local_k = local_k + self.rel_mlp(torch.cat([rel, knn_dist[..., None]], dim=-1))
        score = (q[:, :, None, :] * local_k).sum(dim=-1) / (H ** 0.5) - knn_dist
        if anchor_metadata is not None:
            meta = anchor_metadata[batch_index, knn_idx]
            source_bonus = torch.where(meta[..., 0] > 0.5, 0.15, 0.0)
            uncertainty_penalty = meta[..., 6].clamp(0, 1)
            support_bonus = torch.log1p(meta[..., 4].clamp_min(0))
            conflict_penalty = meta[..., 5].clamp_min(0)
            score = score + source_bonus + support_bonus - uncertainty_penalty - conflict_penalty
            local_conf = (local_conf * meta[..., 2:3].clamp(0, 1) * torch.exp(-conflict_penalty[..., None])).clamp(0, 1)
        weights = torch.softmax(score + local_conf.squeeze(-1).clamp_min(1e-4).log(), dim=-1)
        return (weights[..., None] * local_k).sum(dim=2), (weights[..., None] * local_conf).sum(dim=2).clamp(0, 1)
