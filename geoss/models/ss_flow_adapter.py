"""3-D-aligned residual velocity adapter for frozen TRELLIS SS Flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class SSFlowAdapterOutput:
    v_final: torch.Tensor
    delta_v_geo: torch.Tensor
    gate: torch.Tensor
    alpha_t: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


class _VoxelCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.kv_norm = nn.LayerNorm(cond_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(cond_dim, hidden_dim)
        self.v_proj = nn.Linear(cond_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(nn.Linear(hidden_dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, hidden_dim))

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor, condition_valid: torch.Tensor) -> torch.Tensor:
        B, Lq, H = hidden.shape
        Lk = condition.shape[1]
        q = self.q_proj(self.q_norm(hidden)).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        normalized_condition = self.kv_norm(condition)
        k = self.k_proj(normalized_condition).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normalized_condition).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        # SDPA dispatches to Flash/memory-efficient CUDA kernels when supported.
        allowed = condition_valid[:, None, None, :]
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
        context = context.transpose(1, 2).reshape(B, Lq, H)
        hidden = hidden + self.out_proj(context)
        return hidden + self.ff(self.ff_norm(hidden))


class SSFlowAdapter(nn.Module):
    """Predict a confidence-gated residual; never replace TRELLIS velocity."""

    def __init__(
        self,
        *,
        latent_dim: int = 8,
        condition_dim: int = 283,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_blocks: int = 2,
        mlp_ratio: float = 4.0,
        g_observed: float = 1.0,
        g_unobserved: float = 0.0,
        alpha_schedule: str = "cosine",
        trust_region: float = 0.25,
        gradient_checkpointing: bool = True,
        use_observation_gate: bool = True,
    ) -> None:
        super().__init__()
        if alpha_schedule not in {"cosine", "linear", "constant"}:
            raise ValueError(f"Unsupported alpha_schedule={alpha_schedule!r}")
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.g_observed = float(g_observed)
        self.g_unobserved = float(g_unobserved)
        self.alpha_schedule = alpha_schedule
        self.trust_region = float(trust_region)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.use_observation_gate = bool(use_observation_gate)
        self.input_norm = nn.LayerNorm(latent_dim)
        self.input_projection = nn.Linear(latent_dim, hidden_dim)
        self.time_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList(
            [_VoxelCrossAttentionBlock(hidden_dim, condition_dim, num_heads, mlp_ratio) for _ in range(num_blocks)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.residual_head = nn.Linear(hidden_dim, latent_dim)
        # Mandatory base-preservation invariant: the adapter is exactly zero at initialization.
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self,
        x_t_tokens: torch.Tensor,
        voxel_condition: torch.Tensor,
        timestep: torch.Tensor,
        observation_mask: torch.Tensor,
        voxel_confidence: torch.Tensor,
        *,
        v_base: torch.Tensor,
        enabled: bool = True,
    ) -> SSFlowAdapterOutput:
        if x_t_tokens.ndim != 3 or x_t_tokens.shape[-1] != self.latent_dim:
            raise ValueError(f"x_t_tokens must be [B,L,{self.latent_dim}], got {tuple(x_t_tokens.shape)}")
        if voxel_condition.ndim != 3 or voxel_condition.shape[-1] != self.condition_dim:
            raise ValueError(
                f"voxel_condition must be [B,M,{self.condition_dim}], got {tuple(voxel_condition.shape)}"
            )
        B, L, _ = x_t_tokens.shape
        if v_base.shape != x_t_tokens.shape:
            raise ValueError(f"v_base must match x_t_tokens, got {tuple(v_base.shape)}")
        observation_mask = _as_gate_tensor(observation_mask, B, L, "observation_mask").bool()
        voxel_confidence = _as_gate_tensor(voxel_confidence, B, L, "voxel_confidence").float().clamp(0, 1)
        if not enabled:
            zero = torch.zeros_like(v_base)
            zero_gate = torch.zeros(B, L, 1, device=v_base.device, dtype=v_base.dtype)
            return SSFlowAdapterOutput(v_base, zero, zero_gate, zero_gate[:, :1], _diagnostics(v_base, zero, zero_gate, observation_mask, voxel_confidence))
        condition_valid = observation_mask[..., 0]
        if bool((condition_valid.sum(dim=1) == 0).any()):
            if self.g_unobserved == 0.0:
                zero = torch.zeros_like(v_base)
                zero_gate = torch.zeros(B, L, 1, device=v_base.device, dtype=v_base.dtype)
                return SSFlowAdapterOutput(
                    v_base,
                    zero,
                    zero_gate,
                    zero_gate[:, :1],
                    _diagnostics(v_base, zero, zero_gate, observation_mask, voxel_confidence),
                )
            raise RuntimeError("Nonzero unobserved leakage requires at least one valid voxel condition")

        t_norm = normalize_trellis_timestep(timestep, B).to(device=x_t_tokens.device)
        time_embedding = timestep_embedding(t_norm, self.hidden_dim).to(dtype=x_t_tokens.dtype)
        hidden = self.input_projection(self.input_norm(x_t_tokens)) + self.time_mlp(time_embedding)[:, None]
        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                hidden = checkpoint(block, hidden, voxel_condition, condition_valid, use_reentrant=False)
            else:
                hidden = block(hidden, voxel_condition, condition_valid)
        delta_raw = self.residual_head(self.output_norm(hidden))
        tau = self.trust_region * (0.25 + 0.75 * t_norm).view(B, 1, 1)
        raw_norm = torch.linalg.vector_norm(delta_raw.float(), dim=-1, keepdim=True)
        clip_scale = torch.minimum(torch.ones_like(raw_norm), tau / raw_norm.clamp_min(1e-8))
        delta = (delta_raw.float() * clip_scale).to(delta_raw.dtype)
        alpha = self._alpha(t_norm).view(B, 1, 1).to(dtype=delta.dtype)
        observed = observation_mask.to(dtype=delta.dtype)
        confidence = voxel_confidence.to(dtype=delta.dtype)
        if self.use_observation_gate:
            spatial_strength = self.g_unobserved + (self.g_observed - self.g_unobserved) * observed
            q_effective = observed * confidence + (1.0 - observed)
            gate = alpha * spatial_strength * q_effective
        else:
            gate = alpha.expand(B, L, 1)
        v_final = v_base + gate * delta
        diagnostics = _diagnostics(v_base, delta, gate, observation_mask, voxel_confidence)
        diagnostics.update(
            {
                "raw_adapter_norm": raw_norm.mean(),
                "trust_region_clip_fraction": (clip_scale < 1.0).float().mean(),
                "observed_gate_mean": _masked_mean(gate, observation_mask),
                "unobserved_gate_mean": _masked_mean(gate, ~observation_mask),
            }
        )
        return SSFlowAdapterOutput(v_final, delta, gate, alpha, diagnostics)

    def _alpha(self, t_norm: torch.Tensor) -> torch.Tensor:
        if self.alpha_schedule == "cosine":
            return torch.cos(0.5 * torch.pi * t_norm).clamp(0, 1)
        if self.alpha_schedule == "linear":
            return 1.0 - t_norm
        return torch.ones_like(t_norm)


def normalize_trellis_timestep(timestep: torch.Tensor, batch_size: int) -> torch.Tensor:
    if not isinstance(timestep, torch.Tensor):
        timestep = torch.as_tensor(timestep, dtype=torch.float32)
    timestep = timestep.float().reshape(-1)
    if timestep.numel() == 1:
        timestep = timestep.expand(batch_size)
    if timestep.numel() != batch_size:
        raise ValueError(f"timestep must be scalar or [B], got {tuple(timestep.shape)} for B={batch_size}")
    # TRELLIS model calls use [0,1000]; training-pair construction uses [0,1].
    normalized = torch.where(timestep > 1.0 + 1e-6, timestep / 1000.0, timestep)
    if bool(((normalized < 0) | (normalized > 1)).any()):
        raise ValueError(f"TRELLIS timestep lies outside [0,1]/[0,1000]: {timestep}")
    return normalized


def timestep_embedding(timestep: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -torch.log(torch.tensor(max_period, device=timestep.device))
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(half, 1)
    )
    angles = timestep[:, None].float() * frequencies[None] * 1000.0
    embedding = torch.cat([angles.cos(), angles.sin()], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def _as_gate_tensor(value: torch.Tensor, B: int, L: int, name: str) -> torch.Tensor:
    if value.ndim == 2:
        value = value[..., None]
    if value.shape != (B, L, 1):
        raise ValueError(f"{name} must be [B,L] or [B,L,1], got {tuple(value.shape)}")
    return value


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.to(dtype=value.dtype)
    return (value * mask_float).sum() / mask_float.sum().clamp_min(1)


def _diagnostics(
    base: torch.Tensor,
    delta: torch.Tensor,
    gate: torch.Tensor,
    observed: torch.Tensor,
    confidence: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    base_norm = torch.linalg.vector_norm(base.float(), dim=-1).mean()
    effective = gate.float() * delta.float()
    adapter_norm = torch.linalg.vector_norm(effective, dim=-1).mean()
    return {
        "mean_adapter_norm": adapter_norm,
        "base_velocity_norm": base_norm,
        "residual_base_ratio": adapter_norm / base_norm.clamp_min(1e-8),
        "confidence_mean": confidence.float().mean(),
        "percentage_observed": observed.float().mean() * 100.0,
        "percentage_zero_gated": (gate.abs() <= 1e-8).float().mean() * 100.0,
    }


__all__ = ["SSFlowAdapter", "SSFlowAdapterOutput", "normalize_trellis_timestep"]
