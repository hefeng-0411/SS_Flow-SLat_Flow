from __future__ import annotations

from typing import Dict

import torch


def nearest_anchor_map(
    active_xyz: torch.Tensor,
    anchor_xyz: torch.Tensor,
    anchor_tokens: torch.Tensor | None = None,
    anchor_confidence: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Map sparse SS active voxels to nearest GeoSS anchors in canonical space."""
    if active_xyz.ndim != 3 or anchor_xyz.ndim != 3:
        raise ValueError("active_xyz and anchor_xyz must be [B,L,3] and [B,M,3]")
    dist = torch.cdist(active_xyz.float(), anchor_xyz.float())
    nn_index = dist.argmin(dim=-1)
    out: Dict[str, torch.Tensor] = {"nearest_anchor_index": nn_index, "nearest_anchor_distance": dist.gather(2, nn_index[..., None])}
    if anchor_tokens is not None:
        gather = nn_index[..., None].expand(-1, -1, anchor_tokens.shape[-1])
        out["mapped_tokens"] = anchor_tokens.gather(1, gather)
    if anchor_confidence is not None:
        out["mapped_confidence"] = anchor_confidence.gather(1, nn_index[..., None]).clamp(0, 1)
    return out


def soft_anchor_map(
    active_xyz: torch.Tensor,
    anchor_xyz: torch.Tensor,
    anchor_tokens: torch.Tensor | None = None,
    anchor_confidence: torch.Tensor | None = None,
    k: int = 8,
) -> Dict[str, torch.Tensor]:
    """Distance/confidence-weighted anchor interpolation for SS -> SLAT context."""
    if active_xyz.ndim != 3 or anchor_xyz.ndim != 3:
        raise ValueError("active_xyz and anchor_xyz must be [B,L,3] and [B,M,3]")
    dist = torch.cdist(active_xyz.float(), anchor_xyz.float()).clamp_min(1e-6)
    k = min(k, anchor_xyz.shape[1])
    knn_dist, knn_idx = dist.topk(k, dim=-1, largest=False)
    inv = 1.0 / knn_dist
    if anchor_confidence is not None:
        conf = anchor_confidence.gather(1, knn_idx.reshape(anchor_xyz.shape[0], -1)[..., None].expand(-1, -1, 1)).reshape(*knn_idx.shape, 1)
        inv = inv * conf.squeeze(-1).clamp_min(1e-4)
    weights = inv / inv.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    out: Dict[str, torch.Tensor] = {"knn_anchor_index": knn_idx, "knn_anchor_distance": knn_dist, "knn_anchor_weight": weights}
    if anchor_tokens is not None:
        B, L, _ = active_xyz.shape
        C = anchor_tokens.shape[-1]
        gathered = anchor_tokens[:, None].expand(B, L, -1, C).gather(2, knn_idx[..., None].expand(-1, -1, -1, C))
        out["mapped_tokens"] = (gathered * weights[..., None]).sum(dim=2)
    if anchor_confidence is not None:
        gathered_conf = anchor_confidence[:, None].expand(anchor_xyz.shape[0], active_xyz.shape[1], -1, 1).gather(2, knn_idx[..., None])
        out["mapped_confidence"] = (gathered_conf * weights[..., None]).sum(dim=2).clamp(0, 1)
    out["local_geo_uncertainty"] = knn_dist.mean(dim=-1, keepdim=True)
    return out


def align_token_dim(tokens: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Zero-pad only; truncation is forbidden for SS -> SLAT context."""
    if tokens.shape[-1] == target_dim:
        return tokens
    if tokens.shape[-1] > target_dim:
        raise ValueError(
            f"Refusing to truncate SS->SLAT context from {tokens.shape[-1]} to {target_dim}. "
            "Use GeoVisSLATAggregator learned projection / soft_anchor_map instead."
        )
    pad = tokens.new_zeros(*tokens.shape[:-1], target_dim - tokens.shape[-1])
    return torch.cat([tokens, pad], dim=-1)
