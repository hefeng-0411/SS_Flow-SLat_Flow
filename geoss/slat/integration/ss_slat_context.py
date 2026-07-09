from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from geoss.slat.utils.active_voxel_utils import indices_to_active_xyz, pad_sparse_tensor_tokens
from geoss.slat.utils.slat_token_mapping import align_token_dim, soft_anchor_map


def build_ss_slat_context(
    *,
    ss_active_indices: Optional[torch.Tensor] = None,
    ss_active_xyz: Optional[torch.Tensor] = None,
    ss_geo_tokens: Optional[torch.Tensor] = None,
    ss_confidence: Optional[torch.Tensor] = None,
    geoss_output: Optional[Dict[str, torch.Tensor]] = None,
    slat_sparse_tensor: Any = None,
    resolution: int = 64,
    target_dim: int = 8,
) -> Dict[str, torch.Tensor | Dict[str, Any]]:
    """Normalize Sparse-Ray GeoSS/SS outputs for GeoVis-SLAT active token order."""
    debug: Dict[str, Any] = {"warnings": []}
    if slat_sparse_tensor is not None:
        _, sparse_mask, _, padded_indices = pad_sparse_tensor_tokens(slat_sparse_tensor)
        ss_active_indices = padded_indices
        debug["source"] = "slat_sparse_tensor"
        debug["sparse_mask"] = sparse_mask
    if ss_active_indices is None and ss_active_xyz is None:
        raise ValueError("ss_active_indices, ss_active_xyz, or slat_sparse_tensor is required")
    if ss_active_xyz is None:
        ss_active_xyz = indices_to_active_xyz(ss_active_indices, resolution)
    if ss_active_indices is None:
        from geoss.slat.utils.active_voxel_utils import active_xyz_to_indices

        ss_active_indices = active_xyz_to_indices(ss_active_xyz, resolution)

    B, L = ss_active_xyz.shape[:2]
    device = ss_active_xyz.device
    dtype = ss_active_xyz.dtype

    if ss_geo_tokens is None or ss_confidence is None:
        if geoss_output is not None and "anchor_xyz" in geoss_output:
            mapped = soft_anchor_map(
                ss_active_xyz,
                geoss_output["anchor_xyz"].to(device=device, dtype=dtype),
                geoss_output.get("geo_tokens", None).to(device=device, dtype=dtype) if geoss_output.get("geo_tokens", None) is not None else None,
                geoss_output.get("geo_confidence", None).to(device=device, dtype=dtype) if geoss_output.get("geo_confidence", None) is not None else None,
            )
            if ss_geo_tokens is None and "mapped_tokens" in mapped:
                ss_geo_tokens = mapped["mapped_tokens"]
            if ss_confidence is None and "mapped_confidence" in mapped:
                ss_confidence = mapped["mapped_confidence"]
            debug["nearest_anchor_distance_mean"] = mapped["knn_anchor_distance"].mean()
            debug["ss_to_slat_token_map"] = mapped["knn_anchor_index"]
            debug["local_geo_uncertainty"] = mapped["local_geo_uncertainty"]

    if ss_geo_tokens is None:
        ss_geo_tokens = torch.zeros(B, L, target_dim, device=device, dtype=dtype)
        debug["warnings"].append("ss_geo_tokens missing; using zeros")
    else:
        ss_geo_tokens = ss_geo_tokens.to(device=device, dtype=dtype)
        if ss_geo_tokens.shape[:2] != (B, L):
            ss_geo_tokens = _resize_tokens(ss_geo_tokens, L)

    if ss_confidence is None:
        ss_confidence = torch.ones(B, L, 1, device=device, dtype=dtype)
        debug["warnings"].append("ss_confidence missing; using ones")
    else:
        ss_confidence = ss_confidence.to(device=device, dtype=dtype)
        if ss_confidence.ndim == 2:
            ss_confidence = ss_confidence[..., None]
        if ss_confidence.shape[1] != L:
            ss_confidence = _resize_tokens(ss_confidence, L)

    return {
        "ss_active_indices": ss_active_indices.to(device=device).long(),
        "ss_active_xyz": ss_active_xyz,
        "ss_geo_tokens": ss_geo_tokens,
        "ss_confidence": ss_confidence.clamp(0, 1),
        "ss_to_slat_token_map": debug.get("ss_to_slat_token_map"),
        "debug": debug,
    }


def _resize_tokens(tokens: torch.Tensor, target_len: int) -> torch.Tensor:
    if tokens.shape[1] > target_len:
        return tokens[:, :target_len]
    return torch.cat([tokens, tokens[:, -1:].expand(-1, target_len - tokens.shape[1], -1)], dim=1)
