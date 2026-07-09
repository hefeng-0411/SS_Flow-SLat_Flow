from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from geoss.slat.models.slat_velocity_adapter import SLATVelocityAdapter
from geoss.slat.utils.active_voxel_utils import pad_sparse_tensor_tokens, unpad_sparse_tensor_tokens


class GeoVisTrellisSLATWrapper(nn.Module):
    """Minimal wrapper around TRELLIS SLatFlowModel velocity output."""

    def __init__(
        self,
        slat_flow_model: nn.Module,
        velocity_adapter: SLATVelocityAdapter,
        *,
        use_geovis_slat: bool = True,
        geovis_slat_apply_to_uncond: bool = False,
    ) -> None:
        super().__init__()
        self.slat_flow_model = slat_flow_model
        self.velocity_adapter = velocity_adapter
        self.use_geovis_slat = use_geovis_slat
        self.geovis_slat_apply_to_uncond = geovis_slat_apply_to_uncond
        self.last_debug: Dict[str, Any] = {}
        for p in self.slat_flow_model.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        x,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        *,
        geovis_slat_context: Optional[Dict[str, torch.Tensor]] = None,
        use_geovis_slat: Optional[bool] = None,
        geovis_branch: str = "cond",
        **kwargs,
    ):
        v_base = self.slat_flow_model(x, t, cond, **kwargs)
        enabled = self.use_geovis_slat if use_geovis_slat is None else bool(use_geovis_slat)
        if not enabled or geovis_slat_context is None:
            self.last_debug = {"enabled": False, "identity_error": torch.zeros((), device=_device_from_velocity(v_base))}
            return v_base
        if geovis_branch == "uncond" and not self.geovis_slat_apply_to_uncond:
            self.last_debug = {"enabled": False, "branch": geovis_branch, "reason": "uncond_disabled"}
            return v_base
        if hasattr(v_base, "feats") and hasattr(v_base, "coords"):
            return self._forward_sparse(x, v_base, t, geovis_slat_context, geovis_branch)
        return self._forward_dense(x, v_base, t, geovis_slat_context, geovis_branch)

    def _forward_sparse(self, x, v_base, t, context: Dict[str, torch.Tensor], branch: str):
        slat_latent_tokens, valid_mask, layout, active_indices = pad_sparse_tensor_tokens(x)
        v_base_tokens, _, _, _ = pad_sparse_tensor_tokens(v_base)
        context = _context_to_padded(context, slat_latent_tokens, active_indices)
        out = self.velocity_adapter(
            slat_latent_tokens,
            context["slat_cond_tokens"],
            context["slat_confidence"],
            context["ss_confidence"],
            t,
            v_base_tokens,
            use_geovis_slat=True,
        )
        v_geo_feats = unpad_sparse_tensor_tokens(out["v_slat_geo"], layout)
        v_geo = v_base.replace(v_geo_feats)
        self.last_debug = _make_debug(out, branch, valid_mask)
        return v_geo

    def _forward_dense(self, x, v_base, t, context: Dict[str, torch.Tensor], branch: str):
        if x.ndim != 3:
            raise ValueError(f"dense SLAT hook expects [B,L,C], got {tuple(x.shape)}")
        context = _context_to_padded(context, x, None)
        out = self.velocity_adapter(
            x,
            context["slat_cond_tokens"],
            context["slat_confidence"],
            context["ss_confidence"],
            t,
            v_base,
            use_geovis_slat=True,
        )
        self.last_debug = _make_debug(out, branch, None)
        return out["v_slat_geo"]

    @torch.no_grad()
    def identity_error(self, x, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        base = self.slat_flow_model(x, t, cond)
        wrapped = self.forward(x, t, cond, use_geovis_slat=False)
        if hasattr(base, "feats"):
            return (base.feats - wrapped.feats).abs().max()
        return (base - wrapped).abs().max()


def make_cfg_geovis_context(
    context: Dict[str, torch.Tensor],
    *,
    branch: str,
    apply_to_uncond: bool = False,
) -> Optional[Dict[str, torch.Tensor]]:
    """Return context for CFG branch; default injects only conditional branch."""
    if branch == "uncond" and not apply_to_uncond:
        return None
    return context


def _context_to_padded(context: Dict[str, torch.Tensor], tokens: torch.Tensor, active_indices: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
    B, L, C = tokens.shape
    device, dtype = tokens.device, tokens.dtype
    out: Dict[str, torch.Tensor] = {}
    if "slat_cond_tokens" in context:
        out["slat_cond_tokens"] = _fit_tokens(context["slat_cond_tokens"].to(device=device, dtype=dtype), B, L, C)
    elif "ss_geo_tokens" in context:
        out["slat_cond_tokens"] = _fit_tokens(context["ss_geo_tokens"].to(device=device, dtype=dtype), B, L, C)
    else:
        out["slat_cond_tokens"] = torch.zeros(B, L, C, device=device, dtype=dtype)
    if "slat_confidence" in context:
        out["slat_confidence"] = _fit_tokens(context["slat_confidence"].to(device=device, dtype=dtype), B, L, 1).clamp(0, 1)
    else:
        out["slat_confidence"] = torch.ones(B, L, 1, device=device, dtype=dtype)
    if "ss_confidence" in context:
        out["ss_confidence"] = _fit_tokens(context["ss_confidence"].to(device=device, dtype=dtype), B, L, 1).clamp(0, 1)
    else:
        out["ss_confidence"] = torch.ones(B, L, 1, device=device, dtype=dtype)
    if active_indices is not None:
        out["ss_active_indices"] = active_indices
    return out


def _fit_tokens(tokens: torch.Tensor, B: int, L: int, C: int) -> torch.Tensor:
    if tokens.ndim == 2:
        tokens = tokens[None].expand(B, -1, -1)
    if tokens.shape[0] == 1 and B > 1:
        tokens = tokens.expand(B, -1, -1)
    if tokens.shape[1] > L:
        tokens = tokens[:, :L]
    elif tokens.shape[1] < L:
        tokens = torch.cat([tokens, tokens[:, -1:].expand(-1, L - tokens.shape[1], -1)], dim=1)
    if tokens.shape[-1] > C:
        raise ValueError(
            f"Refusing to truncate SLAT context from {tokens.shape[-1]} to {C}. "
            "Pass learned slat_cond_tokens with the correct dimension."
        )
    elif tokens.shape[-1] < C:
        tokens = torch.cat([tokens, tokens.new_zeros(*tokens.shape[:-1], C - tokens.shape[-1])], dim=-1)
    return tokens


def _make_debug(out: Dict[str, torch.Tensor], branch: str, valid_mask: Optional[torch.Tensor]) -> Dict[str, Any]:
    debug = {
        "enabled": True,
        "branch": branch,
        "delta_v_slat": out["delta_v_slat"].detach(),
        "v_slat_geo": out["v_slat_geo"].detach(),
        "beta_t": out["beta_t"].detach(),
        "joint_confidence": out["joint_confidence"].detach(),
        "clipping_ratio": out["clipping_ratio"].detach(),
        "delta_norm": out["debug"]["delta_norm"].detach(),
        "confidence_mean": out["debug"]["confidence_mean"].detach(),
    }
    if valid_mask is not None:
        debug["valid_token_ratio"] = valid_mask.float().mean().detach()
    return debug


def _device_from_velocity(v_base) -> torch.device:
    return v_base.feats.device if hasattr(v_base, "feats") else v_base.device
