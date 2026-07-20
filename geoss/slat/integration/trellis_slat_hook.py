from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from geoss.slat.models.slat_velocity_adapter import SLATVelocityAdapter
from geoss.slat.utils.active_voxel_utils import pad_sparse_tensor_tokens, unpad_sparse_tensor_tokens


class GeoVisTrellisSLATWrapper(nn.Module):
    """Adapter wrapper that preserves the public TRELLIS SLatFlowModel contract.

    TRELLIS samplers inspect model metadata such as ``in_channels`` before
    calling ``forward``.  A plain :class:`nn.Module` wrapper hides those fields,
    which otherwise makes a healthy adapter checkpoint fail only at decode time.
    """

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

    def __getattr__(self, name: str):
        """Delegate unknown TRELLIS model metadata to the wrapped flow model.

        ``nn.Module`` keeps child modules in ``_modules`` rather than the
        instance dictionary, so the lookup deliberately uses that registry to
        avoid recursive attribute access.  This preserves sampler-facing
        attributes (notably ``in_channels``) without duplicating TRELLIS
        internals or weakening normal module/state-dict behavior.
        """
        try:
            return super().__getattr__(name)
        except AttributeError as original_error:
            wrapped = self.__dict__.get("_modules", {}).get("slat_flow_model")
            if wrapped is not None:
                try:
                    return getattr(wrapped, name)
                except AttributeError:
                    pass
            raise original_error

    def forward(
        self,
        x,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        *,
        geovis_slat_context: Optional[Dict[str, torch.Tensor]] = None,
        use_geovis_slat: Optional[bool] = None,
        geovis_branch: str = "cond",
        geovis_residual_scale: float = 1.0,
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
            return self._forward_sparse(
                x, v_base, t, geovis_slat_context, geovis_branch, float(geovis_residual_scale)
            )
        return self._forward_dense(
            x, v_base, t, geovis_slat_context, geovis_branch, float(geovis_residual_scale)
        )

    def _forward_sparse(self, x, v_base, t, context: Dict[str, torch.Tensor], branch: str, residual_scale: float):
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
            token_valid_mask=valid_mask.unsqueeze(-1).to(slat_latent_tokens.dtype),
            correction_demand=context.get("correction_demand"),
            residual_variance=context.get("residual_variance"),
        )
        v_geo_tokens = v_base_tokens + residual_scale * (out["v_slat_geo"] - v_base_tokens)
        v_geo_feats = unpad_sparse_tensor_tokens(v_geo_tokens, layout)
        v_geo = v_base.replace(v_geo_feats)
        self.last_debug = _make_debug(out, branch, valid_mask, residual_scale=residual_scale)
        return v_geo

    def _forward_dense(self, x, v_base, t, context: Dict[str, torch.Tensor], branch: str, residual_scale: float):
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
            token_valid_mask=context.get("slat_token_valid_mask"),
            correction_demand=context.get("correction_demand"),
            residual_variance=context.get("residual_variance"),
        )
        v_geo = v_base + residual_scale * (out["v_slat_geo"] - v_base)
        self.last_debug = _make_debug(out, branch, None, residual_scale=residual_scale)
        return v_geo

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
    context_indices = context.get("ss_active_indices")
    if context_indices is not None and active_indices is not None:
        expected = context_indices.to(device=active_indices.device, dtype=active_indices.dtype)
        if expected.shape != active_indices.shape or not torch.equal(expected, active_indices):
            raise RuntimeError(
                "GeoVis context coordinates do not match the live TRELLIS sparse state; "
                "refusing to apply token controls to different voxels."
            )
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
    if "correction_demand" in context:
        out["correction_demand"] = _fit_tokens(context["correction_demand"].to(device=device, dtype=dtype), B, L, 1).clamp(0, 1)
    if "residual_variance" in context:
        out["residual_variance"] = _fit_tokens(context["residual_variance"].to(device=device, dtype=dtype), B, L, 1).clamp_min(0)
    if active_indices is not None:
        out["ss_active_indices"] = active_indices
    return out


def _fit_tokens(tokens: torch.Tensor, B: int, L: int, C: int) -> torch.Tensor:
    if tokens.ndim == 2:
        tokens = tokens[None].expand(B, -1, -1)
    if tokens.shape[0] == 1 and B > 1:
        tokens = tokens.expand(B, -1, -1)
    if tokens.ndim != 3 or tokens.shape[0] != B or tokens.shape[1] != L:
        raise ValueError(
            f"SLAT context must align exactly with live tokens [B,L,*]=[{B},{L},*], got {tuple(tokens.shape)}."
        )
    if tokens.shape[-1] != C:
        raise ValueError(f"SLAT context channel width must be {C}, got {tokens.shape[-1]}.")
    return tokens


def _make_debug(
    out: Dict[str, torch.Tensor],
    branch: str,
    valid_mask: Optional[torch.Tensor],
    *,
    residual_scale: float = 1.0,
) -> Dict[str, Any]:
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
        "cfg_residual_scale": float(residual_scale),
        "effective_delta_norm": (
            (out["v_slat_geo"] - out["v_slat_base"]) * residual_scale
        ).norm(dim=-1).mean().detach(),
    }
    if valid_mask is not None:
        debug["valid_token_ratio"] = valid_mask.float().mean().detach()
    return debug


def _device_from_velocity(v_base) -> torch.device:
    return v_base.feats.device if hasattr(v_base, "feats") else v_base.device
