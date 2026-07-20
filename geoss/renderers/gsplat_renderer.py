from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from geoss.utils.optional_deps import require_dependency


SH_C0 = 0.28209479177387814


def render_gaussians(
    gaussians: Any,
    cameras: Dict[str, torch.Tensor],
    image_size: Tuple[int, int],
    backgrounds: Optional[torch.Tensor] = None,
    return_alpha: bool = True,
    return_depth: bool = True,
    return_visibility: bool = True,
) -> Dict[str, torch.Tensor]:
    """Render 3D Gaussians through gsplat with a stable project-level schema."""
    require_dependency("gsplat", real_mode=True, feature="3DGS render/eval")
    from gsplat.rendering import rasterization

    means, quats, scales, opacities, colors = _gaussian_tensors(gaussians)
    viewmats = cameras.get("w2c")
    Ks = cameras.get("K")
    if viewmats is None or Ks is None:
        raise KeyError("cameras must contain OpenCV K and w2c tensors.")
    H, W = int(image_size[0]), int(image_size[1])
    render_mode = "RGB+ED" if return_depth else "RGB"
    outputs = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        backgrounds=backgrounds,
        render_mode=render_mode,
    )
    if len(outputs) == 3:
        renders, alphas, meta = outputs
    else:
        renders, alphas = outputs[:2]
        meta = {}
    rgb = renders[..., :3]
    out: Dict[str, torch.Tensor] = {"rendered_rgb": rgb}
    if return_depth:
        out["rendered_depth"] = renders[..., 3:4] if renders.shape[-1] > 3 else torch.zeros_like(alphas)
    if return_alpha:
        out["rendered_alpha"] = alphas
    if return_visibility:
        radii = meta.get("radii") if isinstance(meta, dict) else None
        if radii is not None:
            out["per_gaussian_visibility"] = (radii > 0).float()
            out["visibility_map"] = alphas
        else:
            out["per_gaussian_visibility"] = torch.ones(means.shape[0], device=means.device, dtype=means.dtype)
            out["visibility_map"] = alphas
    out["meta"] = meta
    return out


def _gaussian_tensors(gaussians: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # TRELLIS' in-memory Gaussian object exposes activated physical attributes
    # through get_* properties. Reading the private tensors would omit scale and
    # opacity biases and would render a different asset from TRELLIS itself.
    if not isinstance(gaussians, dict) and all(
        hasattr(gaussians, name) for name in ("get_xyz", "get_scaling", "get_rotation", "get_opacity", "get_features")
    ):
        means = gaussians.get_xyz
        quats = gaussians.get_rotation
        scales = gaussians.get_scaling
        opacities = gaussians.get_opacity.reshape(-1)
        colors = _dc_sh_to_rgb(gaussians.get_features)
        return means.float(), quats.float(), scales.float(), opacities.float(), colors.float()
    scale_parameterization = None
    opacity_parameterization = None
    if isinstance(gaussians, dict):
        get = gaussians.get
        scale_parameterization = gaussians.get("scaling_parameterization")
        opacity_parameterization = gaussians.get("opacity_parameterization")
    else:
        get = lambda name, default=None: getattr(gaussians, name, getattr(gaussians, f"_{name}", default))
    means = _first(get, "means", "xyz")
    quats = _first(get, "quats", "rotation")
    scales = _first(get, "scales", "scaling")
    opacities = _first(get, "opacities", "opacity")
    colors = get("colors", None)
    colors_are_sh = colors is None
    if colors_are_sh:
        colors = get("features_dc", None)
    if means is None or scales is None or opacities is None:
        raise ValueError("gaussians must expose means/xyz, scales/scaling, and opacities/opacity.")
    if quats is None:
        quats = torch.zeros(means.shape[0], 4, device=means.device, dtype=means.dtype)
        quats[:, 0] = 1.0
    if colors is None:
        colors = torch.ones(means.shape[0], 3, device=means.device, dtype=means.dtype)
    if colors_are_sh:
        colors = _dc_sh_to_rgb(colors)
    elif colors.ndim == 3:
        colors = colors[:, 0]
    if colors.ndim != 2 or colors.shape[-1] != 3:
        raise ValueError(f"Gaussian colors must be [G,3] or [G,1,3], got {tuple(colors.shape)}")
    opacities = opacities.reshape(-1)
    # PLY exports retain the optimizer parameterization (log-scales/logits).
    # Decode it here so evaluation renders the same physical Gaussians as TRELLIS.
    if scale_parameterization == "log" or (scale_parameterization is None and scales.median() < 0):
        scales = scales.exp()
    if opacity_parameterization == "logit" or (
        opacity_parameterization is None and (opacities.min() < 0 or opacities.max() > 1)
    ):
        opacities = opacities.sigmoid()
    quats = F.normalize(quats.float(), dim=-1, eps=1e-8)
    return means.float(), quats, scales.float(), opacities.float(), colors.float()


def _dc_sh_to_rgb(colors: torch.Tensor) -> torch.Tensor:
    """Convert TRELLIS/3DGS degree-zero SH coefficients to linear RGB values."""
    if colors.ndim == 3:
        colors = colors[:, 0]
    if colors.ndim != 2 or colors.shape[-1] != 3:
        raise ValueError(f"Gaussian colors must be [G,3] or [G,1,3], got {tuple(colors.shape)}")
    return (0.5 + SH_C0 * colors).clamp(0.0, 1.0)


def _first(get, *names: str):
    for name in names:
        value = get(name, None)
        if value is not None:
            return value
    return None
