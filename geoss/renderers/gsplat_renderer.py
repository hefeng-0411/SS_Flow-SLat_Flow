from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from geoss.utils.optional_deps import require_dependency


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
    if isinstance(gaussians, dict):
        get = gaussians.get
    else:
        get = lambda name, default=None: getattr(gaussians, name, getattr(gaussians, f"_{name}", default))
    means = _first(get, "means", "xyz")
    quats = _first(get, "quats", "rotation")
    scales = _first(get, "scales", "scaling")
    opacities = _first(get, "opacities", "opacity")
    colors = _first(get, "colors", "features_dc")
    if means is None or scales is None or opacities is None:
        raise ValueError("gaussians must expose means/xyz, scales/scaling, and opacities/opacity.")
    if quats is None:
        quats = torch.zeros(means.shape[0], 4, device=means.device, dtype=means.dtype)
        quats[:, 0] = 1.0
    if colors is None:
        colors = torch.ones(means.shape[0], 3, device=means.device, dtype=means.dtype)
    if colors.ndim == 3:
        colors = colors[:, 0]
    opacities = opacities.reshape(-1)
    return means.float(), quats.float(), scales.float(), opacities.float(), colors.float()


def _first(get, *names: str):
    for name in names:
        value = get(name, None)
        if value is not None:
            return value
    return None
