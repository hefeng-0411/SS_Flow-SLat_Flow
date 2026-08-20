from __future__ import annotations

from typing import Any, Dict

import torch


def gaussian_statistics(gaussians: Any) -> Dict[str, float | int | list[float]]:
    # TRELLIS native objects expose activated world-space properties, whereas
    # their PLY stores opacity logits and log-scales. Select the representation
    # explicitly so pre-export and post-load statistics are comparable.
    native = not isinstance(gaussians, dict)
    xyz = _first_attr(gaussians, "get_xyz", "means", "xyz", "_xyz")
    opacity = _first_attr(gaussians, "get_opacity", "opacities", "opacity", "_opacity")
    scaling = _first_attr(gaussians, "get_scaling", "scales", "scaling", "_scaling")
    stats: Dict[str, float | int | list[float]] = {}
    if xyz is not None:
        xyz = xyz.float()
        # Stable snake_case names prevent CSV schemas from containing duplicate
        # null aliases such as both "Gaussian count" and "Gaussian_count".
        stats["Gaussian_count"] = int(xyz.shape[0])
        stats["floating_splat_ratio"] = float((xyz.norm(dim=-1) > 2.0).float().mean().detach().cpu())
        stats["view_coverage"] = float((xyz.abs().amax(dim=-1) <= 1.5).float().mean().detach().cpu())
    if opacity is not None:
        op = opacity.float().reshape(-1)
        # Native get_opacity is already sigmoid-activated. PLY `opacity` and
        # private `_opacity` values are logits and must be decoded exactly once.
        if not (native and hasattr(gaussians, "get_opacity")):
            op = torch.sigmoid(op) if ("opacity" in gaussians if isinstance(gaussians, dict) else True) else op
        stats["opacity_mean"] = float(op.mean().detach().cpu())
        stats["opacity_std"] = float(op.std(unbiased=False).detach().cpu())
        hist = torch.histc(op.detach().cpu(), bins=10, min=0.0, max=1.0)
        stats["opacity_histogram"] = [float(v) for v in hist]
    if scaling is not None:
        sc = scaling.float()
        # Native get_scaling is physical; the TRELLIS PLY `scaling` field is
        # log-space. This removes heuristic double transforms from asset stats.
        if not (native and hasattr(gaussians, "get_scaling")):
            sc = sc.exp() if ("scaling" in gaussians if isinstance(gaussians, dict) else True) else sc
        stats["scale_mean"] = float(sc.mean().detach().cpu())
        stats["scale_max"] = float(sc.max().detach().cpu())
        stats["scale_abnormal_ratio"] = float((sc.amax(dim=-1) > 0.25).float().mean().detach().cpu())
    return stats


def _first_attr(obj: Any, *names: str):
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
        return None
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None
