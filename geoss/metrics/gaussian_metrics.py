from __future__ import annotations

from typing import Any, Dict

import torch


def gaussian_statistics(gaussians: Any) -> Dict[str, float | int | list[float]]:
    xyz = _first_attr(gaussians, "means", "xyz", "_xyz")
    opacity = _first_attr(gaussians, "opacities", "opacity", "_opacity")
    scaling = _first_attr(gaussians, "scales", "scaling", "_scaling")
    stats: Dict[str, float | int | list[float]] = {}
    if xyz is not None:
        xyz = xyz.float()
        stats["Gaussian count"] = int(xyz.shape[0])
        stats["floating splat ratio"] = float((xyz.norm(dim=-1) > 2.0).float().mean().detach().cpu())
        stats["view coverage"] = float((xyz.abs().amax(dim=-1) <= 1.5).float().mean().detach().cpu())
    if opacity is not None:
        op = torch.sigmoid(opacity.float()).reshape(-1)
        stats["opacity mean"] = float(op.mean().detach().cpu())
        stats["opacity std"] = float(op.std(unbiased=False).detach().cpu())
        hist = torch.histc(op.detach().cpu(), bins=10, min=0.0, max=1.0)
        stats["opacity histogram"] = [float(v) for v in hist]
    if scaling is not None:
        sc = scaling.float().exp() if scaling.float().median() < 0 else scaling.float()
        stats["scale mean"] = float(sc.mean().detach().cpu())
        stats["scale max"] = float(sc.max().detach().cpu())
        stats["scale abnormal ratio"] = float((sc.amax(dim=-1) > 0.25).float().mean().detach().cpu())
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
