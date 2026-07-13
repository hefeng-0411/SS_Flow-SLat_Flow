from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from geoss.utils.optional_deps import require_dependency


def read_asset(path: str | Path, *, real_mode: bool = False) -> Any:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".ply":
        require_dependency("plyfile", real_mode=real_mode, feature="PLY asset IO")
        from plyfile import PlyData

        return PlyData.read(str(path))
    if suffix in {".obj", ".glb", ".gltf"}:
        require_dependency("trimesh", real_mode=real_mode, feature="mesh/GLB asset IO")
        import trimesh

        return trimesh.load(str(path), force="scene" if suffix in {".glb", ".gltf"} else None)
    raise ValueError(f"Unsupported asset format: {path}")


def write_asset(asset: Any, path: str | Path, *, real_mode: bool = False) -> None:
    path = Path(path)
    if hasattr(asset, "export"):
        require_dependency("trimesh", real_mode=real_mode, feature="asset export")
        asset.export(str(path))
        return
    if hasattr(asset, "save_ply") and path.suffix.lower() == ".ply":
        asset.save_ply(str(path))
        return
    raise TypeError(f"Cannot export asset type {type(asset)!r} to {path}")


def read_gaussian_ply(path: str | Path, *, real_mode: bool = False) -> Dict[str, torch.Tensor]:
    ply = read_asset(path, real_mode=real_mode)
    vertex = ply["vertex"].data
    out = {
        "xyz": torch.tensor([[row["x"], row["y"], row["z"]] for row in vertex], dtype=torch.float32),
    }
    names = set(vertex.dtype.names or [])
    if {"opacity"} <= names:
        out["opacity"] = torch.tensor([row["opacity"] for row in vertex], dtype=torch.float32).view(-1, 1)
    scale_names = ["scale_0", "scale_1", "scale_2"]
    if all(name in names for name in scale_names):
        out["scaling"] = torch.tensor([[row[name] for name in scale_names] for row in vertex], dtype=torch.float32)
    rotation_names = [f"rot_{i}" for i in range(4)]
    if all(name in names for name in rotation_names):
        out["rotation"] = torch.tensor([[row[name] for name in rotation_names] for row in vertex], dtype=torch.float32)
    dc_names = [f"f_dc_{i}" for i in range(3)]
    if all(name in names for name in dc_names):
        # TRELLIS exports spherical-harmonic DC coefficients; convert them to
        # RGB before gsplat rasterization instead of treating them as colors.
        dc = torch.tensor([[row[name] for name in dc_names] for row in vertex], dtype=torch.float32)
        out["colors"] = (0.5 + 0.28209479177387814 * dc).clamp(0.0, 1.0)
    elif {"red", "green", "blue"} <= names:
        rgb = torch.tensor([[row["red"], row["green"], row["blue"]] for row in vertex], dtype=torch.float32)
        out["colors"] = rgb / 255.0 if rgb.max() > 1.0 else rgb
    return out
