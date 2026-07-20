from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from geoss.utils.optional_deps import require_dependency


# TRELLIS writes public PLY/GLB assets after rotating its internal z-up frame
# into a y-up exchange frame.  MeshFleet cameras in this project remain in the
# internal reconstruction frame, so evaluation must invert this exact rotation.
TRELLIS_INTERNAL_TO_EXPORT = torch.tensor(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=torch.float32,
)


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


def write_internal_mesh(mesh: Any, path: str | Path, *, real_mode: bool = True) -> None:
    """Persist a TRELLIS decoded mesh without applying its public y-up rotation."""
    vertices = getattr(mesh, "vertices", None)
    faces = getattr(mesh, "faces", None)
    if vertices is None or faces is None:
        raise TypeError(f"Decoded mesh {type(mesh)!r} does not expose vertices/faces.")
    require_dependency("trimesh", real_mode=real_mode, feature="internal TRELLIS mesh export")
    import trimesh

    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().float().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().long().cpu().numpy()
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(str(path))


def read_gaussian_ply(path: str | Path, *, real_mode: bool = False) -> Dict[str, Any]:
    ply = read_asset(path, real_mode=real_mode)
    vertex = ply["vertex"].data
    out = {
        "xyz": torch.tensor([[row["x"], row["y"], row["z"]] for row in vertex], dtype=torch.float32),
    }
    names = set(vertex.dtype.names or [])
    if {"opacity"} <= names:
        out["opacity"] = torch.tensor([row["opacity"] for row in vertex], dtype=torch.float32).view(-1, 1)
        out["opacity_parameterization"] = "logit"
    scale_names = ["scale_0", "scale_1", "scale_2"]
    if all(name in names for name in scale_names):
        out["scaling"] = torch.tensor([[row[name] for name in scale_names] for row in vertex], dtype=torch.float32)
        out["scaling_parameterization"] = "log"
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


def trellis_export_gaussian_to_internal(gaussians: Dict[str, Any]) -> Dict[str, Any]:
    """Undo the coordinate transform applied by ``TRELLIS Gaussian.save_ply``.

    TRELLIS exports row-vector positions as ``p_export = p_internal @ T.T``
    and orientation matrices as ``R_export = T @ R_internal``.  Rendering an
    exported PLY with the original MeshFleet cameras without this inverse
    transform rotates the object by 90 degrees about x and invalidates both
    appearance and silhouette metrics.
    """
    if "xyz" not in gaussians:
        raise KeyError("A TRELLIS Gaussian dictionary must contain 'xyz'.")
    out = dict(gaussians)
    xyz = gaussians["xyz"]
    transform = TRELLIS_INTERNAL_TO_EXPORT.to(device=xyz.device, dtype=xyz.dtype)
    out["xyz"] = xyz @ transform
    rotation = gaussians.get("rotation")
    if rotation is not None:
        if rotation.ndim != 2 or rotation.shape[-1] != 4:
            raise ValueError(f"TRELLIS rotations must be wxyz quaternions [P,4], got {tuple(rotation.shape)}")
        # T is +90 degrees around x.  Since q_export = q_T * q_internal,
        # q_internal = conjugate(q_T) * q_export.
        half = rotation.new_tensor(2.0).sqrt().reciprocal()
        q_inv = torch.stack([half, -half, half.new_zeros(()), half.new_zeros(())])
        out["rotation"] = _quaternion_multiply(q_inv.expand_as(rotation), rotation)
        out["rotation"] = torch.nn.functional.normalize(out["rotation"], dim=-1)
    out["coordinate_frame"] = "trellis_internal_z_up"
    return out


def trellis_export_points_to_internal(points: torch.Tensor) -> torch.Tensor:
    """Undo TRELLIS' public-asset transform for row-vector point arrays."""
    if points.shape[-1] != 3:
        raise ValueError(f"points must end in xyz, got {tuple(points.shape)}")
    transform = TRELLIS_INTERNAL_TO_EXPORT.to(device=points.device, dtype=points.dtype)
    return points @ transform


def update_gaussian_ply_parameters(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    colors: torch.Tensor,
    opacity_logits: torch.Tensor,
    scaling_logits: torch.Tensor | None = None,
    real_mode: bool = True,
) -> None:
    """Write refined appearance parameters while preserving export-frame geometry."""
    require_dependency("plyfile", real_mode=real_mode, feature="refined Gaussian PLY export")
    from plyfile import PlyData

    ply = PlyData.read(str(source_path))
    vertex = ply["vertex"].data
    count = len(vertex)
    if colors.shape != (count, 3) or opacity_logits.numel() != count:
        raise ValueError(
            f"Refined Gaussian count mismatch: PLY={count}, colors={tuple(colors.shape)}, "
            f"opacity={tuple(opacity_logits.shape)}"
        )
    names = set(vertex.dtype.names or [])
    if not {"f_dc_0", "f_dc_1", "f_dc_2", "opacity"} <= names:
        raise ValueError("TRELLIS PLY lacks SH-DC/opacity fields required for refinement export.")
    colors_cpu = colors.detach().float().clamp(0, 1).cpu()
    dc = (colors_cpu - 0.5) / 0.28209479177387814
    for channel in range(3):
        vertex[f"f_dc_{channel}"] = dc[:, channel].numpy()
    vertex["opacity"] = opacity_logits.detach().float().reshape(-1).cpu().numpy()
    if scaling_logits is not None:
        if scaling_logits.shape != (count, 3):
            raise ValueError(f"scaling_logits must be [{count},3], got {tuple(scaling_logits.shape)}")
        for channel in range(3):
            name = f"scale_{channel}"
            if name not in names:
                raise ValueError(f"TRELLIS PLY lacks {name}.")
            vertex[name] = scaling_logits.detach().float().cpu().numpy()[:, channel]
    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
    ply.write(str(destination_path))


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product for wxyz quaternions with broadcastable shapes."""
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dim=-1,
    )
