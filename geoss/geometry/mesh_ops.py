from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from geoss.utils.optional_deps import require_dependency


def load_mesh(path: str | Path, *, real_mode: bool = False) -> Any:
    require_dependency("trimesh", real_mode=real_mode, feature="mesh loading")
    import trimesh

    return trimesh.load(str(path), force="mesh")


def cleanup_mesh(mesh: Any, *, real_mode: bool = False) -> Any:
    require_dependency("trimesh", real_mode=real_mode, feature="mesh cleanup")
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def mesh_stats(mesh: Any) -> Dict[str, float | int | bool]:
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(getattr(mesh, "is_watertight", False)),
        "area": float(getattr(mesh, "area", 0.0)),
    }
