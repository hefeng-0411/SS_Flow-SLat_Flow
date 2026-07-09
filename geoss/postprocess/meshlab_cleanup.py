from __future__ import annotations

from pathlib import Path

from geoss.utils.optional_deps import require_dependency


def cleanup_mesh_with_meshlab(input_mesh: str | Path, output_mesh: str | Path, *, real_mode: bool = False) -> str:
    require_dependency("pymeshlab", real_mode=real_mode, feature="MeshLab cleanup")
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(input_mesh))
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()
    ms.save_current_mesh(str(output_mesh))
    return str(output_mesh)
