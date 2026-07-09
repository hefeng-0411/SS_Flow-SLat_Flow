from __future__ import annotations

from pathlib import Path
from typing import Optional


def find_shapenet_mesh(shapenet_root: str | Path, object_id: str) -> Optional[Path]:
    """Find a ShapeNet mesh for an object id using common paths."""
    root = Path(shapenet_root)
    candidates = [
        root / object_id / "models" / "model_normalized.obj",
        root / object_id / "model.obj",
        root / object_id / f"{object_id}.obj",
        root / "02958343" / object_id / "models" / "model_normalized.obj",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = list(root.rglob(f"{object_id}*.obj"))
    return matches[0] if matches else None
