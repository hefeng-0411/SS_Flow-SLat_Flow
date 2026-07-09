from __future__ import annotations

import importlib.util
from functools import lru_cache
from typing import Dict


_MODULES = {
    "gsplat": ("gsplat",),
    "pycolmap": ("pycolmap",),
    "kornia": ("kornia",),
    "pytorch3d": ("pytorch3d",),
    "nvdiffrast": ("nvdiffrast",),
    "kaolin": ("kaolin",),
    "open3d": ("open3d",),
    "trimesh": ("trimesh",),
    "plyfile": ("plyfile",),
    "point_cloud_utils": ("point_cloud_utils",),
    "pymeshlab": ("pymeshlab",),
    "mcubes": ("mcubes",),
    "pygltflib": ("pygltflib",),
    "viser": ("viser",),
    "rerun": ("rerun", "rerun_sdk"),
    "pyvista": ("pyvista",),
    "poselib": ("poselib",),
    "lightglue": ("lightglue",),
    "hloc": ("hloc",),
    "nerfstudio": ("nerfstudio",),
    "nerfacc": ("nerfacc",),
    "lpips": ("lpips",),
}


@lru_cache(maxsize=None)
def is_available(name: str) -> bool:
    aliases = _MODULES.get(name, (name,))
    return any(importlib.util.find_spec(alias) is not None for alias in aliases)


def is_gsplat_available() -> bool:
    return is_available("gsplat")


def is_pycolmap_available() -> bool:
    return is_available("pycolmap")


def is_kornia_available() -> bool:
    return is_available("kornia")


def is_pytorch3d_available() -> bool:
    return is_available("pytorch3d")


def is_nvdiffrast_available() -> bool:
    return is_available("nvdiffrast")


def is_kaolin_available() -> bool:
    return is_available("kaolin")


def is_open3d_available() -> bool:
    return is_available("open3d")


def availability_report() -> Dict[str, bool]:
    return {name: is_available(name) for name in sorted(_MODULES)}


def require_dependency(name: str, real_mode: bool = False, feature: str | None = None) -> bool:
    ok = is_available(name)
    if ok:
        return True
    msg = f"Missing optional dependency '{name}'"
    if feature:
        msg += f" required for {feature}"
    msg += "."
    if real_mode:
        raise ImportError(msg + " Install the matching requirements extra or disable this real-mode feature.")
    return False


def require_many(names: list[str], *, real_mode: bool, feature: str) -> None:
    for name in names:
        require_dependency(name, real_mode=real_mode, feature=feature)
