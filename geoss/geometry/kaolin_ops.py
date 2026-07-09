from __future__ import annotations

from typing import Any

from geoss.utils.optional_deps import require_dependency


def require_kaolin_voxel_ops(*, real_mode: bool = False) -> Any:
    require_dependency("kaolin", real_mode=real_mode, feature="voxel/mesh/SDF conversion")
    import kaolin

    return kaolin
