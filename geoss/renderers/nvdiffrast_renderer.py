from __future__ import annotations

from typing import Dict, Tuple

import torch

from geoss.utils.optional_deps import require_dependency


def rasterize_triangles(vertices_clip: torch.Tensor, faces: torch.Tensor, image_size: Tuple[int, int], *, real_mode: bool = True) -> Dict[str, torch.Tensor]:
    require_dependency("nvdiffrast", real_mode=real_mode, feature="advanced differentiable mesh rasterization")
    import nvdiffrast.torch as dr

    ctx = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(ctx, vertices_clip, faces.int(), resolution=image_size)
    return {"rast": rast, "mask": (rast[..., 3:4] > 0).float()}
