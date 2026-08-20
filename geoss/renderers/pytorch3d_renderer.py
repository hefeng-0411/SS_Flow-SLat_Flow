from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from geoss.utils.optional_deps import require_dependency


def render_mesh_pytorch3d(mesh: Any, cameras: Dict[str, torch.Tensor], image_size: Tuple[int, int], *, real_mode: bool = True) -> Dict[str, torch.Tensor]:
    require_dependency("pytorch3d", real_mode=real_mode, feature="mesh render evaluation")
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings

    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
    rasterizer = MeshRasterizer(raster_settings=raster_settings)
    fragments = rasterizer(mesh)
    zbuf = fragments.zbuf[..., :1]
    silhouette = (fragments.pix_to_face[..., :1] >= 0).float()
    return {"depth": zbuf, "mask": silhouette}
