from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def parse_srn_intrinsics(path: str | Path, image_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Parse SRN intrinsics.txt as `f cx cy 0` into a 3x3 OpenCV K matrix."""
    values = [float(x) for x in Path(path).read_text().strip().split()]
    if len(values) < 3:
        raise ValueError(f"SRN intrinsics require at least f cx cy, got {values}")
    f, cx, cy = values[:3]
    if image_size is not None and max(cx, cy, f) <= 2.0:
        h, w = image_size
        f = f * max(h, w)
        cx = cx * w
        cy = cy * h
    return torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)


def parse_srn_pose(path: str | Path) -> torch.Tensor:
    """Parse SRN pose text as a camera-to-world 4x4 matrix in OpenCV convention."""
    values = [float(x) for x in Path(path).read_text().strip().split()]
    if len(values) == 16:
        c2w = torch.tensor(values, dtype=torch.float32).view(4, 4)
    elif len(values) == 12:
        c2w = torch.eye(4, dtype=torch.float32)
        c2w[:3, :4] = torch.tensor(values, dtype=torch.float32).view(3, 4)
    else:
        raise ValueError(f"Unsupported SRN pose length {len(values)} in {path}")
    return c2w


def opengl_to_opencv(
    c2w_opengl: torch.Tensor,
    *,
    input_is_c2w: bool = True,
) -> torch.Tensor:
    """Convert Blender/OpenGL camera axes to OpenCV axes using diag(1,-1,-1,1)."""
    convert = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=c2w_opengl.dtype, device=c2w_opengl.device))
    if input_is_c2w:
        return c2w_opengl @ convert
    return convert @ c2w_opengl


def c2w_to_w2c(c2w: torch.Tensor) -> torch.Tensor:
    """Invert camera-to-world matrices with arbitrary leading dimensions."""
    if c2w.shape[-2:] != (4, 4):
        raise ValueError(f"c2w must end in 4x4, got {tuple(c2w.shape)}")
    return torch.linalg.inv(c2w)


def w2c_to_c2w(w2c: torch.Tensor) -> torch.Tensor:
    """Invert world-to-camera matrices with arbitrary leading dimensions."""
    if w2c.shape[-2:] != (4, 4):
        raise ValueError(f"w2c must end in 4x4, got {tuple(w2c.shape)}")
    return torch.linalg.inv(w2c)


def parse_objaverse_camera(
    camera_json: str | Path | Dict[str, Any],
    image_size: Optional[Tuple[int, int]] = None,
    *,
    assume_opengl: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parse Objaverse per-view or transforms.json camera data into OpenCV c2w and K."""
    if isinstance(camera_json, (str, Path)):
        data = json.loads(Path(camera_json).read_text())
    else:
        data = camera_json

    matrix = (
        data.get("transform_matrix")
        or data.get("camera_to_world")
        or data.get("c2w")
        or data.get("camera_matrix")
    )
    if matrix is None:
        raise ValueError("Objaverse camera json must contain transform_matrix/camera_to_world/c2w")
    c2w = torch.tensor(matrix, dtype=torch.float32)
    if c2w.numel() == 16:
        c2w = c2w.view(4, 4)
    elif c2w.shape == (3, 4):
        pad = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        c2w = torch.cat([c2w, pad], dim=0)
    else:
        raise ValueError(f"Unsupported Objaverse c2w shape {tuple(c2w.shape)}")
    if assume_opengl:
        c2w = opengl_to_opencv(c2w, input_is_c2w=True)

    source_h, source_w = _source_image_size(data, image_size)
    h, w = _image_size_from_data(data, image_size)
    scale_x = float(w) / max(float(source_w), 1.0)
    scale_y = float(h) / max(float(source_h), 1.0)
    if "K" in data:
        K = torch.tensor(data["K"], dtype=torch.float32).view(3, 3)
        K = _rescale_intrinsics(K, scale_x, scale_y, h, w)
    elif "intrinsics" in data:
        K = torch.tensor(data["intrinsics"], dtype=torch.float32).view(3, 3)
        K = _rescale_intrinsics(K, scale_x, scale_y, h, w)
    else:
        angle_x = data.get("camera_angle_x") or data.get("fov_x")
        if angle_x is None and "fl_x" in data:
            fx_source = float(data["fl_x"])
            fy_source = float(data.get("fl_y", fx_source))
            fx = fx_source * scale_x
            fy = fy_source * scale_y
        elif angle_x is not None:
            fx = 0.5 * w / math.tan(0.5 * float(angle_x))
            fy = fx
        else:
            fx = fy = float(data.get("focal_length", max(h, w)))
        cx = float(data["cx"]) * scale_x if "cx" in data else w / 2.0
        cy = float(data["cy"]) * scale_y if "cy" in data else h / 2.0
        K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
    return c2w, K


def normalize_points_to_canonical(
    points: torch.Tensor,
    *,
    source_range: Tuple[float, float] = (0.0, 1.0),
    target_range: Tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """Linearly map points between cube conventions, e.g. TRELLIS [0,1] to GeoSS [-1,1]."""
    src_min, src_max = source_range
    dst_min, dst_max = target_range
    return (points - src_min) / (src_max - src_min) * (dst_max - dst_min) + dst_min


def canonical_to_trellis_unit(points: torch.Tensor) -> torch.Tensor:
    return normalize_points_to_canonical(points, source_range=(-1.0, 1.0), target_range=(0.0, 1.0))


def trellis_unit_to_canonical(points: torch.Tensor) -> torch.Tensor:
    return normalize_points_to_canonical(points, source_range=(0.0, 1.0), target_range=(-1.0, 1.0))


def canonical_to_trellis_grid(points: torch.Tensor, resolution: int) -> torch.Tensor:
    """Map canonical_geo [-1,1]^3 points to continuous TRELLIS grid coordinates [0,R)."""
    unit = canonical_to_trellis_unit(points)
    return unit * float(resolution)


def trellis_grid_to_canonical(grid_xyz: torch.Tensor, resolution: int) -> torch.Tensor:
    """Map continuous TRELLIS grid coordinates [0,R) to canonical_geo [-1,1]^3."""
    unit = grid_xyz / float(resolution)
    return trellis_unit_to_canonical(unit)


def anchor_to_occ_index(anchor_xyz: torch.Tensor, resolution: int) -> torch.Tensor:
    """Map canonical anchors to nearest occupancy indices in [0,R-1]."""
    grid = canonical_to_trellis_grid(anchor_xyz, resolution)
    return grid.floor().clamp(0, resolution - 1).long()


def occ_index_to_anchor_center(indices: torch.Tensor, resolution: int) -> torch.Tensor:
    """Map occupancy indices to canonical voxel centers."""
    centers = indices.to(torch.float32) + 0.5
    return trellis_grid_to_canonical(centers, resolution)


def _image_size_from_data(data: Dict[str, Any], fallback: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if fallback is not None:
        return int(fallback[0]), int(fallback[1])
    h = data.get("h") or data.get("height") or data.get("H")
    w = data.get("w") or data.get("width") or data.get("W")
    if h is None or w is None:
        raise ValueError("image_size is required when camera json lacks width/height")
    return int(h), int(w)


def _source_image_size(data: Dict[str, Any], fallback: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    h = data.get("h") or data.get("height") or data.get("H")
    w = data.get("w") or data.get("width") or data.get("W")
    if h is not None and w is not None:
        return int(h), int(w)
    if fallback is None:
        raise ValueError("image_size is required when camera json lacks width/height")
    return int(fallback[0]), int(fallback[1])


def _rescale_intrinsics(K: torch.Tensor, scale_x: float, scale_y: float, h: int, w: int) -> torch.Tensor:
    K = K.clone()
    # Some preprocessors serialize normalized intrinsics. Detect this before
    # applying source-pixel resizing.
    if float(K[:2].abs().max()) <= 2.0:
        K[0, :] *= float(w)
        K[1, :] *= float(h)
    else:
        K[0, :] *= scale_x
        K[1, :] *= scale_y
    K[2] = torch.tensor([0.0, 0.0, 1.0], dtype=K.dtype, device=K.device)
    return K
