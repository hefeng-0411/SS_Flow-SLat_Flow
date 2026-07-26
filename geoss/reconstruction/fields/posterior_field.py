from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


EVIDENCE_STATES: Tuple[str, ...] = (
    "observed_free",
    "observed_surface",
    "occluded",
    "unobserved",
    "contradictory",
)

# One explicit volume owns geometry, uncertainty, state responsibilities, and
# appearance.  Keeping the schema fixed makes every decoded asset traceable to
# the same continuous state.
FIELD_CHANNELS: Dict[str, slice] = {
    "sdf_mean": slice(0, 1),
    "sdf_log_variance": slice(1, 2),
    "outlier_logit": slice(2, 3),
    "evidence_logits": slice(3, 8),
    "diffuse_logits": slice(8, 11),
    "directional_sh": slice(11, 38),  # RGB x 9 real SH coefficients.
}
FIELD_CHANNEL_COUNT = 38


@dataclass(frozen=True)
class SparseBrickLevel:
    """Sparse residual lattice at one refinement level.

    `brick_indices` are xyz integer coordinates in a regular grid of
    `bricks_per_axis`.  Every brick stores `(brick_resolution + 1)^3` nodes so
    adjacent cells can share a continuous trilinear boundary.  Values are
    residuals in the same channel schema as the coarse volume.
    """

    brick_indices: torch.Tensor  # [B, M, 3], xyz
    values: torch.Tensor  # [B, M, C, G+1, G+1, G+1], zyx storage
    valid: torch.Tensor  # [B, M]
    bricks_per_axis: int
    brick_resolution: int

    def validate(self, batch_size: int, channels: int) -> None:
        if self.brick_indices.ndim != 3 or self.brick_indices.shape[-1] != 3:
            raise ValueError(
                f"brick_indices must be [B,M,3], got {tuple(self.brick_indices.shape)}"
            )
        if self.values.ndim != 6:
            raise ValueError(f"brick values must be [B,M,C,G+1,G+1,G+1], got {tuple(self.values.shape)}")
        if self.brick_indices.shape[:2] != self.values.shape[:2]:
            raise ValueError("brick_indices and values must use identical [B,M] dimensions")
        expected_nodes = self.brick_resolution + 1
        if self.values.shape[2] != channels or self.values.shape[-3:] != (
            expected_nodes,
            expected_nodes,
            expected_nodes,
        ):
            raise ValueError(
                f"Expected brick values [B,M,{channels},{expected_nodes},{expected_nodes},{expected_nodes}], "
                f"got {tuple(self.values.shape)}"
            )
        if self.valid.shape != self.brick_indices.shape[:2]:
            raise ValueError(f"brick valid mask must be [B,M], got {tuple(self.valid.shape)}")
        if self.brick_indices.shape[0] != batch_size:
            raise ValueError("Sparse brick batch size does not match the coarse field")


@dataclass(frozen=True)
class PosteriorFieldState:
    """A continuous field induced by a coarse lattice and sparse residual bricks."""

    volume: torch.Tensor  # [B,38,D,H,W], zyx storage, Ω coordinates in xyz.
    bounds: Tuple[float, float] = (-0.5, 0.5)
    residual_levels: Tuple[SparseBrickLevel, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if self.volume.ndim != 5:
            raise ValueError(f"field volume must be [B,C,D,H,W], got {tuple(self.volume.shape)}")
        if self.volume.shape[1] != FIELD_CHANNEL_COUNT:
            raise ValueError(
                f"field volume requires exactly {FIELD_CHANNEL_COUNT} channels, got {self.volume.shape[1]}"
            )
        if not self.bounds[1] > self.bounds[0]:
            raise ValueError(f"Invalid field bounds {self.bounds}")
        for level in self.residual_levels:
            level.validate(self.volume.shape[0], self.volume.shape[1])

    def replace_volume(self, volume: torch.Tensor) -> "PosteriorFieldState":
        return PosteriorFieldState(volume=volume, bounds=self.bounds, residual_levels=self.residual_levels)

    def with_residual_levels(self, levels: Iterable[SparseBrickLevel]) -> "PosteriorFieldState":
        return PosteriorFieldState(volume=self.volume, bounds=self.bounds, residual_levels=tuple(levels))


@dataclass
class FieldSamples:
    sdf_mean: torch.Tensor  # [B,N,1], world-length units
    sdf_variance: torch.Tensor  # [B,N,1], squared world-length units
    outlier_probability: torch.Tensor  # [B,N,1]
    state_probability: torch.Tensor  # [B,N,5]
    diffuse: torch.Tensor  # [B,N,3]
    directional_sh: torch.Tensor  # [B,N,3,9]
    spatial_gradient: Optional[torch.Tensor] = None  # [B,N,3]
    normal: Optional[torch.Tensor] = None  # [B,N,3]
    color: Optional[torch.Tensor] = None  # [B,N,3]
    raw: Optional[torch.Tensor] = None  # [B,N,38]


def pack_field_volume(
    *,
    sdf_mean: torch.Tensor,
    sdf_log_variance: torch.Tensor,
    outlier_logit: torch.Tensor,
    evidence_logits: torch.Tensor,
    diffuse_logits: torch.Tensor,
    directional_sh: torch.Tensor,
) -> torch.Tensor:
    """Pack named `[B,C,D,H,W]` tensors into the invariant field schema."""

    tensors = (
        sdf_mean,
        sdf_log_variance,
        outlier_logit,
        evidence_logits,
        diffuse_logits,
        directional_sh,
    )
    if any(t.ndim != 5 for t in tensors):
        raise ValueError("Every field component must be a [B,C,D,H,W] tensor")
    spatial = sdf_mean.shape[0:1] + sdf_mean.shape[-3:]
    if any(t.shape[0:1] + t.shape[-3:] != spatial for t in tensors):
        raise ValueError("Field components must share batch and spatial dimensions")
    volume = torch.cat(tensors, dim=1)
    if volume.shape[1] != FIELD_CHANNEL_COUNT:
        raise ValueError(f"Packed field has {volume.shape[1]} channels, expected {FIELD_CHANNEL_COUNT}")
    return volume


def make_canonical_grid(
    resolution: int | Sequence[int],
    *,
    batch_size: int = 1,
    bounds: Tuple[float, float] = (-0.5, 0.5),
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return cell-node query positions `[B,D*H*W,3]` in xyz order."""

    if isinstance(resolution, int):
        d = h = w = int(resolution)
    else:
        if len(resolution) != 3:
            raise ValueError("resolution must be an integer or (D,H,W)")
        d, h, w = (int(v) for v in resolution)
    if min(d, h, w) < 2:
        raise ValueError("Every grid dimension must contain at least two nodes")
    lo, hi = bounds
    z, y, x = torch.meshgrid(
        torch.linspace(lo, hi, d, device=device, dtype=dtype),
        torch.linspace(lo, hi, h, device=device, dtype=dtype),
        torch.linspace(lo, hi, w, device=device, dtype=dtype),
        indexing="ij",
    )
    xyz = torch.stack((x, y, z), dim=-1).reshape(1, d * h * w, 3)
    return xyz.expand(batch_size, -1, -1).contiguous()


def query_posterior_field(
    state: PosteriorFieldState,
    points: torch.Tensor,
    *,
    directions: Optional[torch.Tensor] = None,
    require_gradient: bool = True,
    normal_epsilon: float = 1e-8,
) -> FieldSamples:
    """Evaluate the posterior field with exact trilinear spatial derivatives."""

    state.validate()
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must be [B,N,3], got {tuple(points.shape)}")
    if points.shape[0] != state.volume.shape[0]:
        raise ValueError("Point and field batch dimensions differ")
    raw, raw_gradient = _query_dense_trilinear(
        state.volume,
        points,
        state.bounds,
        require_gradient=require_gradient,
    )
    for level in state.residual_levels:
        residual, residual_gradient = _query_sparse_bricks(
            level,
            points,
            state.bounds,
            require_gradient=require_gradient,
        )
        raw = raw + residual
        if require_gradient:
            if raw_gradient is None or residual_gradient is None:
                raise RuntimeError("A requested spatial derivative was not computed")
            raw_gradient = raw_gradient + residual_gradient

    sdf_mean = raw[..., FIELD_CHANNELS["sdf_mean"]]
    log_variance = raw[..., FIELD_CHANNELS["sdf_log_variance"]].clamp(-14.0, 2.0)
    sdf_variance = log_variance.exp()
    outlier = raw[..., FIELD_CHANNELS["outlier_logit"]].sigmoid()
    state_probability = raw[..., FIELD_CHANNELS["evidence_logits"]].softmax(dim=-1)
    diffuse_logits = raw[..., FIELD_CHANNELS["diffuse_logits"]]
    diffuse = diffuse_logits.sigmoid()
    sh = raw[..., FIELD_CHANNELS["directional_sh"]].reshape(*raw.shape[:-1], 3, 9)

    sdf_gradient = None
    normal = None
    if require_gradient:
        if raw_gradient is None:
            raise RuntimeError("Missing field derivative")
        sdf_gradient = raw_gradient[..., FIELD_CHANNELS["sdf_mean"], :].squeeze(-2)
        # FP32 normalization is important near locally flat or uninitialized
        # regions; the result is cast back only after a finite normal exists.
        grad32 = sdf_gradient.float()
        normal = grad32 / grad32.square().sum(dim=-1, keepdim=True).add(normal_epsilon).sqrt()
        normal = normal.to(dtype=sdf_gradient.dtype)

    color = diffuse
    if directions is not None:
        if directions.shape != points.shape:
            raise ValueError(f"directions must match points [B,N,3], got {tuple(directions.shape)}")
        basis = real_sh_basis_degree2(directions)
        directional = torch.einsum("bnck,bnk->bnc", sh, basis)
        color = (diffuse_logits + directional).sigmoid()

    return FieldSamples(
        sdf_mean=sdf_mean,
        sdf_variance=sdf_variance,
        outlier_probability=outlier,
        state_probability=state_probability,
        diffuse=diffuse,
        directional_sh=sh,
        spatial_gradient=sdf_gradient,
        normal=normal,
        color=color,
        raw=raw,
    )


def real_sh_basis_degree2(directions: torch.Tensor) -> torch.Tensor:
    """Nine orthonormal real spherical-harmonic basis functions."""

    d = F.normalize(directions.float(), dim=-1, eps=1e-8)
    x, y, z = d.unbind(dim=-1)
    basis = torch.stack(
        (
            0.28209479177387814 * torch.ones_like(x),
            -0.4886025119029199 * y,
            0.4886025119029199 * z,
            -0.4886025119029199 * x,
            1.0925484305920792 * x * y,
            -1.0925484305920792 * y * z,
            0.31539156525252005 * (3.0 * z.square() - 1.0),
            -1.0925484305920792 * x * z,
            0.5462742152960396 * (x.square() - y.square()),
        ),
        dim=-1,
    )
    return basis.to(dtype=directions.dtype)


def _query_dense_trilinear(
    volume: torch.Tensor,
    points: torch.Tensor,
    bounds: Tuple[float, float],
    *,
    require_gradient: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    b, c, d, h, w = volume.shape
    lo, hi = bounds
    scale_xyz = points.new_tensor((w - 1, h - 1, d - 1)) / float(hi - lo)
    coordinate = ((points - lo) * scale_xyz).clamp_min(0.0)
    max_xyz = points.new_tensor((w - 1, h - 1, d - 1))
    coordinate = torch.minimum(coordinate, max_xyz)
    lower = coordinate.floor().long()
    upper = torch.minimum(lower + 1, max_xyz.long())
    fraction = coordinate - lower.to(coordinate.dtype)
    flat = volume.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, c)
    return _gather_trilinear(
        flat,
        lower,
        upper,
        fraction,
        node_shape=(d, h, w),
        derivative_scale=scale_xyz,
        require_gradient=require_gradient,
    )


def _query_sparse_bricks(
    level: SparseBrickLevel,
    points: torch.Tensor,
    bounds: Tuple[float, float],
    *,
    require_gradient: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    b, n, _ = points.shape
    _, m, c, nodes, _, _ = level.values.shape
    lo, hi = bounds
    extent = float(hi - lo)
    brick_scale = float(level.bricks_per_axis) / extent
    brick_coordinate = (points - lo) * brick_scale
    query_brick = brick_coordinate.floor().long().clamp(0, level.bricks_per_axis - 1)
    query_key = _linear_xyz_key(query_brick, level.bricks_per_axis)

    stored_key = _linear_xyz_key(level.brick_indices.long(), level.bricks_per_axis)
    invalid_key = level.bricks_per_axis**3
    stored_key = torch.where(level.valid.bool(), stored_key, stored_key.new_full((), invalid_key))
    sorted_key, order = stored_key.sort(dim=1)
    search = torch.searchsorted(sorted_key, query_key).clamp_max(max(m - 1, 0))
    selected_key = sorted_key.gather(1, search)
    selected_brick = order.gather(1, search)
    found = selected_key.eq(query_key) & selected_key.ne(invalid_key)

    local = (brick_coordinate - query_brick.to(brick_coordinate.dtype)) * level.brick_resolution
    max_local = local.new_full((3,), float(level.brick_resolution))
    local = torch.minimum(local.clamp_min(0.0), max_local)
    lower = local.floor().long()
    upper = torch.minimum(lower + 1, max_local.long())
    fraction = local - lower.to(local.dtype)

    nodes_per_brick = nodes**3
    flat = level.values.permute(0, 1, 3, 4, 5, 2).reshape(b, m * nodes_per_brick, c)
    brick_offset = selected_brick * nodes_per_brick
    values, gradients = _gather_trilinear(
        flat,
        lower,
        upper,
        fraction,
        node_shape=(nodes, nodes, nodes),
        derivative_scale=points.new_full((3,), level.brick_resolution * brick_scale),
        require_gradient=require_gradient,
        flat_offset=brick_offset,
    )
    valid = found[..., None].to(values.dtype)
    values = values * valid
    if gradients is not None:
        gradients = gradients * valid[..., None]
    return values, gradients


def _gather_trilinear(
    flat: torch.Tensor,
    lower_xyz: torch.Tensor,
    upper_xyz: torch.Tensor,
    fraction_xyz: torch.Tensor,
    *,
    node_shape: Tuple[int, int, int],
    derivative_scale: torch.Tensor,
    require_gradient: bool,
    flat_offset: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    b, n, _ = lower_xyz.shape
    c = flat.shape[-1]
    d, h, w = node_shape
    values = flat.new_zeros((b, n, c))
    gradients = flat.new_zeros((b, n, c, 3)) if require_gradient else None
    if flat_offset is None:
        flat_offset = lower_xyz.new_zeros((b, n))

    fx, fy, fz = fraction_xyz.unbind(dim=-1)
    for oz in (0, 1):
        iz = lower_xyz[..., 2] if oz == 0 else upper_xyz[..., 2]
        wz = (1.0 - fz) if oz == 0 else fz
        sz = -1.0 if oz == 0 else 1.0
        for oy in (0, 1):
            iy = lower_xyz[..., 1] if oy == 0 else upper_xyz[..., 1]
            wy = (1.0 - fy) if oy == 0 else fy
            sy = -1.0 if oy == 0 else 1.0
            for ox in (0, 1):
                ix = lower_xyz[..., 0] if ox == 0 else upper_xyz[..., 0]
                wx = (1.0 - fx) if ox == 0 else fx
                sx = -1.0 if ox == 0 else 1.0
                index = flat_offset + iz * (h * w) + iy * w + ix
                corner = flat.gather(1, index[..., None].expand(-1, -1, c))
                weight = wx * wy * wz
                values = values + weight[..., None] * corner
                if gradients is not None:
                    derivative = torch.stack(
                        (
                            sx * derivative_scale[0] * wy * wz,
                            sy * derivative_scale[1] * wx * wz,
                            sz * derivative_scale[2] * wx * wy,
                        ),
                        dim=-1,
                    )
                    gradients = gradients + corner[..., None] * derivative[..., None, :]
    return values, gradients


def _linear_xyz_key(indices_xyz: torch.Tensor, size: int) -> torch.Tensor:
    x, y, z = indices_xyz.unbind(dim=-1)
    return z * (size * size) + y * size + x
