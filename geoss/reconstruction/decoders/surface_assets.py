from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoss.reconstruction.fields import (
    PosteriorFieldState,
    make_canonical_grid,
    query_posterior_field,
    real_sh_basis_degree2,
)


@dataclass
class MeshAsset:
    vertices: torch.Tensor
    faces: torch.Tensor
    vertex_normals: torch.Tensor


@dataclass
class SurfaceGaussianAsset:
    means: torch.Tensor  # [B,G,3]
    quaternions: torch.Tensor  # [B,G,4], wxyz
    scales: torch.Tensor  # [B,G,3], positive
    covariance: torch.Tensor  # [B,G,3,3]
    opacity_logits: torch.Tensor  # [B,G,1]
    diffuse: torch.Tensor  # [B,G,3]
    directional_sh: torch.Tensor  # [B,G,3,9]
    normals: torch.Tensor  # [B,G,3]
    valid: torch.Tensor  # [B,G]
    sdf_at_means: torch.Tensor  # [B,G,1]

    def colors(self, view_directions: Optional[torch.Tensor] = None) -> torch.Tensor:
        if view_directions is None:
            return self.diffuse
        basis = real_sh_basis_degree2(view_directions)
        diffuse_logit = torch.logit(self.diffuse.clamp(1e-5, 1.0 - 1e-5))
        directional = torch.einsum("bgck,bgk->bgc", self.directional_sh, basis)
        return (diffuse_logit + directional).sigmoid()

    def as_gsplat_dict(
        self,
        batch_index: int = 0,
        *,
        view_directions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor | str]:
        valid = self.valid[batch_index]
        directions = None
        if view_directions is not None:
            directions = view_directions[batch_index : batch_index + 1]
        colors = self.colors(directions)[0] if directions is not None else self.diffuse[batch_index]
        return {
            "means": self.means[batch_index, valid],
            "quats": self.quaternions[batch_index, valid],
            "scales": self.scales[batch_index, valid],
            "opacities": self.opacity_logits[batch_index, valid].sigmoid().reshape(-1),
            "colors": colors[valid],
            "scaling_parameterization": "activated",
            "opacity_parameterization": "activated",
        }


class UnifiedSurfaceDecoder(nn.Module):
    """Decode mesh and Gaussians from one posterior zero level set."""

    def __init__(
        self,
        *,
        max_gaussians: int = 24_576,
        newton_steps: int = 2,
        surface_band_cells: float = 2.0,
        tangent_scale_cells: float = 0.65,
        minimum_normal_scale_cells: float = 0.05,
        maximum_normal_scale_cells: float = 0.45,
        gradient_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.max_gaussians = int(max_gaussians)
        self.newton_steps = int(newton_steps)
        self.surface_band_cells = float(surface_band_cells)
        self.tangent_scale_cells = float(tangent_scale_cells)
        self.minimum_normal_scale_cells = float(minimum_normal_scale_cells)
        self.maximum_normal_scale_cells = float(maximum_normal_scale_cells)
        self.gradient_epsilon = float(gradient_epsilon)
        self.scale_predictor = nn.Sequential(
            nn.Linear(4, 32),
            nn.SiLU(),
            nn.Linear(32, 3),
        )
        self.opacity_predictor = nn.Sequential(
            nn.Linear(7, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.scale_predictor[-1].weight)
        nn.init.zeros_(self.scale_predictor[-1].bias)
        nn.init.zeros_(self.opacity_predictor[-1].weight)
        nn.init.constant_(self.opacity_predictor[-1].bias, 2.0)

    def forward(self, field: PosteriorFieldState) -> SurfaceGaussianAsset:
        field.validate()
        b, _, d, h, w = field.volume.shape
        points, candidate_valid = _field_candidate_points(field)
        coarse = query_posterior_field(field, points, require_gradient=False)
        cell_size = float(field.bounds[1] - field.bounds[0]) / max(min(d, h, w) - 1, 1)
        state = coarse.state_probability
        surface_probability = state[..., 1]
        uncertainty = coarse.sdf_variance[..., 0].sqrt()
        priority = (
            -coarse.sdf_mean[..., 0].abs() / max(cell_size, 1e-8)
            + surface_probability
            + 0.1 * uncertainty / max(cell_size, 1e-8)
        )
        priority = torch.where(candidate_valid, priority, torch.full_like(priority, -torch.inf))
        count = min(self.max_gaussians, points.shape[1])
        selected = priority.topk(count, dim=1, largest=True, sorted=False).indices
        candidate = points.gather(1, selected[..., None].expand(-1, -1, 3))
        selected_candidate_valid = candidate_valid.gather(1, selected)
        selected_abs_sdf = coarse.sdf_mean[..., 0].abs().gather(1, selected)
        selected_surface_probability = surface_probability.gather(1, selected)
        valid = selected_candidate_valid & (
            (selected_abs_sdf <= self.surface_band_cells * cell_size)
            | (selected_surface_probability >= 0.25)
        )
        means = project_to_surface(
            field,
            candidate,
            steps=self.newton_steps,
            gradient_epsilon=self.gradient_epsilon,
        )
        samples = query_posterior_field(field, means, require_gradient=True)
        if samples.normal is None or samples.spatial_gradient is None:
            raise RuntimeError("Surface Gaussian decoding requires differentiable normals")
        normal = samples.normal
        tangent1, tangent2 = _stable_tangent_frame(normal)
        rotation = torch.stack((tangent1, tangent2, normal), dim=-1)
        quaternions = _rotation_matrix_to_quaternion(rotation)
        curvature = _normal_variation_curvature(
            field,
            means,
            tangent1,
            tangent2,
            step=cell_size,
        )
        standard_deviation = samples.sdf_variance.sqrt().clamp_min(1e-6)
        scale_features = torch.cat(
            (
                (standard_deviation / max(cell_size, 1e-8)).clamp_max(10.0),
                (curvature * cell_size).clamp_max(10.0),
                samples.state_probability[..., 1:2],
                samples.state_probability[..., 4:5],
            ),
            dim=-1,
        )
        scale_adjustment = 0.35 * torch.tanh(self.scale_predictor(scale_features))
        tangent_base = self.tangent_scale_cells * cell_size / (1.0 + curvature * cell_size)
        normal_base = cell_size * (
            self.minimum_normal_scale_cells
            + 0.25 * (standard_deviation / max(cell_size, 1e-8)).clamp(0.0, 1.0)
        )
        normal_base = normal_base.clamp(
            self.minimum_normal_scale_cells * cell_size,
            self.maximum_normal_scale_cells * cell_size,
        )
        scales = torch.cat((tangent_base.expand(-1, -1, 2), normal_base), dim=-1)
        scales = scales * scale_adjustment.exp()
        minimum_scale = self.minimum_normal_scale_cells * cell_size
        maximum_scale = 2.0 * cell_size
        scales = scales.clamp(minimum_scale, maximum_scale)
        covariance = rotation @ torch.diag_embed(scales.square().float()) @ rotation.transpose(-1, -2)
        covariance = covariance.to(scales.dtype)

        opacity_features = torch.cat(
            (
                samples.state_probability,
                samples.outlier_probability,
                (samples.sdf_mean.abs() / max(cell_size, 1e-8)).clamp_max(10.0),
            ),
            dim=-1,
        )
        opacity_logits = self.opacity_predictor(opacity_features)
        invalid_logit = torch.full_like(opacity_logits, -20.0)
        opacity_logits = torch.where(valid[..., None], opacity_logits, invalid_logit)
        return SurfaceGaussianAsset(
            means=means,
            quaternions=quaternions,
            scales=scales,
            covariance=covariance,
            opacity_logits=opacity_logits,
            diffuse=samples.diffuse,
            directional_sh=samples.directional_sh,
            normals=normal,
            valid=valid,
            sdf_at_means=samples.sdf_mean,
        )

    @torch.no_grad()
    def extract_meshes(
        self,
        field: PosteriorFieldState,
        *,
        resolution: Optional[int] = None,
    ) -> List[MeshAsset]:
        return extract_marching_tetrahedra(field, resolution=resolution)


def project_to_surface(
    field: PosteriorFieldState,
    points: torch.Tensor,
    *,
    steps: int = 2,
    gradient_epsilon: float = 1e-8,
) -> torch.Tensor:
    """Stabilized differentiable Newton projection onto `s(x)=0`."""

    x = points
    lo, hi = field.bounds
    max_step = float(hi - lo) / max(min(field.volume.shape[-3:]) - 1, 1) * 2.0
    for _ in range(max(steps, 1)):
        samples = query_posterior_field(field, x, require_gradient=True)
        if samples.spatial_gradient is None:
            raise RuntimeError("SDF projection requires spatial gradients")
        gradient = samples.spatial_gradient.float()
        denominator = gradient.square().sum(dim=-1, keepdim=True).clamp_min(gradient_epsilon)
        step = samples.sdf_mean.float() * gradient / denominator
        step_norm = step.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        step = step * torch.minimum(torch.ones_like(step_norm), step.new_tensor(max_step) / step_norm)
        x = (x.float() - step).clamp(lo, hi).to(points.dtype)
    return x


@torch.no_grad()
def extract_marching_tetrahedra(
    field: PosteriorFieldState,
    *,
    level: float = 0.0,
    resolution: Optional[int] = None,
    query_chunk_size: int = 131_072,
) -> List[MeshAsset]:
    """Extract a deterministic triangle mesh from the dense posterior mean.

    Topology selection is discrete; vertex interpolation is the exact
    piecewise-linear zero set.  Differentiable training uses `project_to_surface`
    while this routine provides the final renderable topology.
    """

    field.validate()
    if resolution is None:
        resolution = field.volume.shape[-1]
        if field.residual_levels:
            resolution = max(
                resolution,
                max(
                    (field.volume.shape[-1] - 1) * refinement.brick_resolution + 1
                    for refinement in field.residual_levels
                ),
            )
        resolution = min(int(resolution), 128)
    if resolution == field.volume.shape[-1] and not field.residual_levels:
        volume = field.volume[:, 0]
    else:
        query_points = make_canonical_grid(
            resolution,
            batch_size=field.volume.shape[0],
            bounds=field.bounds,
            device=field.volume.device,
            dtype=field.volume.dtype,
        )
        values = []
        for start in range(0, query_points.shape[1], query_chunk_size):
            stop = min(start + query_chunk_size, query_points.shape[1])
            values.append(
                query_posterior_field(
                    field,
                    query_points[:, start:stop],
                    require_gradient=False,
                ).sdf_mean
            )
        volume = torch.cat(values, dim=1)[..., 0].reshape(
            field.volume.shape[0],
            resolution,
            resolution,
            resolution,
        )
    b, d, h, w = volume.shape
    lo, hi = field.bounds
    z, y, x = torch.meshgrid(
        torch.linspace(lo, hi, d, device=volume.device, dtype=volume.dtype),
        torch.linspace(lo, hi, h, device=volume.device, dtype=volume.dtype),
        torch.linspace(lo, hi, w, device=volume.device, dtype=volume.dtype),
        indexing="ij",
    )
    positions = torch.stack((x, y, z), dim=-1)
    cube_offset = torch.tensor(
        (
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ),
        device=volume.device,
        dtype=torch.long,
    )
    tetrahedra = (
        (0, 5, 1, 6),
        (0, 1, 2, 6),
        (0, 2, 3, 6),
        (0, 3, 7, 6),
        (0, 7, 4, 6),
        (0, 4, 5, 6),
    )
    meshes: List[MeshAsset] = []
    cell_positions = []
    cell_indices = []
    for ox, oy, oz in cube_offset.tolist():
        cell_positions.append(
            positions[
                oz : oz + d - 1,
                oy : oy + h - 1,
                ox : ox + w - 1,
            ].reshape(-1, 3)
        )
        cell_indices.append((oz, oy, ox))
    cell_positions_tensor = torch.stack(cell_positions, dim=1)
    for batch_index in range(b):
        triangles = []
        corner_sdf = torch.stack(
            [
                volume[
                    batch_index,
                    oz : oz + d - 1,
                    oy : oy + h - 1,
                    ox : ox + w - 1,
                ].reshape(-1)
                for oz, oy, ox in cell_indices
            ],
            dim=-1,
        ) - float(level)
        active = (corner_sdf.amin(dim=-1) < 0.0) & (corner_sdf.amax(dim=-1) >= 0.0)
        active_position = cell_positions_tensor[active]
        active_sdf = corner_sdf[active]
        for tet in tetrahedra:
            tet_index = torch.tensor(tet, device=volume.device, dtype=torch.long)
            triangles.extend(
                _polygonize_tetrahedra_batch(
                    active_position[:, tet_index],
                    active_sdf[:, tet_index],
                )
            )
        if triangles:
            triangle_tensor = torch.cat(triangles, dim=0)
            vertices = triangle_tensor.reshape(-1, 3)
            faces = torch.arange(vertices.shape[0], device=vertices.device, dtype=torch.long).reshape(-1, 3)
            face_normal = F.normalize(
                torch.cross(
                    triangle_tensor[:, 1] - triangle_tensor[:, 0],
                    triangle_tensor[:, 2] - triangle_tensor[:, 0],
                    dim=-1,
                ),
                dim=-1,
                eps=1e-8,
            )
            vertex_normals = face_normal[:, None, :].expand(-1, 3, -1).reshape(-1, 3)
        else:
            vertices = volume.new_zeros((0, 3))
            faces = torch.zeros((0, 3), device=volume.device, dtype=torch.long)
            vertex_normals = volume.new_zeros((0, 3))
        meshes.append(MeshAsset(vertices=vertices, faces=faces, vertex_normals=vertex_normals))
    return meshes


def _polygonize_tetrahedra_batch(position: torch.Tensor, sdf: torch.Tensor) -> List[torch.Tensor]:
    edge_vertices = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
    table = (
        (),
        (0, 3, 2),
        (0, 1, 4),
        (1, 4, 2, 2, 4, 3),
        (1, 2, 5),
        (0, 3, 5, 0, 5, 1),
        (0, 2, 5, 0, 5, 4),
        (5, 4, 3),
        (3, 4, 5),
        (4, 5, 0, 5, 2, 0),
        (1, 5, 0, 5, 3, 0),
        (5, 2, 1),
        (3, 4, 2, 2, 4, 1),
        (4, 1, 0),
        (2, 3, 0),
        (),
    )
    if position.shape[0] == 0:
        return []
    case = sum(((sdf[:, index] < 0).long() << index) for index in range(4))
    triangles = []
    for case_index in range(1, 15):
        selected = case == case_index
        if not bool(selected.any()):
            continue
        selected_position = position[selected]
        selected_sdf = sdf[selected]
        edge_ids = table[case_index]
        intersections = {}
        for edge_id in set(edge_ids):
            a, b = edge_vertices[edge_id]
            denominator = selected_sdf[:, a] - selected_sdf[:, b]
            safe_denominator = torch.where(
                denominator.abs() >= 1e-12,
                denominator,
                torch.full_like(denominator, 1e-12),
            )
            t = (selected_sdf[:, a] / safe_denominator).clamp(0.0, 1.0)
            intersections[edge_id] = selected_position[:, a] + t[:, None] * (
                selected_position[:, b] - selected_position[:, a]
            )
        for offset in range(0, len(edge_ids), 3):
            triangles.append(
                torch.stack(
                tuple(intersections[edge_ids[offset + j]] for j in range(3)),
                dim=1,
            )
            )
    return triangles


def _stable_tangent_frame(normal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    axis_index = normal.abs().argmin(dim=-1)
    reference = F.one_hot(axis_index, num_classes=3).to(normal.dtype)
    tangent1 = F.normalize(torch.cross(reference, normal, dim=-1), dim=-1, eps=1e-8)
    tangent2 = F.normalize(torch.cross(normal, tangent1, dim=-1), dim=-1, eps=1e-8)
    return tangent1, tangent2


def _normal_variation_curvature(
    field: PosteriorFieldState,
    points: torch.Tensor,
    tangent1: torch.Tensor,
    tangent2: torch.Tensor,
    *,
    step: float,
) -> torch.Tensor:
    lo, hi = field.bounds
    samples = []
    for tangent in (tangent1, tangent2):
        plus = query_posterior_field(
            field,
            (points + step * tangent).clamp(lo, hi),
            require_gradient=True,
        )
        minus = query_posterior_field(
            field,
            (points - step * tangent).clamp(lo, hi),
            require_gradient=True,
        )
        if plus.normal is None or minus.normal is None:
            raise RuntimeError("Curvature requires normals")
        samples.append((plus.normal - minus.normal).norm(dim=-1, keepdim=True) / (2.0 * step))
    return 0.5 * (samples[0] + samples[1])


def _rotation_matrix_to_quaternion(rotation: torch.Tensor) -> torch.Tensor:
    r00 = rotation[..., 0, 0]
    r11 = rotation[..., 1, 1]
    r22 = rotation[..., 2, 2]
    qw = 0.5 * torch.sqrt((1.0 + r00 + r11 + r22).clamp_min(0.0))
    qx = 0.5 * torch.copysign(
        torch.sqrt((1.0 + r00 - r11 - r22).clamp_min(0.0)),
        rotation[..., 2, 1] - rotation[..., 1, 2],
    )
    qy = 0.5 * torch.copysign(
        torch.sqrt((1.0 - r00 + r11 - r22).clamp_min(0.0)),
        rotation[..., 0, 2] - rotation[..., 2, 0],
    )
    qz = 0.5 * torch.copysign(
        torch.sqrt((1.0 - r00 - r11 + r22).clamp_min(0.0)),
        rotation[..., 1, 0] - rotation[..., 0, 1],
    )
    return F.normalize(torch.stack((qw, qx, qy, qz), dim=-1), dim=-1, eps=1e-8)


def _field_candidate_points(field: PosteriorFieldState) -> Tuple[torch.Tensor, torch.Tensor]:
    b, _, d, h, w = field.volume.shape
    points = [
        make_canonical_grid(
            (d, h, w),
            batch_size=b,
            bounds=field.bounds,
            device=field.volume.device,
            dtype=field.volume.dtype,
        )
    ]
    valid = [torch.ones(points[0].shape[:2], device=field.volume.device, dtype=torch.bool)]
    lo, hi = field.bounds
    for level in field.residual_levels:
        local = torch.linspace(
            0.0,
            1.0,
            level.brick_resolution + 1,
            device=field.volume.device,
            dtype=field.volume.dtype,
        )
        lz, ly, lx = torch.meshgrid(local, local, local, indexing="ij")
        local_xyz = torch.stack((lx, ly, lz), dim=-1)
        brick_size = float(hi - lo) / level.bricks_per_axis
        origin = lo + level.brick_indices.to(field.volume.dtype) * brick_size
        brick_points = origin[:, :, None, None, None, :] + brick_size * local_xyz[None, None]
        points.append(brick_points.reshape(b, -1, 3))
        valid.append(
            level.valid[:, :, None]
            .expand(b, level.valid.shape[1], (level.brick_resolution + 1) ** 3)
            .reshape(b, -1)
        )
    return torch.cat(points, dim=1), torch.cat(valid, dim=1)
