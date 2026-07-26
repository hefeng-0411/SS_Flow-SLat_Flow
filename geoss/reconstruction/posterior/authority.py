from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from geoss.reconstruction.fields import PosteriorFieldState, SparseBrickLevel


@dataclass(frozen=True)
class SparseLinearConstraints:
    """Local linear constraints `J x = target` over a flattened SDF lattice."""

    indices: torch.Tensor  # [B,G,K]
    jacobian: torch.Tensor  # [B,G,K]
    target: torch.Tensor  # [B,G]
    valid: torch.Tensor  # [B,G]
    num_parameters: int

    def validate(self) -> None:
        if self.indices.ndim != 3:
            raise ValueError(f"constraint indices must be [B,G,K], got {tuple(self.indices.shape)}")
        if self.jacobian.shape != self.indices.shape:
            raise ValueError("constraint Jacobian and indices must have identical shapes")
        if self.target.shape != self.indices.shape[:2] or self.valid.shape != self.indices.shape[:2]:
            raise ValueError("constraint target/valid tensors must be [B,G]")
        if self.indices.numel() and (
            int(self.indices.min()) < 0 or int(self.indices.max()) >= self.num_parameters
        ):
            raise ValueError("constraint index lies outside the flattened field")

    def apply(self, vector: torch.Tensor) -> torch.Tensor:
        self.validate()
        if vector.shape != (self.indices.shape[0], self.num_parameters):
            raise ValueError(
                f"constraint vector must be [B,{self.num_parameters}], got {tuple(vector.shape)}"
            )
        gathered = vector.gather(
            1,
            self.indices.reshape(self.indices.shape[0], -1),
        ).reshape_as(self.jacobian)
        value = (gathered * self.jacobian).sum(dim=-1)
        return value * self.valid.to(value.dtype)

    def transpose_apply(self, multiplier: torch.Tensor) -> torch.Tensor:
        self.validate()
        if multiplier.shape != self.target.shape:
            raise ValueError(f"constraint multiplier must be {tuple(self.target.shape)}")
        contribution = (
            self.jacobian
            * multiplier[..., None]
            * self.valid[..., None].to(self.jacobian.dtype)
        )
        out = contribution.new_zeros((self.indices.shape[0], self.num_parameters))
        out.scatter_add_(
            1,
            self.indices.reshape(self.indices.shape[0], -1),
            contribution.reshape(self.indices.shape[0], -1),
        )
        return out


@dataclass
class AuthorityUpdateResult:
    value: torch.Tensor
    observation_correction: torch.Tensor
    projected_prior_update: torch.Tensor
    feasible_step: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


class MatrixFreeAuthorityProjector(nn.Module):
    """Observation-authoritative projected posterior update.

    The Schur system is never materialized:

        A y = (J M^{-1} Jᵀ + λI)y.

    This permits thousands of local trilinear surface constraints while
    retaining the exact first-order authority invariant up to CG tolerance.
    """

    def __init__(
        self,
        *,
        cg_iterations: int = 48,
        cg_tolerance: float = 1e-6,
        damping: float = 1e-6,
        projection_passes: int = 2,
        inequality_passes: int = 4,
        metric_floor: float = 1e-5,
    ) -> None:
        super().__init__()
        self.cg_iterations = int(cg_iterations)
        self.cg_tolerance = float(cg_tolerance)
        self.damping = float(damping)
        self.projection_passes = int(projection_passes)
        self.inequality_passes = int(inequality_passes)
        self.metric_floor = float(metric_floor)

    def forward(
        self,
        current: torch.Tensor,
        prior_proposal: torch.Tensor,
        *,
        metric_diagonal: torch.Tensor,
        surface_constraints: SparseLinearConstraints,
        free_constraints: Optional[SparseLinearConstraints] = None,
        free_margin: float = 0.002,
    ) -> AuthorityUpdateResult:
        if current.ndim != 2 or prior_proposal.shape != current.shape:
            raise ValueError("current and prior_proposal must both be [B,P]")
        if metric_diagonal.shape != current.shape:
            raise ValueError("metric_diagonal must match [B,P]")
        inverse_metric = metric_diagonal.float().clamp_min(self.metric_floor).reciprocal()
        current32 = current.float()
        proposal32 = prior_proposal.float()

        provisional_correction, _ = self.minimum_norm_correction(
            current32,
            surface_constraints,
            inverse_metric,
        )
        active = None
        if free_constraints is not None:
            free_value = free_constraints.apply(current32 + provisional_correction)
            violation = (float(free_margin) - free_value).clamp_min(0.0)
            active = free_constraints.valid & (violation > 0.0)
            if active.any():
                active_constraints = SparseLinearConstraints(
                    indices=free_constraints.indices,
                    jacobian=free_constraints.jacobian,
                    target=torch.full_like(free_constraints.target, float(free_margin)),
                    valid=active,
                    num_parameters=free_constraints.num_parameters,
                )
                joint_constraints = _concatenate_constraints(
                    surface_constraints,
                    active_constraints,
                )
                observation_correction, observation_diag = self.minimum_norm_correction(
                    current32,
                    joint_constraints,
                    inverse_metric,
                )
                observation_diag["free_active_count"] = active.sum(dim=-1)
            else:
                observation_correction, observation_diag = self.minimum_norm_correction(
                    current32,
                    surface_constraints,
                    inverse_metric,
                )
        else:
            observation_correction, observation_diag = self.minimum_norm_correction(
                current32,
                surface_constraints,
                inverse_metric,
            )
        corrected = current32 + observation_correction

        projected, projection_diag = self.project_null_space(
            proposal32,
            surface_constraints,
            inverse_metric,
        )
        feasible_step = corrected.new_ones((corrected.shape[0], 1))
        if free_constraints is not None:
            projected, inequality_diag = self.project_free_space_feasible(
                corrected,
                projected,
                surface_constraints,
                free_constraints,
                lower_bound=float(free_margin),
                inverse_metric=inverse_metric,
            )
        else:
            inequality_diag = {}
        value = corrected + projected
        diagnostics = {
            **observation_diag,
            **projection_diag,
            **inequality_diag,
            "feasible_step": feasible_step.squeeze(-1),
            "surface_residual_after": (
                surface_constraints.apply(value) - surface_constraints.target
            ).abs().amax(dim=-1),
        }
        if free_constraints is not None:
            valid_free = free_constraints.valid
            free_value = free_constraints.apply(value)
            diagnostics["minimum_free_sdf"] = torch.where(
                valid_free,
                free_value,
                torch.full_like(free_value, float("inf")),
            ).amin(dim=-1)
        return AuthorityUpdateResult(
            value=value.to(current.dtype),
            observation_correction=observation_correction.to(current.dtype),
            projected_prior_update=projected.to(current.dtype),
            feasible_step=feasible_step.to(current.dtype),
            diagnostics=diagnostics,
        )

    def project_free_space_feasible(
        self,
        current: torch.Tensor,
        proposal: torch.Tensor,
        surface_constraints: SparseLinearConstraints,
        free_constraints: SparseLinearConstraints,
        *,
        lower_bound: float,
        inverse_metric: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Sparse active-set projection for `J_free(current + delta) >= lower`.

        Unlike a global line search, this correction leaves parameter
        directions unrelated to the violated ray constraints unchanged.
        """

        value = proposal
        surface_tangent = SparseLinearConstraints(
            indices=surface_constraints.indices,
            jacobian=surface_constraints.jacobian,
            target=torch.zeros_like(surface_constraints.target),
            valid=surface_constraints.valid,
            num_parameters=surface_constraints.num_parameters,
        )
        active_counts = []
        correction_norms = []
        for _ in range(max(self.inequality_passes, 1)):
            free_current = free_constraints.apply(current)
            free_after = free_constraints.apply(current + value)
            active = free_constraints.valid & (free_after < float(lower_bound))
            active_counts.append(active.sum(dim=-1))
            if not bool(active.any().item()):
                correction_norms.append(value.new_zeros((value.shape[0],)))
                break
            active_free = SparseLinearConstraints(
                indices=free_constraints.indices,
                jacobian=free_constraints.jacobian,
                target=(
                    torch.full_like(free_constraints.target, float(lower_bound))
                    - free_current
                ),
                valid=active,
                num_parameters=free_constraints.num_parameters,
            )
            joint = _concatenate_constraints(surface_tangent, active_free)
            correction, _ = self.minimum_norm_correction(
                value,
                joint,
                inverse_metric,
            )
            value = value + correction
            correction_norms.append(correction.norm(dim=-1))
        free_after = free_constraints.apply(current + value)
        minimum = torch.where(
            free_constraints.valid,
            free_after,
            torch.full_like(free_after, float("inf")),
        ).amin(dim=-1)
        return value, {
            "free_projection_active_count": torch.stack(active_counts, dim=-1),
            "free_projection_correction_norm": torch.stack(correction_norms, dim=-1),
            "free_projection_minimum": minimum,
        }

    def project_null_space(
        self,
        proposal: torch.Tensor,
        constraints: SparseLinearConstraints,
        inverse_metric: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        projected = proposal
        before = constraints.apply(proposal)
        residual_norms = []
        for _ in range(max(self.projection_passes, 1)):
            rhs = constraints.apply(projected)
            multiplier, solve_diag = self._solve_schur(rhs, constraints, inverse_metric)
            correction = inverse_metric * constraints.transpose_apply(multiplier)
            projected = projected - correction
            residual_norms.append(constraints.apply(projected).norm(dim=-1))
        after = constraints.apply(projected)
        return projected, {
            "prior_constraint_before": before.norm(dim=-1),
            "prior_constraint_after": after.norm(dim=-1),
            "prior_projection_ratio": after.norm(dim=-1) / before.norm(dim=-1).clamp_min(1e-12),
            "cg_relative_residual": solve_diag["relative_residual"],
            "projection_pass_residual": torch.stack(residual_norms, dim=-1),
        }

    def minimum_norm_correction(
        self,
        current: torch.Tensor,
        constraints: SparseLinearConstraints,
        inverse_metric: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        residual = constraints.apply(current) - constraints.target * constraints.valid.to(current.dtype)
        multiplier, solve_diag = self._solve_schur(residual, constraints, inverse_metric)
        correction = -inverse_metric * constraints.transpose_apply(multiplier)
        corrected_residual = constraints.apply(current + correction) - constraints.target * constraints.valid.to(
            current.dtype
        )
        return correction, {
            "surface_residual_before": residual.norm(dim=-1),
            "surface_residual_corrected": corrected_residual.norm(dim=-1),
            "observation_correction_norm": correction.norm(dim=-1),
            "observation_cg_relative_residual": solve_diag["relative_residual"],
        }

    def maximum_feasible_step(
        self,
        current: torch.Tensor,
        proposal: torch.Tensor,
        constraints: SparseLinearConstraints,
        *,
        lower_bound: float,
    ) -> torch.Tensor:
        value = constraints.apply(current)
        direction = constraints.apply(proposal)
        decreasing = constraints.valid & (direction < -1e-12)
        candidate = (value - float(lower_bound)).clamp_min(0.0) / (-direction).clamp_min(1e-12)
        candidate = torch.where(decreasing, candidate, torch.ones_like(candidate))
        # A small safety contraction prevents floating-point roundoff from
        # crossing an observed-free inequality.
        alpha = candidate.amin(dim=-1, keepdim=True).clamp(0.0, 1.0)
        return torch.where(alpha < 1.0, alpha * 0.999, alpha)

    def _solve_schur(
        self,
        rhs: torch.Tensor,
        constraints: SparseLinearConstraints,
        inverse_metric: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rhs32 = rhs.float() * constraints.valid.to(rhs.dtype)

        def apply_system(vector: torch.Tensor) -> torch.Tensor:
            jt = constraints.transpose_apply(vector)
            j_minv_jt = constraints.apply(inverse_metric * jt)
            return j_minv_jt + self.damping * vector

        x = torch.zeros_like(rhs32)
        residual = rhs32.clone()
        direction = residual.clone()
        squared = residual.square().sum(dim=-1, keepdim=True)
        initial = squared.sqrt().clamp_min(1e-20)
        for _ in range(self.cg_iterations):
            system_direction = apply_system(direction)
            denominator = (direction * system_direction).sum(dim=-1, keepdim=True).clamp_min(1e-20)
            alpha = squared / denominator
            x = x + alpha * direction
            residual = residual - alpha * system_direction
            next_squared = residual.square().sum(dim=-1, keepdim=True)
            if bool((next_squared.sqrt() <= self.cg_tolerance * initial).all().item()):
                squared = next_squared
                break
            beta = next_squared / squared.clamp_min(1e-20)
            direction = residual + beta * direction
            squared = next_squared
        return x, {
            "relative_residual": (squared.sqrt() / initial).squeeze(-1),
        }


def make_trilinear_constraints(
    points: torch.Tensor,
    *,
    resolution: int | Tuple[int, int, int],
    target: Optional[torch.Tensor] = None,
    valid: Optional[torch.Tensor] = None,
    reliability: Optional[torch.Tensor] = None,
    bounds: Tuple[float, float] = (-0.5, 0.5),
) -> SparseLinearConstraints:
    """Create sparse interpolation rows over a dense zyx-stored lattice."""

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"constraint points must be [B,G,3], got {tuple(points.shape)}")
    if isinstance(resolution, int):
        d = h = w = int(resolution)
    else:
        d, h, w = (int(value) for value in resolution)
    b, g, _ = points.shape
    lo, hi = bounds
    inside = ((points >= lo) & (points <= hi)).all(dim=-1)
    scale = points.new_tensor((w - 1, h - 1, d - 1)) / float(hi - lo)
    coordinate = ((points - lo) * scale).clamp_min(0.0)
    coordinate = torch.minimum(coordinate, points.new_tensor((w - 1, h - 1, d - 1)))
    lower = coordinate.floor().long()
    upper = torch.minimum(lower + 1, lower.new_tensor((w - 1, h - 1, d - 1)))
    fraction = coordinate - lower.to(coordinate.dtype)
    indices = []
    weights = []
    fx, fy, fz = fraction.unbind(dim=-1)
    for oz in (0, 1):
        iz = lower[..., 2] if oz == 0 else upper[..., 2]
        wz = 1.0 - fz if oz == 0 else fz
        for oy in (0, 1):
            iy = lower[..., 1] if oy == 0 else upper[..., 1]
            wy = 1.0 - fy if oy == 0 else fy
            for ox in (0, 1):
                ix = lower[..., 0] if ox == 0 else upper[..., 0]
                wx = 1.0 - fx if ox == 0 else fx
                indices.append(iz * h * w + iy * w + ix)
                weights.append(wx * wy * wz)
    index_tensor = torch.stack(indices, dim=-1)
    jacobian = torch.stack(weights, dim=-1)
    if reliability is not None:
        if reliability.shape == (b, g, 1):
            reliability = reliability[..., 0]
        if reliability.shape != (b, g):
            raise ValueError("constraint reliability must be [B,G] or [B,G,1]")
        sqrt_reliability = reliability.float().clamp_min(0.0).sqrt().to(jacobian.dtype)
        jacobian = jacobian * sqrt_reliability[..., None]
    else:
        sqrt_reliability = jacobian.new_ones((b, g))
    if target is None:
        target_tensor = points.new_zeros((b, g))
    else:
        target_tensor = target[..., 0] if target.shape == (b, g, 1) else target
        if target_tensor.shape != (b, g):
            raise ValueError("constraint target must be [B,G] or [B,G,1]")
    target_tensor = target_tensor * sqrt_reliability
    valid_tensor = inside if valid is None else (inside & valid.bool().reshape(b, g))
    return SparseLinearConstraints(
        indices=index_tensor,
        jacobian=jacobian,
        target=target_tensor,
        valid=valid_tensor,
        num_parameters=d * h * w,
    )


def flatten_hierarchical_sdf(field: PosteriorFieldState) -> torch.Tensor:
    """Flatten coarse and sparse-residual SDF nodes into one parameter vector."""

    field.validate()
    values = [field.volume[:, 0].reshape(field.volume.shape[0], -1)]
    values.extend(
        level.values[:, :, 0].reshape(field.volume.shape[0], -1)
        for level in field.residual_levels
    )
    return torch.cat(values, dim=1)


def replace_hierarchical_sdf(
    field: PosteriorFieldState,
    flattened: torch.Tensor,
) -> PosteriorFieldState:
    """Functional inverse of :func:`flatten_hierarchical_sdf`."""

    b, _, d, h, w = field.volume.shape
    coarse_count = d * h * w
    if flattened.shape[0] != b:
        raise ValueError("Flattened SDF batch does not match the field")
    cursor = coarse_count
    volume = field.volume.clone()
    volume[:, 0] = flattened[:, :coarse_count].reshape(b, d, h, w)
    levels = []
    for level in field.residual_levels:
        count = level.values[:, :, 0].numel() // b
        values = level.values.clone()
        values[:, :, 0] = flattened[:, cursor : cursor + count].reshape_as(values[:, :, 0])
        cursor += count
        levels.append(
            SparseBrickLevel(
                brick_indices=level.brick_indices,
                values=values,
                valid=level.valid,
                bricks_per_axis=level.bricks_per_axis,
                brick_resolution=level.brick_resolution,
            )
        )
    if cursor != flattened.shape[1]:
        raise ValueError(f"Flattened SDF has {flattened.shape[1]} values, expected {cursor}")
    return PosteriorFieldState(
        volume=volume,
        bounds=field.bounds,
        residual_levels=tuple(levels),
    )


def make_hierarchical_constraints(
    field: PosteriorFieldState,
    points: torch.Tensor,
    *,
    target: Optional[torch.Tensor] = None,
    valid: Optional[torch.Tensor] = None,
) -> SparseLinearConstraints:
    """Trilinear constraints over coarse nodes and every active residual brick."""

    field.validate()
    _, _, d, h, w = field.volume.shape
    coarse = make_trilinear_constraints(
        points,
        resolution=(d, h, w),
        target=target,
        valid=valid,
        bounds=field.bounds,
    )
    indices = [coarse.indices]
    jacobian = [coarse.jacobian]
    parameter_offset = d * h * w
    for level in field.residual_levels:
        level_indices, level_jacobian, found = _sparse_brick_constraint_stencil(
            level,
            points,
            field.bounds,
            parameter_offset,
        )
        indices.append(level_indices)
        jacobian.append(level_jacobian * found[..., None].to(level_jacobian.dtype))
        parameter_offset += level.values[:, :, 0].numel() // field.volume.shape[0]
    return SparseLinearConstraints(
        indices=torch.cat(indices, dim=-1),
        jacobian=torch.cat(jacobian, dim=-1),
        target=coarse.target,
        valid=coarse.valid,
        num_parameters=parameter_offset,
    )


def _sparse_brick_constraint_stencil(
    level: SparseBrickLevel,
    points: torch.Tensor,
    bounds: Tuple[float, float],
    parameter_offset: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, g, _ = points.shape
    m = level.brick_indices.shape[1]
    nodes = level.brick_resolution + 1
    nodes_per_brick = nodes**3
    lo, hi = bounds
    coordinate = (points - lo) * (float(level.bricks_per_axis) / float(hi - lo))
    brick = coordinate.floor().long().clamp(0, level.bricks_per_axis - 1)
    query_key = _linear_key(brick, level.bricks_per_axis)
    stored_key = _linear_key(level.brick_indices.long(), level.bricks_per_axis)
    invalid_key = level.bricks_per_axis**3
    stored_key = torch.where(level.valid.bool(), stored_key, stored_key.new_full((), invalid_key))
    sorted_key, order = stored_key.sort(dim=1)
    search = torch.searchsorted(sorted_key, query_key).clamp_max(max(m - 1, 0))
    selected_key = sorted_key.gather(1, search)
    selected_brick = order.gather(1, search)
    found = selected_key.eq(query_key) & selected_key.ne(invalid_key)
    local = (coordinate - brick.to(coordinate.dtype)) * level.brick_resolution
    local = local.clamp(0.0, float(level.brick_resolution))
    lower = local.floor().long()
    upper = torch.minimum(lower + 1, lower.new_full((3,), level.brick_resolution))
    fraction = local - lower.to(local.dtype)
    fx, fy, fz = fraction.unbind(dim=-1)
    indices = []
    weights = []
    for oz in (0, 1):
        iz = lower[..., 2] if oz == 0 else upper[..., 2]
        wz = 1.0 - fz if oz == 0 else fz
        for oy in (0, 1):
            iy = lower[..., 1] if oy == 0 else upper[..., 1]
            wy = 1.0 - fy if oy == 0 else fy
            for ox in (0, 1):
                ix = lower[..., 0] if ox == 0 else upper[..., 0]
                wx = 1.0 - fx if ox == 0 else fx
                node = iz * nodes * nodes + iy * nodes + ix
                indices.append(parameter_offset + selected_brick * nodes_per_brick + node)
                weights.append(wx * wy * wz)
    return torch.stack(indices, dim=-1), torch.stack(weights, dim=-1), found


def _linear_key(indices: torch.Tensor, size: int) -> torch.Tensor:
    x, y, z = indices.unbind(dim=-1)
    return z * size * size + y * size + x


def _concatenate_constraints(
    first: SparseLinearConstraints,
    second: SparseLinearConstraints,
) -> SparseLinearConstraints:
    if first.num_parameters != second.num_parameters:
        raise ValueError("Cannot combine constraints over different parameter spaces")
    if first.indices.shape[0] != second.indices.shape[0] or first.indices.shape[2] != second.indices.shape[2]:
        raise ValueError("Combined constraints must share batch and local stencil width")
    return SparseLinearConstraints(
        indices=torch.cat((first.indices, second.indices), dim=1),
        jacobian=torch.cat((first.jacobian, second.jacobian), dim=1),
        target=torch.cat((first.target, second.target), dim=1),
        valid=torch.cat((first.valid, second.valid), dim=1),
        num_parameters=first.num_parameters,
    )
