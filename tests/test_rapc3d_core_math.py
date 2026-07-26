from __future__ import annotations

import torch

from geoss.reconstruction.decoders import UnifiedSurfaceDecoder
from geoss.reconstruction.fields import (
    PosteriorFieldState,
    make_canonical_grid,
    query_posterior_field,
)
from geoss.reconstruction.posterior import (
    MatrixFreeAuthorityProjector,
    SparseLinearConstraints,
)
from geoss.reconstruction.visibility import SDFRayRenderer


def _linear_field(resolution: int = 7) -> PosteriorFieldState:
    points = make_canonical_grid(resolution, batch_size=1)
    sdf = points[..., 0] + 2.0 * points[..., 1] + 3.0 * points[..., 2]
    volume = torch.zeros(1, 38, resolution, resolution, resolution)
    volume[:, 0] = sdf.reshape(1, resolution, resolution, resolution)
    volume[:, 1] = -8.0
    volume[:, 4] = 5.0
    return PosteriorFieldState(volume)


def _sphere_field(resolution: int = 17, requires_grad: bool = False) -> PosteriorFieldState:
    points = make_canonical_grid(resolution, batch_size=1)
    sdf = points.norm(dim=-1) - 0.28
    volume = torch.zeros(1, 38, resolution, resolution, resolution)
    volume[:, 0] = sdf.reshape(1, resolution, resolution, resolution)
    volume[:, 1] = -8.0
    volume[:, 4] = 6.0
    volume[:, 8:11] = torch.logit(torch.tensor(0.5))
    volume.requires_grad_(requires_grad)
    return PosteriorFieldState(volume)


def test_trilinear_sdf_has_exact_linear_spatial_jacobian():
    field = _linear_field()
    points = torch.tensor(
        [[[-0.31, -0.17, 0.22], [0.11, 0.29, -0.27], [0.0, 0.0, 0.0]]]
    )
    samples = query_posterior_field(field, points, require_gradient=True)
    expected_value = points[..., 0:1] + 2.0 * points[..., 1:2] + 3.0 * points[..., 2:3]
    assert torch.allclose(samples.sdf_mean, expected_value, atol=1e-5)
    assert samples.spatial_gradient is not None
    expected_gradient = torch.tensor([1.0, 2.0, 3.0]).reshape(1, 1, 3).expand_as(
        samples.spatial_gradient
    )
    assert torch.allclose(samples.spatial_gradient, expected_gradient, atol=1e-5)


def test_prior_update_is_observation_tangent_and_free_space_feasible():
    surface = SparseLinearConstraints(
        indices=torch.tensor([[[0]]]),
        jacobian=torch.ones(1, 1, 1),
        target=torch.zeros(1, 1),
        valid=torch.ones(1, 1, dtype=torch.bool),
        num_parameters=4,
    )
    free = SparseLinearConstraints(
        indices=torch.tensor([[[1]]]),
        jacobian=torch.ones(1, 1, 1),
        target=torch.zeros(1, 1),
        valid=torch.ones(1, 1, dtype=torch.bool),
        num_parameters=4,
    )
    projector = MatrixFreeAuthorityProjector(
        cg_iterations=64,
        cg_tolerance=1e-8,
        damping=1e-8,
        projection_passes=2,
    )
    result = projector(
        torch.tensor([[0.2, 0.01, 0.0, 0.0]]),
        torch.tensor([[1.0, -1.0, 2.0, 3.0]]),
        metric_diagonal=torch.ones(1, 4),
        surface_constraints=surface,
        free_constraints=free,
        free_margin=0.002,
    )
    assert surface.apply(result.value).abs().max() < 1e-5
    assert free.apply(result.value).min() >= 0.002 - 1e-5
    assert surface.apply(result.projected_prior_update).abs().max() < 1e-5
    assert torch.allclose(result.projected_prior_update[:, 2:], torch.tensor([[2.0, 3.0]]), atol=1e-5)


def test_physical_visibility_orders_surface_hit_before_background_ray():
    field = _sphere_field()
    origins = torch.tensor([[[0.0, 0.0, -1.0], [0.48, 0.0, -1.0]]])
    directions = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
    renderer = SDFRayRenderer(num_samples=128, beta=0.01, eta=12.0)
    rendered = renderer(field, origins, directions)
    assert rendered["opacity"][0, 0, 0] > 0.9
    assert rendered["opacity"][0, 1, 0] < 0.1
    assert rendered["depth"][0, 0, 0] < 1.0


def test_surface_gaussians_are_tethered_and_covariance_is_positive_definite():
    field = _sphere_field(resolution=17)
    decoder = UnifiedSurfaceDecoder(max_gaussians=1024, newton_steps=3)
    gaussian = decoder(field)
    assert gaussian.valid.any()
    valid = gaussian.valid
    assert gaussian.sdf_at_means[valid].abs().mean() < 2e-3
    eigenvalues = torch.linalg.eigvalsh(gaussian.covariance[valid])
    assert torch.isfinite(eigenvalues).all()
    assert eigenvalues.min() > 0.0
    identity = gaussian.quaternions.square().sum(dim=-1)
    assert torch.allclose(identity[valid], torch.ones_like(identity[valid]), atol=1e-5)
    mesh = decoder.extract_meshes(field, resolution=17)[0]
    assert mesh.vertices.numel() > 0 and mesh.faces.numel() > 0
    mesh_samples = query_posterior_field(
        field,
        mesh.vertices[None],
        require_gradient=False,
    )
    assert mesh_samples.sdf_mean.abs().mean() < 3e-3


def test_surface_asset_loss_reaches_the_continuous_sdf_state():
    field = _sphere_field(resolution=13, requires_grad=True)
    decoder = UnifiedSurfaceDecoder(max_gaussians=512, newton_steps=2)
    gaussian = decoder(field)
    loss = gaussian.sdf_at_means[gaussian.valid].abs().mean() + 0.01 * gaussian.scales.square().mean()
    loss.backward()
    assert field.volume.grad is not None
    assert torch.isfinite(field.volume.grad).all()
    assert field.volume.grad[:, 0].abs().sum() > 0.0
