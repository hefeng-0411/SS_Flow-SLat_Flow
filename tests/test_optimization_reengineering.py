from __future__ import annotations

from types import SimpleNamespace

import torch

from geoss.integration.real_trellis_pipeline import _sample_final_without_trajectory
from geoss.integration.trellis_ss_hook import ss_grid_to_tokens
from geoss.models.cross_view_evidence_aggregator import CrossViewEvidenceAggregator
from geoss.models.ss_velocity_adapter import SSVelocityAdapter
from geoss.ops.flow_matching import flow_matching_pair
from geoss.utils.early_stopping import EarlyStopper
from geoss.utils.training_control import ControlConfig, WarmupCosineController
from scripts.train_sparse_ray_geoss import _stationary_geoss_loss
from scripts.train_sparse_ray_ss_velocity import (
    _bound_scheduler_intervention,
    _build_lr_scheduler,
    _sample_flow_timesteps,
    _stage2_residual_objective,
)


def test_flow_matching_fallback_matches_trellis_equations() -> None:
    torch.manual_seed(4)
    x0 = torch.randn(3, 2, 4, 4, 4)
    noise = torch.randn_like(x0)
    timestep = torch.tensor([0.1, 0.5, 0.9])
    sigma_min = 1.0e-5

    x_t, velocity, backend = flow_matching_pair(
        x0, noise, timestep, sigma_min, backend="torch"
    )
    t = timestep.view(3, 1, 1, 1, 1)
    expected_x_t = (1.0 - t) * x0 + (sigma_min + (1.0 - sigma_min) * t) * noise
    expected_velocity = (1.0 - sigma_min) * noise - x0

    assert backend == "torch"
    assert torch.allclose(x_t, expected_x_t)
    assert torch.allclose(velocity, expected_velocity)
    assert not x_t.requires_grad
    assert not velocity.requires_grad


def test_grid_to_token_routing_is_zero_copy() -> None:
    grid = torch.randn(2, 8, 4, 4, 4)
    tokens = ss_grid_to_tokens(grid)

    assert tokens.shape == (2, 64, 8)
    assert tokens.untyped_storage().data_ptr() == grid.untyped_storage().data_ptr()
    assert not tokens.is_contiguous()


def test_smooth_trust_projection_keeps_gradient_beyond_boundary() -> None:
    adapter = SSVelocityAdapter(
        latent_dim=2,
        geo_dim=4,
        hidden_dim=8,
        num_heads=2,
        local_attention=False,
        activation_checkpointing=False,
    )
    with torch.no_grad():
        adapter.delta_head[-1].weight.zero_()
        adapter.delta_head[-1].bias.fill_(0.5)
    out = adapter(
        torch.randn(1, 3, 2),
        torch.randn(1, 5, 4),
        torch.ones(1, 5, 1),
        torch.tensor([1000.0]),
        torch.zeros(1, 3, 2),
    )
    out["delta_v_geo"].sum().backward()

    assert bool(out["debug"]["clipping_ratio"] > 0)
    assert adapter.delta_head[-1].bias.grad is not None
    assert bool((adapter.delta_head[-1].bias.grad.abs() > 0).all())


def test_gate_consistent_auxiliary_has_same_optimum_as_effective_residual() -> None:
    gate = torch.full((1, 4, 1), 0.25)
    target = torch.full((1, 4, 2), 0.5)
    raw = torch.full((1, 4, 2), 2.0, requires_grad=True)
    effective = gate * raw
    terms = _stage2_residual_objective(
        raw,
        effective,
        target,
        alpha_t=torch.ones(1, 1, 1),
        token_confidence=gate,
        token_mask=None,
        frozen_base_mse=target.square().mean(),
        auxiliary_weight=0.25,
        normalize=True,
    )
    terms["optimization_residual"].backward()

    assert torch.allclose(terms["effective_mse"], torch.zeros(()))
    assert torch.allclose(terms["unclipped_effective_mse"], torch.zeros(()))
    assert torch.allclose(raw.grad, torch.zeros_like(raw.grad))


def test_early_stop_control_ignores_known_scheduler_warmup() -> None:
    stopper = EarlyStopper(
        enabled=True,
        metric="loss",
        min_control_updates=10,
    )
    for step in range(1, 11):
        status = stopper.update(
            {"step": step, "loss": float(step), "global_batch_size": 4}
        )
        assert status.reason == "control_warmup"
        assert status.lr_multiplier is None
        assert not status.should_stop
    assert stopper.total_updates == 0

    status = stopper.update({"step": 11, "loss": 11.0, "global_batch_size": 4})
    assert status.reason != "control_warmup"
    assert stopper.total_updates == 1


def test_lr_interventions_cannot_collapse_or_explode_schedule() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-3)
    controller = WarmupCosineController(
        optimizer,
        ControlConfig(
            min_stability_multiplier=0.1,
            max_stability_multiplier=2.0,
        ),
    )
    for _ in range(20):
        controller.intervene(0.1)
    assert controller.stability_multiplier == 0.1
    for _ in range(20):
        controller.intervene(10.0)
    assert controller.stability_multiplier == 2.0

    lambda_scheduler = _build_lr_scheduler(
        optimizer,
        total_updates=100,
        warmup_updates=10,
        min_lr_ratio=0.05,
    )
    requested = 0.01
    for _ in range(5):
        applied = _bound_scheduler_intervention(lambda_scheduler, requested)
        lambda_scheduler.control_multiplier *= applied
    assert lambda_scheduler.control_multiplier == 0.1


def test_logit_normal_timestep_sampler_matches_trellis_support() -> None:
    torch.manual_seed(1)
    timesteps = _sample_flow_timesteps(
        20_000,
        torch.device("cpu"),
        mode="logit_normal",
        mean=0.0,
        std=1.0,
    )
    assert bool(((timesteps > 0.0) & (timesteps < 1.0)).all())
    assert abs(float(timesteps.mean()) - 0.5) < 0.01


def test_final_only_sampler_does_not_call_trajectory_retaining_api() -> None:
    class FakeFlowEuler:
        sigma_min = 1.0e-5

        def __init__(self) -> None:
            self.calls = 0

        def sample_once(self, model, sample, t, t_prev, **kwargs):
            self.calls += 1
            return SimpleNamespace(pred_x_prev=sample + (t - t_prev))

        def sample(self, *args, **kwargs):  # pragma: no cover - must remain unused
            raise AssertionError("trajectory-retaining sampler API was called")

    sampler = FakeFlowEuler()
    result = _sample_final_without_trajectory(
        sampler,
        object(),
        torch.zeros(1),
        steps=4,
        rescale_t=1.0,
        verbose=False,
    )

    assert sampler.calls == 4
    assert torch.allclose(result, torch.ones(1))


def test_stage1_stationary_loss_has_fixed_physical_weights() -> None:
    terms = {
        "occupancy": torch.tensor(2.0),
        "free_space": torch.tensor(3.0),
        "projection": torch.tensor(5.0),
        "confidence": torch.tensor(7.0),
        "sparsity": torch.tensor(11.0),
    }
    expected = 2.0 + 0.75 * 3.0 + 5.0 + 0.5 * 7.0 + 0.25 * 11.0
    assert torch.allclose(_stationary_geoss_loss(terms), torch.tensor(expected))


def test_stage1_attention_recomputation_preserves_gradients() -> None:
    module = CrossViewEvidenceAggregator(
        anchor_dim=8,
        evidence_dim=6,
        hidden_dim=8,
        num_heads=2,
        activation_checkpointing=True,
    ).train()
    views = torch.randn(2, 5, 3, 6, requires_grad=True)
    anchors = torch.randn(2, 5, 8, requires_grad=True)
    valid = torch.ones(2, 5, 3, 1)
    output = module(views, anchors, valid)
    output["occ_evidence"].mean().backward()

    assert views.grad is not None and bool(torch.isfinite(views.grad).all())
    assert anchors.grad is not None and bool(torch.isfinite(anchors.grad).all())
