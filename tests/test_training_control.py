from __future__ import annotations

import torch

from geoss.losses.confidence_loss import confidence_calibration_loss
from geoss.utils.training_control import BoundedLossBalancer, ControlConfig, WarmupCosineController


def test_confidence_calibration_rejects_collapsed_gate_on_correct_anchors():
    error = torch.zeros(16, 1)
    collapsed = confidence_calibration_loss(torch.full_like(error, 0.01), error)["loss"]
    calibrated = confidence_calibration_loss(torch.full_like(error, 0.90), error)["loss"]
    assert calibrated < collapsed


def test_lr_contraction_survives_following_scheduler_steps():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([{"params": [parameter], "lr": 1e-3}])
    control = WarmupCosineController(optimizer, ControlConfig(warmup_steps=10, total_steps=100))
    before = control.step(20)[0]
    control.contract(0.5)
    after = control.step(21)[0]
    unconstrained_next = 1e-3 * control.factor(21)
    assert after == unconstrained_next * 0.5
    assert after < before


def test_loss_balancer_bounds_authority_for_collapsing_loss():
    config = ControlConfig(loss_weight_min=0.25, loss_weight_max=4.0)
    balancer = BoundedLossBalancer(("geometry", "free_space"), config)
    total, weights = balancer.combine(
        {"geometry": torch.tensor(1.0), "free_space": torch.tensor(1e-12)}
    )
    assert torch.isfinite(total)
    assert weights["free_space"] == config.loss_weight_max
    assert config.loss_weight_min <= weights["geometry"] <= config.loss_weight_max
