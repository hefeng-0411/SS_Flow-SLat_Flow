from __future__ import annotations

import math

import pytest

from geoss.utils.early_stopping import (
    EarlyStopper,
    handoff_contract,
    quarantine_legacy_best_checkpoint,
)


def _record(
    step: int,
    loss: float,
    *,
    raw_residual: float = 0.3,
    effective_residual: float = 0.2,
) -> dict:
    return {
        "step": step,
        "loss": loss,
        "loss_aux": 0.4 * loss,
        "global_batch_size": 10,
        "loss_raw_residual": raw_residual,
        "loss_effective_residual": effective_residual,
        "adapter_grad_norms": {"weight": 1.0e-3},
    }


def test_step_303_descent_is_not_a_plateau() -> None:
    stopper = EarlyStopper(enabled=True, metric="loss")
    for step in range(1, 304):
        loss = 2.0 * math.exp(-step / 300.0) + 0.2 + 0.01 * math.sin(step)
        status = stopper.update(_record(step, loss))

    assert not status.should_stop
    assert status.regime == "rapid_descent"
    assert status.opportunity_ratio is not None
    assert status.opportunity_ratio > 1.0


def test_stationarity_must_survive_a_causal_lr_probe_before_stopping() -> None:
    stopper = EarlyStopper(enabled=True, metric="loss")
    saw_probe = False
    stopped = None
    for step in range(1, 1600):
        baseline = (
            2.0 * math.exp(-step / 45.0) + 0.25 if step < 180 else 0.25
        )
        status = stopper.update(
            _record(step, baseline + 0.003 * math.sin(1.7 * step))
        )
        saw_probe = saw_probe or status.lr_multiplier is not None
        if status.should_stop:
            stopped = status
            break

    assert saw_probe
    assert stopped is not None
    assert stopped.reason in {
        "evidence_limited_convergence",
        "lr_probe_rejected_convergence",
    }
    assert stopped.selected_checkpoint == "best"
    assert stopped.handoff_ready


def test_single_adverse_spike_is_quarantined_not_counted_as_degradation() -> None:
    stopper = EarlyStopper(enabled=True, metric="loss")
    statuses = {}
    for step in range(1, 100):
        loss = 1.0 - 0.003 * step + 0.002 * math.sin(step)
        if step == 75:
            loss = 4.0
        status = stopper.update(_record(step, loss))
        if step in {75, 76}:
            statuses[step] = status

    assert statuses[75].reason == "transient_candidate"
    assert not statuses[75].should_stop
    assert statuses[76].reason == "transient_spike_rejected"
    assert not statuses[76].should_stop


def test_persistent_degradation_requires_an_independent_recovery_segment() -> None:
    stopper = EarlyStopper(enabled=True, metric="loss")
    reasons = []
    for step in range(1, 200):
        loss = 1.0 - 0.004 * step if step < 65 else 0.74 + 0.01 * (step - 65)
        status = stopper.update(_record(step, loss))
        reasons.append(status.reason)
        if status.should_stop:
            break

    assert "self_calibrating_recovery_probe" in reasons
    assert status.reason == "confirmed_degeneration"
    assert status.selected_checkpoint == "best"
    assert not status.handoff_ready


def test_numerically_null_adapter_effect_blocks_cross_stage_handoff() -> None:
    stopper = EarlyStopper(enabled=True, metric="loss")
    for step in range(1, 80):
        status = stopper.update(
            _record(
                step,
                0.7 - 0.002 * step,
                raw_residual=0.29041650891304016,
                effective_residual=0.2903973162174225,
            )
        )

    assert not status.handoff_ready
    assert not status.diagnostics["causal_significant"]
    ready, reasons = handoff_contract(status.as_dict())
    assert not ready
    assert "adapter_effect_below_numerical_resolution" in reasons


def test_state_round_trip_preserves_the_next_decision() -> None:
    first = EarlyStopper(enabled=True, metric="loss")
    for step in range(1, 96):
        first.update(_record(step, 1.0 / (1.0 + 0.01 * step)))

    resumed = EarlyStopper(enabled=True, metric="loss")
    resumed.load_state_dict(first.state_dict())
    next_record = _record(96, 1.0 / 1.96)
    expected = first.update(next_record)
    actual = resumed.update(next_record)

    assert actual.as_dict() == expected.as_dict()


def test_legacy_best_is_quarantined_before_adaptive_replacement(tmp_path) -> None:
    best_path = tmp_path / "adapter_best.pt"
    best_path.write_bytes(b"legacy")
    legacy_state = {
        "early_stopper": {
            "metric": "loss",
            "mode": "min",
            "best_score": 0.25337448716163635,
            "best_step": 177,
            "seen_steps": 301,
        }
    }
    quarantined = quarantine_legacy_best_checkpoint(best_path, legacy_state)

    assert quarantined is not None
    assert best_path.with_name("adapter_best_legacy_unverified.pt").read_bytes() == b"legacy"

    stopper = EarlyStopper(enabled=True, metric="loss")
    stopper.load_state_dict(legacy_state["early_stopper"])
    status = stopper.update(_record(302, 0.5808138251304626))

    assert status.is_best


@pytest.mark.parametrize(
    ("record_update", "reason"),
    [
        ({"loss": float("nan")}, "nonfinite_primary_metric"),
        (
            {"adapter_grad_norms": {"weight": 0.0}},
            "structural_zero_gradient",
        ),
        (
            {"adapter_grad_norms": {"weight": float("inf")}},
            "nonfinite_gradient",
        ),
    ],
)
def test_structural_failures_terminate_safely(
    record_update: dict,
    reason: str,
) -> None:
    stopper = EarlyStopper(enabled=True, metric="loss")
    record = _record(1, 1.0)
    record.update(record_update)
    status = stopper.update(record)

    assert status.should_stop
    assert status.reason == reason
    assert status.selected_checkpoint == "best"
