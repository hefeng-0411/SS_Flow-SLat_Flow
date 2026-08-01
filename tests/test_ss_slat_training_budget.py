from __future__ import annotations

from pathlib import Path

import pytest
import torch

import argparse
import json

from geoss.utils.config import apply_config_mappings, load_config, str2bool
from geoss.utils.training_budget import (
    compute_training_budget,
    defer_nonfatal_early_stop_until_minimum_exposure,
    enforce_minimum_dataset_passes,
)
from geoss.utils.early_stopping import EarlyStopStatus
from geoss.integration.trellis_hub import resolve_dinov2_repo, resolve_torch_hub_dir


ROOT = Path(__file__).resolve().parents[1]


def test_meshfleet_budget_matches_the_declared_distributed_topologies() -> None:
    ss = compute_training_budget(
        dataset_objects=1357,
        world_size=2,
        microbatch_per_rank=1,
        grad_accum_steps=1,
        planned_optimizer_updates=1000,
    )
    assert ss.samples_per_rank_per_sampler_epoch == 679
    assert ss.padded_samples_per_sampler_epoch == 1
    assert ss.optimizer_updates_per_sampler_epoch == 679
    assert ss.nominal_sample_presentations == 2000
    assert ss.nominal_dataset_passes == pytest.approx(2000 / 1357)

    slat = compute_training_budget(
        dataset_objects=1357,
        world_size=4,
        microbatch_per_rank=1,
        grad_accum_steps=1,
        planned_optimizer_updates=2,
    )
    assert slat.samples_per_rank_per_sampler_epoch == 340
    assert slat.padded_samples_per_sampler_epoch == 3
    assert slat.optimizer_updates_per_sampler_epoch == 340
    assert slat.nominal_sample_presentations == 8
    assert slat.nominal_dataset_passes == pytest.approx(8 / 1357)


def test_corrected_manifest_budgets_match_production_configs() -> None:
    ss = compute_training_budget(
        dataset_objects=1205,
        world_size=2,
        microbatch_per_rank=1,
        grad_accum_steps=1,
        planned_optimizer_updates=1000,
    )
    assert ss.samples_per_rank_per_sampler_epoch == 603
    assert ss.padded_samples_per_sampler_epoch == 1
    assert ss.optimizer_updates_per_sampler_epoch == 603
    assert ss.nominal_dataset_passes == pytest.approx(2000 / 1205)

    slat = compute_training_budget(
        dataset_objects=1205,
        world_size=4,
        microbatch_per_rank=1,
        grad_accum_steps=1,
        planned_optimizer_updates=1000,
    )
    assert slat.samples_per_rank_per_sampler_epoch == 302
    assert slat.padded_samples_per_sampler_epoch == 3
    assert slat.optimizer_updates_per_sampler_epoch == 302
    assert slat.nominal_dataset_passes == pytest.approx(4000 / 1205)


def test_two_step_real_slat_budget_is_rejected_with_actionable_arithmetic() -> None:
    budget = compute_training_budget(
        dataset_objects=1357,
        world_size=4,
        microbatch_per_rank=1,
        grad_accum_steps=1,
        planned_optimizer_updates=2,
    )
    with pytest.raises(RuntimeError, match=r"[Aa]t least 340 updates"):
        enforce_minimum_dataset_passes(
            budget,
            minimum_dataset_passes=1.0,
            stage="SLAT",
        )
    enforce_minimum_dataset_passes(
        budget,
        minimum_dataset_passes=0.0,
        stage="SLAT smoke test",
    )


def test_real_configs_have_explicit_total_budgets_and_existing_local_roots() -> None:
    for name in ("real_train_ss.yaml", "real_train_slat_only.yaml"):
        cfg = load_config(ROOT / "configs" / name)
        assert cfg["steps"] >= 1000
        assert cfg["steps_are_total"] is True
        assert cfg["minimum_dataset_passes"] >= 1.0
        assert Path(cfg["trellis"]["root"]).is_dir()
        assert Path(cfg["dataset"]["root"]).exists()
        manifest = json.loads(Path(cfg["dataset"]["train_manifest"]).read_text(encoding="utf-8"))
        assert manifest["count"] == 1205
        assert Path(cfg["vggt"]["root"]).is_dir()


def test_real_ss_and_slat_clis_have_no_implicit_two_step_budget() -> None:
    for relative in (
        "scripts/train_sparse_ray_ss_velocity.py",
        "scripts/train_geovis_slat.py",
        "scripts/train_geovis_slat_joint.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert 'parser.add_argument("--steps", type=int, default=2)' not in source


def test_trellis_active_support_boundary_is_nondifferentiable() -> None:
    occupancy_logits = torch.tensor([-1.0, 2.0, 0.5], requires_grad=True)
    integer_support = torch.argwhere(occupancy_logits > 0).int()
    assert integer_support.requires_grad is False
    assert integer_support.grad_fn is None


def test_slat_trainer_records_teacher_support_and_absent_ss_connection() -> None:
    source = (ROOT / "scripts" / "train_geovis_slat.py").read_text(encoding="utf-8")
    assert '"coordinate_source": "cached_trellis_slat_teacher"' in source
    assert '"upstream_ss_checkpoint": None' in source
    assert '"train_inference_support_match": False' in source
    assert '"vggt_root": cfg.get("vggt_root") or vggt.get("root")' in source


def test_statistical_early_stop_cannot_preempt_the_minimum_data_exposure() -> None:
    budget = compute_training_budget(
        dataset_objects=1357,
        world_size=4,
        microbatch_per_rank=1,
        grad_accum_steps=1,
        planned_optimizer_updates=1000,
    )
    status = EarlyStopStatus(
        enabled=True,
        should_stop=True,
        is_best=False,
        metric="loss",
        raw_value=1.0,
        score=1.0,
        best_score=1.0,
        bad_steps=0,
        reason="evidence_limited_convergence",
    )
    gate = defer_nonfatal_early_stop_until_minimum_exposure(
        status,
        budget=budget,
        step=50,
        minimum_dataset_passes=1.0,
    )
    assert gate == {
        "deferred": True,
        "original_reason": "evidence_limited_convergence",
        "required_updates": 340,
        "current_step": 50,
    }
    assert status.should_stop is False
    assert status.reason == "minimum_dataset_exposure_gate"


def test_minimum_exposure_gate_never_suppresses_a_structural_failure() -> None:
    budget = compute_training_budget(
        dataset_objects=1357,
        world_size=4,
        microbatch_per_rank=1,
        grad_accum_steps=1,
        planned_optimizer_updates=1000,
    )
    status = EarlyStopStatus(
        enabled=True,
        should_stop=True,
        is_best=False,
        metric="loss",
        raw_value=1.0,
        score=1.0,
        best_score=1.0,
        bad_steps=0,
        reason="nonfinite_gradient",
    )
    gate = defer_nonfatal_early_stop_until_minimum_exposure(
        status,
        budget=budget,
        step=1,
        minimum_dataset_passes=1.0,
    )
    assert gate["deferred"] is False
    assert status.should_stop is True


def test_local_dinov2_checkout_is_resolved_without_a_network_probe(tmp_path) -> None:
    hub = tmp_path / "hub"
    repo = hub / "facebookresearch_dinov2_main"
    repo.mkdir(parents=True)
    (repo / "hubconf.py").write_text("# local test repository\n", encoding="utf-8")

    class Args:
        torch_hub_dir = str(hub)
        dinov2_repo = None

    resolved_hub = resolve_torch_hub_dir(Args())
    assert resolved_hub == hub
    assert resolve_dinov2_repo(Args(), resolved_hub) == repo


def test_explicit_cli_boolean_wins_even_when_it_equals_parser_default() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--early_stop", type=str2bool, default=False)
    args = parser.parse_args(["--early_stop", "false"])
    apply_config_mappings(
        args,
        parser,
        {"early_stop": True},
        argv=["--early_stop", "false"],
    )
    assert args.early_stop is False
