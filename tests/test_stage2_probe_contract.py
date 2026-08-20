from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.launch_meshfleet_multigpu_sequence import (
    FailureKind,
    Stage,
    _resolved_grad_accum_steps,
    _probe_too_hot,
    _torchrun_command,
    classify_failure,
)
from scripts.train_sparse_ray_ss_velocity import (
    _build_lr_scheduler,
    _save_velocity_checkpoint,
)


def _stage(tmp_path: Path) -> Stage:
    return Stage(
        name="stage2",
        script="scripts/train_sparse_ray_ss_velocity.py",
        config="configs/sparse_ray_ss_velocity.yaml",
        output_dir=tmp_path / "stage2_ss_velocity",
        steps=100_000,
        max_batch_size=8,
        resume_path=tmp_path / "stage2_ss_velocity" / "ss_velocity_adapter_last.pt",
        best_path=tmp_path / "stage2_ss_velocity" / "ss_velocity_adapter_best.pt",
        extra_args=[
            "--minimum_dataset_passes", "1",
            "--auto_expand_training_budget", "true",
            "--grad_accum_steps", "1",
            "--early_stop", "true",
            "--save_best", "true",
            "--fault_tolerant_save_every", "25",
            "--adaptive_batch", "true",
            "--adaptive_max_batch_size", "8",
        ],
    )


def _value(command: list[str], flag: str) -> str:
    positions = [index for index, token in enumerate(command) if token == flag]
    assert len(positions) == 1, f"{flag} must be serialized exactly once: {command}"
    return command[positions[0] + 1]


def test_probe_command_has_first_class_nonpromotable_semantics(tmp_path: Path) -> None:
    stage = _stage(tmp_path)
    command = _torchrun_command(
        stage,
        nproc=4,
        batch_size=4,
        steps=3,
        output_dir=tmp_path / "_probe_bs4",
        resume=False,
        execution_mode="probe",
        grad_accum_steps=2,
    )
    assert _value(command, "--execution_mode") == "probe"
    assert _value(command, "--minimum_dataset_passes") == "0"
    assert _value(command, "--auto_expand_training_budget") == "false"
    assert _value(command, "--grad_accum_steps") == "2"
    assert _value(command, "--adaptive_batch") == "false"
    assert _value(command, "--early_stop") == "false"
    assert _value(command, "--save_best") == "false"
    assert _value(command, "--fault_tolerant_save_every") == "0"
    assert "--resume" not in command


def test_real_command_restores_training_floor_and_horizon(tmp_path: Path) -> None:
    stage = _stage(tmp_path)
    command = _torchrun_command(
        stage,
        nproc=4,
        batch_size=8,
        steps=100_000,
        output_dir=stage.output_dir,
        resume=False,
        execution_mode="train",
        grad_accum_steps=1,
    )
    assert _value(command, "--execution_mode") == "train"
    assert _value(command, "--minimum_dataset_passes") == "1"
    assert _value(command, "--auto_expand_training_budget") == "true"
    assert _value(command, "--steps") == "100000"
    assert _value(command, "--adaptive_max_batch_size") == "8"


@pytest.mark.parametrize(
    ("microbatch", "expected_accumulation"),
    [(8, 1), (4, 2), (2, 4), (1, 8)],
)
def test_batch_fallback_preserves_target_effective_scale(
    tmp_path: Path, microbatch: int, expected_accumulation: int
) -> None:
    assert _resolved_grad_accum_steps(_stage(tmp_path), microbatch) == expected_accumulation


@pytest.mark.parametrize(
    ("text", "returncode", "expected"),
    [
        ("SS training budget is not a real dataset pass: minimum_dataset_passes=1", 1, FailureKind.CONFIGURATION),
        ("UID manifest contains unavailable objects", 1, FailureKind.DATASET),
        ("Missing key(s) in state_dict", 1, FailureKind.CHECKPOINT),
        ("torch.cuda.OutOfMemoryError: CUDA out of memory", 1, FailureKind.CUDA_OOM),
        ("ProcessGroupNCCL collective operation timeout", 1, FailureKind.DISTRIBUTED),
        ("adapter contains NaN", 1, FailureKind.NONFINITE),
        ("KeyboardInterrupt", 130, FailureKind.INTERRUPTED),
        ("arbitrary failure", 1, FailureKind.UNKNOWN),
    ],
)
def test_typed_failure_classification(text: str, returncode: int, expected: FailureKind) -> None:
    assert classify_failure(text, returncode) is expected


def test_nccl_failure_is_never_classified_as_oom() -> None:
    assert classify_failure("ProcessGroupNCCL: nccl error", 1) is FailureKind.DISTRIBUTED


def test_probe_headroom_uses_logged_vram_utilization() -> None:
    class Args:
        probe_max_vram_util = 0.96

    assert _probe_too_hot('{"memory":{"vram_utilization":0.97}}', Args()) is True
    assert _probe_too_hot('{"memory":{"vram_utilization":0.90}}', Args()) is False


def test_probe_checkpoint_write_is_rejected_before_serialization(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Refusing to write"):
        _save_velocity_checkpoint(
            tmp_path / "ss_velocity_adapter_best.pt",
            adapter=None,
            optimizer=None,
            scheduler=None,
            step=1,
            cfg={},
            early_stopper=None,
            early_status=None,
            execution_mode="probe",
        )
    assert not (tmp_path / "ss_velocity_adapter_best.pt").exists()


def test_scheduler_uses_resolved_horizon_and_lr_floor() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = _build_lr_scheduler(
        optimizer,
        total_updates=100,
        warmup_updates=10,
        min_lr_ratio=0.05,
    )
    assert scheduler.resolved_total_updates == 100
    assert scheduler.resolved_warmup_updates == 10
    for _ in range(100):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05, abs=2e-3)
