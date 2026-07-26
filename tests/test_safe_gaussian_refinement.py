from __future__ import annotations

from scripts.refine_gaussian_conditioning import (
    _interleaved_validation_indices,
    _is_safe_validation_improvement,
)
from scripts.infer_native_trellis_multiview import _parse_candidate_seeds


def _metrics(objective: float, foreground: float, ssim: float, mask: float):
    return {
        "objective": objective,
        "foreground_l1": foreground,
        "ssim_loss": ssim,
        "mask_loss": mask,
    }


def test_interleaved_selection_views_are_unique_and_distributed():
    assert _interleaved_validation_indices(8, 2).tolist() == [2, 6]
    assert _interleaved_validation_indices(12, 3).tolist() == [2, 6, 10]


def test_safe_selection_requires_joint_non_regression():
    baseline = _metrics(1.0, 0.4, 0.3, 0.2)
    assert _is_safe_validation_improvement(
        _metrics(0.95, 0.39, 0.29, 0.2),
        baseline,
        baseline,
        min_relative_improvement=0.002,
        tolerance=1e-4,
    )
    assert not _is_safe_validation_improvement(
        _metrics(0.95, 0.41, 0.29, 0.2),
        baseline,
        baseline,
        min_relative_improvement=0.002,
        tolerance=1e-4,
    )
    assert not _is_safe_validation_improvement(
        _metrics(0.999, 0.39, 0.29, 0.2),
        baseline,
        baseline,
        min_relative_improvement=0.002,
        tolerance=1e-4,
    )


def test_candidate_seed_parser_is_deterministic_and_deduplicated():
    assert _parse_candidate_seeds(None, 42) == [42]
    assert _parse_candidate_seeds("42, 7,42", 1) == [42, 7]
