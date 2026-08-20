from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_training_summaries_mark_not_for_metrics():
    for rel in (
        "scripts/train_sparse_ray_ss_velocity.py",
        "scripts/train_geovis_slat.py",
        "scripts/train_sparse_ray_geoss.py",
    ):
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "not_for_paper_metrics" in src
