from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_main_pipeline_keeps_trellis_backbone_and_no_baseline_import():
    src = _read("geoss/integration/real_trellis_pipeline.py")
    assert "TrellisImageTo3DPipeline" in src
    assert "sample_sparse_structure" in src
    assert "sample_slat" in src
    assert "decode_slat" in src
    assert "geoss.baselines" not in src
    assert "splatfacto" not in src.lower()


def test_real_training_paths_do_not_create_random_condition_or_zero_base():
    ss = _read("scripts/train_sparse_ray_ss_velocity.py")
    assert "cond = torch.randn" not in ss
    assert "torch.zeros_like(ss_latent_tokens)" not in ss
    assert "target_residual = (target_v - v_base).detach()" in ss
    slat = _read("scripts/train_geovis_slat.py")
    assert "trellis_slat_base_velocity" in slat
    assert "zero base velocity is only allowed in --dry_run" in slat


def test_sparse_adapter_refuses_missing_base_velocity_when_tokens_exist():
    src = _read("geoss/models/sparse_ray_geoss_adapter.py")
    assert "requires v_base when ss_latent_tokens are provided" in src
    assert "torch.zeros_like(ss_latent_tokens)" not in src


def test_gsplat_and_nerfstudio_are_auxiliary_only():
    renderer = _read("geoss/renderers/gsplat_renderer.py")
    assert "rasterization" in renderer
    assert "optimizer" not in renderer.lower()
    baseline = _read("geoss/baselines/run_splatfacto_baseline.py")
    assert "ns-train" in baseline
    pipeline = _read("geoss/integration/real_trellis_pipeline.py")
    assert "ns-train" not in pipeline
