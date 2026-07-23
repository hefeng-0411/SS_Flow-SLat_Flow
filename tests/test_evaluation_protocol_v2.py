from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from geoss.datasets.meshfleet_trellis_dataset import MeshFleetTrellisDataset
from geoss.eval import render_metrics
from geoss.integration.vggt_geometry_wrapper import _expp1_confidence_to_probability
from geoss.io.asset_io import trellis_export_gaussian_to_internal
from geoss.metrics.geometry_metrics import geometry_metrics
from geoss.losses.render_losses import render_level_losses
from geoss.renderers.gsplat_renderer import SH_C0, _gaussian_tensors
from scripts.eval_geovis_slat import _camera_overlap_count
from scripts.evaluate_meshfleet_sequence import (
    _aggregate,
    _merge_asset_evaluation_rows,
    _parse_stage_vram_estimates,
    _run_with_peak_vram,
    _stage_capacities,
    _tag_sample_rows,
    _validate_parallel_launch,
    _validate_requested_checkpoints,
)
from scripts.infer_geovis_slat import _inference_only_batch
from scripts.infer_sparse_ray_geoss_ss import _context_only_batch


def test_windowed_ssim_identity_and_perturbation():
    image = torch.rand(2, 3, 32, 32)
    assert torch.allclose(render_metrics._ssim(image, image), torch.tensor(1.0), atol=1e-6)
    shifted = (image + 0.2).clamp(0, 1)
    assert render_metrics._ssim(image, shifted) < 1.0


def test_render_metrics_use_true_lpips_for_masked_and_full(monkeypatch):
    calls = []

    def fake_lpips(pred, gt):
        calls.append((pred.clone(), gt.clone()))
        return (pred - gt).square().mean()

    monkeypatch.setattr(render_metrics, "_lpips", fake_lpips)
    pred = torch.zeros(1, 3, 16, 16)
    gt = torch.ones_like(pred)
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, 4:12, 4:12] = 1
    metrics = render_metrics.image_render_metrics(pred, gt, mask, mask)
    assert len(calls) == 2
    assert metrics["LPIPS"] == pytest.approx(1.0)
    assert metrics["masked_LPIPS"] == pytest.approx(0.25)
    assert "foreground_L1" in metrics
    assert "DINO_similarity" not in metrics
    assert "multi_view_consistency" not in metrics


def test_geometry_metrics_are_symmetric_and_thresholded():
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    identical = geometry_metrics(points, points, threshold=0.01)
    assert identical["Chamfer Distance"] == pytest.approx(0.0)
    assert identical["F-score"] == pytest.approx(1.0)
    shifted = geometry_metrics(points + torch.tensor([0.1, 0.0, 0.0]), points, threshold=0.05)
    assert shifted["Chamfer Distance"] == pytest.approx(0.1, abs=1e-5)
    assert shifted["F-score"] == pytest.approx(0.0)


def test_camera_overlap_is_explicit():
    cond = torch.eye(4).repeat(2, 1, 1)
    cond[1, 0, 3] = 1.0
    heldout = torch.eye(4).repeat(2, 1, 1)
    heldout[0, 1, 3] = 2.0
    heldout[1, 0, 3] = 1.0
    assert _camera_overlap_count(cond, heldout) == 1


def test_dataset_explicit_heldout_set_and_background_compositing(tmp_path: Path):
    split = tmp_path / "test"
    uid = "object"
    for render_set, rgba in (("renders", (255, 0, 0, 0)), ("renders_eval_70", (255, 0, 0, 0))):
        folder = split / render_set / uid
        folder.mkdir(parents=True)
        Image.new("RGBA", (8, 8), rgba).save(folder / "000.png")
        transforms = {
            "frames": [
                {
                    "file_path": "000.png",
                    "camera_angle_x": 0.7,
                    "transform_matrix": np.eye(4).tolist(),
                }
            ]
        }
        (folder / "transforms.json").write_text(json.dumps(transforms), encoding="utf-8")
    voxels = split / "voxels"
    voxels.mkdir(parents=True)
    (voxels / f"{uid}.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n",
        encoding="utf-8",
    )
    dataset = MeshFleetTrellisDataset(
        tmp_path,
        split="test",
        render_set="renders_eval_70",
        num_views=1,
        image_size=8,
        background_color=(1.0, 1.0, 1.0),
        repeat_views_if_insufficient=False,
    )
    sample = dataset[0]
    assert sample["metadata"]["render_set"] == "renders_eval_70"
    assert torch.allclose(sample["images"], torch.ones_like(sample["images"]))
    assert sample["masks"].max() == 0


def test_dataset_filters_unusable_render_without_substituting_uid(tmp_path: Path):
    split = tmp_path / "test"
    for uid, has_image in (("a_bad", False), ("b_good", True)):
        folder = split / "renders" / uid
        folder.mkdir(parents=True)
        if has_image:
            Image.new("RGBA", (8, 8), (0, 0, 0, 255)).save(folder / "000.png")
        transforms = {
            "frames": [
                {
                    "file_path": "000.png",
                    "camera_angle_x": 0.7,
                    "transform_matrix": np.eye(4).tolist(),
                }
            ]
        }
        (folder / "transforms.json").write_text(json.dumps(transforms), encoding="utf-8")
    voxels = split / "voxels"
    voxels.mkdir(parents=True)
    for uid in ("a_bad", "b_good"):
        (voxels / f"{uid}.ply").write_text(
            "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n",
            encoding="utf-8",
        )
    dataset = MeshFleetTrellisDataset(tmp_path, split="test", num_views=1, image_size=8)
    assert len(dataset) == 1
    assert dataset[0]["uid"] == "b_good"
    assert {item["uid"] for item in dataset.discovery_skips} == {"a_bad"}
    with pytest.raises(KeyError, match="a_bad"):
        dataset.get_by_uid("a_bad")


def test_trellis_export_rotation_is_exactly_inverted():
    root_half = 2.0 ** -0.5
    exported = {
        "xyz": torch.tensor([[0.0, 0.0, 1.0]]),
        "rotation": torch.tensor([[root_half, root_half, 0.0, 0.0]]),
    }
    internal = trellis_export_gaussian_to_internal(exported)
    assert torch.allclose(internal["xyz"], torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)
    assert torch.allclose(internal["rotation"].abs(), torch.tensor([[1.0, 0.0, 0.0, 0.0]]), atol=1e-6)


def test_vggt_expp1_confidence_keeps_uncertainty_ordering():
    raw_expp1 = torch.tensor([1.0, 1.2, 2.0, 11.0])
    probability = _expp1_confidence_to_probability(raw_expp1)
    assert torch.all(probability[1:] > probability[:-1])
    assert torch.allclose(probability, torch.tensor([0.0, 1 / 6, 0.5, 10 / 11]), atol=1e-6)


def test_inference_boundary_removes_all_dataset_3d_supervision():
    batch = {
        "images": torch.zeros(1, 2, 3, 8, 8),
        "K": torch.eye(3).view(1, 1, 3, 3),
        "gt_occ": torch.ones(1, 8, 8, 8),
        "mesh_path": ["ground_truth.glb"],
        "ss_latent_grid": torch.ones(1, 8, 2, 2, 2),
        "ss_latent_tokens": torch.ones(1, 8, 8),
        "trellis_slat_feats": torch.ones(1, 4, 64),
        "trellis_slat_indices": torch.ones(1, 4, 4),
    }
    context = _context_only_batch(batch)
    slat_context = _inference_only_batch(batch)
    assert "images" in context and "K" in context
    for key in ("gt_occ", "mesh_path", "ss_latent_grid", "ss_latent_tokens", "trellis_slat_feats", "trellis_slat_indices"):
        assert key not in context
        assert key not in slat_context


def test_official_aggregate_excludes_invalid_protocol_rows():
    rows = [
        {"index": 0, "ablation": "method", "status": "ok", "asset_PSNR": 99.0, "asset_official_metrics": False, "population_manifested": True},
        {
            "index": 1,
            "ablation": "method",
            "status": "ok",
            "asset_PSNR": 20.0,
            "asset_SSIM": 0.8,
            "asset_LPIPS": 0.2,
            "asset_CD": 0.1,
            "asset_F-score": 0.9,
            "asset_official_metrics": True,
            "population_manifested": True,
        },
    ]
    summary = _aggregate(rows, expected_indices=[0, 1], expected_ablations=["method"])["by_ablation"]["method"]
    assert summary["official_num_objects"] == 1
    assert summary["official_complete"] is False
    assert summary["official_metrics"]["PSNR"]["mean"] == pytest.approx(20.0)


def test_unmanifested_population_cannot_become_official():
    rows = _tag_sample_rows(
        [{
            "ablation": "method",
            "status": "ok",
            "asset_PSNR": 20.0,
            "asset_SSIM": 0.8,
            "asset_LPIPS": 0.2,
            "asset_CD": 0.1,
            "asset_F-score": 0.9,
            "asset_official_metrics": True,
        }],
        index=7,
        gpu=None,
        uid="exact_uid",
        population_manifested=False,
    )
    assert rows[0]["uid"] == "exact_uid"
    summary = _aggregate(rows, expected_indices=[7], expected_ablations=["method"])["by_ablation"]["method"]
    assert summary["official_num_objects"] == 0
    assert summary["official_complete"] is False


def test_trellis_dc_sh_is_not_mistaken_for_rgb():
    coeff = torch.tensor([[1.0, 0.0, -1.0]])
    gaussian = {
        "xyz": torch.zeros(1, 3),
        "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "scaling": torch.ones(1, 3) * 0.01,
        "opacity": torch.ones(1, 1) * 0.5,
        "features_dc": coeff[:, None],
    }
    *_, rgb = _gaussian_tensors(gaussian)
    assert torch.allclose(rgb, 0.5 + SH_C0 * coeff)


def test_explicit_gaussian_rgb_is_not_reinterpreted_as_sh():
    expected = torch.tensor([[0.1, 0.4, 0.9]])
    gaussian = {
        "xyz": torch.zeros(1, 3),
        "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "scaling": torch.ones(1, 3) * 0.01,
        "opacity": torch.ones(1, 1) * 0.5,
        "colors": expected,
    }
    *_, rgb = _gaussian_tensors(gaussian)
    assert torch.equal(rgb, expected)


def test_export_parameterization_metadata_overrides_unsafe_range_heuristics():
    gaussian = {
        "xyz": torch.zeros(1, 3),
        "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "scaling": torch.full((1, 3), 0.2),
        "scaling_parameterization": "log",
        "opacity": torch.full((1, 1), 0.2),
        "opacity_parameterization": "logit",
        "colors": torch.full((1, 3), 0.5),
    }
    _, _, scales, opacities, _ = _gaussian_tensors(gaussian)
    assert torch.allclose(scales, torch.full((1, 3), float(torch.exp(torch.tensor(0.2)))))
    assert torch.allclose(opacities, torch.full((1,), float(torch.sigmoid(torch.tensor(0.2)))))


def test_ssim_render_loss_has_a_gradient():
    prediction = torch.full((1, 3, 16, 16), 0.3, requires_grad=True)
    target = torch.full_like(prediction, 0.7)
    loss = render_level_losses(prediction, target)["L_ssim"]
    loss.backward()
    assert prediction.grad is not None
    assert prediction.grad.abs().sum() > 0


def test_foreground_rgb_loss_is_normalized_and_differentiable():
    prediction = torch.zeros(1, 3, 8, 8, requires_grad=True)
    target = torch.ones_like(prediction)
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, 2:6, 2:6] = 1
    loss = render_level_losses(prediction, target, target_mask=mask, rendered_alpha=mask)["L_rgb_foreground"]
    assert torch.allclose(loss, torch.tensor(1.0))
    loss.backward()
    assert prediction.grad is not None
    assert prediction.grad[:, :, 2:6, 2:6].abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast regression requires a CUDA device")
def test_decoded_mask_loss_is_cuda_autocast_safe():
    prediction = torch.full((1, 3, 8, 8), 0.4, device="cuda", requires_grad=True)
    target = torch.full_like(prediction, 0.6)
    alpha_logits = torch.zeros(1, 1, 8, 8, device="cuda", requires_grad=True)
    target_mask = torch.ones_like(alpha_logits)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        terms = render_level_losses(
            prediction,
            target,
            rendered_alpha=alpha_logits.sigmoid(),
            target_mask=target_mask,
        )
    terms["L_mask"].backward()
    assert alpha_logits.grad is not None
    assert torch.isfinite(alpha_logits.grad).all()


def test_evaluator_rejects_missing_enabled_stage_checkpoint_before_workers(tmp_path: Path):
    run_root = tmp_path / "foundation"
    valid_geoss = run_root / "stage1_geoss" / "geoss_adapter_best.pt"
    valid_ss = run_root / "stage2_ss_velocity" / "ss_velocity_adapter_best.pt"
    valid_geoss.parent.mkdir(parents=True)
    valid_ss.parent.mkdir(parents=True)
    valid_geoss.write_bytes(b"checkpoint")
    valid_ss.write_bytes(b"checkpoint")
    args = SimpleNamespace(
        run_stage1=True,
        run_stage2=True,
        run_stage3=False,
        run_stage4=True,
        geoss_checkpoint=str(valid_geoss),
        ss_checkpoint=str(valid_ss),
        slat_checkpoint=str(run_root / "stage3_geovis_slat" / "geovis_slat_adapter_best.pt"),
        slat_joint_checkpoint=str(tmp_path / "wrong" / "geovis_slat_adapter_best.pt"),
    )
    with pytest.raises(FileNotFoundError, match="slat_joint_checkpoint"):
        _validate_requested_checkpoints(args, run_root)


def test_stage_scheduler_uses_per_gpu_free_memory_and_stage_reservation(monkeypatch):
    free = {"0": (72.0, 80.0), "1": (40.0, 80.0)}
    monkeypatch.setattr(
        "scripts.evaluate_meshfleet_sequence._query_gpu_memory_gb",
        lambda gpu: free[gpu],
    )
    args = SimpleNamespace(
        auto_workers_per_gpu=True,
        max_workers_per_gpu=6,
        workers_per_gpu=1,
        min_free_vram_gb=8.0,
        eval_worker_vram_gb=18.0,
        _stage_vram_estimates=_parse_stage_vram_estimates(
            "stage4_geovis_slat_joint=18", 18.0
        ),
    )
    capacities = _stage_capacities(
        "stage4_geovis_slat_joint",
        ["0", "1"],
        args,
    )
    assert capacities == {"0": 3, "1": 1}


def test_parallel_cuda_launch_requires_visible_gpu_list(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    args = SimpleNamespace(parallel=True, device="cuda", gpus=None)
    with pytest.raises(ValueError, match="--gpus"):
        _validate_parallel_launch(args)


def test_split_asset_evaluation_preserves_inference_gpu_and_failure():
    inference = [{
        "ablation": "stage4_geovis_slat_joint",
        "index": 7,
        "uid": "sample",
        "gpu": "0",
        "population_manifested": True,
        "status": "ok",
        "peak_vram_gb": 13.0,
    }]
    asset = [{
        "ablation": "stage4_geovis_slat_joint",
        "index": 7,
        "uid": "sample",
        "gpu": "1",
        "population_manifested": True,
        "status": "failed",
        "asset_eval_status": "failed",
        "asset_eval_error": "render timeout",
    }]
    merged = _merge_asset_evaluation_rows(inference, asset)
    assert merged[0]["gpu"] == "0"
    assert merged[0]["asset_eval_gpu"] == "1"
    assert merged[0]["peak_vram_gb"] == pytest.approx(13.0)
    assert merged[0]["asset_eval_status"] == "failed"
    assert merged[0]["status"] == "failed"


def test_supervised_worker_hard_timeout_reaps_process_group(tmp_path: Path):
    log_path = tmp_path / "worker.log"
    runtime = SimpleNamespace(
        worker_timeout_seconds=0.25,
        worker_stall_timeout_seconds=0.0,
        worker_terminate_grace_seconds=0.1,
        worker_monitor_interval_seconds=0.05,
    )
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        result = _run_with_peak_vram(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            log,
            dict(os.environ),
            None,
            runtime_args=runtime,
        )
    assert result.returncode != 0
    assert result.timed_out is True
    assert result.termination_reason == "hard_timeout_after_0.25_seconds"
    assert time.monotonic() - started < 5.0


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup contract")
def test_supervisor_reaps_descendant_after_normal_leader_exit(tmp_path: Path):
    log_path = tmp_path / "descendant.log"
    runtime = SimpleNamespace(
        worker_timeout_seconds=10.0,
        worker_stall_timeout_seconds=0.0,
        worker_terminate_grace_seconds=0.1,
        worker_monitor_interval_seconds=0.05,
    )
    code = (
        "import subprocess,sys;"
        "subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
        "print('leader exiting', flush=True)"
    )
    with log_path.open("w", encoding="utf-8") as log:
        result = _run_with_peak_vram(
            [sys.executable, "-c", code],
            log,
            dict(os.environ),
            None,
            runtime_args=runtime,
        )
    assert result.returncode == 0
    assert "reaping lingering process-group descendants" in log_path.read_text(encoding="utf-8")
