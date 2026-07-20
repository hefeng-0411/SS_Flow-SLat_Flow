from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def test_auditor_separates_strict_stage_and_evaluation_manifests(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_object(root / "train", "complete_train")
    _write_object(root / "train", "missing_features", omit={"features"})
    _write_object(root / "train", "missing_slat", omit={"latents"})
    _write_object(root / "test", "complete_test")
    _write_object(root / "test", "eval_only_test", omit={"features", "latents", "ss_latents", "voxels"})
    output = tmp_path / "audit"
    script = Path(__file__).resolve().parents[1] / "scripts" / "inspect_meshfleet_dataset.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data_root",
            str(root),
            "--output_dir",
            str(output),
            "--validation_percent",
            "0",
            "--min_train_views",
            "2",
            "--min_eval_views",
            "2",
            "--validate_payloads",
            "--strict",
            "--strict_scope",
            "manifests",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    assert _uids(output / "train_uids.json") == ["complete_train"]
    assert _uids(output / "stage1_train_uids.json") == ["complete_train", "missing_features", "missing_slat"]
    assert _uids(output / "stage2_train_uids.json") == ["complete_train", "missing_features", "missing_slat"]
    assert _uids(output / "stage3_train_uids.json") == ["complete_train", "missing_features"]
    assert _uids(output / "test_uids.json") == ["complete_test"]
    assert _uids(output / "test_evaluation_uids.json") == ["complete_test", "eval_only_test"]

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["all_discovered_valid"] is False
    assert summary["manifests_valid"] is True
    assert summary["invalid_reason_counts"]["missing:features"] == 2

    profile_output = tmp_path / "profile"
    profile_script = Path(__file__).resolve().parents[1] / "scripts" / "profile_meshfleet_distribution.py"
    profile = subprocess.run(
        [
            sys.executable,
            str(profile_script),
            "--data_root",
            str(root),
            "--output_dir",
            str(profile_output),
            "--splits",
            "train,test",
            "--images_per_set",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert profile.returncode == 0, profile.stdout + profile.stderr
    profile_summary = json.loads((profile_output / "summary.json").read_text(encoding="utf-8"))
    assert profile_summary["objects"] == 5
    assert profile_summary["preprocessing_models"]["latents"] == "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"


def _uids(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))["uids"]


def _write_object(split_root: Path, uid: str, *, omit: set[str] | None = None) -> None:
    omit = omit or set()
    for render_set in ("renders", "renders_cond", "renders_eval_70", "renders_eval_90"):
        render_dir = split_root / render_set / uid
        render_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        count = 1 if render_set == "renders_cond" else 2
        base = {"renders": 0.0, "renders_cond": 10.0, "renders_eval_70": 20.0, "renders_eval_90": 30.0}[render_set]
        for index in range(count):
            filename = f"{index:03d}.png"
            Image.new("RGBA", (8, 8), (64, 96, 128, 255)).save(render_dir / filename)
            camera = np.eye(4, dtype=np.float32)
            camera[0, 3] = base + index
            frames.append({"file_path": filename, "transform_matrix": camera.tolist(), "camera_angle_x": 0.7})
        (render_dir / "transforms.json").write_text(json.dumps({"frames": frames}), encoding="utf-8")

    file_paths = {
        "features": split_root / "features" / "dinov2_vitl14_reg" / f"{uid}.npz",
        "latents": split_root / "latents" / "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16" / f"{uid}.npz",
        "ss_latents": split_root / "ss_latents" / "ss_enc_conv3d_16l8_fp16" / f"{uid}.npz",
        "voxels": split_root / "voxels" / f"{uid}.ply",
    }
    for modality, path in file_paths.items():
        if modality in omit:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if modality == "features":
            np.savez_compressed(path, patchtokens=np.zeros((1, 2), dtype=np.float32))
        elif modality == "latents":
            np.savez_compressed(path, feats=np.zeros((1, 2), dtype=np.float32), coords=np.zeros((1, 3), dtype=np.uint8))
        elif modality == "ss_latents":
            np.savez_compressed(path, mean=np.zeros((1, 2, 2, 2), dtype=np.float32))
        else:
            path.write_text(
                "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n",
                encoding="ascii",
            )
    mesh = split_root / "mesh_normalized" / uid / "mesh.glb"
    mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.write_bytes(b"glb")
