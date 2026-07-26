from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoss.datasets.meshfleet_trellis_dataset import MeshFleetTrellisDataset
from geoss.reconstruction.visual_hull import (
    VisualHullConfig,
    carve_visual_hull,
    extract_visual_hull_mesh,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-free conditioning-view visual-hull reconstruction."
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_gaussian", required=True)
    parser.add_argument("--meshfleet_root", required=True)
    parser.add_argument("--meshfleet_split", default="test")
    parser.add_argument("--meshfleet_category", default=None)
    parser.add_argument("--meshfleet_index", type=int, default=0)
    parser.add_argument("--meshfleet_uid", default=None)
    parser.add_argument("--conditioning_view_set", choices=("renders", "renders_cond"), default="renders")
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--min_view_fraction", type=float, default=1.0)
    parser.add_argument("--min_valid_views", type=int, default=2)
    parser.add_argument("--mask_dilation_pixels", type=int, default=2)
    parser.add_argument("--closing_iterations", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=262144)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.conditioning_view_set.startswith("renders_eval_"):
        raise ValueError("Held-out evaluation renders are forbidden during reconstruction.")
    source_gaussian = Path(args.source_gaussian)
    if not source_gaussian.is_file():
        raise FileNotFoundError(f"Source Gaussian does not exist: {source_gaussian}")
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = MeshFleetTrellisDataset(
        args.meshfleet_root,
        split=args.meshfleet_split,
        category=args.meshfleet_category,
        num_views=args.num_views,
        image_size=args.image_size,
        render_set=args.conditioning_view_set,
        repeat_views_if_insufficient=False,
        uid_manifest=[args.meshfleet_uid] if args.meshfleet_uid else None,
        load_3d_modalities=False,
    )
    if args.meshfleet_uid:
        sample = dataset.get_by_uid(args.meshfleet_uid)
    else:
        if not 0 <= args.meshfleet_index < len(dataset):
            raise IndexError(
                f"meshfleet_index={args.meshfleet_index} outside dataset length {len(dataset)}"
            )
        sample = dataset[args.meshfleet_index]
    if sample.get("mesh_path") is not None or "gt_occ" in sample:
        raise RuntimeError("Conditioning-only reconstruction received a forbidden 3D target.")

    device = torch.device(args.device)
    config = VisualHullConfig(
        resolution=args.resolution,
        mask_threshold=args.mask_threshold,
        min_view_fraction=args.min_view_fraction,
        min_valid_views=args.min_valid_views,
        mask_dilation_pixels=args.mask_dilation_pixels,
        closing_iterations=args.closing_iterations,
        chunk_size=args.chunk_size,
    )
    occupancy, hull_report = carve_visual_hull(
        sample["masks"].to(device=device, dtype=torch.float32),
        sample["K"].to(device=device, dtype=torch.float32),
        sample["w2c"].to(device=device, dtype=torch.float32),
        config,
    )
    mesh = extract_visual_hull_mesh(
        occupancy,
        bounds_min=config.bounds_min,
        bounds_max=config.bounds_max,
    )
    mesh_path = output_dir / "asset_mesh_internal.ply"
    mesh.export(str(mesh_path))
    occupancy_path = output_dir / "visual_hull_occupancy.npz"
    np.savez_compressed(
        occupancy_path,
        occupancy=occupancy.detach().cpu().numpy().astype(np.uint8),
        bounds_min=np.asarray(config.bounds_min, dtype=np.float32),
        bounds_max=np.asarray(config.bounds_max, dtype=np.float32),
    )
    gaussian_path = output_dir / "asset_gaussian.ply"
    shutil.copy2(source_gaussian, gaussian_path)
    source_hash = _sha256(source_gaussian)
    copied_hash = _sha256(gaussian_path)
    if source_hash != copied_hash:
        raise RuntimeError("Copied TRELLIS Gaussian failed the byte-identity check.")
    elapsed = time.perf_counter() - started
    metrics = {
        "status": "ok",
        "mode": "conditioning_visual_hull",
        "uid": sample["uid"],
        "saved_assets": {
            "gaussian_ply": str(gaussian_path),
            "mesh_internal_ply": str(mesh_path),
            "visual_hull_occupancy": str(occupancy_path),
        },
        "visual_hull": hull_report,
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_watertight": bool(mesh.is_watertight),
        "conditioning_view_set": args.conditioning_view_set,
        "conditioning_frame_ids": sample["metadata"]["selected_frame_ids"],
        "conditioning_image_size": list(sample["images"].shape[-2:]),
        "test_time_ground_truth_latents_used": False,
        "test_time_ground_truth_mesh_used": False,
        "test_time_ground_truth_voxels_used": False,
        "evaluation_views_used": False,
        "inference_context_source": "conditioning_masks_and_calibrated_cameras_plus_source_gaussian_appearance",
        "appearance_source": str(source_gaussian),
        "appearance_source_sha256": source_hash,
        "output_gaussian_sha256": copied_hash,
        "appearance_byte_identical": True,
        "geometry_source": "conditioning_silhouettes_only",
        "latency_seconds": elapsed,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2), flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
