from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoss.datasets.meshfleet_trellis_dataset import MeshFleetTrellisDataset
from geoss.geometry.alignment import align_vggt_batch
from geoss.integration.vggt_geometry_wrapper import VGGTGeometryWrapper
from geoss.reconstruction import (
    VGGTDepthFusionConfig,
    VisualHullConfig,
    aligned_pointmap_depth_evidence,
    carve_visual_hull,
    carve_visual_hull_with_depth,
    extract_visual_hull_mesh,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-free reconstruction from MeshFleet silhouettes, calibrated "
            "cameras, confidence-filtered VGGT geometry, and unchanged TRELLIS appearance."
        )
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_gaussian", required=True)
    parser.add_argument("--meshfleet_root", required=True)
    parser.add_argument("--meshfleet_split", default="test")
    parser.add_argument("--meshfleet_category", default=None)
    parser.add_argument("--meshfleet_index", type=int, default=0)
    parser.add_argument("--meshfleet_uid", default=None)
    parser.add_argument(
        "--conditioning_view_set",
        choices=("renders", "renders_cond"),
        default="renders",
    )
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--hull_min_view_fraction", type=float, default=1.0)
    parser.add_argument("--mask_dilation_pixels", type=int, default=2)
    parser.add_argument("--closing_iterations", type=int, default=1)
    parser.add_argument("--confidence_threshold", type=float, default=0.15)
    parser.add_argument("--free_space_margin", type=float, default=0.0125)
    parser.add_argument("--min_depth_views", type=int, default=2)
    parser.add_argument("--min_not_free_fraction", type=float, default=0.75)
    parser.add_argument("--reprojection_sigma_pixels", type=float, default=4.0)
    parser.add_argument("--max_reprojection_error_pixels", type=float, default=12.0)
    parser.add_argument("--chunk_size", type=int, default=262144)
    parser.add_argument("--vggt_root", default=None)
    parser.add_argument("--vggt_checkpoint", default=None)
    parser.add_argument("--vggt_pretrained", default="facebook/VGGT-1B")
    parser.add_argument("--vggt_image_size", type=int, default=518)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    source_gaussian = Path(args.source_gaussian)
    if not source_gaussian.is_file():
        raise FileNotFoundError(f"Source Gaussian does not exist: {source_gaussian}")
    if args.conditioning_view_set.startswith("renders_eval_"):
        raise ValueError("Held-out evaluation renders are forbidden during reconstruction.")

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
    context = {
        key: sample[key].unsqueeze(0).to(device=device, dtype=torch.float32)
        for key in ("images", "masks", "K", "c2w", "w2c")
    }
    vggt = VGGTGeometryWrapper(
        vggt_root=args.vggt_root,
        checkpoint=args.vggt_checkpoint,
        pretrained_name=args.vggt_pretrained,
        mock=False,
        cache_features=False,
        vggt_image_size=args.vggt_image_size,
    ).to(device)
    with torch.inference_mode():
        predictions = vggt(context["images"], use_cache=False)
    if predictions.get("vggt_pointmap") is None:
        raise RuntimeError("VGGT did not produce a world point map.")
    if predictions.get("vggt_camera") is None:
        raise RuntimeError(
            "VGGT did not produce cameras; a projection-consistent canonical "
            "alignment cannot be established."
        )
    context.update(predictions)
    aligned = align_vggt_batch(context)
    aligned_debug = aligned["alignment_debug"]
    camera_alignment_valid = aligned_debug["camera_alignment_valid"]
    if not bool(camera_alignment_valid.all().item()):
        raise RuntimeError(
            "VGGT-to-MeshFleet camera Sim(3) alignment is invalid; refusing "
            "bbox-normalized pseudo-geometry for official reconstruction."
        )

    point_confidence = predictions.get(
        "vggt_point_confidence", predictions.get("vggt_confidence")
    )
    if point_confidence is None:
        point_confidence = torch.ones(
            1,
            context["images"].shape[1],
            context["images"].shape[-2],
            context["images"].shape[-1],
            device=device,
        )
    alignment_confidence = aligned["alignment_confidence"]
    combined_confidence = point_confidence
    if combined_confidence.ndim == 4:
        combined_confidence = combined_confidence.unsqueeze(2)
    combined_confidence = combined_confidence * alignment_confidence

    depth_map, evidence_confidence, reprojection_report = (
        aligned_pointmap_depth_evidence(
            aligned["aligned_pointmap"],
            context["K"],
            context["w2c"],
            context["masks"],
            combined_confidence,
            reprojection_sigma_pixels=args.reprojection_sigma_pixels,
            max_reprojection_error_pixels=args.max_reprojection_error_pixels,
        )
    )
    hull_config = VisualHullConfig(
        resolution=args.resolution,
        mask_threshold=args.mask_threshold,
        min_view_fraction=args.hull_min_view_fraction,
        min_valid_views=2,
        mask_dilation_pixels=args.mask_dilation_pixels,
        closing_iterations=args.closing_iterations,
        chunk_size=args.chunk_size,
    )
    visual_hull, hull_report = carve_visual_hull(
        context["masks"][0],
        context["K"][0],
        context["w2c"][0],
        hull_config,
    )
    fusion_config = VGGTDepthFusionConfig(
        bounds_min=hull_config.bounds_min,
        bounds_max=hull_config.bounds_max,
        mask_threshold=args.mask_threshold,
        confidence_threshold=args.confidence_threshold,
        free_space_margin=args.free_space_margin,
        min_depth_views=args.min_depth_views,
        min_not_free_fraction=args.min_not_free_fraction,
        reprojection_sigma_pixels=args.reprojection_sigma_pixels,
        max_reprojection_error_pixels=args.max_reprojection_error_pixels,
        chunk_size=args.chunk_size,
    )
    fused, fusion_report = carve_visual_hull_with_depth(
        visual_hull,
        depth_map,
        evidence_confidence,
        context["masks"][0],
        context["K"][0],
        context["w2c"][0],
        fusion_config,
    )
    mesh = extract_visual_hull_mesh(
        fused,
        bounds_min=hull_config.bounds_min,
        bounds_max=hull_config.bounds_max,
    )

    mesh_path = output_dir / "asset_mesh_internal.ply"
    mesh.export(str(mesh_path))
    evidence_path = output_dir / "vggt_depth_evidence.npz"
    np.savez_compressed(
        evidence_path,
        visual_hull=visual_hull.detach().cpu().numpy().astype(np.uint8),
        fused_occupancy=fused.detach().cpu().numpy().astype(np.uint8),
        depth=depth_map.detach().cpu().numpy().astype(np.float32),
        confidence=evidence_confidence.detach().cpu().numpy().astype(np.float32),
        bounds_min=np.asarray(hull_config.bounds_min, dtype=np.float32),
        bounds_max=np.asarray(hull_config.bounds_max, dtype=np.float32),
    )
    gaussian_path = output_dir / "asset_gaussian.ply"
    shutil.copy2(source_gaussian, gaussian_path)
    source_hash = _sha256(source_gaussian)
    copied_hash = _sha256(gaussian_path)
    if source_hash != copied_hash:
        raise RuntimeError("Copied TRELLIS Gaussian failed the byte-identity check.")

    metrics = {
        "status": "ok",
        "mode": "vggt_depth_fused_visual_hull_with_trellis_appearance",
        "uid": sample["uid"],
        "saved_assets": {
            "gaussian_ply": str(gaussian_path),
            "mesh_internal_ply": str(mesh_path),
            "vggt_depth_evidence": str(evidence_path),
        },
        "visual_hull": hull_report,
        "vggt_reprojection": reprojection_report,
        "vggt_depth_fusion": fusion_report,
        "vggt_alignment": _jsonable_alignment_debug(aligned_debug),
        "vggt_root": (
            str(Path(args.vggt_root).resolve()) if args.vggt_root else None
        ),
        "vggt_pretrained": args.vggt_pretrained,
        "vggt_checkpoint": (
            str(Path(args.vggt_checkpoint).resolve())
            if args.vggt_checkpoint
            else None
        ),
        "vggt_checkpoint_sha256": (
            _sha256(Path(args.vggt_checkpoint))
            if args.vggt_checkpoint and Path(args.vggt_checkpoint).is_file()
            else None
        ),
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
        "inference_context_source": (
            "conditioning_images_masks_calibrated_cameras_vggt_geometry_"
            "and_original_trellis_gaussian_appearance"
        ),
        "appearance_source": str(source_gaussian),
        "appearance_source_sha256": source_hash,
        "output_gaussian_sha256": copied_hash,
        "appearance_byte_identical": True,
        "geometry_source": (
            "conditioning_silhouettes_plus_confidence_filtered_aligned_vggt_pointmap"
        ),
        "latency_seconds": time.perf_counter() - started,
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


def _jsonable_alignment_debug(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        if detached.numel() == 1:
            return detached.item()
        return detached.tolist()
    if isinstance(value, dict):
        return {key: _jsonable_alignment_debug(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_alignment_debug(item) for item in value]
    return value


if __name__ == "__main__":
    main()
