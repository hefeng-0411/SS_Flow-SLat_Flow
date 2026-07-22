from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoss.datasets.meshfleet_trellis_dataset import MeshFleetTrellisDataset
from geoss.utils.config import str2bool


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch inference/evaluation suite for a completed MeshFleet 4-stage run.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--run_root", type=str, required=True, help="Training output root containing stage1_geoss ... stage4_geovis_slat_joint.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=0, help="Maximum objects; 0 evaluates the complete selected split.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--uid_manifest", type=str, default=None, help="Optional validation/test UID manifest from the dataset auditor.")
    parser.add_argument(
        "--allow_unmanifested",
        type=str2bool,
        default=False,
        help="Diagnostic escape hatch. Official evaluation requires a frozen UID manifest.",
    )
    parser.add_argument(
        "--require_audited_manifest",
        type=str2bool,
        default=True,
        help="Require a v3 auditor manifest whose role matches the selected split.",
    )
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--eval_num_views", type=int, default=12)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--occ_resolution", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config_geoss", type=str, default="configs/sparse_ray_geoss.yaml")
    parser.add_argument("--config_slat", type=str, default="configs/geovis_slat.yaml")
    parser.add_argument("--config_slat_joint", type=str, default="configs/phase2_decoded_asset.yaml")
    parser.add_argument("--vggt_root", type=str, default=None)
    parser.add_argument("--vggt_pretrained", type=str, default="facebook/VGGT-1B")
    parser.add_argument("--trellis_root", type=str, default=None)
    parser.add_argument("--trellis_model_path", type=str, default="microsoft/TRELLIS-image-large")
    parser.add_argument("--geoss_checkpoint", type=str, default=None)
    parser.add_argument("--ss_checkpoint", type=str, default=None)
    parser.add_argument("--slat_checkpoint", type=str, default=None)
    parser.add_argument("--slat_joint_checkpoint", type=str, default=None)
    parser.add_argument("--run_original_trellis", type=str2bool, default=True)
    parser.add_argument("--run_stage1", type=str2bool, default=True)
    parser.add_argument("--run_stage2", type=str2bool, default=True)
    parser.add_argument("--run_stage3", type=str2bool, default=True)
    parser.add_argument("--run_stage4", type=str2bool, default=True)
    parser.add_argument("--run_refined_final", type=str2bool, default=True)
    parser.add_argument("--refinement_steps", type=int, default=150)
    parser.add_argument("--refinement_views_per_step", type=int, default=2)
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated physical CUDA ids for parallel evaluation, e.g. 4,5,6,7.")
    parser.add_argument("--parallel", type=str2bool, default=True)
    parser.add_argument("--auto_workers_per_gpu", type=str2bool, default=True)
    parser.add_argument("--workers_per_gpu", type=int, default=1)
    parser.add_argument("--max_workers_per_gpu", type=int, default=2)
    parser.add_argument("--eval_worker_vram_gb", type=float, default=34.0, help="Estimated peak VRAM per single-object TRELLIS eval worker.")
    parser.add_argument("--min_free_vram_gb", type=float, default=8.0, help="Free VRAM reserve kept on every GPU.")
    parser.add_argument("--oom_retry_limit", type=int, default=2)
    parser.add_argument("--render_eval", type=str2bool, default=True)
    parser.add_argument("--render_background_color", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--eval_view_set", choices=("renders_eval_70", "renders_eval_90"), default="renders_eval_70")
    parser.add_argument("--conditioning_view_set", choices=("renders", "renders_cond"), default="renders")
    parser.add_argument("--geometry_samples", type=int, default=100000)
    parser.add_argument("--geometry_seed", type=int, default=20260720)
    parser.add_argument("--fscore_threshold", type=float, default=0.01)
    parser.add_argument("--save_visuals", type=str2bool, default=True)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir) if args.output_dir else run_root / "evaluation_suite"
    output_dir.mkdir(parents=True, exist_ok=True)
    _fill_checkpoint_defaults(args, run_root)
    _validate_requested_checkpoints(args, run_root)

    dataset = MeshFleetTrellisDataset(
        args.data_root,
        split=args.split,
        category=args.category,
        num_views=args.num_views,
        image_size=args.image_size,
        occ_resolution=args.occ_resolution,
        render_set="renders",
        background_color=args.render_background_color,
    )
    if len(dataset) == 0:
        raise FileNotFoundError(f"No MeshFleet samples found under {args.data_root} split={args.split}.")
    uid_manifest_provenance = None
    if args.uid_manifest:
        manifest_path = Path(args.uid_manifest)
        raw_manifest = manifest_path.read_bytes()
        payload = json.loads(raw_manifest.decode("utf-8"))
        if args.require_audited_manifest:
            if not isinstance(payload, dict) or payload.get("audit_protocol_version") != "meshfleet_dataset_audit_v3":
                raise ValueError(
                    "Official evaluation requires a meshfleet_dataset_audit_v3 manifest. "
                    "Regenerate it with scripts/inspect_meshfleet_dataset.py; v2 manifests included invalid UIDs."
                )
            allowed_roles = {"test", "test_evaluation"} if args.split == "test" else {"validation", "validation_evaluation"}
            if payload.get("role") not in allowed_roles:
                raise ValueError(
                    f"Manifest role={payload.get('role')!r} is incompatible with split={args.split!r}; "
                    f"expected one of {sorted(allowed_roles)}."
                )
        requested_uids = payload.get("uids", payload) if isinstance(payload, dict) else payload
        if not isinstance(requested_uids, list) or not all(isinstance(uid, str) for uid in requested_uids):
            raise ValueError("--uid_manifest must be a JSON string list or an object containing 'uids'.")
        if len(set(requested_uids)) != len(requested_uids):
            raise ValueError("--uid_manifest contains duplicate UIDs; evaluation population must be unique.")
        index_by_uid = {sample["uid"]: index for index, sample in enumerate(dataset.samples)}
        missing_uids = [uid for uid in requested_uids if uid not in index_by_uid]
        if missing_uids:
            raise KeyError(f"UID manifest contains {len(missing_uids)} objects absent from split {args.split}: {missing_uids[:10]}")
        selected_indices = [index_by_uid[uid] for uid in requested_uids]
        uid_manifest_provenance = {
            "path": str(manifest_path.resolve()),
            "sha256": hashlib.sha256(raw_manifest).hexdigest(),
            "declared_count": len(requested_uids),
            "role": payload.get("role") if isinstance(payload, dict) else None,
            "audit_protocol_version": payload.get("audit_protocol_version") if isinstance(payload, dict) else None,
        }
    else:
        if not args.allow_unmanifested:
            raise ValueError(
                "Official evaluation requires --uid_manifest generated before model evaluation. "
                "Use --allow_unmanifested true only for diagnostic smoke runs."
            )
        selected_indices = list(range(len(dataset)))
    args._uid_by_index = {index: sample["uid"] for index, sample in enumerate(dataset.samples)}
    selected_indices = selected_indices[args.start_index :]
    indices = selected_indices[: args.max_samples] if args.max_samples > 0 else selected_indices
    manifest = {
        "data_root": args.data_root,
        "run_root": str(run_root),
        "split": args.split,
        "num_dataset_samples": len(dataset),
        "evaluated_indices": indices,
        "evaluated_uids": [dataset.samples[index]["uid"] for index in indices],
        "uid_manifest": uid_manifest_provenance,
        "protocol": {
            "version": "meshfleet_heldout_v2",
            "conditioning_view_set": args.conditioning_view_set,
            "conditioning_num_views": args.num_views,
            "evaluation_view_set": args.eval_view_set,
            "evaluation_num_views": args.eval_num_views,
            "image_size": args.image_size,
            "background_color": list(args.render_background_color),
            "geometry_samples": args.geometry_samples,
            "geometry_seed": args.geometry_seed,
            "fscore_threshold": args.fscore_threshold,
        },
        "checkpoints": {
            "geoss": args.geoss_checkpoint,
            "ss_velocity": args.ss_checkpoint,
            "slat": args.slat_checkpoint,
            "slat_joint": args.slat_joint_checkpoint,
        },
        "parallel": {
            "enabled": bool(args.parallel),
            "gpus": _visible_gpus(args),
            "auto_workers_per_gpu": bool(args.auto_workers_per_gpu),
            "workers_per_gpu": args.workers_per_gpu,
            "max_workers_per_gpu": args.max_workers_per_gpu,
            "oom_retry_limit": args.oom_retry_limit,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.parallel and _visible_gpus(args):
        rows = _run_parallel(args, output_dir, indices)
    else:
        rows = []
        for index in indices:
            try:
                rows.extend(_run_sample(args, output_dir, index, gpu=None))
            except Exception as exc:
                rows.append({
                    "ablation": "sample",
                    "status": "failed",
                    "index": index,
                    "uid": _uid_for_index(args, index),
                    "gpu": None,
                    "population_manifested": bool(args.uid_manifest),
                    "error": repr(exc),
                })

    _write_jsonl(output_dir / "per_sample_metrics.jsonl", rows)
    expected_ablations = [
        name
        for name, enabled in (
            ("original_trellis", args.run_original_trellis),
            ("stage1_geoss_context", args.run_stage1),
            ("stage2_geoss_ss", args.run_stage2),
            ("stage3_geovis_slat", args.run_stage3),
            ("stage4_geovis_slat_joint", args.run_stage4),
            ("final_conditioning_refined", args.run_refined_final),
        )
        if enabled
    ]
    summary = _aggregate(rows, expected_indices=indices, expected_ablations=expected_ablations)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_rows_csv(output_dir / "per_sample_metrics.csv", rows)
    _write_summary_csv(output_dir / "summary.csv", summary)
    print(json.dumps({"samples": len(indices), "rows": len(rows), "output_dir": str(output_dir)}, indent=2))


def _run_parallel(args: argparse.Namespace, output_dir: Path, indices: list[int]) -> list[dict[str, Any]]:
    gpus = _visible_gpus(args)
    capacities = {gpu: _initial_workers_for_gpu(gpu, args) for gpu in gpus}
    retry_counts = {index: 0 for index in indices}
    pending = list(indices)
    rows: list[dict[str, Any]] = []
    futures = {}
    max_total_workers = max(1, len(gpus) * max(1, args.max_workers_per_gpu))
    with ThreadPoolExecutor(max_workers=max_total_workers) as pool:
        while pending or futures:
            _fill_gpu_slots(pool, futures, pending, capacities, args, output_dir)
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                index, gpu = futures.pop(fut)
                try:
                    sample_rows = fut.result()
                except Exception as exc:
                    sample_rows = [{
                        "ablation": "sample",
                        "status": "failed",
                        "index": index,
                        "uid": _uid_for_index(args, index),
                        "gpu": gpu,
                        "population_manifested": bool(args.uid_manifest),
                        "error": repr(exc),
                    }]
                oom = any(bool(row.get("oom")) for row in sample_rows)
                if oom and retry_counts[index] < args.oom_retry_limit:
                    old_capacity = capacities[gpu]
                    capacities[gpu] = max(1, capacities[gpu] // 2)
                    retry_counts[index] += 1
                    _mark_retry(output_dir, index, gpu, old_capacity, capacities[gpu], retry_counts[index])
                    pending.insert(0, index)
                else:
                    rows.extend(sample_rows)
                    if not oom and args.auto_workers_per_gpu:
                        capacities[gpu] = _maybe_grow_capacity(gpu, capacities[gpu], args)
    return rows


def _fill_gpu_slots(
    pool: ThreadPoolExecutor,
    futures: dict,
    pending: list[int],
    capacities: dict[str, int],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    while pending:
        slot_gpu = None
        for gpu, capacity in capacities.items():
            running = sum(1 for _, active_gpu in futures.values() if active_gpu == gpu)
            if running < capacity:
                slot_gpu = gpu
                break
        if slot_gpu is None:
            return
        index = pending.pop(0)
        fut = pool.submit(_run_sample, args, output_dir, index, slot_gpu)
        futures[fut] = (index, slot_gpu)


def _run_sample(args: argparse.Namespace, output_dir: Path, index: int, gpu: str | None) -> list[dict[str, Any]]:
    local_args = copy.copy(args)
    uid = _uid_for_index(args, index)
    local_args.device = "cuda" if gpu is not None and str(args.device).startswith("cuda") else args.device
    sample_root = output_dir / f"{args.split}_{index:06d}"
    sample_root.mkdir(parents=True, exist_ok=True)
    identity_path = sample_root / "sample_identity.json"
    if identity_path.is_file():
        recorded = _read_json(identity_path)
        if recorded.get("uid") != uid and not args.overwrite:
            raise RuntimeError(
                f"Evaluation directory {sample_root} belongs to uid={recorded.get('uid')!r}, not uid={uid!r}. "
                "Use --overwrite true or a new --output_dir; cached metrics must never be reassigned to another UID."
            )
    elif any(sample_root.iterdir()) and not args.overwrite:
        raise RuntimeError(
            f"Evaluation directory {sample_root} predates exact-UID provenance. "
            "Use --overwrite true or a new --output_dir before trusting cached metrics."
        )
    identity_path.write_text(json.dumps({"index": index, "uid": uid, "split": args.split}, indent=2), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    if local_args.run_original_trellis:
        rows.append(_run_original_trellis(local_args, sample_root, index, gpu))
        if _is_oom_row(rows[-1]):
            return _tag_sample_rows(rows, index, gpu, uid, bool(args.uid_manifest))
    stage2_dir = sample_root / "stage2_geoss_ss"
    if local_args.run_stage1:
        rows.append(_run_stage1(local_args, sample_root, index, gpu))
        if _is_oom_row(rows[-1]):
            return _tag_sample_rows(rows, index, gpu, uid, bool(args.uid_manifest))
    if local_args.run_stage2:
        rows.append(_run_stage2(local_args, sample_root, index, stage2_dir, gpu))
        if _is_oom_row(rows[-1]):
            return _tag_sample_rows(rows, index, gpu, uid, bool(args.uid_manifest))
    geoss_context = stage2_dir / "geoss_context.pt"
    if local_args.run_stage3:
        rows.append(_run_slat_stage(local_args, sample_root, index, "stage3_geovis_slat", local_args.slat_checkpoint, geoss_context, gpu))
        if _is_oom_row(rows[-1]):
            return _tag_sample_rows(rows, index, gpu, uid, bool(args.uid_manifest))
    if local_args.run_stage4:
        rows.append(_run_slat_stage(local_args, sample_root, index, "stage4_geovis_slat_joint", local_args.slat_joint_checkpoint, geoss_context, gpu))
        if _is_oom_row(rows[-1]):
            return _tag_sample_rows(rows, index, gpu, uid, bool(args.uid_manifest))
    if local_args.run_refined_final:
        rows.append(_run_refined_final(local_args, sample_root, index, gpu))
        if _is_oom_row(rows[-1]):
            return _tag_sample_rows(rows, index, gpu, uid, bool(args.uid_manifest))
    return _tag_sample_rows(rows, index, gpu, uid, bool(args.uid_manifest))


def _tag_sample_rows(
    rows: list[dict[str, Any]],
    index: int,
    gpu: str | None,
    uid: str | None = None,
    population_manifested: bool = False,
) -> list[dict[str, Any]]:
    for row in rows:
        row["index"] = index
        row["uid"] = uid
        row["population_manifested"] = population_manifested
        row["gpu"] = gpu
        review = []
        opacity = row.get("asset_opacity_mean", row.get("opacity_mean"))
        scale_ratio = row.get("asset_scale_abnormal_ratio", row.get("scale_abnormal_ratio"))
        mask_iou = row.get("asset_render_Mask_IoU", row.get("render_Mask_IoU"))
        if isinstance(opacity, (int, float)) and (opacity > 0.95 or opacity < 0.01):
            review.append("opacity_distribution_outlier")
        if isinstance(scale_ratio, (int, float)) and scale_ratio > 0.05:
            review.append("abnormal_gaussian_scale_ratio")
        if isinstance(mask_iou, (int, float)) and mask_iou < 0.25:
            review.append("low_heldout_silhouette_iou")
        if review:
            row["manual_review_flags"] = review
    return rows


def _fill_checkpoint_defaults(args: argparse.Namespace, run_root: Path) -> None:
    defaults = {
        "geoss_checkpoint": run_root / "stage1_geoss" / "geoss_adapter_best.pt",
        "ss_checkpoint": run_root / "stage2_ss_velocity" / "ss_velocity_adapter_best.pt",
        "slat_checkpoint": run_root / "stage3_geovis_slat" / "geovis_slat_adapter_best.pt",
        "slat_joint_checkpoint": run_root / "stage4_geovis_slat_joint" / "geovis_slat_adapter_best.pt",
    }
    for name, path in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, str(path))


def _validate_requested_checkpoints(args: argparse.Namespace, run_root: Path) -> None:
    """Fail before spawning workers when an enabled ablation cannot run."""
    requirements = (
        ("geoss_checkpoint", bool(args.run_stage1 or args.run_stage2), "Stage 1/2"),
        ("ss_checkpoint", bool(args.run_stage2), "Stage 2"),
        ("slat_checkpoint", bool(args.run_stage3), "Stage 3"),
        ("slat_joint_checkpoint", bool(args.run_stage4), "Stage 4"),
    )
    missing = []
    for attribute, required, consumer in requirements:
        if not required:
            continue
        configured = Path(str(getattr(args, attribute))).expanduser()
        if not configured.is_file():
            missing.append((attribute, consumer, configured))
    if not missing:
        return
    defaults = {
        "geoss_checkpoint": run_root / "stage1_geoss" / "geoss_adapter_best.pt",
        "ss_checkpoint": run_root / "stage2_ss_velocity" / "ss_velocity_adapter_best.pt",
        "slat_checkpoint": run_root / "stage3_geovis_slat" / "geovis_slat_adapter_best.pt",
        "slat_joint_checkpoint": run_root / "stage4_geovis_slat_joint" / "geovis_slat_adapter_best.pt",
    }
    details = []
    for attribute, consumer, configured in missing:
        default = defaults[attribute]
        hint = (
            f"; run-root default exists at {default}—remove the explicit --{attribute} argument or use that path"
            if default.is_file()
            else f"; expected run-root default is {default}"
        )
        details.append(f"  {consumer} --{attribute}: {configured}{hint}")
    raise FileNotFoundError(
        "Evaluation checkpoint preflight failed before launching GPU workers:\n"
        + "\n".join(details)
    )


def _run_original_trellis(args: argparse.Namespace, sample_root: Path, index: int, gpu: str | None = None) -> dict[str, Any]:
    out_dir = sample_root / "original_trellis"
    command = _geoss_command(args, out_dir, index, decode=True, disable_ss_adapter=True)
    return _run_and_collect("original_trellis", command, out_dir, args.overwrite, evaluate_assets=True, eval_args=args, gpu=gpu)


def _run_stage1(args: argparse.Namespace, sample_root: Path, index: int, gpu: str | None = None) -> dict[str, Any]:
    out_dir = sample_root / "stage1_geoss_context"
    command = _geoss_command(args, out_dir, index, decode=False, geoss_checkpoint=args.geoss_checkpoint)
    return _run_and_collect("stage1_geoss_context", command, out_dir, args.overwrite, gpu=gpu)


def _run_stage2(args: argparse.Namespace, sample_root: Path, index: int, out_dir: Path, gpu: str | None = None) -> dict[str, Any]:
    command = _geoss_command(
        args,
        out_dir,
        index,
        decode=True,
        geoss_checkpoint=args.geoss_checkpoint,
        ss_checkpoint=args.ss_checkpoint,
    )
    return _run_and_collect("stage2_geoss_ss", command, out_dir, args.overwrite, evaluate_assets=True, eval_args=args, gpu=gpu)


def _run_slat_stage(args: argparse.Namespace, sample_root: Path, index: int, name: str, checkpoint: str, geoss_context: Path, gpu: str | None = None) -> dict[str, Any]:
    out_dir = sample_root / name
    config_path = args.config_slat_joint if name == "stage4_geovis_slat_joint" else args.config_slat
    command = [
        sys.executable,
        "scripts/infer_geovis_slat.py",
        "--config",
        config_path,
        "--output_dir",
        str(out_dir),
        "--device",
        args.device,
        "--meshfleet_root",
        args.data_root,
        "--meshfleet_split",
        args.split,
        "--meshfleet_index",
        str(index),
        "--meshfleet_uid",
        _uid_for_index(args, index),
        "--num_views",
        str(args.num_views),
        "--image_size",
        str(args.image_size),
        "--trellis_model_path",
        str(args.trellis_model_path),
        "--slat_adapter_checkpoint",
        str(checkpoint),
        "--geoss_context",
        str(geoss_context),
        "--trellis_latents",
        str(geoss_context.parent / "trellis_latents.pt"),
        "--decode",
        "true",
        "--render_eval",
        str(bool(args.render_eval)).lower(),
        "--real_infer",
    ]
    if args.trellis_root:
        command += ["--trellis_root", str(args.trellis_root)]
    if args.vggt_root:
        command += ["--vggt_root", str(args.vggt_root)]
    if args.vggt_pretrained:
        command += ["--vggt_pretrained", str(args.vggt_pretrained)]
    if args.category:
        command += ["--meshfleet_category", args.category]
    return _run_and_collect(name, command, out_dir, args.overwrite, evaluate_assets=True, eval_args=args, gpu=gpu)


def _run_refined_final(args: argparse.Namespace, sample_root: Path, index: int, gpu: str | None = None) -> dict[str, Any]:
    source_dir = sample_root / "stage4_geovis_slat_joint"
    source_gaussian = source_dir / "asset_gaussian.ply"
    source_mesh = source_dir / "asset_mesh_internal.ply"
    out_dir = sample_root / "final_conditioning_refined"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_gaussian = out_dir / "asset_gaussian.ply"
    log_path = out_dir / "command.log"
    if not source_gaussian.is_file():
        return {
            "ablation": "final_conditioning_refined",
            "status": "failed",
            "error": f"Missing Stage-4 Gaussian: {source_gaussian}",
            "run_path": str(out_dir),
        }
    command = [
        sys.executable,
        "scripts/refine_gaussian_conditioning.py",
        "--gaussian_ply", str(source_gaussian),
        "--output_ply", str(output_gaussian),
        "--meshfleet_root", args.data_root,
        "--meshfleet_split", args.split,
        "--meshfleet_index", str(index),
        "--meshfleet_uid", _uid_for_index(args, index),
        "--conditioning_view_set", args.conditioning_view_set,
        "--num_views", str(args.num_views),
        "--image_size", str(args.image_size),
        "--background_color", *(str(value) for value in args.render_background_color),
        "--steps", str(args.refinement_steps),
        "--views_per_step", str(args.refinement_views_per_step),
        "--device", args.device,
    ]
    if args.category:
        command += ["--meshfleet_category", args.category]
    latency_seconds = None
    peak_vram_gb = None
    if args.overwrite or not output_gaussian.is_file():
        with log_path.open("w", encoding="utf-8") as log:
            log.write("==== COMMAND ====\n" + " ".join(command) + "\n\n")
            started = time.perf_counter()
            returncode, peak_vram_gb = _run_with_peak_vram(command, log, _child_env(gpu), gpu)
            latency_seconds = time.perf_counter() - started
        if returncode != 0:
            return {
                "ablation": "final_conditioning_refined",
                "status": "failed",
                "returncode": returncode,
                "run_path": str(out_dir),
                "log": str(log_path),
                "oom": _log_looks_like_oom(log_path),
                "latency_seconds": latency_seconds,
                "peak_vram_gb": peak_vram_gb,
            }
    if source_mesh.is_file() and (args.overwrite or not (out_dir / source_mesh.name).is_file()):
        shutil.copy2(source_mesh, out_dir / source_mesh.name)
    report = _read_json(output_gaussian.with_suffix(".refinement.json"))
    metrics = {
        "ablation": "final_conditioning_refined",
        "status": "ok",
        "run_path": str(out_dir),
        "log": str(log_path),
        "refinement_protocol": report.get("protocol"),
        "refinement_steps": report.get("steps"),
        "evaluation_views_used_for_refinement": report.get("evaluation_views_used"),
        "test_time_ground_truth_latents_used": False,
        "inference_context_source": "stage4_prediction_plus_conditioning_images_only",
    }
    if latency_seconds is not None:
        metrics.update({"latency_seconds": latency_seconds, "peak_vram_gb": peak_vram_gb})
    # The evaluator reads this file before rendering so it can reject any
    # inference artifact whose provenance declares test-time GT latent use.
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics.update(_eval_assets("final_conditioning_refined", out_dir, args, gpu=gpu))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _geoss_command(
    args: argparse.Namespace,
    out_dir: Path,
    index: int,
    *,
    decode: bool,
    disable_ss_adapter: bool = False,
    geoss_checkpoint: str | None = None,
    ss_checkpoint: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/infer_sparse_ray_geoss_ss.py",
        "--config",
        args.config_geoss,
        "--output_dir",
        str(out_dir),
        "--device",
        args.device,
        "--meshfleet_root",
        args.data_root,
        "--meshfleet_split",
        args.split,
        "--meshfleet_index",
        str(index),
        "--meshfleet_uid",
        _uid_for_index(args, index),
        "--num_views",
        str(args.num_views),
        "--image_size",
        str(args.image_size),
        "--meshfleet_occ_resolution",
        str(args.occ_resolution),
        "--vggt_pretrained",
        str(args.vggt_pretrained),
        "--trellis_model_path",
        str(args.trellis_model_path),
        "--decode",
        str(bool(decode)).lower(),
        "--render_eval",
        str(bool(args.render_eval)).lower(),
        "--disable_ss_adapter",
        str(bool(disable_ss_adapter)).lower(),
        "--real_infer",
    ]
    if args.vggt_root:
        command += ["--vggt_root", str(args.vggt_root)]
    if args.trellis_root:
        command += ["--trellis_root", str(args.trellis_root)]
    if args.category:
        command += ["--meshfleet_category", args.category]
    if geoss_checkpoint:
        command += ["--geoss_checkpoint", str(geoss_checkpoint)]
    if ss_checkpoint:
        command += ["--ss_adapter_checkpoint", str(ss_checkpoint)]
    return command


def _run_and_collect(
    ablation: str,
    command: list[str],
    out_dir: Path,
    overwrite: bool,
    *,
    evaluate_assets: bool = False,
    eval_args: argparse.Namespace | None = None,
    gpu: str | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    log_path = out_dir / "command.log"
    if overwrite or not metrics_path.exists():
        with log_path.open("w", encoding="utf-8") as log:
            log.write("==== COMMAND ====\n" + " ".join(command) + "\n\n")
            log.write(f"CUDA_VISIBLE_DEVICES={gpu if gpu is not None else os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}\n\n")
            started = time.perf_counter()
            returncode, peak_vram_gb = _run_with_peak_vram(command, log, _child_env(gpu), gpu)
            latency_seconds = time.perf_counter() - started
        if returncode != 0:
            return {
                "ablation": ablation,
                "status": "failed",
                "returncode": returncode,
                "run_path": str(out_dir),
                "log": str(log_path),
                "oom": _log_looks_like_oom(log_path),
                "latency_seconds": latency_seconds,
                "peak_vram_gb": peak_vram_gb,
            }
    row = _read_json(metrics_path)
    # Inference can intentionally publish a blocked diagnostic (for example a
    # missing SLAT checkpoint) without crashing the worker; preserve that state.
    row.update({"ablation": ablation, "status": row.get("status", "ok"), "run_path": str(out_dir), "log": str(log_path)})
    if "latency_seconds" in locals():
        row.update({"latency_seconds": latency_seconds, "peak_vram_gb": peak_vram_gb})
    if evaluate_assets:
        row.update(_eval_assets(ablation, out_dir, eval_args, gpu=gpu))
    return row


def _eval_assets(ablation: str, out_dir: Path, args: argparse.Namespace | None, *, gpu: str | None = None) -> dict[str, Any]:
    gaussian = out_dir / "asset_gaussian.ply"
    if not gaussian.exists():
        return {}
    eval_dir = out_dir / "asset_eval"
    eval_dir.mkdir(exist_ok=True)
    command = [
        sys.executable,
        "scripts/eval_geovis_slat.py",
        "--input_dir",
        str(out_dir),
        "--output_dir",
        str(eval_dir),
        "--ablation",
        ablation,
        "--gaussian_ply",
        str(gaussian),
        "--inference_metrics",
        str(out_dir / "metrics.json"),
        "--real_eval",
    ]
    if args is not None and args.render_eval:
        command += [
            "--meshfleet_root", args.data_root,
            "--meshfleet_split", args.split,
            "--meshfleet_index", str(_index_from_sample_dir(out_dir)),
            "--meshfleet_uid", _uid_for_index(args, _index_from_sample_dir(out_dir)),
            "--num_views", str(args.eval_num_views),
            "--conditioning_num_views", str(args.num_views),
            "--image_size", str(args.image_size),
            "--device", args.device,
            "--background_color", *(str(v) for v in args.render_background_color),
            "--eval_view_set", args.eval_view_set,
            "--conditioning_view_set", args.conditioning_view_set,
            "--geometry_samples", str(args.geometry_samples),
            "--geometry_seed", str(args.geometry_seed),
            "--fscore_threshold", str(args.fscore_threshold),
            "--save_visuals", str(bool(args.save_visuals)).lower(),
        ]
        mesh = out_dir / "asset_mesh_internal.ply"
        if mesh.is_file():
            command += ["--pred_mesh", str(mesh)]
        if args.category:
            command += ["--meshfleet_category", args.category]
    log_path = eval_dir / "command.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("==== COMMAND ====\n" + " ".join(command) + "\n\n")
        log.write(f"CUDA_VISIBLE_DEVICES={gpu if gpu is not None else os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}\n\n")
        returncode, _ = _run_with_peak_vram(command, log, _child_env(gpu), gpu)
    if returncode != 0:
        return {"asset_eval_status": "failed", "asset_eval_log": str(log_path), "oom": _log_looks_like_oom(log_path)}
    metrics = _read_json(eval_dir / "geovis_slat_metrics.json")
    # Preserve the per-view render breakdown in JSONL/CSV while aggregation only
    # consumes the scalar render means above.
    return {f"asset_{k}": v for k, v in metrics.items() if isinstance(v, (int, float, str, bool, list)) or v is None}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = []
    for ablation, metrics in summary["by_ablation"].items():
        row = {"ablation": ablation, "num_ok": metrics.get("num_ok", 0), "num_failed": metrics.get("num_failed", 0)}
        for name, stats in metrics.get("metrics", {}).items():
            row[f"{name}/mean"] = stats["mean"]
            row[f"{name}/std"] = stats["std"]
            row[f"{name}/ci95"] = stats["ci95"]
        rows.append(row)
    _write_rows_csv(path, rows)


def _aggregate(
    rows: list[dict[str, Any]],
    *,
    expected_indices: list[int] | None = None,
    expected_ablations: list[str] | None = None,
) -> dict[str, Any]:
    expected_indices = list(expected_indices or [])
    ablations = sorted(set(expected_ablations or []) | {str(row.get("ablation")) for row in rows})
    out: dict[str, Any] = {
        "by_ablation": {},
        "expected_object_count": len(expected_indices),
        "aggregation_policy": "failed or missing objects are reported and never silently dropped",
        "official_metric_policy": "PSNR/SSIM/LPIPS/CD/F-score aggregate only rows whose evaluator marks asset_official_metrics=true",
    }
    for ablation in ablations:
        subset = [row for row in rows if row.get("ablation") == ablation]
        ok = [row for row in subset if row.get("status") == "ok"]
        completed_indices = {int(row["index"]) for row in subset if isinstance(row.get("index"), int)}
        missing_indices = sorted(set(expected_indices) - completed_indices)
        metric_names = sorted({key for row in ok for key, value in row.items() if isinstance(value, (int, float)) and not isinstance(value, bool)})
        metrics = {}
        for name in metric_names:
            values = [float(row[name]) for row in ok if isinstance(row.get(name), (int, float)) and math.isfinite(float(row[name]))]
            if not values:
                continue
            mean = sum(values) / len(values)
            var = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
            std = math.sqrt(var)
            metrics[name] = _distribution_stats(values, mean=mean, std=std)
        official_rows = [
            row for row in ok
            if row.get("asset_official_metrics") is True and row.get("population_manifested") is True
        ]
        official_metrics = {}
        for name in ("asset_PSNR", "asset_SSIM", "asset_LPIPS", "asset_CD", "asset_F-score"):
            values = [float(row[name]) for row in official_rows if isinstance(row.get(name), (int, float)) and math.isfinite(float(row[name]))]
            if not values:
                continue
            mean = sum(values) / len(values)
            var = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
            std = math.sqrt(var)
            official_metrics[name.removeprefix("asset_")] = _distribution_stats(values, mean=mean, std=std)
        out["by_ablation"][ablation] = {
            "num_ok": len(ok),
            "num_failed": len(subset) - len(ok),
            "num_missing": len(missing_indices),
            "missing_indices": missing_indices,
            "complete": not missing_indices and len(ok) == len(expected_indices),
            "metrics": metrics,
            "official_num_objects": len(official_rows),
            "official_complete": len(official_rows) == len(expected_indices) and not missing_indices,
            "official_metrics": official_metrics,
        }
    return out


def _distribution_stats(values: list[float], *, mean: float | None = None, std: float | None = None) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    if count == 0:
        raise ValueError("Cannot summarize an empty metric population.")
    mean = sum(ordered) / count if mean is None else mean
    if std is None:
        variance = sum((value - mean) ** 2 for value in ordered) / max(1, count - 1)
        std = math.sqrt(variance)

    def percentile(fraction: float) -> float:
        if count == 1:
            return ordered[0]
        position = fraction * (count - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "mean": mean,
        "median": percentile(0.5),
        "std": std,
        "ci95": 1.96 * std / math.sqrt(count),
        "min": ordered[0],
        "p10": percentile(0.1),
        "p90": percentile(0.9),
        "max": ordered[-1],
        "n": count,
    }


def _visible_gpus(args: argparse.Namespace) -> list[str]:
    raw = args.gpus or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _initial_workers_for_gpu(gpu: str, args: argparse.Namespace) -> int:
    base = max(1, int(args.workers_per_gpu))
    if not args.auto_workers_per_gpu:
        return min(base, max(1, int(args.max_workers_per_gpu)))
    free_gb, _ = _query_gpu_memory_gb(gpu)
    if free_gb is None:
        return min(base, max(1, int(args.max_workers_per_gpu)))
    by_memory = int(max(1.0, math.floor((free_gb - args.min_free_vram_gb) / max(args.eval_worker_vram_gb, 1.0))))
    return min(max(base, by_memory), max(1, int(args.max_workers_per_gpu)))


def _maybe_grow_capacity(gpu: str, current: int, args: argparse.Namespace) -> int:
    max_workers = max(1, int(args.max_workers_per_gpu))
    if current >= max_workers:
        return current
    free_gb, _ = _query_gpu_memory_gb(gpu)
    if free_gb is None:
        return current
    needed = (current + 1) * max(args.eval_worker_vram_gb, 1.0) + args.min_free_vram_gb
    return current + 1 if free_gb >= needed else current


def _query_gpu_memory_gb(gpu: str) -> tuple[float | None, float | None]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return None, None
    if proc.returncode != 0:
        return None, None
    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 2:
        return None, None
    try:
        return float(parts[0]) / 1024.0, float(parts[1]) / 1024.0
    except ValueError:
        return None, None


def _run_with_peak_vram(command: list[str], log, env: dict[str, str], gpu: str | None) -> tuple[int, float | None]:
    """Sample device memory while a child runs so peak VRAM is per-stage, not an estimate."""
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
    peak = 0.0
    measured = False
    while proc.poll() is None:
        used = _query_gpu_used_memory_gb(gpu)
        if used is not None:
            peak = max(peak, used)
            measured = True
        time.sleep(0.2)
    used = _query_gpu_used_memory_gb(gpu)
    if used is not None:
        peak = max(peak, used)
        measured = True
    return proc.returncode, peak if measured else None


def _query_gpu_used_memory_gb(gpu: str | None) -> float | None:
    if gpu is None:
        return None
    try:
        proc = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=2)
        return float(proc.stdout.strip().splitlines()[0]) / 1024.0 if proc.returncode == 0 and proc.stdout.strip() else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _index_from_sample_dir(out_dir: Path) -> int:
    try:
        return int(out_dir.parent.name.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Cannot infer MeshFleet index from {out_dir.parent}") from exc


def _uid_for_index(args: argparse.Namespace, index: int) -> str:
    mapping = getattr(args, "_uid_by_index", None)
    if not isinstance(mapping, dict) or index not in mapping:
        raise KeyError(f"No exact MeshFleet UID recorded for evaluation index {index}.")
    return str(mapping[index])


def _child_env(gpu: str | None) -> dict[str, str]:
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.setdefault("NCCL_P2P_DISABLE", "0")
    env.setdefault("NCCL_SHM_DISABLE", "0")
    env.pop("NCCL_ASYNC_ERROR_HANDLING", None)
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512,garbage_collection_threshold:0.9")
    return env


def _log_looks_like_oom(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    return any(
        needle in text
        for needle in (
            "cuda out of memory",
            "torch.cuda.outofmemoryerror",
            "cublas_status_alloc_failed",
            "cudnn_status_alloc_failed",
            "out of memory",
        )
    )


def _is_oom_row(row: dict[str, Any]) -> bool:
    return bool(row.get("oom"))


def _mark_retry(output_dir: Path, index: int, gpu: str, old_capacity: int, new_capacity: int, retry: int) -> None:
    path = output_dir / "adaptive_scheduler.jsonl"
    record = {
        "event": "oom_retry",
        "time": time.time(),
        "index": index,
        "gpu": gpu,
        "old_workers_per_gpu": old_capacity,
        "new_workers_per_gpu": new_capacity,
        "retry": retry,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


if __name__ == "__main__":
    main()
