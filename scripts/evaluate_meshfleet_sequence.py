from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
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
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--occ_resolution", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config_geoss", type=str, default="configs/sparse_ray_geoss.yaml")
    parser.add_argument("--config_slat", type=str, default="configs/geovis_slat.yaml")
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
    parser.add_argument("--overwrite", type=str2bool, default=False)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir) if args.output_dir else run_root / "evaluation_suite"
    output_dir.mkdir(parents=True, exist_ok=True)
    _fill_checkpoint_defaults(args, run_root)

    dataset = MeshFleetTrellisDataset(
        args.data_root,
        split=args.split,
        category=args.category,
        num_views=args.num_views,
        image_size=args.image_size,
        occ_resolution=args.occ_resolution,
        require_voxels=True,
    )
    if len(dataset) == 0:
        raise FileNotFoundError(f"No MeshFleet samples found under {args.data_root} split={args.split}.")
    indices = list(range(args.start_index, min(len(dataset), args.start_index + args.max_samples)))
    manifest = {
        "data_root": args.data_root,
        "run_root": str(run_root),
        "split": args.split,
        "num_dataset_samples": len(dataset),
        "evaluated_indices": indices,
        "checkpoints": {
            "geoss": args.geoss_checkpoint,
            "ss_velocity": args.ss_checkpoint,
            "slat": args.slat_checkpoint,
            "slat_joint": args.slat_joint_checkpoint,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for index in indices:
        sample_root = output_dir / f"{args.split}_{index:06d}"
        sample_root.mkdir(parents=True, exist_ok=True)
        if args.run_original_trellis:
            rows.append(_run_original_trellis(args, sample_root, index))
        stage2_dir = sample_root / "stage2_geoss_ss"
        if args.run_stage1:
            rows.append(_run_stage1(args, sample_root, index))
        if args.run_stage2:
            rows.append(_run_stage2(args, sample_root, index, stage2_dir))
        geoss_context = stage2_dir / "geoss_context.pt"
        if args.run_stage3:
            rows.append(_run_slat_stage(args, sample_root, index, "stage3_geovis_slat", args.slat_checkpoint, geoss_context))
        if args.run_stage4:
            rows.append(_run_slat_stage(args, sample_root, index, "stage4_geovis_slat_joint", args.slat_joint_checkpoint, geoss_context))

    _write_jsonl(output_dir / "per_sample_metrics.jsonl", rows)
    summary = _aggregate(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_rows_csv(output_dir / "per_sample_metrics.csv", rows)
    _write_summary_csv(output_dir / "summary.csv", summary)
    print(json.dumps({"samples": len(indices), "rows": len(rows), "output_dir": str(output_dir)}, indent=2))


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


def _run_original_trellis(args: argparse.Namespace, sample_root: Path, index: int) -> dict[str, Any]:
    out_dir = sample_root / "original_trellis"
    command = _geoss_command(args, out_dir, index, decode=True, disable_ss_adapter=True)
    return _run_and_collect("original_trellis", command, out_dir, args.overwrite)


def _run_stage1(args: argparse.Namespace, sample_root: Path, index: int) -> dict[str, Any]:
    out_dir = sample_root / "stage1_geoss_context"
    command = _geoss_command(args, out_dir, index, decode=False, geoss_checkpoint=args.geoss_checkpoint)
    return _run_and_collect("stage1_geoss_context", command, out_dir, args.overwrite)


def _run_stage2(args: argparse.Namespace, sample_root: Path, index: int, out_dir: Path) -> dict[str, Any]:
    command = _geoss_command(
        args,
        out_dir,
        index,
        decode=True,
        geoss_checkpoint=args.geoss_checkpoint,
        ss_checkpoint=args.ss_checkpoint,
    )
    return _run_and_collect("stage2_geoss_ss", command, out_dir, args.overwrite, evaluate_assets=True)


def _run_slat_stage(args: argparse.Namespace, sample_root: Path, index: int, name: str, checkpoint: str, geoss_context: Path) -> dict[str, Any]:
    out_dir = sample_root / name
    command = [
        sys.executable,
        "scripts/infer_geovis_slat.py",
        "--config",
        args.config_slat,
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
        "--decode",
        "true",
        "--real_infer",
    ]
    if args.trellis_root:
        command += ["--trellis_root", str(args.trellis_root)]
    if args.category:
        command += ["--meshfleet_category", args.category]
    return _run_and_collect(name, command, out_dir, args.overwrite, evaluate_assets=True)


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
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    log_path = out_dir / "command.log"
    if overwrite or not metrics_path.exists():
        with log_path.open("w", encoding="utf-8") as log:
            log.write("==== COMMAND ====\n" + " ".join(command) + "\n\n")
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            return {"ablation": ablation, "status": "failed", "returncode": proc.returncode, "run_path": str(out_dir), "log": str(log_path)}
    row = _read_json(metrics_path)
    row.update({"ablation": ablation, "status": "ok", "run_path": str(out_dir), "log": str(log_path)})
    if evaluate_assets:
        row.update(_eval_assets(ablation, out_dir))
    return row


def _eval_assets(ablation: str, out_dir: Path) -> dict[str, Any]:
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
        "--real_eval",
    ]
    log_path = eval_dir / "command.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("==== COMMAND ====\n" + " ".join(command) + "\n\n")
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        return {"asset_eval_status": "failed", "asset_eval_log": str(log_path)}
    metrics = _read_json(eval_dir / "geovis_slat_metrics.json")
    return {f"asset_{k}": v for k, v in metrics.items() if isinstance(v, (int, float, str, bool)) or v is None}


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


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"by_ablation": {}}
    for ablation in sorted({str(row.get("ablation")) for row in rows}):
        subset = [row for row in rows if row.get("ablation") == ablation]
        ok = [row for row in subset if row.get("status") == "ok"]
        metric_names = sorted({key for row in ok for key, value in row.items() if isinstance(value, (int, float)) and not isinstance(value, bool)})
        metrics = {}
        for name in metric_names:
            values = [float(row[name]) for row in ok if isinstance(row.get(name), (int, float)) and math.isfinite(float(row[name]))]
            if not values:
                continue
            mean = sum(values) / len(values)
            var = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
            std = math.sqrt(var)
            metrics[name] = {"mean": mean, "std": std, "ci95": 1.96 * std / math.sqrt(len(values)), "n": len(values)}
        out["by_ablation"][ablation] = {"num_ok": len(ok), "num_failed": len(subset) - len(ok), "metrics": metrics}
    return out


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


if __name__ == "__main__":
    main()
