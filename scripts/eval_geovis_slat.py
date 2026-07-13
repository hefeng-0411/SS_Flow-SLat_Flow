from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from geoss.io.asset_io import read_gaussian_ply
from geoss.metrics.gaussian_metrics import gaussian_statistics
from geoss.metrics.geometry_metrics import geometry_metrics
from geoss.metrics.render_metrics import image_render_metrics
from geoss.utils.config import str2bool
from geoss.utils.run_mode import validate_real_mode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="outputs/geovis_slat_infer_dry")
    parser.add_argument("--output_dir", type=str, default="outputs/geovis_slat_eval")
    parser.add_argument("--ablation", type=str, default="full_geovis_slat")
    parser.add_argument("--dry_run", type=str2bool, default=False)
    parser.add_argument("--real_eval", action="store_true")
    parser.add_argument("--prediction", type=str, default=None)
    parser.add_argument("--gt_render", type=str, default=None)
    parser.add_argument("--gaussian_ply", type=str, default=None)
    parser.add_argument("--pred_points", type=str, default=None)
    parser.add_argument("--gt_points", type=str, default=None)
    args = parser.parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_modes = validate_real_mode(cfg={}, args=args, mode="real_eval", required=("render",)) if not args.dry_run else {
        "vggt_mode": "mock",
        "trellis_mode": "mock",
        "data_mode": "synthetic",
        "decoder_enabled": False,
        "render_eval_enabled": False,
        "official_metrics": False,
    }
    metrics = {"ablation": args.ablation, **run_modes}
    controlled = in_dir / "geovis_controlled_slat.npz"
    original = in_dir / "original_slat.npz"
    has_asset_eval = bool(args.gaussian_ply or args.pred_points or (controlled.exists() and original.exists()))
    if args.prediction and args.gt_render:
        pred = np.load(args.prediction)
        gt = np.load(args.gt_render)
        metrics.update(
            image_render_metrics(
                torch.tensor(pred["rgb"]),
                torch.tensor(gt["rgb"]),
                torch.tensor(pred["mask"]) if "mask" in pred else None,
                torch.tensor(gt["mask"]) if "mask" in gt else None,
            )
        )
    elif not args.dry_run and not has_asset_eval:
        raise FileNotFoundError("real_eval requires render npz files or asset/latent outputs such as --gaussian_ply, --pred_points, or geovis_controlled_slat.npz.")
    if args.gaussian_ply:
        metrics.update(gaussian_statistics(read_gaussian_ply(args.gaussian_ply, real_mode=not args.dry_run)))
    if args.pred_points:
        pred_points = torch.tensor(np.load(args.pred_points)["points"]).float()
        gt_points = torch.tensor(np.load(args.gt_points)["points"]).float() if args.gt_points else None
        metrics.update(geometry_metrics(pred_points, gt_points, real_mode=not args.dry_run))
    if controlled.exists() and original.exists():
        c = np.load(controlled)["feats"]
        o = np.load(original)["feats"]
        metrics["slat_latent_l2"] = float(np.mean((c - o) ** 2))
        metrics["prior_preservation_error"] = float(np.mean(np.abs(c - o)))
    visibility = in_dir / "slat_visibility_debug.npz"
    if visibility.exists():
        v = np.load(visibility)["visibility"]
        metrics["visible_region_consistency"] = float(np.mean(v))
        metrics["occluded_region_stability"] = float(1.0 - np.std(v))
    velocity = in_dir / "slat_velocity_debug.npz"
    if velocity.exists():
        d = np.load(velocity)["delta_v_slat"]
        metrics["slat_velocity_residual_norm"] = float(np.linalg.norm(d, axis=-1).mean())
    metrics.setdefault("Gaussian_count", None)
    metrics.setdefault("opacity_mean", None)
    metrics.setdefault("scale_abnormal_ratio", None)
    metrics.setdefault("floating_splat_ratio", None)
    (out_dir / "geovis_slat_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_csv(out_dir / "geovis_slat_metrics.csv", metrics)
    print(json.dumps(metrics, indent=2))


def _write_csv(path: Path, metrics: dict) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(metrics))
        writer.writeheader()
        writer.writerow(metrics)


if __name__ == "__main__":
    main()
