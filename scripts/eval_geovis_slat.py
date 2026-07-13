from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from geoss.datasets.meshfleet_trellis_dataset import MeshFleetTrellisDataset
from geoss.io.asset_io import read_gaussian_ply
from geoss.metrics.gaussian_metrics import gaussian_statistics
from geoss.metrics.geometry_metrics import geometry_metrics
from geoss.metrics.render_metrics import image_render_metrics
from geoss.renderers.gsplat_renderer import render_gaussians
from geoss.utils.config import str2bool
from geoss.utils.run_mode import validate_real_mode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="outputs/geovis_slat_infer_dry")
    parser.add_argument("--output_dir", type=str, default="outputs/geovis_slat_eval")
    parser.add_argument("--ablation", type=str, default="full_geovis_slat")
    parser.add_argument("--dry_run", type=str2bool, default=False)
    parser.add_argument("--render_eval", type=str2bool, default=True)
    parser.add_argument("--real_eval", action="store_true")
    parser.add_argument("--prediction", type=str, default=None)
    parser.add_argument("--gt_render", type=str, default=None)
    parser.add_argument("--gaussian_ply", type=str, default=None)
    parser.add_argument("--pred_points", type=str, default=None)
    parser.add_argument("--gt_points", type=str, default=None)
    parser.add_argument("--meshfleet_root", type=str, default=None)
    parser.add_argument("--meshfleet_split", type=str, default="test")
    parser.add_argument("--meshfleet_category", type=str, default=None)
    parser.add_argument("--meshfleet_index", type=int, default=0)
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--background_color", type=float, nargs=3, default=(0.0, 0.0, 0.0))
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
    elif args.gaussian_ply and args.meshfleet_root:
        # Asset rendering is evaluated independently of SLAT success, which lets
        # Stage 2 receive PSNR/SSIM/LPIPS even while a later stage is unhealthy.
        metrics.update(_evaluate_gaussian_renders(args))
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


@torch.no_grad()
def _evaluate_gaussian_renders(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    dataset = MeshFleetTrellisDataset(
        args.meshfleet_root,
        split=args.meshfleet_split,
        category=args.meshfleet_category,
        num_views=args.num_views,
        image_size=args.image_size,
    )
    if not len(dataset):
        raise FileNotFoundError(f"No renderable MeshFleet samples under {args.meshfleet_root} ({args.meshfleet_split}).")
    sample = dataset[min(args.meshfleet_index, len(dataset) - 1)]
    gaussian = {key: value.to(device) for key, value in read_gaussian_ply(args.gaussian_ply, real_mode=True).items()}
    images = sample["images"].to(device=device, dtype=torch.float32)
    cameras = {"K": sample["K"].to(device=device, dtype=torch.float32), "w2c": sample["w2c"].to(device=device, dtype=torch.float32)}
    backgrounds = torch.tensor(args.background_color, device=device, dtype=torch.float32).view(1, 3).expand(images.shape[0], -1)
    rendered = render_gaussians(gaussian, cameras, tuple(images.shape[-2:]), backgrounds=backgrounds)
    view_metrics = []
    for view in range(images.shape[0]):
        item = image_render_metrics(rendered["rendered_rgb"][view : view + 1], images[view : view + 1])
        item["view"] = view
        view_metrics.append(item)
    aggregate = {f"render_{name}": float(np.mean([item[name] for item in view_metrics])) for name in ("PSNR", "SSIM", "LPIPS")}
    aggregate.update({"render_num_views": len(view_metrics), "render_view_metrics": view_metrics})
    return aggregate


def _write_csv(path: Path, metrics: dict) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(metrics))
        writer.writeheader()
        writer.writerow(metrics)


if __name__ == "__main__":
    main()
