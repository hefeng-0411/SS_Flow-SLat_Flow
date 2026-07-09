from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from geoss.utils.config import add_common_args
from geoss.utils.run_mode import validate_real_mode


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--prediction", type=str, default=None)
    parser.add_argument("--gt_occ", type=str, default=None)
    parser.add_argument("--real_eval", action="store_true")
    args = parser.parse_args()
    run_modes = validate_real_mode(cfg={}, args=args, mode="real_eval", required=("prediction", "gt")) if not args.dry_run else {
        "vggt_mode": "mock",
        "trellis_mode": "mock",
        "data_mode": "synthetic",
        "decoder_enabled": False,
        "render_eval_enabled": False,
        "official_metrics": False,
    }
    pred, gt = _load_or_make_eval_tensors(args)
    metrics = _occupancy_metrics(pred, gt)
    metrics.update(_component_metrics(pred))
    metrics.update(_no_gt_proxy_metrics(pred, gt))
    metrics = {
        **metrics,
        **run_modes,
        "ablations": [
            {"name": "original_trellis_ss", "status": "requires_real_trellis_output"},
            {"name": "vggt_pointmap_voxelize", "status": "requires_real_vggt_pointmap"},
            {"name": "dense_voxel_query_adapter", "status": "not_selected"},
            {"name": "sparse_anchor_query_adapter", "status": "implemented"},
            {"name": "no_ray_free_space", "status": "config_switch_expected"},
            {"name": "no_confidence_gate", "status": "config_switch_expected"},
            {"name": "no_trust_region_clipping", "status": "config_switch_expected"},
            {"name": "no_timestep_alpha", "status": "config_switch_expected"},
            {"name": "no_velocity_level_control", "status": "config_switch_expected"},
            {"name": "full_sparse_ray_geoss", "status": "implemented"},
        ],
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "eval_sparse_ray_geoss.json").write_text(json.dumps(metrics, indent=2))
    _write_csv(Path(args.output_dir) / "eval_sparse_ray_geoss.csv", metrics)
    print(json.dumps(metrics, indent=2))


def _write_csv(path: Path, metrics: dict) -> None:
    import csv

    flat = {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in metrics.items()}
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(flat))
        writer.writeheader()
        writer.writerow(flat)


def _load_or_make_eval_tensors(args):
    if args.prediction and Path(args.prediction).exists():
        data = np.load(args.prediction)
        key = "occ" if "occ" in data else list(data.keys())[0]
        pred = torch.tensor(data[key]).float()
    else:
        if not args.dry_run:
            raise FileNotFoundError("real_eval requires --prediction; fake occupancy fallback is allowed only with --dry_run true.")
        pred = torch.zeros(1, 32, 32, 32)
        pred[:, 10:22, 12:20, 12:20] = 1
    if args.gt_occ and Path(args.gt_occ).exists():
        obj = torch.load(args.gt_occ, map_location="cpu")
        gt = obj["gt_occ"] if isinstance(obj, dict) and "gt_occ" in obj else obj
        gt = gt.float()
    else:
        if not args.dry_run:
            raise FileNotFoundError("real_eval requires --gt_occ or a real dataset-backed evaluator; fake GT fallback is allowed only with --dry_run true.")
        gt = torch.zeros_like(pred)
        gt[:, 11:23, 12:20, 12:20] = 1
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
    if gt.ndim == 3:
        gt = gt.unsqueeze(0)
    return pred > 0.5, gt > 0.5


def _occupancy_metrics(pred, gt):
    pred = pred.bool()
    gt = gt.bool()
    tp = (pred & gt).sum().float()
    fp = (pred & ~gt).sum().float()
    fn = (~pred & gt).sum().float()
    union = (pred | gt).sum().float().clamp_min(1)
    iou = tp / union
    dice = (2 * tp) / (pred.sum().float() + gt.sum().float()).clamp_min(1)
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    chamfer = _chamfer_distance(pred, gt)
    fscore = (2 * precision * recall / (precision + recall).clamp_min(1e-6))
    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "chamfer_distance": float(chamfer),
        "f_score": float(fscore),
    }


def _component_metrics(pred):
    count = _connected_components_count(pred[0])
    surface = _surface_voxels(pred[0]).sum().float()
    volume = pred[0].sum().float().clamp_min(1)
    return {"connected_component_count": int(count), "surface_compactness": float(surface / volume)}


def _no_gt_proxy_metrics(pred, gt):
    proj_pred = pred.any(dim=1)
    proj_gt = gt.any(dim=1)
    sil_iou = ((proj_pred & proj_gt).sum().float() / (proj_pred | proj_gt).sum().float().clamp_min(1))
    violation = (pred & ~gt).sum().float() / pred.sum().float().clamp_min(1)
    return {
        "silhouette_iou": float(sil_iou),
        "depth_consistency": float(1.0 - violation.clamp(0, 1)),
        "reprojection_consistency": float(sil_iou),
        "free_space_violation_rate": float(violation),
        "confidence_error_correlation": None,
        "connected_component_statistics": {"count": int(_connected_components_count(pred[0]))},
    }


def _surface_voxels(occ):
    padded = torch.nn.functional.pad(occ.float()[None, None], (1, 1, 1, 1, 1, 1))[0, 0]
    center = padded[1:-1, 1:-1, 1:-1] > 0.5
    eroded = center.clone()
    for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        eroded &= padded[1 + dx : 1 + dx + occ.shape[0], 1 + dy : 1 + dy + occ.shape[1], 1 + dz : 1 + dz + occ.shape[2]] > 0.5
    return center & ~eroded


def _chamfer_distance(pred, gt):
    p = torch.nonzero(_surface_voxels(pred[0]), as_tuple=False).float()
    g = torch.nonzero(_surface_voxels(gt[0]), as_tuple=False).float()
    if p.numel() == 0 or g.numel() == 0:
        return torch.tensor(float("inf"))
    d = torch.cdist(p, g)
    return d.min(dim=1).values.mean() + d.min(dim=0).values.mean()


def _connected_components_count(occ):
    occ_np = occ.cpu().numpy().astype(bool)
    visited = np.zeros_like(occ_np, dtype=bool)
    count = 0
    dims = occ_np.shape
    for start in zip(*np.nonzero(occ_np & ~visited)):
        if visited[start]:
            continue
        count += 1
        stack = [start]
        visited[start] = True
        while stack:
            x, y, z = stack.pop()
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < dims[0] and 0 <= ny < dims[1] and 0 <= nz < dims[2] and occ_np[nx, ny, nz] and not visited[nx, ny, nz]:
                    visited[nx, ny, nz] = True
                    stack.append((nx, ny, nz))
    return count


if __name__ == "__main__":
    main()
