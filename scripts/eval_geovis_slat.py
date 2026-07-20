from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from geoss.datasets.meshfleet_trellis_dataset import MeshFleetTrellisDataset
from geoss.io.asset_io import read_gaussian_ply, trellis_export_gaussian_to_internal
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
    parser.add_argument("--meshfleet_uid", type=str, default=None, help="Exact UID; preferred over layout-dependent --meshfleet_index.")
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--conditioning_num_views", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--background_color", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--eval_view_set", choices=("renders_eval_70", "renders_eval_90"), default="renders_eval_70")
    parser.add_argument("--conditioning_view_set", choices=("renders", "renders_cond"), default="renders")
    parser.add_argument("--allow_conditioning_overlap", type=str2bool, default=False)
    parser.add_argument("--pred_mesh", type=str, default=None)
    parser.add_argument("--gt_mesh", type=str, default=None)
    parser.add_argument("--geometry_samples", type=int, default=100000)
    parser.add_argument("--geometry_seed", type=int, default=20260720)
    parser.add_argument("--fscore_threshold", type=float, default=0.01)
    parser.add_argument("--distance_chunk_size", type=int, default=2048)
    parser.add_argument("--save_visuals", type=str2bool, default=True)
    parser.add_argument("--inference_metrics", type=str, default=None, help="Optional inference provenance JSON checked for GT-latent leakage.")
    args = parser.parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        # Validate that this is a real artifact evaluation, then describe only
        # resources used by this process. VGGT/TRELLIS already produced the PLY
        # upstream; calling them "mock" here is false provenance.
        validate_real_mode(cfg={}, args=args, mode="real_eval", required=("render",))
        run_modes = {
            "vggt_mode": "not_used_by_asset_evaluator",
            "trellis_mode": "not_used_by_asset_evaluator",
            "data_mode": "real",
            "decoder_enabled": False,
            "render_eval_enabled": bool(args.render_eval),
            "official_metrics": False,
            "evaluation_protocol": "meshfleet_heldout_v2",
        }
    else:
        run_modes = {
            "vggt_mode": "mock",
            "trellis_mode": "mock",
            "data_mode": "synthetic",
            "decoder_enabled": False,
            "render_eval_enabled": False,
            "official_metrics": False,
            "evaluation_protocol": "dry_run",
        }
    metrics = {"ablation": args.ablation, **run_modes}
    appearance_valid = False
    geometry_valid = False
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
        appearance_valid = bool(args.real_eval)
    elif args.gaussian_ply and args.meshfleet_root:
        # Asset rendering is evaluated independently of SLAT success, which lets
        # Stage 2 receive PSNR/SSIM/LPIPS even while a later stage is unhealthy.
        render_metrics = _evaluate_gaussian_renders(args, out_dir)
        metrics.update(render_metrics)
        appearance_valid = bool(args.real_eval and render_metrics.get("appearance_protocol_valid"))
    elif not args.dry_run and not has_asset_eval:
        raise FileNotFoundError("real_eval requires render npz files or asset/latent outputs such as --gaussian_ply, --pred_points, or geovis_controlled_slat.npz.")
    if args.gaussian_ply:
        metrics.update(gaussian_statistics(read_gaussian_ply(args.gaussian_ply, real_mode=not args.dry_run)))
    if args.pred_points:
        pred_points = torch.tensor(np.load(args.pred_points)["points"]).float()
        gt_points = torch.tensor(np.load(args.gt_points)["points"]).float() if args.gt_points else None
        metrics.update(
            geometry_metrics(
                pred_points,
                gt_points,
                threshold=args.fscore_threshold,
                real_mode=not args.dry_run,
                chunk_size=args.distance_chunk_size,
            )
        )
        geometry_valid = gt_points is not None
    elif not args.dry_run:
        pred_mesh = Path(args.pred_mesh) if args.pred_mesh else in_dir / "asset_mesh_internal.ply"
        gt_mesh = Path(args.gt_mesh) if args.gt_mesh else _meshfleet_gt_mesh(args)
        if pred_mesh.is_file() and gt_mesh is not None and gt_mesh.is_file():
            mesh_metrics = _evaluate_mesh_geometry(pred_mesh, gt_mesh, args)
            metrics.update(mesh_metrics)
            geometry_valid = bool(mesh_metrics.get("geometry_protocol_valid"))
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
    if "render_PSNR" in metrics:
        metrics["PSNR"] = metrics["render_PSNR"]
        metrics["SSIM"] = metrics["render_SSIM"]
        metrics["LPIPS"] = metrics["render_LPIPS"]
    if "Chamfer Distance" in metrics:
        metrics["CD"] = metrics["Chamfer Distance"]
    metrics["appearance_metrics_valid"] = appearance_valid
    metrics["geometry_metrics_valid"] = geometry_valid
    metrics["official_metrics"] = bool(args.real_eval and appearance_valid and geometry_valid)
    required = ("PSNR", "SSIM", "LPIPS", "CD", "F-score")
    metrics["missing_official_metrics"] = [name for name in required if name not in metrics]
    metrics.setdefault("Gaussian_count", None)
    metrics.setdefault("opacity_mean", None)
    metrics.setdefault("scale_abnormal_ratio", None)
    metrics.setdefault("floating_splat_ratio", None)
    (out_dir / "geovis_slat_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_csv(out_dir / "geovis_slat_metrics.csv", metrics)
    print(json.dumps(metrics, indent=2))


@torch.no_grad()
def _evaluate_gaussian_renders(args: argparse.Namespace, out_dir: Path) -> dict:
    device = torch.device(args.device)
    dataset = MeshFleetTrellisDataset(
        args.meshfleet_root,
        split=args.meshfleet_split,
        category=args.meshfleet_category,
        num_views=args.num_views,
        image_size=args.image_size,
        render_set=args.eval_view_set,
        background_color=args.background_color,
        repeat_views_if_insufficient=False,
        uid_manifest=[args.meshfleet_uid] if args.meshfleet_uid else None,
    )
    if not len(dataset):
        raise FileNotFoundError(f"No renderable MeshFleet samples under {args.meshfleet_root} ({args.meshfleet_split}).")
    if args.meshfleet_uid:
        sample = dataset.get_by_uid(args.meshfleet_uid)
    else:
        if not 0 <= args.meshfleet_index < len(dataset):
            raise IndexError(f"meshfleet_index={args.meshfleet_index} is outside dataset length {len(dataset)}")
        sample = dataset[args.meshfleet_index]
    conditioning = MeshFleetTrellisDataset(
        args.meshfleet_root,
        split=args.meshfleet_split,
        category=args.meshfleet_category,
        num_views=args.conditioning_num_views or args.num_views,
        image_size=args.image_size,
        render_set=args.conditioning_view_set,
        background_color=args.background_color,
        repeat_views_if_insufficient=False,
        uid_manifest=[sample["uid"]],
    )
    cond_sample = conditioning.get_by_uid(sample["uid"])
    if cond_sample["uid"] != sample["uid"]:
        raise RuntimeError(f"conditioning/evaluation UID mismatch: {cond_sample['uid']} != {sample['uid']}")
    overlap = _camera_overlap_count(cond_sample["c2w"], sample["c2w"])
    if overlap and not args.allow_conditioning_overlap:
        raise RuntimeError(f"Detected {overlap} evaluation cameras duplicated in conditioning views.")
    # Official appearance numbers require positive provenance: absence of a
    # record is unknown, not evidence that test-time GT 3D products were unused.
    leakage_free = False
    inference_provenance = {}
    if args.inference_metrics:
        provenance_path = Path(args.inference_metrics)
        if provenance_path.is_file():
            inference_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            leakage_free = inference_provenance.get("test_time_ground_truth_latents_used") is False
    gaussian_cpu = trellis_export_gaussian_to_internal(read_gaussian_ply(args.gaussian_ply, real_mode=True))
    gaussian = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in gaussian_cpu.items()}
    images = sample["images"].to(device=device, dtype=torch.float32)
    gt_masks = sample.get("masks")
    if isinstance(gt_masks, torch.Tensor):
        gt_masks = gt_masks.to(device=device, dtype=torch.float32)
    cameras = {"K": sample["K"].to(device=device, dtype=torch.float32), "w2c": sample["w2c"].to(device=device, dtype=torch.float32)}
    backgrounds = torch.tensor(args.background_color, device=device, dtype=torch.float32).view(1, 3).expand(images.shape[0], -1)
    rendered = render_gaussians(gaussian, cameras, tuple(images.shape[-2:]), backgrounds=backgrounds)
    pred_masks = rendered.get("rendered_alpha")
    if isinstance(pred_masks, torch.Tensor) and pred_masks.ndim == 4 and pred_masks.shape[-1] == 1:
        pred_masks = pred_masks.permute(0, 3, 1, 2).contiguous()
    view_metrics = []
    for view in range(images.shape[0]):
        item = image_render_metrics(
            rendered["rendered_rgb"][view : view + 1],
            images[view : view + 1],
            pred_masks[view : view + 1] if isinstance(pred_masks, torch.Tensor) else None,
            gt_masks[view : view + 1] if isinstance(gt_masks, torch.Tensor) else None,
        )
        item["view"] = view
        item["frame_id"] = sample["metadata"]["selected_frame_ids"][view]
        view_metrics.append(item)
    numeric_names = sorted(set.intersection(*(set(item) for item in view_metrics)) - {"view"})
    aggregate = {
        f"render_{name}": float(np.mean([item[name] for item in view_metrics]))
        for name in numeric_names
        if all(isinstance(item.get(name), (int, float)) for item in view_metrics)
    }
    if args.save_visuals:
        _save_render_artifacts(out_dir / "visuals", rendered, images, gt_masks)
        _save_conditioning_artifacts(out_dir / "visuals" / "conditioning", cond_sample["images"])
    aggregate.update(
        {
            "render_num_views": len(view_metrics),
            "render_view_metrics": view_metrics,
            "render_uid": sample["uid"],
            "render_view_set": args.eval_view_set,
            "conditioning_view_set": args.conditioning_view_set,
            "conditioning_num_views": int(cond_sample["images"].shape[0]),
            "conditioning_camera_overlap": overlap,
            "background_color": [float(v) for v in args.background_color],
            "appearance_protocol_valid": bool(overlap == 0 and args.eval_view_set.startswith("renders_eval_") and leakage_free),
            "test_time_ground_truth_latents_used": inference_provenance.get("test_time_ground_truth_latents_used"),
            "inference_context_source": inference_provenance.get("inference_context_source"),
            "render_metric_protocol": {
                "color_space": "sRGB [0,1]",
                "alpha": "GT RGBA composited on declared background",
                "PSNR": "per-view RGB MSE then object mean",
                "SSIM": "11x11 Gaussian window sigma=1.5",
                "LPIPS": "lpips.LPIPS(net='vgg') on [-1,1]",
                "view_role": "held-out cameras only",
                "leakage_check": "conditioning/evaluation cameras disjoint and inference provenance contains no GT 3D latents",
            },
        }
    )
    return aggregate


def _meshfleet_gt_mesh(args: argparse.Namespace) -> Optional[Path]:
    if not args.meshfleet_root:
        return None
    dataset = MeshFleetTrellisDataset(
        args.meshfleet_root,
        split=args.meshfleet_split,
        category=args.meshfleet_category,
        num_views=1,
        image_size=max(1, args.image_size),
        render_set=args.eval_view_set,
        background_color=args.background_color,
        repeat_views_if_insufficient=False,
        uid_manifest=[args.meshfleet_uid] if args.meshfleet_uid else None,
    )
    if args.meshfleet_uid:
        index = dataset.index_for_uid(args.meshfleet_uid)
    else:
        if not 0 <= args.meshfleet_index < len(dataset):
            raise IndexError(f"meshfleet_index={args.meshfleet_index} is outside dataset length {len(dataset)}")
        index = args.meshfleet_index
    path = dataset.samples[index].get("mesh_path")
    return Path(path) if path is not None else None


def _evaluate_mesh_geometry(pred_path: Path, gt_path: Path, args: argparse.Namespace) -> dict:
    from geoss.utils.optional_deps import require_dependency

    require_dependency("trimesh", real_mode=True, feature="deterministic mesh surface evaluation")
    import trimesh

    pred_mesh = trimesh.load(str(pred_path), force="mesh", process=False)
    gt_mesh = trimesh.load(str(gt_path), force="mesh", process=False)
    if isinstance(pred_mesh, trimesh.Scene):
        pred_mesh = trimesh.util.concatenate(tuple(pred_mesh.geometry.values()))
    if isinstance(gt_mesh, trimesh.Scene):
        gt_mesh = trimesh.util.concatenate(tuple(gt_mesh.geometry.values()))
    if len(pred_mesh.faces) == 0 or len(gt_mesh.faces) == 0:
        raise ValueError("Geometry evaluation requires non-empty triangle meshes.")
    rng_state = np.random.get_state()
    try:
        np.random.seed(int(args.geometry_seed))
        pred_points, _ = trimesh.sample.sample_surface(pred_mesh, int(args.geometry_samples))
        np.random.seed(int(args.geometry_seed) + 1)
        gt_points, _ = trimesh.sample.sample_surface(gt_mesh, int(args.geometry_samples))
    finally:
        np.random.set_state(rng_state)
    pred = torch.from_numpy(np.asarray(pred_points, dtype=np.float32)).to(args.device)
    gt = torch.from_numpy(np.asarray(gt_points, dtype=np.float32)).to(args.device)
    pred_lo, pred_hi = pred.amin(dim=0), pred.amax(dim=0)
    gt_lo, gt_hi = gt.amin(dim=0), gt.amax(dim=0)
    pred_extent = (pred_hi - pred_lo).amax()
    gt_extent = (gt_hi - gt_lo).amax().clamp_min(1e-8)
    extent_ratio = pred_extent / gt_extent
    # Do not independently normalize or ICP-align the prediction. A gross
    # extent mismatch invalidates the shared-canonical-frame protocol instead
    # of being hidden by metric-time fitting.
    extent_ratio_value = float(extent_ratio.detach().cpu().item())
    protocol_valid = bool(torch.isfinite(extent_ratio).item() and 0.25 <= extent_ratio_value <= 4.0)
    result = geometry_metrics(
        pred,
        gt,
        threshold=args.fscore_threshold,
        real_mode=True,
        chunk_size=args.distance_chunk_size,
    )
    if args.save_visuals:
        _write_geometry_overlay(Path(args.output_dir) / "visuals" / "geometry_overlay.ply", pred, gt)
    result.update(
        {
            "geometry_protocol_valid": protocol_valid,
            "geometry_num_surface_samples": int(args.geometry_samples),
            "geometry_sampling_seed": int(args.geometry_seed),
            "geometry_coordinate_protocol": "shared MeshFleet/TRELLIS canonical frame; no per-shape alignment or normalization",
            "geometry_pred_mesh": str(pred_path),
            "geometry_gt_mesh": str(gt_path),
            "geometry_pred_bounds": [pred_lo.detach().cpu().tolist(), pred_hi.detach().cpu().tolist()],
            "geometry_gt_bounds": [gt_lo.detach().cpu().tolist(), gt_hi.detach().cpu().tolist()],
            "geometry_extent_ratio": extent_ratio_value,
        }
    )
    return result


def _camera_overlap_count(conditioning_c2w: torch.Tensor, evaluation_c2w: torch.Tensor, tolerance: float = 1e-5) -> int:
    cond = conditioning_c2w.detach().cpu().float().reshape(-1, 1, 4, 4)
    eval_ = evaluation_c2w.detach().cpu().float().reshape(1, -1, 4, 4)
    duplicate = (cond - eval_).abs().amax(dim=(-2, -1)) <= tolerance
    return int(duplicate.any(dim=0).sum().item())


def _save_render_artifacts(out_dir: Path, rendered: dict, gt_rgb: torch.Tensor, gt_mask: Optional[torch.Tensor]) -> None:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_device = rendered["rendered_rgb"].detach().float().clamp(0, 1)
    if pred_device.ndim == 4 and pred_device.shape[-1] == 3:
        pred_device = pred_device.permute(0, 3, 1, 2)
    gt_device = gt_rgb.detach().float().to(pred_device.device).clamp(0, 1)
    perceptual, perceptual_status = _spatial_lpips_maps(pred_device, gt_device)
    (out_dir / "perceptual_error_status.json").write_text(json.dumps(perceptual_status, indent=2), encoding="utf-8")
    pred = pred_device.cpu()
    gt = gt_rgb.detach().float().clamp(0, 1).cpu()
    depth = rendered.get("rendered_depth")
    alpha = rendered.get("rendered_alpha")
    pred_frames, comparison_frames = [], []
    for view in range(pred.shape[0]):
        pred_np = (pred[view].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        gt_np = (gt[view].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        error = (pred[view] - gt[view]).abs().mean(dim=0).numpy()
        error_scaled = np.clip(error / max(float(np.percentile(error, 99)), 1e-6), 0.0, 1.0)
        error_np = np.stack([error_scaled, np.zeros_like(error_scaled), 1.0 - error_scaled], axis=-1)
        error_np = (error_np * 255.0).round().astype(np.uint8)
        Image.fromarray(pred_np).save(out_dir / f"view_{view:03d}_pred.png")
        Image.fromarray(gt_np).save(out_dir / f"view_{view:03d}_gt.png")
        Image.fromarray(error_np).save(out_dir / f"view_{view:03d}_error.png")
        comparison = np.concatenate([gt_np, pred_np, error_np], axis=1)
        Image.fromarray(comparison).save(out_dir / f"view_{view:03d}_comparison.png")
        pred_frames.append(Image.fromarray(pred_np))
        comparison_frames.append(Image.fromarray(comparison))
        if isinstance(perceptual, torch.Tensor):
            perceptual_view = perceptual[view].squeeze().cpu().numpy()
            perceptual_view = np.clip(perceptual_view / max(float(np.percentile(perceptual_view, 99)), 1e-6), 0, 1)
            perceptual_rgb = np.stack([perceptual_view, np.sqrt(perceptual_view), 1.0 - perceptual_view], axis=-1)
            Image.fromarray((perceptual_rgb * 255).round().astype(np.uint8)).save(out_dir / f"view_{view:03d}_perceptual_error.png")
        if isinstance(alpha, torch.Tensor):
            alpha_view = alpha[view].detach().float().cpu().squeeze().clamp(0, 1).numpy()
            Image.fromarray((alpha_view * 255.0).round().astype(np.uint8)).save(out_dir / f"view_{view:03d}_alpha.png")
        if isinstance(depth, torch.Tensor):
            depth_view = depth[view].detach().float().cpu().squeeze().numpy()
            valid = np.isfinite(depth_view) & (depth_view > 0)
            depth_vis = np.zeros_like(depth_view, dtype=np.float32)
            if valid.any():
                lo, hi = np.percentile(depth_view[valid], [2, 98])
                depth_vis[valid] = np.clip((depth_view[valid] - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
            Image.fromarray((depth_vis * 255.0).round().astype(np.uint8)).save(out_dir / f"view_{view:03d}_depth.png")
            normal_vis = _depth_normals(depth_view, valid)
            Image.fromarray((normal_vis * 255.0).round().astype(np.uint8)).save(out_dir / f"view_{view:03d}_normal.png")
        if isinstance(gt_mask, torch.Tensor):
            mask_np = gt_mask[view].detach().float().cpu().squeeze().clamp(0, 1).numpy()
            Image.fromarray((mask_np * 255.0).round().astype(np.uint8)).save(out_dir / f"view_{view:03d}_gt_mask.png")
    if pred_frames:
        pred_frames[0].save(out_dir / "turntable_pred.gif", save_all=True, append_images=pred_frames[1:], duration=120, loop=0)
        comparison_frames[0].save(out_dir / "turntable_comparison.gif", save_all=True, append_images=comparison_frames[1:], duration=120, loop=0)


def _save_conditioning_artifacts(out_dir: Path, images: torch.Tensor) -> None:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    values = images.detach().float().clamp(0, 1).cpu()
    for view in range(values.shape[0]):
        array = (values[view].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(array).save(out_dir / f"view_{view:03d}_input.png")


def _spatial_lpips_maps(pred: torch.Tensor, gt: torch.Tensor) -> tuple[Optional[torch.Tensor], dict]:
    try:
        import lpips
        model = lpips.LPIPS(net="vgg", spatial=True).to(pred.device).eval()
        with torch.no_grad():
            maps = model(pred * 2.0 - 1.0, gt * 2.0 - 1.0)
        maps = torch.nn.functional.interpolate(maps, size=pred.shape[-2:], mode="bilinear", align_corners=False).detach().cpu()
        return maps, {"available": True, "backbone": "vgg", "spatial": True}
    except Exception as exc:
        return None, {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def _depth_normals(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    filled = np.where(valid, depth, 0.0).astype(np.float32)
    dy, dx = np.gradient(filled)
    normals = np.stack([-dx, -dy, np.ones_like(filled)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True).clip(1e-6)
    normals = normals * 0.5 + 0.5
    normals[~valid] = 0.0
    return normals


def _write_geometry_overlay(path: Path, pred: torch.Tensor, gt: torch.Tensor, max_points_each: int = 50000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pred_np = pred.detach().float().cpu().numpy()[:: max(1, pred.shape[0] // max_points_each)][:max_points_each]
    gt_np = gt.detach().float().cpu().numpy()[:: max(1, gt.shape[0] // max_points_each)][:max_points_each]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(pred_np) + len(gt_np)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point in pred_np:
            handle.write(f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} 255 64 64\n")
        for point in gt_np:
            handle.write(f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} 64 255 64\n")


def _write_csv(path: Path, metrics: dict) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(metrics))
        writer.writeheader()
        writer.writerow(metrics)


if __name__ == "__main__":
    main()
