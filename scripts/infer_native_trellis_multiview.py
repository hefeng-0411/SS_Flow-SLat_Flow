from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoss.datasets.meshfleet_trellis_dataset import MeshFleetTrellisDataset
from geoss.eval.render_metrics import _ssim
from geoss.integration.real_trellis_pipeline import RealTrellisGeoPipeline
from geoss.renderers.gsplat_renderer import render_gaussians
from geoss.utils.config import str2bool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Native leakage-free multi-view TRELLIS inference without VGGT/adapter overhead."
    )
    parser.add_argument("--output_dir", required=True)
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
    parser.add_argument("--trellis_root", default=None)
    parser.add_argument(
        "--trellis_model_path",
        default="microsoft/TRELLIS-image-large",
    )
    parser.add_argument("--mask_aware_crop", type=str2bool, default=False)
    parser.add_argument("--crop_padding", type=float, default=1.2)
    parser.add_argument(
        "--multi_image_mode",
        choices=("multidiffusion", "stochastic"),
        default="multidiffusion",
    )
    parser.add_argument("--ss_steps", type=int, default=12)
    parser.add_argument("--ss_cfg_strength", type=float, default=7.5)
    parser.add_argument("--slat_steps", type=int, default=12)
    parser.add_argument("--slat_cfg_strength", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--candidate_seeds",
        default=None,
        help=(
            "Optional comma-separated deterministic seed set. Candidates are "
            "selected using conditioning-view renders only; held-out evaluation "
            "views are never read."
        ),
    )
    parser.add_argument("--export_textured_glb", type=str2bool, default=False)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

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
        raise RuntimeError("Native TRELLIS inference received a forbidden 3D target.")
    if sample["images"].shape[0] != args.num_views:
        raise RuntimeError(
            f"UID {sample['uid']} provides {sample['images'].shape[0]} conditioning "
            f"views, but the frozen protocol requires {args.num_views}."
        )

    device = torch.device(args.device)
    pipeline = RealTrellisGeoPipeline(
        args.trellis_root,
        args.trellis_model_path,
        device=args.device,
    )
    images = sample["images"].to(device=device, dtype=torch.float32)
    masks = sample["masks"].to(device=device, dtype=torch.float32)
    cameras = {
        "K": sample["K"].to(device=device, dtype=torch.float32),
        "w2c": sample["w2c"].to(device=device, dtype=torch.float32),
    }
    candidate_seeds = _parse_candidate_seeds(args.candidate_seeds, args.seed)
    candidate_reports = []
    saved = None
    selected_seed = None
    selected_score = None
    for candidate_seed in candidate_seeds:
        outputs = pipeline.run(
            images,
            masks=masks,
            mask_aware_crop=args.mask_aware_crop,
            crop_padding=args.crop_padding,
            formats=("gaussian", "mesh"),
            seed=candidate_seed,
            multi_image_mode=args.multi_image_mode,
            ss_sampler_params={
                "steps": args.ss_steps,
                "cfg_strength": args.ss_cfg_strength,
            },
            slat_sampler_params={
                "steps": args.slat_steps,
                "cfg_strength": args.slat_cfg_strength,
            },
        )
        if len(candidate_seeds) == 1:
            selection = {"score": None, "selection_skipped": True}
        else:
            selection = _conditioning_selection_score(
                outputs,
                images,
                masks,
                cameras,
            )
        candidate_reports.append({"seed": candidate_seed, **selection})
        if (
            selected_seed is None
            or (
                selection["score"] is not None
                and (selected_score is None or selection["score"] < selected_score)
            )
        ):
            saved = pipeline.save_outputs(
                outputs,
                output_dir,
                export_textured_glb=args.export_textured_glb,
            )
            selected_seed = candidate_seed
            selected_score = (
                float(selection["score"])
                if selection["score"] is not None
                else None
            )
        del outputs
    if saved is None or selected_seed is None:
        raise RuntimeError("TRELLIS seed selection produced no valid decoded candidate.")
    metrics = {
        "status": "ok",
        "mode": (
            "native_multiview_trellis_mask_cropped"
            if args.mask_aware_crop
            else "native_multiview_trellis"
        ),
        "uid": sample["uid"],
        "saved_assets": saved,
        "conditioning_view_set": args.conditioning_view_set,
        "conditioning_frame_ids": sample["metadata"]["selected_frame_ids"],
        "num_conditioning_views": int(sample["images"].shape[0]),
        "conditioning_image_size": list(sample["images"].shape[-2:]),
        "multi_image_mode": args.multi_image_mode,
        "ss_sampler_params": {
            "steps": int(args.ss_steps),
            "cfg_strength": float(args.ss_cfg_strength),
        },
        "slat_sampler_params": {
            "steps": int(args.slat_steps),
            "cfg_strength": float(args.slat_cfg_strength),
        },
        "seed": int(selected_seed),
        "candidate_seeds": candidate_seeds,
        "candidate_selection": candidate_reports,
        "candidate_selection_metric": (
            "conditioning-only foreground L1 + 0.25 full L1 + "
            "0.2 (1-SSIM) + 0.25 alpha L1"
        ),
        "candidate_selected_score": selected_score,
        "trellis_root": (
            str(Path(args.trellis_root).resolve()) if args.trellis_root else None
        ),
        "trellis_model_path": args.trellis_model_path,
        "mask_aware_crop": bool(args.mask_aware_crop),
        "crop_padding": float(args.crop_padding) if args.mask_aware_crop else None,
        "vggt_loaded": False,
        "adapter_loaded": False,
        "test_time_ground_truth_latents_used": False,
        "test_time_ground_truth_mesh_used": False,
        "test_time_ground_truth_voxels_used": False,
        "evaluation_views_used": False,
        "inference_context_source": "conditioning_multiview_rgb_and_masks_only",
        "latency_seconds": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2), flush=True)


def _parse_candidate_seeds(value: str | None, fallback: int) -> list[int]:
    if value is None or not value.strip():
        return [int(fallback)]
    seeds = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        seed = int(raw)
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise ValueError("--candidate_seeds did not contain any integer seed.")
    return seeds


@torch.inference_mode()
def _conditioning_selection_score(
    outputs,
    images: torch.Tensor,
    masks: torch.Tensor,
    cameras,
):
    gaussian = outputs.get("gaussian")
    if not isinstance(gaussian, list) or not gaussian:
        raise RuntimeError("TRELLIS candidate has no Gaussian for conditioning selection.")
    foreground_values = []
    full_values = []
    ssim_values = []
    alpha_values = []
    for view in range(images.shape[0]):
        rendered = render_gaussians(
            gaussian[0],
            {
                "K": cameras["K"][view : view + 1],
                "w2c": cameras["w2c"][view : view + 1],
            },
            tuple(images.shape[-2:]),
            backgrounds=torch.zeros(1, 3, device=images.device),
            return_depth=False,
            return_visibility=False,
        )
        pred = rendered["rendered_rgb"].permute(0, 3, 1, 2).clamp(0.0, 1.0)
        alpha = rendered["rendered_alpha"].permute(0, 3, 1, 2).clamp(0.0, 1.0)
        target = images[view : view + 1]
        mask = masks[view : view + 1]
        absolute = (pred - target).abs()
        foreground_values.append(
            (absolute * mask).sum() / (mask.sum().clamp_min(1.0) * pred.shape[1])
        )
        full_values.append(absolute.mean())
        ssim_values.append(_ssim(pred, target))
        alpha_values.append((alpha - mask).abs().mean())
    foreground_l1 = torch.stack(foreground_values).mean()
    full_l1 = torch.stack(full_values).mean()
    ssim = torch.stack(ssim_values).mean()
    alpha_l1 = torch.stack(alpha_values).mean()
    score = foreground_l1 + 0.25 * full_l1 + 0.2 * (1.0 - ssim) + 0.25 * alpha_l1
    return {
        "score": float(score.cpu()),
        "foreground_l1": float(foreground_l1.cpu()),
        "full_l1": float(full_l1.cpu()),
        "ssim": float(ssim.cpu()),
        "alpha_l1": float(alpha_l1.cpu()),
    }


if __name__ == "__main__":
    main()
