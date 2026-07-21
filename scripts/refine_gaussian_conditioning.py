from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from geoss.datasets.meshfleet_trellis_dataset import MeshFleetTrellisDataset
from geoss.eval.render_metrics import _ssim
from geoss.losses.stable_bce import probability_binary_cross_entropy
from geoss.io.asset_io import (
    read_gaussian_ply,
    trellis_export_gaussian_to_internal,
    update_gaussian_ply_parameters,
)
from geoss.renderers.gsplat_renderer import render_gaussians
from geoss.utils.config import str2bool


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-free per-object Gaussian appearance refinement.")
    parser.add_argument("--gaussian_ply", required=True)
    parser.add_argument("--output_ply", required=True)
    parser.add_argument("--meshfleet_root", required=True)
    parser.add_argument("--meshfleet_split", default="test")
    parser.add_argument("--meshfleet_category", default=None)
    parser.add_argument("--meshfleet_index", type=int, default=0)
    parser.add_argument("--meshfleet_uid", default=None, help="Exact UID; preferred over layout-dependent --meshfleet_index.")
    parser.add_argument("--conditioning_view_set", choices=("renders", "renders_cond"), default="renders")
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--background_color", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--views_per_step", type=int, default=2)
    parser.add_argument("--lr_color", type=float, default=2e-2)
    parser.add_argument("--lr_opacity", type=float, default=5e-3)
    parser.add_argument("--optimize_scaling", type=str2bool, default=False)
    parser.add_argument("--lr_scaling", type=float, default=5e-4)
    parser.add_argument("--lambda_ssim", type=float, default=0.2)
    parser.add_argument("--lambda_mask", type=float, default=0.5)
    parser.add_argument("--lambda_prior", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.conditioning_view_set.startswith("renders_eval_"):
        raise ValueError("Evaluation views are forbidden during test-time refinement.")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = MeshFleetTrellisDataset(
        args.meshfleet_root,
        split=args.meshfleet_split,
        category=args.meshfleet_category,
        num_views=args.num_views,
        image_size=args.image_size,
        render_set=args.conditioning_view_set,
        background_color=args.background_color,
        repeat_views_if_insufficient=False,
        uid_manifest=[args.meshfleet_uid] if args.meshfleet_uid else None,
    )
    if args.meshfleet_uid:
        sample = dataset.get_by_uid(args.meshfleet_uid)
    else:
        if not 0 <= args.meshfleet_index < len(dataset):
            raise IndexError(f"meshfleet_index={args.meshfleet_index} outside dataset length {len(dataset)}")
        sample = dataset[args.meshfleet_index]
    images = sample["images"].to(device=device, dtype=torch.float32)
    masks = sample["masks"].to(device=device, dtype=torch.float32)
    cameras = {
        "K": sample["K"].to(device=device, dtype=torch.float32),
        "w2c": sample["w2c"].to(device=device, dtype=torch.float32),
    }
    backgrounds = torch.tensor(args.background_color, device=device).view(1, 3).expand(images.shape[0], -1)

    source_export = read_gaussian_ply(args.gaussian_ply, real_mode=True)
    fixed = trellis_export_gaussian_to_internal(source_export)
    fixed = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in fixed.items()}
    initial_colors = fixed["colors"].detach().clamp(1e-4, 1 - 1e-4)
    color_logits = torch.nn.Parameter(torch.logit(initial_colors))
    initial_opacity = fixed["opacity"].detach()
    opacity_logits = torch.nn.Parameter(initial_opacity.clone())
    initial_scaling = fixed["scaling"].detach()
    scaling_logits = torch.nn.Parameter(initial_scaling.clone(), requires_grad=args.optimize_scaling)
    groups = [
        {"params": [color_logits], "lr": args.lr_color},
        {"params": [opacity_logits], "lr": args.lr_opacity},
    ]
    if args.optimize_scaling:
        groups.append({"params": [scaling_logits], "lr": args.lr_scaling})
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.steps), eta_min=1e-5)
    history = []
    num_views = images.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    for step in range(1, args.steps + 1):
        take = min(max(1, args.views_per_step), num_views)
        selection = torch.randperm(num_views, generator=generator)[:take].to(device)
        gaussian = {
            "xyz": fixed["xyz"],
            "rotation": fixed["rotation"],
            "scaling": scaling_logits,
            "scaling_parameterization": "log",
            "opacity": opacity_logits,
            "opacity_parameterization": "logit",
            "colors": color_logits.sigmoid(),
        }
        rendered = render_gaussians(
            gaussian,
            {"K": cameras["K"][selection], "w2c": cameras["w2c"][selection]},
            tuple(images.shape[-2:]),
            backgrounds=backgrounds[selection],
        )
        pred = rendered["rendered_rgb"].permute(0, 3, 1, 2)
        alpha = rendered["rendered_alpha"].permute(0, 3, 1, 2)
        target = images[selection]
        target_mask = masks[selection]
        rgb_l1 = (pred - target).abs().mean()
        ssim_loss = 1.0 - _ssim(pred.clamp(0, 1), target)
        mask_loss = probability_binary_cross_entropy(
            alpha.clamp(1e-5, 1 - 1e-5),
            target_mask,
        )
        prior = (color_logits.sigmoid() - initial_colors).square().mean()
        prior = prior + 0.25 * (opacity_logits - initial_opacity).square().mean()
        if args.optimize_scaling:
            prior = prior + 0.25 * (scaling_logits - initial_scaling).square().mean()
        loss = rgb_l1 + args.lambda_ssim * ssim_loss + args.lambda_mask * mask_loss + args.lambda_prior * prior
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 10 == 0 or step == args.steps:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "rgb_l1": float(rgb_l1.detach().cpu()),
                    "ssim_loss": float(ssim_loss.detach().cpu()),
                    "mask_loss": float(mask_loss.detach().cpu()),
                }
            )

    update_gaussian_ply_parameters(
        args.gaussian_ply,
        args.output_ply,
        colors=color_logits.sigmoid(),
        opacity_logits=opacity_logits,
        scaling_logits=scaling_logits if args.optimize_scaling else None,
        real_mode=True,
    )
    report = {
        "protocol": "conditioning_only_gaussian_refinement_v1",
        "uid": sample["uid"],
        "source_ply": str(Path(args.gaussian_ply).resolve()),
        "output_ply": str(Path(args.output_ply).resolve()),
        "conditioning_view_set": args.conditioning_view_set,
        "conditioning_frame_ids": sample["metadata"]["selected_frame_ids"],
        "evaluation_views_used": False,
        "steps": args.steps,
        "seed": args.seed,
        "optimize_scaling": bool(args.optimize_scaling),
        "history": history,
    }
    report_path = Path(args.output_ply).with_suffix(".refinement.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
