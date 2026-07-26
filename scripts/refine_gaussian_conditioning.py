from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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
    parser.add_argument("--optimize_opacity", type=str2bool, default=False)
    parser.add_argument("--optimize_scaling", type=str2bool, default=False)
    parser.add_argument("--lr_scaling", type=float, default=5e-4)
    parser.add_argument("--lambda_ssim", type=float, default=0.2)
    parser.add_argument("--lambda_mask", type=float, default=0.5)
    parser.add_argument("--lambda_prior", type=float, default=1e-3)
    parser.add_argument("--max_color_delta", type=float, default=0.08)
    parser.add_argument("--max_opacity_logit_delta", type=float, default=0.25)
    parser.add_argument("--max_scaling_log_delta", type=float, default=0.05)
    parser.add_argument("--validation_views", type=int, default=2)
    parser.add_argument("--validation_every", type=int, default=10)
    parser.add_argument("--validation_patience", type=int, default=5)
    parser.add_argument("--min_relative_improvement", type=float, default=0.002)
    parser.add_argument("--validation_tolerance", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.conditioning_view_set.startswith("renders_eval_"):
        raise ValueError("Evaluation views are forbidden during test-time refinement.")
    output_path = Path(args.output_ply)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.num_views < 3:
        raise ValueError(
            "Safe refinement requires at least three conditioning views so that "
            "optimization and model-selection views remain disjoint."
        )
    if not 1 <= args.validation_views < args.num_views:
        raise ValueError("--validation_views must be in [1, num_views-1].")
    if args.validation_every < 1 or args.validation_patience < 1:
        raise ValueError("Validation cadence and patience must be positive.")
    if not 0.0 <= args.min_relative_improvement < 1.0:
        raise ValueError("--min_relative_improvement must be in [0,1).")
    if min(
        args.max_color_delta,
        args.max_opacity_logit_delta,
        args.max_scaling_log_delta,
    ) < 0.0:
        raise ValueError("Refinement residual bounds must be non-negative.")
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
        load_3d_modalities=False,
    )
    if args.meshfleet_uid:
        sample = dataset.get_by_uid(args.meshfleet_uid)
    else:
        if not 0 <= args.meshfleet_index < len(dataset):
            raise IndexError(f"meshfleet_index={args.meshfleet_index} outside dataset length {len(dataset)}")
        sample = dataset[args.meshfleet_index]
    if sample.get("mesh_path") is not None or "gt_occ" in sample:
        raise RuntimeError("Conditioning-only Gaussian refinement received a forbidden 3D target.")
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
    initial_colors = fixed["colors"].detach().clamp(0.0, 1.0)
    color_residual = torch.nn.Parameter(torch.zeros_like(initial_colors))
    initial_opacity = fixed["opacity"].detach()
    opacity_residual = torch.nn.Parameter(
        torch.zeros_like(initial_opacity),
        requires_grad=args.optimize_opacity,
    )
    initial_scaling = fixed["scaling"].detach()
    scaling_residual = torch.nn.Parameter(
        torch.zeros_like(initial_scaling),
        requires_grad=args.optimize_scaling,
    )
    groups = [{"params": [color_residual], "lr": args.lr_color}]
    if args.optimize_opacity:
        groups.append({"params": [opacity_residual], "lr": args.lr_opacity})
    if args.optimize_scaling:
        groups.append({"params": [scaling_residual], "lr": args.lr_scaling})
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.steps), eta_min=1e-5)
    history = []
    num_views = images.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    validation_indices = _interleaved_validation_indices(
        num_views,
        args.validation_views,
    )
    validation_set = set(validation_indices.tolist())
    training_indices = torch.tensor(
        [index for index in range(num_views) if index not in validation_set],
        dtype=torch.long,
    )
    if training_indices.numel() == 0:
        raise RuntimeError("No conditioning views remain for refinement.")

    initial_gaussian = _bounded_gaussian(
        fixed,
        initial_colors,
        initial_opacity,
        initial_scaling,
        color_residual.detach(),
        opacity_residual.detach(),
        scaling_residual.detach(),
        args,
    )
    with torch.no_grad():
        baseline_validation = _render_objective(
            initial_gaussian,
            images[validation_indices.to(device)],
            masks[validation_indices.to(device)],
            cameras["K"][validation_indices.to(device)],
            cameras["w2c"][validation_indices.to(device)],
            backgrounds[validation_indices.to(device)],
            lambda_ssim=args.lambda_ssim,
            lambda_mask=args.lambda_mask,
        )
    best_validation = dict(baseline_validation)
    best_parameters = None
    validations_without_improvement = 0

    for step in range(1, args.steps + 1):
        take = min(max(1, args.views_per_step), int(training_indices.numel()))
        permutation = torch.randperm(
            int(training_indices.numel()),
            generator=generator,
        )[:take]
        selection = training_indices[permutation].to(device)
        gaussian = _bounded_gaussian(
            fixed,
            initial_colors,
            initial_opacity,
            initial_scaling,
            color_residual,
            opacity_residual,
            scaling_residual,
            args,
        )
        train_metrics = _render_objective(
            gaussian,
            images[selection],
            masks[selection],
            cameras["K"][selection],
            cameras["w2c"][selection],
            backgrounds[selection],
            lambda_ssim=args.lambda_ssim,
            lambda_mask=args.lambda_mask,
        )
        prior = color_residual.square().mean()
        if args.optimize_opacity:
            prior = prior + 0.25 * opacity_residual.square().mean()
        if args.optimize_scaling:
            prior = prior + 0.25 * scaling_residual.square().mean()
        loss = train_metrics["objective_tensor"] + args.lambda_prior * prior
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        validate_now = step == 1 or step % args.validation_every == 0 or step == args.steps
        if validate_now:
            with torch.no_grad():
                validation_gaussian = _bounded_gaussian(
                    fixed,
                    initial_colors,
                    initial_opacity,
                    initial_scaling,
                    color_residual,
                    opacity_residual,
                    scaling_residual,
                    args,
                )
                validation = _render_objective(
                    validation_gaussian,
                    images[validation_indices.to(device)],
                    masks[validation_indices.to(device)],
                    cameras["K"][validation_indices.to(device)],
                    cameras["w2c"][validation_indices.to(device)],
                    backgrounds[validation_indices.to(device)],
                    lambda_ssim=args.lambda_ssim,
                    lambda_mask=args.lambda_mask,
                )
            eligible = _is_safe_validation_improvement(
                validation,
                baseline_validation,
                best_validation,
                min_relative_improvement=args.min_relative_improvement,
                tolerance=args.validation_tolerance,
            )
            if eligible:
                best_validation = {
                    key: value
                    for key, value in validation.items()
                    if key != "objective_tensor"
                }
                best_parameters = {
                    "colors": validation_gaussian["colors"].detach().cpu().clone(),
                    "opacity": validation_gaussian["opacity"].detach().cpu().clone(),
                    "scaling": validation_gaussian["scaling"].detach().cpu().clone(),
                    "step": step,
                }
                validations_without_improvement = 0
            else:
                validations_without_improvement += 1
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "training": _json_metrics(train_metrics),
                    "validation": _json_metrics(validation),
                    "eligible": bool(eligible),
                }
            )
            if validations_without_improvement >= args.validation_patience:
                break

    source_hash = _sha256(Path(args.gaussian_ply))
    if best_parameters is None:
        shutil.copy2(args.gaussian_ply, output_path)
        selection_status = "reverted_to_source"
        selected_step = 0
    else:
        update_gaussian_ply_parameters(
            args.gaussian_ply,
            output_path,
            colors=best_parameters["colors"],
            opacity_logits=best_parameters["opacity"],
            scaling_logits=(
                best_parameters["scaling"] if args.optimize_scaling else None
            ),
            real_mode=True,
        )
        selection_status = "conditioning_validation_selected"
        selected_step = int(best_parameters["step"])
    output_hash = _sha256(output_path)
    report = {
        "protocol": "conditioning_only_gaussian_refinement_v2_safe_selection",
        "uid": sample["uid"],
        "source_ply": str(Path(args.gaussian_ply).resolve()),
        "output_ply": str(output_path.resolve()),
        "source_sha256": source_hash,
        "output_sha256": output_hash,
        "output_byte_identical_to_source": source_hash == output_hash,
        "conditioning_view_set": args.conditioning_view_set,
        "conditioning_frame_ids": sample["metadata"]["selected_frame_ids"],
        "optimization_frame_indices": training_indices.tolist(),
        "validation_frame_indices": validation_indices.tolist(),
        "evaluation_views_used": False,
        "test_time_ground_truth_latents_used": False,
        "test_time_ground_truth_mesh_used": False,
        "test_time_ground_truth_voxels_used": False,
        "steps": args.steps,
        "executed_steps": history[-1]["step"] if history else 0,
        "seed": args.seed,
        "optimize_opacity": bool(args.optimize_opacity),
        "optimize_scaling": bool(args.optimize_scaling),
        "selection_status": selection_status,
        "selected_step": selected_step,
        "baseline_validation": _json_metrics(baseline_validation),
        "selected_validation": _json_metrics(best_validation),
        "history": history,
    }
    report_path = output_path.with_suffix(".refinement.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _interleaved_validation_indices(num_views: int, count: int) -> torch.Tensor:
    """Choose angularly distributed selection views without random leakage."""
    positions = (
        (torch.arange(count, dtype=torch.float64) + 0.5)
        * float(num_views)
        / float(count)
    )
    indices = torch.floor(positions).long().clamp(0, num_views - 1)
    if indices.unique().numel() != count:
        raise RuntimeError("Could not construct unique validation view indices.")
    return indices


def _bounded_gaussian(
    fixed,
    initial_colors,
    initial_opacity,
    initial_scaling,
    color_residual,
    opacity_residual,
    scaling_residual,
    args,
):
    colors = (
        initial_colors
        + float(args.max_color_delta) * torch.tanh(color_residual)
    ).clamp(0.0, 1.0)
    opacity = (
        initial_opacity
        + float(args.max_opacity_logit_delta) * torch.tanh(opacity_residual)
        if args.optimize_opacity
        else initial_opacity
    )
    scaling = (
        initial_scaling
        + float(args.max_scaling_log_delta) * torch.tanh(scaling_residual)
        if args.optimize_scaling
        else initial_scaling
    )
    return {
        "xyz": fixed["xyz"],
        "rotation": fixed["rotation"],
        "scaling": scaling,
        "scaling_parameterization": "log",
        "opacity": opacity,
        "opacity_parameterization": "logit",
        "colors": colors,
    }


def _render_objective(
    gaussian,
    target,
    target_mask,
    K,
    w2c,
    backgrounds,
    *,
    lambda_ssim: float,
    lambda_mask: float,
):
    rendered = render_gaussians(
        gaussian,
        {"K": K, "w2c": w2c},
        tuple(target.shape[-2:]),
        backgrounds=backgrounds,
    )
    pred = rendered["rendered_rgb"].permute(0, 3, 1, 2).clamp(0.0, 1.0)
    alpha = rendered["rendered_alpha"].permute(0, 3, 1, 2)
    absolute = (pred - target).abs()
    foreground_l1 = (
        (absolute * target_mask).sum()
        / (target_mask.sum().clamp_min(1.0) * pred.shape[1])
    )
    full_l1 = absolute.mean()
    rgb_l1 = 0.8 * foreground_l1 + 0.2 * full_l1
    ssim_loss = 1.0 - _ssim(pred, target)
    mask_loss = probability_binary_cross_entropy(
        alpha.clamp(1e-5, 1 - 1e-5),
        target_mask,
    )
    objective = rgb_l1 + lambda_ssim * ssim_loss + lambda_mask * mask_loss
    return {
        "objective_tensor": objective,
        "objective": float(objective.detach().cpu()),
        "foreground_l1": float(foreground_l1.detach().cpu()),
        "full_l1": float(full_l1.detach().cpu()),
        "ssim_loss": float(ssim_loss.detach().cpu()),
        "mask_loss": float(mask_loss.detach().cpu()),
    }


def _is_safe_validation_improvement(
    candidate,
    baseline,
    best,
    *,
    min_relative_improvement: float,
    tolerance: float,
) -> bool:
    candidate_objective = float(candidate["objective"])
    baseline_objective = float(baseline["objective"])
    best_objective = float(best["objective"])
    required = baseline_objective * (1.0 - float(min_relative_improvement))
    return bool(
        candidate_objective < min(required, best_objective - tolerance)
        and float(candidate["foreground_l1"])
        <= float(baseline["foreground_l1"]) + tolerance
        and float(candidate["ssim_loss"])
        <= float(baseline["ssim_loss"]) + tolerance
        and float(candidate["mask_loss"])
        <= float(baseline["mask_loss"]) + tolerance
    )


def _json_metrics(metrics):
    return {
        key: float(value)
        for key, value in metrics.items()
        if key != "objective_tensor"
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
