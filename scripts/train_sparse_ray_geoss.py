from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from geoss.datasets.meshfleet_trellis_dataset import MeshFleetTrellisDataset
from geoss.datasets.objaverse_cars_rendered_dataset import ObjaverseCarsRenderedDataset
from geoss.datasets.srn_cars_dataset import SRNCarsDataset
from geoss.datasets.vehicle_multiview_dataset import VehicleMultiViewDataset
from geoss.datasets.vehicle_multiview_dataset import make_dry_run_batch
from geoss.integration.vggt_geometry_wrapper import VGGTGeometryWrapper
from geoss.losses.confidence_loss import confidence_calibration_loss
from geoss.losses.occupancy_loss import occupancy_bce_loss
from geoss.losses.projection_loss import projection_consistency_loss
from geoss.losses.ray_free_space_loss import ray_free_space_loss
from geoss.models.sparse_ray_geoss_adapter import SparseRayGeoSSAdapter
from geoss.utils.checkpoint import save_checkpoint
from geoss.utils.config import add_common_args, load_config
from geoss.utils.run_mode import validate_real_mode
from geoss.utils.distributed import (
    build_dataloader,
    cleanup_distributed,
    init_distributed,
    maybe_wrap_ddp,
    next_from_loader,
    sync_should_stop,
    unwrap_model,
)
from geoss.utils.early_stopping import EarlyStopper
from geoss.utils.visualization import save_npz, save_projected_anchor_debug_png, save_ray_free_space_debug_png, write_point_cloud_ply


def run_dry_run(cfg: dict, device: str) -> dict:
    batch_cfg = cfg.get("dry_run_batch", {})
    model_cfg = cfg.get("model", {})
    batch = make_dry_run_batch(device=device, **batch_cfg)
    vggt = VGGTGeometryWrapper(mock=True)
    batch.update(vggt(batch["images"]))
    model = SparseRayGeoSSAdapter(**model_cfg).to(device)
    out = model(batch)
    gt_occ = torch.zeros(batch["images"].shape[0], 32, 32, 32, device=device)
    occ_terms = occupancy_bce_loss(out["occ_evidence"], out["free_evidence"], out["anchor_xyz"], gt_occ)
    occ_prob = torch.sigmoid(out["occ_evidence"] - out["free_evidence"])
    proj_terms = projection_consistency_loss(out["anchor_xyz"], occ_prob, batch["masks"], batch["K"], batch["w2c"])
    summary = {
        "mode": "dry_run",
        "mock_vggt": True,
        "not_for_paper_metrics": True,
        "anchor_xyz": list(out["anchor_xyz"].shape),
        "geo_tokens": list(out["geo_tokens"].shape),
        "geo_confidence": list(out["geo_confidence"].shape),
        "delta_v_geo": list(out["delta_v_geo"].shape),
        "occupancy_loss": float(occ_terms["loss"].detach().cpu()),
        "projection_loss": float(proj_terms["loss"].detach().cpu()),
    }
    return summary


def run_training(cfg: dict, args: argparse.Namespace) -> dict:
    run_modes = validate_real_mode(cfg=cfg, args=args, mode="real_train", required=("vggt", "dataset"))
    ctx = init_distributed(args)
    device = ctx.device
    model_cfg = cfg.get("model", {})
    model = SparseRayGeoSSAdapter(**model_cfg).to(device)
    for p in model.velocity_adapter.parameters():
        p.requires_grad_(False)
    vggt = VGGTGeometryWrapper(
        vggt_root=args.vggt_root,
        checkpoint=args.vggt_checkpoint,
        pretrained_name=args.vggt_pretrained,
        mock=False,
        cache_features=True,
    ).to(device)
    start_step = 0
    resume_state = None
    if args.resume and Path(args.resume).exists():
        resume_state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(resume_state.get("model", resume_state), strict=False)
        start_step = int(resume_state.get("step", 0))
    model = maybe_wrap_ddp(model, ctx, find_unused_parameters=args.ddp_find_unused_parameters)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    if resume_state is not None and "optimizer" in resume_state:
        opt.load_state_dict(resume_state["optimizer"])
    loader, sampler = _build_real_loader(args, ctx)
    iterator = iter(loader) if loader is not None else None
    if iterator is None:
        raise FileNotFoundError("real_train requires a non-empty real dataset loader; synthetic GeoSS batches are only allowed in --dry_run.")
    data_epoch = 0
    out_dir = Path(args.output_dir)
    if ctx.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_sparse_ray_geoss.jsonl"
    last_summary = {"mode": "real_dataset", **run_modes}
    early_stopper = EarlyStopper.from_args(args, default_metric="loss")
    if resume_state is not None:
        early_stopper.load_state_dict(resume_state.get("early_stopper"))
    end_step = int(args.steps) if args.steps_are_total else start_step + int(args.steps)
    if start_step >= end_step:
        return {
            "step": start_step,
            "target_step": end_step,
            "mode": "already_complete",
            "rank": ctx.rank,
            "world_size": ctx.world_size,
        }
    for step in range(start_step + 1, end_step + 1):
        batch, iterator, data_epoch = next_from_loader(iterator, loader, sampler, data_epoch)
        batch = _move_batch(batch, device)
        with torch.no_grad():
            batch.update(vggt(batch["images"], use_cache=False))
        out = model(batch)
        ray = out["debug"]["ray"]
        occ_prob = torch.sigmoid(torch.nan_to_num(out["occ_evidence"] - out["free_evidence"], nan=0.0, posinf=30.0, neginf=-30.0))
        occ_prob = torch.nan_to_num(occ_prob, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        losses = {}
        if "gt_occ" in batch:
            losses.update(occupancy_bce_loss(out["occ_evidence"], out["free_evidence"], out["anchor_xyz"], batch["gt_occ"]))
            geo_error = (occ_prob - (occ_prob > 0.5).float()).abs().clamp(0.0, 1.0)
        else:
            gt_occ = torch.zeros(batch["images"].shape[0], 32, 32, 32, device=device)
            occ_terms = occupancy_bce_loss(out["occ_evidence"], out["free_evidence"], out["anchor_xyz"], gt_occ)
            losses["occupancy_bce"] = occ_terms["occupancy_bce"] * 0.0
            losses["occupancy_dice"] = occ_terms["occupancy_dice"] * 0.0
            geo_error = occ_prob.detach().abs().clamp(0.0, 1.0)
        ray_terms = ray_free_space_loss(
            ray["free_score"],
            ray["occ_score"],
            ray["ray_valid"],
            ray["depth_residual"],
            signed_depth_residual=ray.get("signed_depth_residual"),
            free_geometry=ray.get("evidence_debug", {}).get("free_geometry"),
        )
        proj_terms = projection_consistency_loss(out["anchor_xyz"], occ_prob, batch["masks"], batch["K"], batch["w2c"])
        conf_terms = confidence_calibration_loss(out["geo_confidence"], geo_error)
        anchor_sparsity = occ_prob.mean() * args.anchor_sparsity_weight
        loss = (
            losses.get("loss", torch.zeros((), device=device))
            + ray_terms["loss"]
            + proj_terms["loss"]
            + conf_terms["loss"]
            + anchor_sparsity
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_summary = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "loss_occ": float(losses.get("occupancy_bce", torch.zeros((), device=device)).detach().cpu()),
            "loss_dice": float(losses.get("occupancy_dice", torch.zeros((), device=device)).detach().cpu()),
            "loss_free": float(ray_terms["loss"].detach().cpu()),
            "loss_conf": float(conf_terms["loss"].detach().cpu()),
            "loss_proj": float(proj_terms["loss"].detach().cpu()),
            "anchor_sparsity": float(anchor_sparsity.detach().cpu()),
            "confidence_mean": float(out["geo_confidence"].mean().detach().cpu()),
            "confidence_std": float(out["geo_confidence"].std(unbiased=False).detach().cpu()),
            "occ_score_mean": float(ray["occ_score"].mean().detach().cpu()),
            "free_score_mean": float(ray["free_score"].mean().detach().cpu()),
            "ray_valid_mean": float(ray["ray_valid"].mean().detach().cpu()),
            "conf_error_corr": float(conf_terms["confidence_error_corr"].detach().cpu()),
            "occ_prob_min": float(occ_prob.min().detach().cpu()),
            "occ_prob_max": float(occ_prob.max().detach().cpu()),
            "geo_error_min": float(geo_error.min().detach().cpu()),
            "geo_error_max": float(geo_error.max().detach().cpu()),
            "mode": "real_dataset",
            **run_modes,
            "rank": ctx.rank,
            "world_size": ctx.world_size,
            "per_gpu_batch_size": args.batch_size,
            "global_batch_size": args.batch_size * ctx.world_size,
        }
        if ctx.is_main and args.val_every > 0 and step % args.val_every == 0:
            last_summary["validation"] = _validation_step(unwrap_model(model), vggt, batch, device)
        early_status = early_stopper.update(last_summary)
        last_summary["early_stop"] = early_status.as_dict()
        if ctx.is_main:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(last_summary) + "\n")
        if ctx.is_main and args.save_best and early_status.is_best:
            _save_geoss_checkpoint(out_dir / "geoss_adapter_best.pt", model, opt, step, cfg, early_stopper, early_status)
        if ctx.is_main and args.visualize_every > 0 and step % args.visualize_every == 0:
            _write_visualization_outputs(out_dir, step, batch, out)
        should_fault_save = args.fault_tolerant_save_every > 0 and step % args.fault_tolerant_save_every == 0
        if ctx.is_main and (should_fault_save or step % args.save_every == 0 or step == end_step):
            _save_geoss_checkpoint(out_dir / "geoss_adapter_last.pt", model, opt, step, cfg, early_stopper, early_status)
        if sync_should_stop(early_status.should_stop, device):
            if ctx.is_main:
                _save_geoss_checkpoint(out_dir / "geoss_adapter_last.pt", model, opt, step, cfg, early_stopper, early_status)
            break
    return last_summary


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--latent_tokens", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--anchor_sparsity_weight", type=float, default=1e-3)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--visualize_every", type=int, default=100)
    parser.add_argument("--val_every", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--srn_root", type=str, default=None)
    parser.add_argument("--objaverse_rendered_root", type=str, default=None)
    parser.add_argument("--meshfleet_root", type=str, default=None)
    parser.add_argument("--meshfleet_split", type=str, default="train")
    parser.add_argument("--meshfleet_category", type=str, default=None)
    parser.add_argument("--meshfleet_occ_resolution", type=int, default=64)
    parser.add_argument("--meshfleet_prefer_cond_render", action="store_true")
    parser.add_argument("--vggt_root", type=str, default=None)
    parser.add_argument("--vggt_checkpoint", type=str, default=None)
    parser.add_argument("--vggt_pretrained", type=str, default=None)
    parser.add_argument("--real_train", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not args.dry_run:
        summary = run_training(cfg, args)
    else:
        summary = run_dry_run(cfg, args.device)
    if getattr(args, "rank", 0) == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "train_sparse_ray_geoss_dry_run.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    cleanup_distributed()


def _build_real_loader(args: argparse.Namespace, ctx):
    datasets = []
    if args.srn_root and Path(args.srn_root).exists():
        datasets.append(SRNCarsDataset(args.srn_root, num_views=args.num_views, image_size=args.image_size))
    if args.objaverse_rendered_root and Path(args.objaverse_rendered_root).exists():
        datasets.append(ObjaverseCarsRenderedDataset(args.objaverse_rendered_root, num_views=args.num_views, image_size=args.image_size))
    if args.meshfleet_root and Path(args.meshfleet_root).exists():
        meshfleet = MeshFleetTrellisDataset(
            args.meshfleet_root,
            split=args.meshfleet_split,
            category=args.meshfleet_category,
            num_views=args.num_views,
            image_size=args.image_size,
            occ_resolution=args.meshfleet_occ_resolution,
            prefer_cond_render=args.meshfleet_prefer_cond_render,
        )
        if len(meshfleet) == 0:
            raise FileNotFoundError(
                "MeshFleet_TRELLIS split has no reconstructed samples. "
                f"Checked root={args.meshfleet_root}, split={args.meshfleet_split}, category={args.meshfleet_category}. "
                "If you downloaded webdataset shards, first reconstruct them with the dataset card's reconstruct_data.py; "
                "for the current local sample use --meshfleet_split test --meshfleet_category sdvas."
            )
        datasets.append(meshfleet)
    if not datasets:
        return None, None
    dataset = VehicleMultiViewDataset(datasets)
    return build_dataloader(dataset, args=args, ctx=ctx, collate_fn=VehicleMultiViewDataset.collate_fn, shuffle=True)


def _move_batch(batch, device):
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


@torch.no_grad()
def _validation_step(model, vggt, batch, device):
    model.eval()
    val_batch = dict(batch)
    val_batch.update(vggt(val_batch["images"], use_cache=False))
    out = model(val_batch)
    occ_prob = torch.sigmoid(out["occ_evidence"] - out["free_evidence"])
    proj = projection_consistency_loss(out["anchor_xyz"], occ_prob, val_batch["masks"], val_batch["K"], val_batch["w2c"])
    summary = {
        "projection": float(proj["loss"].detach().cpu()),
        "confidence_mean": float(out["geo_confidence"].mean().detach().cpu()),
        "confidence_std": float(out["geo_confidence"].std(unbiased=False).detach().cpu()),
    }
    model.train()
    return summary


def _write_visualization_outputs(out_dir: Path, step: int, batch, out) -> None:
    ray = out["debug"]["ray"]
    write_point_cloud_ply(out_dir / "anchor_xyz.ply", out["anchor_xyz"][0])
    write_point_cloud_ply(out_dir / "anchor_confidence.ply", out["anchor_xyz"][0], out["geo_confidence"][0])
    write_point_cloud_ply(out_dir / f"anchor_confidence_step_{step}.ply", out["anchor_xyz"][0], out["geo_confidence"][0])
    save_npz(out_dir / f"occ_evidence_step_{step}.npz", occ_evidence=out["occ_evidence"])
    save_npz(out_dir / f"free_evidence_step_{step}.npz", free_evidence=out["free_evidence"])
    debug = ray.get("evidence_debug", {})
    if "uv" in debug:
        save_projected_anchor_debug_png(
            out_dir / "projection_debug" / f"projected_anchor_step_{step}.png",
            batch["images"][0, 0],
            debug["uv"][0, :, 0],
            ray["ray_valid"][0, :, 0],
        )
    save_ray_free_space_debug_png(out_dir / f"ray_free_space_step_{step}.png", ray["free_score"][0], ray["occ_score"][0])


def _save_geoss_checkpoint(path: Path, model, optimizer, step: int, cfg: dict, early_stopper: EarlyStopper, early_status) -> None:
    save_checkpoint(
        path,
        model=unwrap_model(model).state_dict(),
        optimizer=optimizer.state_dict(),
        step=step,
        config=cfg,
        early_stop=early_status.as_dict() if early_status is not None else None,
        early_stopper=early_stopper.state_dict(),
    )


if __name__ == "__main__":
    main()
