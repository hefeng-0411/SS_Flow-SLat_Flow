from __future__ import annotations

import argparse
import contextlib
import json
import os
from itertools import cycle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
import torch.nn.functional as F

from geoss.datasets.meshfleet_trellis_dataset import MeshFleetTrellisDataset
from geoss.datasets.objaverse_cars_rendered_dataset import ObjaverseCarsRenderedDataset
from geoss.datasets.srn_cars_dataset import SRNCarsDataset
from geoss.datasets.vehicle_multiview_dataset import VehicleMultiViewDataset, make_dry_run_batch
from geoss.slat.integration.ss_slat_context import build_ss_slat_context
from geoss.slat.losses.appearance_feature_loss import appearance_feature_loss
from geoss.slat.losses.slat_flow_loss import slat_flow_matching_loss
from geoss.slat.losses.slat_prior_preservation_loss import slat_prior_preservation_loss
from geoss.slat.losses.slat_velocity_loss import slat_velocity_regularization_loss
from geoss.slat.losses.view_consistency_loss import view_consistency_loss
from geoss.slat.losses.visibility_confidence_loss import visibility_confidence_loss
from geoss.slat.models.geovis_slat_adapter import GeoVisSLATAdapter
from geoss.slat.utils.slat_visualization import save_slat_debug_npz, write_active_voxels_ply
from geoss.utils.adaptive_batch import AdaptiveBatchController, adaptive_config_defaults, add_adaptive_batch_args
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
from geoss.utils.elastic_engine import slice_batch_to_size, train_step_with_oom_retry


def run_dry_run(cfg: dict, args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    batch = make_synthetic_slat_batch(cfg, args, device)
    model = GeoVisSLATAdapter(**cfg.get("model", {})).to(device)
    out = model(batch)
    terms = compute_losses(out, batch)
    summary = summarize(out, terms, "synthetic_dry_run")
    summary.update({"mock_trellis_base_velocity": True, "not_for_paper_metrics": True})
    write_outputs(Path(args.output_dir), out, summary)
    return summary


def run_training(cfg: dict, args: argparse.Namespace) -> dict:
    run_modes = validate_real_mode(cfg=cfg, args=args, mode="real_train", required=("trellis", "dataset"))
    ctx = init_distributed(args)
    device = ctx.device
    batch_controller = AdaptiveBatchController.from_args(args)
    args.batch_size = batch_controller.batch_size
    trellis_pipeline = _load_trellis_pipeline(args, device, ctx)
    model = GeoVisSLATAdapter(**cfg.get("model", {})).to(device)
    start_step = 0
    resume_state = None
    if args.resume and Path(args.resume).exists():
        resume_state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(resume_state.get("model", resume_state), strict=False)
        start_step = int(resume_state.get("step", 0))
    model = maybe_wrap_ddp(model, ctx, find_unused_parameters=args.ddp_find_unused_parameters)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if resume_state is not None and "optimizer" in resume_state:
        opt.load_state_dict(resume_state["optimizer"])
    loader, sampler = build_real_loader(args, ctx)
    iterator = iter(loader) if loader is not None else None
    if iterator is None:
        raise FileNotFoundError("real_train requires a non-empty real dataset loader; synthetic SLAT batches are only allowed in --dry_run.")
    data_epoch = 0
    out_dir = Path(args.output_dir)
    if ctx.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_geovis_slat.jsonl"
    last = {}
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
    step = start_step
    while step < end_step:
        step += 1
        if iterator is not None:
            raw_batch, iterator, data_epoch = next_from_loader(iterator, loader, sampler, data_epoch)
        else:
            raise RuntimeError("real_train unexpectedly has no real dataloader iterator.")

        def rebuild_after_adjustment(adjustment):
            nonlocal raw_batch, loader, sampler, iterator
            args.batch_size = adjustment.new_batch_size
            raw_batch = slice_batch_to_size(raw_batch, args.batch_size)
            loader, sampler = build_real_loader(args, ctx)
            iterator = iter(loader) if loader is not None else None

        def log_oom(record):
            if ctx.is_main:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, **record}) + "\n")

        def step_fn():
            nonlocal raw_batch, iterator, data_epoch
            grad_accum_steps = max(1, int(args.grad_accum_steps))
            opt.zero_grad(set_to_none=True)
            last_batch = None
            last_out = None
            last_terms = None
            total_loss = torch.zeros((), device=device)
            for micro_step in range(grad_accum_steps):
                if micro_step > 0:
                    raw_batch, iterator, data_epoch = next_from_loader(iterator, loader, sampler, data_epoch)
                batch = prepare_batch(raw_batch, cfg, args, device, trellis_pipeline=trellis_pipeline)
                out = model(batch)
                terms = compute_losses(
                    out,
                    batch,
                    raw_residual_weight=args.raw_residual_weight,
                    effective_residual_weight=args.effective_residual_weight,
                )
                loss = (
                    terms["slat_flow"]["loss"]
                    + 0.25 * terms["view"]["loss"]
                    + 0.25 * terms["appearance"]["loss"]
                    + 0.2 * terms["visibility_confidence"]["loss"]
                    + args.velocity_weight * terms["velocity"]["loss"]
                    + args.prior_weight * terms["prior"]["loss"]
                )
                if not torch.isfinite(loss).all().item():
                    raise FloatingPointError("Stage 3 loss became NaN/Inf before backward.")
                sync_context = (
                    model.no_sync()
                    if ctx.distributed and hasattr(model, "no_sync") and micro_step < grad_accum_steps - 1
                    else contextlib.nullcontext()
                )
                with sync_context:
                    (loss / grad_accum_steps).backward()
                total_loss = total_loss + loss.detach()
                last_batch, last_out, last_terms = batch, out, terms
            grad_norms = _assert_geovis_slat_gradients(unwrap_model(model), step)
            opt.step()
            assert last_batch is not None and last_out is not None and last_terms is not None
            last_terms["grad_norms"] = grad_norms
            return last_batch, last_out, last_terms, total_loss / grad_accum_steps

        retry = train_step_with_oom_retry(
            step_fn,
            model=model,
            optimizer=opt,
            sampler=sampler,
            device=device,
            batch_controller=batch_controller,
            rebuild_after_adjustment=rebuild_after_adjustment,
            max_retries=getattr(args, "adaptive_oom_retries", 8),
            log_oom=log_oom,
        )
        batch_adjustment = retry.adjustment
        batch, out, terms, loss = retry.value
        if batch_adjustment.changed:
            args.batch_size = batch_adjustment.new_batch_size
            loader, sampler = build_real_loader(args, ctx)
            iterator = iter(loader) if loader is not None else None
        last = summarize(out, terms, "real_dataset")
        last.update(run_modes)
        last["step"] = step
        last["loss"] = float(loss.detach().cpu())
        last["rank"] = ctx.rank
        last["world_size"] = ctx.world_size
        last["per_gpu_batch_size"] = args.batch_size
        last["global_batch_size"] = args.batch_size * ctx.world_size
        last["effective_global_batch_size"] = args.batch_size * ctx.world_size * max(1, int(args.grad_accum_steps))
        last["grad_accum_steps"] = max(1, int(args.grad_accum_steps))
        last["adapter_grad_norms"] = terms.get("grad_norms", {})
        last["adaptive_batch"] = {**batch_controller.state_dict(), "last_adjustment": batch_adjustment.as_dict()}
        early_status = early_stopper.update(last)
        last["early_stop"] = early_status.as_dict()
        if ctx.is_main:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(last) + "\n")
        if ctx.is_main and args.save_best and early_status.is_best:
            _save_slat_checkpoint(out_dir / "geovis_slat_adapter_best.pt", model, opt, step, cfg, early_stopper, early_status)
        if ctx.is_main and args.visualize_every > 0 and step % args.visualize_every == 0:
            write_outputs(out_dir, out, last)
        should_fault_save = args.fault_tolerant_save_every > 0 and step % args.fault_tolerant_save_every == 0
        if ctx.is_main and (should_fault_save or step % args.save_every == 0 or step == end_step):
            _save_slat_checkpoint(out_dir / "geovis_slat_adapter_last.pt", model, opt, step, cfg, early_stopper, early_status)
        if sync_should_stop(early_status.should_stop, device):
            if ctx.is_main:
                _save_slat_checkpoint(out_dir / "geovis_slat_adapter_last.pt", model, opt, step, cfg, early_stopper, early_status)
            break
    if ctx.is_main:
        write_outputs(out_dir, out, last)
    return last


def compute_losses(
    out: dict,
    batch: dict,
    *,
    raw_residual_weight: float = 1.0,
    effective_residual_weight: float = 1.0,
) -> dict:
    target_residual = batch.get("target_residual", batch["target_velocity"] - batch["v_slat_base"])
    velocity_debug = out.get("debug", {}).get("velocity", {})
    raw_delta = velocity_debug.get("delta_raw", out["delta_v_slat"])
    effective_delta = out["v_slat_geo"] - batch["v_slat_base"]
    _assert_slat_training_contract(batch, out, raw_delta, effective_delta, target_residual)
    raw_flow = slat_flow_matching_loss(raw_delta, target_residual.detach())
    effective_flow = slat_flow_matching_loss(effective_delta, target_residual.detach())
    flow_loss = raw_residual_weight * raw_flow["loss"] + effective_residual_weight * effective_flow["loss"]
    return {
        "slat_flow": {
            "loss": flow_loss,
            "slat_flow_mse": flow_loss,
            "raw_residual_mse": raw_flow["loss"],
            "effective_residual_mse": effective_flow["loss"],
        },
        "view": view_consistency_loss(out["sampled_features"], out["visibility"]),
        "appearance": appearance_feature_loss(out["slat_cond_tokens"], out["sampled_features"], out["visibility"], out["view_weights"]),
        "visibility_confidence": visibility_confidence_loss(
            out["slat_confidence"],
            out["visibility"],
            out["depth_residual"],
            appearance_conflict=out["appearance_conflict"],
            occlusion_score=out["occlusion_score"],
        ),
        "velocity": slat_velocity_regularization_loss(out["delta_v_slat"], batch["timestep"]),
        "prior": slat_prior_preservation_loss(out["v_slat_geo"], batch["v_slat_base"], out["debug"]["joint_confidence"]),
    }


def summarize(out: dict, terms: dict, mode: str) -> dict:
    return {
        "mode": mode,
        "active_xyz": list(out["active_xyz"].shape),
        "slat_cond_tokens": list(out["slat_cond_tokens"].shape),
        "v_slat_geo": list(out["v_slat_geo"].shape),
        "slat_confidence_mean": float(out["slat_confidence"].mean().detach().cpu()),
        "slat_confidence_std": float(out["slat_confidence"].std(unbiased=False).detach().cpu()),
        "visibility_mean": float(out["visibility"].mean().detach().cpu()),
        "view_weights_std": float(out["view_weights"].std(unbiased=False).detach().cpu()),
        "delta_norm": float(out["delta_v_slat"].norm(dim=-1).mean().detach().cpu()),
        "clipping_ratio": float(out["debug"]["clipping_ratio"].detach().cpu()),
        "loss_slat_flow": float(terms["slat_flow"]["loss"].detach().cpu()),
        "loss_slat_raw_residual": float(terms["slat_flow"]["raw_residual_mse"].detach().cpu()),
        "loss_slat_effective_residual": float(terms["slat_flow"]["effective_residual_mse"].detach().cpu()),
        "loss_prior": float(terms["prior"]["loss"].detach().cpu()),
        "loss_velocity": float(terms["velocity"]["loss"].detach().cpu()),
    }


def _assert_slat_training_contract(
    batch: dict,
    out: dict,
    raw_delta: torch.Tensor,
    effective_delta: torch.Tensor,
    target_residual: torch.Tensor,
) -> None:
    required_batch = ("slat_latent_tokens", "v_slat_base", "target_velocity", "timestep", "images", "masks", "K", "w2c")
    for key in required_batch:
        assert key in batch, f"Stage 3 batch missing '{key}'."
    x_t = batch["slat_latent_tokens"]
    v_base = batch["v_slat_base"]
    assert x_t.ndim == 3, f"slat_latent_tokens must be [B,L,C], got {tuple(x_t.shape)}"
    assert v_base.shape == x_t.shape, f"v_slat_base shape {tuple(v_base.shape)} != slat_latent_tokens {tuple(x_t.shape)}"
    assert target_residual.shape == x_t.shape, f"target_residual shape {tuple(target_residual.shape)} != slat_latent_tokens {tuple(x_t.shape)}"
    assert raw_delta.shape == x_t.shape, f"raw SLAT residual shape {tuple(raw_delta.shape)} != slat_latent_tokens {tuple(x_t.shape)}"
    assert effective_delta.shape == x_t.shape, f"effective SLAT residual shape {tuple(effective_delta.shape)} != slat_latent_tokens {tuple(x_t.shape)}"
    for name, tensor in {
        "slat_latent_tokens": x_t,
        "v_slat_base": v_base,
        "target_residual": target_residual,
        "raw_delta": raw_delta,
        "effective_delta": effective_delta,
        "v_slat_geo": out["v_slat_geo"],
        "delta_v_slat": out["delta_v_slat"],
        "slat_cond_tokens": out["slat_cond_tokens"],
        "slat_confidence": out["slat_confidence"],
        "visibility": out["visibility"],
    }.items():
        assert tensor.dtype == torch.float32, f"{name} must be float32, got {tensor.dtype}"
        assert torch.isfinite(tensor).all().item(), f"{name} contains NaN or Inf."
    assert raw_delta.requires_grad, "raw SLAT residual must depend on GeoVisSLATAdapter parameters."
    assert effective_delta.requires_grad, "effective SLAT residual must depend on GeoVisSLATAdapter parameters."
    assert not target_residual.requires_grad, "target_residual must be a frozen flow-matching target."
    assert out["slat_confidence"].shape == (*x_t.shape[:2], 1), f"slat_confidence must be [B,L,1], got {tuple(out['slat_confidence'].shape)}"
    assert out["visibility"].shape[:2] == x_t.shape[:2], f"visibility must align with SLAT tokens, got {tuple(out['visibility'].shape)}"
    assert out["slat_confidence"].min().item() >= 0.0 and out["slat_confidence"].max().item() <= 1.0, "slat_confidence must be in [0, 1]."
    assert out["visibility"].min().item() >= 0.0 and out["visibility"].max().item() <= 1.0, "visibility must be in [0, 1]."


def _assert_geovis_slat_gradients(model: GeoVisSLATAdapter, step: int) -> dict[str, float]:
    critical = {
        "evidence_sampler.token_mlp.0.weight": model.evidence_sampler.token_mlp[0].weight,
        "aggregator.evidence_proj.weight": model.aggregator.evidence_proj.weight,
        "aggregator.slat_proj.weight": model.aggregator.slat_proj.weight,
        "aggregator.out.2.weight": model.aggregator.out[-1].weight,
        "velocity_adapter.latent_proj.weight": model.velocity_adapter.latent_proj.weight,
        "velocity_adapter.cond_proj.weight": model.velocity_adapter.cond_proj.weight,
        "velocity_adapter.delta_head.2.weight": model.velocity_adapter.delta_head[-1].weight,
    }
    missing = [name for name, param in critical.items() if param.grad is None]
    assert not missing, f"Stage 3 DDP graph break at step={step}; missing gradients for {missing}"
    nonfinite = [name for name, param in critical.items() if param.grad is not None and not torch.isfinite(param.grad).all().item()]
    assert not nonfinite, f"Stage 3 non-finite gradients at step={step}: {nonfinite}"
    return {name: float(param.grad.detach().norm().cpu()) for name, param in critical.items()}


def make_synthetic_slat_batch(cfg: dict, args: argparse.Namespace, device: torch.device) -> dict:
    model_cfg = cfg.get("model", {})
    slat_dim = int(model_cfg.get("slat_dim", 8))
    L = int(cfg.get("dry_run_batch", {}).get("active_tokens", args.active_tokens))
    batch = make_dry_run_batch(
        batch_size=args.batch_size,
        num_views=args.num_views,
        image_size=args.image_size,
        latent_tokens=L,
        latent_dim=slat_dim,
        device=device,
    )
    B = batch["images"].shape[0]
    resolution = int(model_cfg.get("resolution", 64))
    indices = torch.randint(20, 44, (B, L, 3), device=device)
    if L >= 3:
        indices[:, 0] = torch.tensor([32, 32, 32], device=device)
        indices[:, 1] = torch.tensor([32, 32, 48], device=device)
        indices[:, 2] = torch.tensor([63, 63, 63], device=device)
    context = build_ss_slat_context(ss_active_indices=indices, resolution=resolution, target_dim=slat_dim)
    x0 = torch.randn(B, L, slat_dim, device=device) * 0.5
    noise = torch.randn_like(x0)
    t = torch.rand(B, device=device)
    sigma_min = float(cfg.get("flow", {}).get("sigma_min", 1e-5))
    x_t = (1 - t.view(B, 1, 1)) * x0 + (sigma_min + (1 - sigma_min) * t.view(B, 1, 1)) * noise
    target_v = (1 - sigma_min) * noise - x0
    batch.update(context)
    batch.update(
        {
            "slat_latent_tokens": x_t,
            "v_slat_base": torch.zeros_like(x_t),
            "target_velocity": target_v,
            "timestep": t,
            "vggt_features": torch.rand(B, args.num_views, 16, args.image_size // 4, args.image_size // 4, device=device),
        }
    )
    return batch


def prepare_batch(raw: dict, cfg: dict, args: argparse.Namespace, device: torch.device, *, trellis_pipeline=None) -> dict:
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in raw.items()}
    model_cfg = cfg.get("model", {})
    slat_dim = int(model_cfg.get("slat_dim", 8))
    resolution = int(model_cfg.get("resolution", 64))
    if "trellis_slat_feats" in batch and "trellis_slat_indices" in batch:
        feats, indices = _pad_latents(batch["trellis_slat_feats"], batch["trellis_slat_indices"], args.active_tokens, device)
        x0 = feats[..., :slat_dim]
        if x0.shape[-1] < slat_dim:
            x0 = torch.cat([x0, x0.new_zeros(*x0.shape[:-1], slat_dim - x0.shape[-1])], dim=-1)
        source = "trellis_native_slat_latents"
    else:
        raise KeyError("real_train requires trellis_slat_feats and trellis_slat_indices; synthetic SLAT latents are only allowed in --dry_run.")
    context = build_ss_slat_context(ss_active_indices=indices, resolution=resolution, target_dim=slat_dim)
    B = x0.shape[0]
    t = torch.rand(B, device=device)
    noise = torch.randn_like(x0)
    sigma_min = float(cfg.get("flow", {}).get("sigma_min", 1e-5))
    x_t = (1 - t.view(B, 1, 1)) * x0 + (sigma_min + (1 - sigma_min) * t.view(B, 1, 1)) * noise
    batch.update(context)
    batch["slat_latent_tokens"] = x_t
    base = batch.get("trellis_slat_base_velocity")
    if base is None:
        base = _compute_trellis_slat_base_velocity(batch, x_t, indices, t, trellis_pipeline, device)
    batch["v_slat_base"] = base[:, : x_t.shape[1], : x_t.shape[2]].to(device=device, dtype=x_t.dtype)
    batch["target_velocity"] = (1 - sigma_min) * noise - x0
    batch["target_residual"] = (batch["target_velocity"] - batch["v_slat_base"]).detach()
    batch["timestep"] = t
    batch["slat_target_source"] = source
    return batch


def build_real_loader(args: argparse.Namespace, ctx):
    datasets = []
    if args.meshfleet_root and Path(args.meshfleet_root).exists():
        datasets.append(
            MeshFleetTrellisDataset(
                args.meshfleet_root,
                split=args.meshfleet_split,
                category=args.meshfleet_category,
                num_views=args.num_views,
                image_size=args.image_size,
                slat_latent_model=args.meshfleet_slat_latent_model,
                require_slat_latents=True,
            )
        )
    if args.srn_root and Path(args.srn_root).exists():
        datasets.append(SRNCarsDataset(args.srn_root, num_views=args.num_views, image_size=args.image_size))
    if args.objaverse_rendered_root and Path(args.objaverse_rendered_root).exists():
        datasets.append(ObjaverseCarsRenderedDataset(args.objaverse_rendered_root, num_views=args.num_views, image_size=args.image_size))
    if not datasets:
        return None, None
    dataset = VehicleMultiViewDataset(datasets)
    return build_dataloader(dataset, args=args, ctx=ctx, collate_fn=VehicleMultiViewDataset.collate_fn, shuffle=True)


def _pad_latents(feats, indices, limit: int, device: torch.device):
    if isinstance(feats, list):
        B = len(feats)
        L = min(limit, max(f.shape[0] for f in feats))
        C = feats[0].shape[-1]
        out_feats = torch.zeros(B, L, C, device=device)
        out_idx = torch.zeros(B, L, 3, dtype=torch.long, device=device)
        for b, (f, idx) in enumerate(zip(feats, indices)):
            take = min(L, f.shape[0])
            out_feats[b, :take] = f[:take].to(device)
            out_idx[b, :take] = idx[:take].to(device)
        return out_feats, out_idx
    return feats[:, :limit].to(device), indices[:, :limit].to(device)


def _pad_indices(indices, limit: int, device: torch.device):
    if isinstance(indices, list):
        B = len(indices)
        L = min(limit, max(max(1, idx.shape[0]) for idx in indices))
        out = torch.zeros(B, L, 3, dtype=torch.long, device=device)
        for b, idx in enumerate(indices):
            take = min(L, idx.shape[0])
            if take:
                out[b, :take] = idx[:take].to(device)
        return out
    return indices[:, :limit].to(device)


def write_outputs(out_dir: Path, out: dict, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_active_voxels_ply(out_dir / "slat_confidence.ply", out["active_xyz"][0], out["slat_confidence"][0])
    write_active_voxels_ply(out_dir / "ss_active_voxels.ply", out["active_xyz"][0], out["ss_confidence"][0])
    save_slat_debug_npz(
        out_dir / "slat_visibility_debug.npz",
        visibility=out["visibility"],
        view_weights=out["view_weights"],
        slat_confidence=out["slat_confidence"],
    )
    save_slat_debug_npz(
        out_dir / "slat_velocity_debug.npz",
        delta_v_slat=out["delta_v_slat"],
        v_slat_geo=out["v_slat_geo"],
        clipping_ratio=out["debug"]["clipping_ratio"],
    )
    (out_dir / "train_geovis_slat_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _save_slat_checkpoint(path: Path, model, optimizer, step: int, cfg: dict, early_stopper: EarlyStopper, early_status) -> None:
    save_checkpoint(
        path,
        model=unwrap_model(model).state_dict(),
        optimizer=optimizer.state_dict(),
        step=step,
        config=cfg,
        early_stop=early_status.as_dict() if early_status is not None else None,
        early_stopper=early_stopper.state_dict(),
    )


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--active_tokens", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--velocity_weight", type=float, default=1e-3)
    parser.add_argument("--prior_weight", type=float, default=1e-2)
    parser.add_argument("--raw_residual_weight", type=float, default=1.0)
    parser.add_argument("--effective_residual_weight", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--visualize_every", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--meshfleet_root", type=str, default=None)
    parser.add_argument("--meshfleet_split", type=str, default="train")
    parser.add_argument("--meshfleet_category", type=str, default=None)
    parser.add_argument("--meshfleet_slat_latent_model", type=str, default="dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16")
    parser.add_argument("--srn_root", type=str, default=None)
    parser.add_argument("--objaverse_rendered_root", type=str, default=None)
    parser.add_argument("--trellis_root", type=str, default=None)
    parser.add_argument("--trellis_model_path", type=str, default=None)
    parser.add_argument("--torch_hub_dir", type=str, default=None)
    parser.add_argument("--dinov2_repo", type=str, default=None)
    parser.add_argument("--real_train", action="store_true")
    add_adaptive_batch_args(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    _apply_config_defaults(args, cfg, parser)
    try:
        summary = run_dry_run(cfg, args) if args.dry_run else run_training(cfg, args)
        if getattr(args, "rank", 0) == 0:
            print(json.dumps(summary, indent=2))
    finally:
        cleanup_distributed()


def _load_trellis_pipeline(args: argparse.Namespace, device: torch.device, ctx):
    if args.trellis_root:
        sys.path.insert(0, args.trellis_root)
    if not args.trellis_model_path:
        raise FileNotFoundError("real_train requires --trellis_model_path or trellis.pipeline in config.")
    _configure_trellis_hub(args)

    if ctx.distributed and not ctx.is_main:
        dist.barrier()
    pipeline = _load_trellis_pipeline_impl(args, device)
    if ctx.distributed and ctx.is_main:
        dist.barrier()
    return pipeline


def _load_trellis_pipeline_impl(args: argparse.Namespace, device: torch.device):
    from trellis.pipelines import TrellisImageTo3DPipeline

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.trellis_model_path)
    pipeline.to(device)
    for name in ("slat_flow_model", "image_cond_model"):
        if name not in pipeline.models or pipeline.models[name] is None:
            raise RuntimeError(f"TRELLIS pipeline is missing {name}, required for real SLAT residual training.")
    for model in pipeline.models.values():
        if model is None:
            continue
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "parameters"):
            for p in model.parameters():
                p.requires_grad_(False)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pipeline


def _configure_trellis_hub(args: argparse.Namespace) -> None:
    hub_dir = _resolve_torch_hub_dir(args)
    if hub_dir is not None:
        torch.hub.set_dir(str(hub_dir))
        os.environ["TORCH_HUB_DIR"] = str(hub_dir)
        os.environ.setdefault("TORCH_HOME", str(hub_dir.parent if hub_dir.name == "hub" else hub_dir))
    dinov2_repo = _resolve_dinov2_repo(args, hub_dir)
    if dinov2_repo is not None:
        _patch_torch_hub_for_local_dinov2(dinov2_repo)


def _resolve_torch_hub_dir(args: argparse.Namespace) -> Path | None:
    candidates: list[Path] = []
    for value in (args.torch_hub_dir, os.environ.get("TORCH_HUB_DIR")):
        if value:
            candidates.append(Path(value).expanduser())
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        home = Path(torch_home).expanduser()
        candidates.extend([home / "hub", home])
    candidates.extend(
        [
            Path.home() / ".cache" / "torch" / "hub",
            Path("/mnt/sda3/yu/checkpoints/hub/hub"),
            Path("/mnt/sda3/yu/checkpoints/hub"),
        ]
    )
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if (candidate / "facebookresearch_dinov2_main" / "hubconf.py").exists():
            return candidate
        nested = candidate / "hub"
        if (nested / "facebookresearch_dinov2_main" / "hubconf.py").exists():
            return nested
        if candidate.exists():
            return candidate
    return None


def _resolve_dinov2_repo(args: argparse.Namespace, hub_dir: Path | None) -> Path | None:
    candidates: list[Path] = []
    if args.dinov2_repo:
        candidates.append(Path(args.dinov2_repo).expanduser())
    if hub_dir is not None:
        candidates.append(hub_dir / "facebookresearch_dinov2_main")
        candidates.extend(sorted(hub_dir.glob("facebookresearch_dinov2*")))
    for candidate in candidates:
        if (candidate / "hubconf.py").exists():
            return candidate
    return None


def _patch_torch_hub_for_local_dinov2(local_repo: Path) -> None:
    if getattr(torch.hub, "_geoss_local_dinov2_patch", False):
        return
    original_load = torch.hub.load

    def load(repo_or_dir, model, *args, **kwargs):
        if str(repo_or_dir).rstrip("/") == "facebookresearch/dinov2":
            local_kwargs = dict(kwargs)
            local_kwargs.pop("trust_repo", None)
            local_kwargs.pop("force_reload", None)
            return original_load(str(local_repo), model, *args, source="local", **local_kwargs)
        return original_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = load
    torch.hub._geoss_local_dinov2_patch = True


@torch.no_grad()
def _compute_trellis_slat_base_velocity(batch: dict, x_t: torch.Tensor, indices: torch.Tensor, t: torch.Tensor, pipeline, device: torch.device) -> torch.Tensor:
    if pipeline is None:
        raise KeyError("real_train requires trellis_slat_base_velocity or a real TRELLIS pipeline to compute frozen SLAT base velocity.")
    from trellis.modules import sparse as sp

    B, L, C = x_t.shape
    batch_ids = torch.arange(B, device=device).view(B, 1, 1).expand(B, L, 1)
    coords = torch.cat([batch_ids, indices.long()], dim=-1).reshape(B * L, 4).int().contiguous()
    sparse = sp.SparseTensor(feats=x_t.reshape(B * L, C).contiguous(), coords=coords)
    cond = _real_slat_condition(batch, device, pipeline)
    v_sparse = pipeline.models["slat_flow_model"](sparse, t * 1000.0, cond)
    if not hasattr(v_sparse, "feats"):
        raise TypeError("TRELLIS slat_flow_model must return a SparseTensor with feats.")
    return v_sparse.feats.reshape(B, L, C).to(device=device, dtype=x_t.dtype).detach()


@torch.no_grad()
def _real_slat_condition(batch: dict, device: torch.device, pipeline) -> torch.Tensor:
    for key in ("trellis_cond", "trellis_cond_tokens", "image_cond", "cond"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.float32)
    cond_image = batch.get("trellis_cond_image")
    if isinstance(cond_image, torch.Tensor):
        return pipeline.encode_image(cond_image.to(device=device, dtype=torch.float32))
    images = batch.get("images")
    if isinstance(images, torch.Tensor):
        first_view = F.interpolate(images[:, 0].to(device=device, dtype=torch.float32), size=(518, 518), mode="bilinear", align_corners=False)
        return pipeline.encode_image(first_view)
    raise KeyError("real_train requires TRELLIS condition image/tokens to compute frozen SLAT base velocity.")


def _apply_config_defaults(args: argparse.Namespace, cfg: dict, parser: argparse.ArgumentParser) -> None:
    if not cfg:
        return
    dataset = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else {}
    trellis = cfg.get("trellis") if isinstance(cfg.get("trellis"), dict) else {}
    mappings = {
        "meshfleet_root": cfg.get("meshfleet_root") or cfg.get("dataset_root") or dataset.get("root"),
        "meshfleet_split": cfg.get("meshfleet_split") or dataset.get("train_split") or dataset.get("split"),
        "meshfleet_category": cfg.get("meshfleet_category") or dataset.get("category"),
        "num_views": cfg.get("num_views") or dataset.get("num_views"),
        "image_size": cfg.get("image_size") or dataset.get("image_size"),
        "trellis_root": cfg.get("trellis_root") or trellis.get("root"),
        "trellis_model_path": cfg.get("trellis_model_path") or cfg.get("trellis_pipeline") or cfg.get("trellis_checkpoint") or trellis.get("model_path") or trellis.get("pipeline") or trellis.get("checkpoint"),
        "steps": cfg.get("steps"),
        "batch_size": cfg.get("batch_size"),
        "lr": cfg.get("lr"),
        "weight_decay": cfg.get("weight_decay"),
        "velocity_weight": cfg.get("velocity_weight"),
        "prior_weight": cfg.get("prior_weight"),
        "raw_residual_weight": cfg.get("raw_residual_weight"),
        "effective_residual_weight": cfg.get("effective_residual_weight"),
        "grad_accum_steps": cfg.get("grad_accum_steps"),
        "save_every": cfg.get("save_every"),
        "output_dir": cfg.get("output_dir"),
        "device": cfg.get("device"),
        "torch_hub_dir": cfg.get("torch_hub_dir") or trellis.get("torch_hub_dir"),
        "dinov2_repo": cfg.get("dinov2_repo") or trellis.get("dinov2_repo"),
        **adaptive_config_defaults(cfg),
    }
    for name, value in mappings.items():
        if value is None or not hasattr(args, name):
            continue
        if getattr(args, name) == parser.get_default(name):
            setattr(args, name, value)


if __name__ == "__main__":
    main()
