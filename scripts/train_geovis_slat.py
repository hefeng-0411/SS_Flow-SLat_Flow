from __future__ import annotations

import argparse
import contextlib
import inspect
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
from geoss.integration.vggt_geometry_wrapper import VGGTGeometryWrapper
from geoss.integration.trellis_residency import configure_trellis_training_residency
from geoss.geometry.alignment import align_vggt_batch
from geoss.slat.integration.ss_slat_context import build_ss_slat_context
from geoss.slat.losses.appearance_feature_loss import appearance_feature_loss
from geoss.slat.losses.slat_flow_loss import slat_flow_matching_loss
from geoss.slat.losses.slat_prior_preservation_loss import slat_prior_preservation_loss
from geoss.slat.losses.slat_velocity_loss import slat_velocity_regularization_loss
from geoss.slat.losses.view_consistency_loss import view_consistency_loss
from geoss.slat.losses.visibility_confidence_loss import visibility_confidence_loss
from geoss.slat.losses.factorized_control_loss import factorized_control_loss
from geoss.slat.losses.decoded_asset_loss import DecodedAssetSupervisor
from geoss.slat.models.geovis_slat_adapter import GeoVisSLATAdapter
from geoss.slat.utils.normalization import SLAT_TENSOR_CONTRACT_VERSION, normalize_slat
from geoss.slat.utils.slat_visualization import save_slat_debug_npz, write_active_voxels_ply
from geoss.utils.adaptive_batch import AdaptiveBatchController, adaptive_config_defaults, add_adaptive_batch_args
from geoss.utils.checkpoint import save_checkpoint
from geoss.utils.config import add_common_args, load_config, str2bool
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
    run_modes = validate_real_mode(cfg=cfg, args=args, mode="real_train", required=("vggt", "trellis", "dataset"))
    ctx = init_distributed(args)
    device = ctx.device
    batch_controller = AdaptiveBatchController.from_args(args)
    args.batch_size = batch_controller.batch_size
    decoded_config = cfg.get("decoded_supervision") if isinstance(cfg.get("decoded_supervision"), dict) else {}
    required_trellis_models = ["slat_flow_model", "image_cond_model"]
    if bool(decoded_config.get("enabled", False)):
        required_trellis_models.append("slat_decoder_gs")
    trellis_pipeline = _load_trellis_pipeline(args, device, ctx, required_models=required_trellis_models)
    decoded_supervisor = DecodedAssetSupervisor(trellis_pipeline, cfg.get("decoded_supervision"))
    vggt_geometry = _load_vggt_geometry(args, device, ctx)
    model_cfg = dict(cfg.get("model", {}))
    actual_slat_dim = int(trellis_pipeline.models["slat_flow_model"].in_channels)
    configured_slat_dim = int(model_cfg.get("slat_dim", actual_slat_dim))
    if configured_slat_dim != actual_slat_dim:
        # A hardcoded eight-channel adapter silently drifts from real TRELLIS
        # checkpoints; construct the adapter with the model's actual interface.
        model_cfg["slat_dim"] = actual_slat_dim
        cfg = {**cfg, "model": model_cfg}
    model = GeoVisSLATAdapter(**model_cfg).to(device)
    model.enable_gradient_checkpointing(args.gradient_checkpointing)
    start_step = 0
    resume_state = None
    if args.resume and Path(args.resume).exists():
        resume_state = torch.load(args.resume, map_location="cpu")
        _validate_checkpoint_tensor_contract(resume_state, args.resume)
        _validate_checkpoint_model_config(resume_state, model_cfg, context="Resume")
        model.load_state_dict(resume_state.get("model", resume_state), strict=True)
        start_step = int(resume_state.get("step", 0))
    elif args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if not init_path.is_file():
            raise FileNotFoundError(f"SLAT initialization checkpoint not found: {init_path}")
        init_state = torch.load(init_path, map_location="cpu")
        _validate_checkpoint_tensor_contract(init_state, init_path)
        _validate_checkpoint_model_config(init_state, model_cfg, context="Initialization")
        model.load_state_dict(init_state.get("model", init_state), strict=True)
    model = maybe_wrap_ddp(model, ctx, find_unused_parameters=args.ddp_find_unused_parameters)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # A800/H100-class GPUs support BF16 natively.  Its wider exponent range
    # avoids the FP16 overflow that previously poisoned the velocity head.
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = _make_grad_scaler(enabled=args.amp and amp_dtype == torch.float16 and device.type == "cuda")
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
                batch = prepare_batch(
                    raw_batch, cfg, args, device,
                    trellis_pipeline=trellis_pipeline,
                    vggt_geometry=vggt_geometry,
                )
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=args.amp and device.type == "cuda",
                ):
                    out = model(batch)
                    terms = compute_losses(
                        out, batch, raw_residual_weight=args.raw_residual_weight,
                        effective_residual_weight=args.effective_residual_weight,
                    )
                    terms["decoded_asset"] = decoded_supervisor(out, batch, step)
                    loss = (
                        terms["slat_flow"]["loss"] + 0.25 * terms["view"]["loss"]
                        + 0.25 * terms["appearance"]["loss"] + 0.2 * terms["visibility_confidence"]["loss"]
                        + float(cfg.get("training", {}).get("factorized_control_weight", 0.2)) * terms["factorized_control"]["loss"]
                        + args.velocity_weight * terms["velocity"]["loss"] + args.prior_weight * terms["prior"]["loss"]
                        + terms["decoded_asset"]["loss"]
                    )
                if not torch.isfinite(loss).all().item():
                    raise FloatingPointError("Stage 3 loss became NaN/Inf before backward.")
                sync_context = (
                    model.no_sync()
                    if ctx.distributed and hasattr(model, "no_sync") and micro_step < grad_accum_steps - 1
                    else contextlib.nullcontext()
                )
                with sync_context:
                    scaler.scale(loss / grad_accum_steps).backward()
                total_loss = total_loss + loss.detach()
                last_batch, last_out, last_terms = batch, out, terms
            scaler.unscale_(opt)
            grad_norms, nonfinite_gradients = _inspect_geovis_slat_gradients(unwrap_model(model), step)
            if nonfinite_gradients:
                # AMP overflow is recoverable.  Do not write NaN/Inf into the
                # checkpoint; discard this update and let GradScaler back off.
                opt.zero_grad(set_to_none=True)
                scaler.update()
                global_grad_norm = float("nan")
            else:
                # Clip only rare large updates.  This preserves normal velocity
                # convergence while bounding the residual head's outliers.
                global_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).detach().cpu())
                scaler.step(opt)
                scaler.update()
            assert last_batch is not None and last_out is not None and last_terms is not None
            last_terms["grad_norms"] = grad_norms
            last_terms["optimizer_step_skipped"] = bool(nonfinite_gradients)
            last_terms["global_grad_norm"] = global_grad_norm
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
        last["optimizer_step_skipped"] = bool(terms.get("optimizer_step_skipped", False))
        last["global_grad_norm"] = terms.get("global_grad_norm")
        last["adaptive_batch"] = {**batch_controller.state_dict(), "last_adjustment": batch_adjustment.as_dict()}
        last["trellis_residency"] = getattr(trellis_pipeline, "training_residency", None)
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
        periodic_save = args.save_every > 0 and step % args.save_every == 0
        if ctx.is_main and (should_fault_save or periodic_save or step == end_step):
            _save_slat_checkpoint(out_dir / "geovis_slat_adapter_last.pt", model, opt, step, cfg, early_stopper, early_status)
        if ctx.is_main and periodic_save:
            _save_slat_checkpoint(
                out_dir / f"geovis_slat_adapter_step_{step:08d}.pt",
                model, opt, step, cfg, early_stopper, early_status,
            )
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
    supervision_mask = batch.get("slat_supervision_mask")
    raw_flow = slat_flow_matching_loss(raw_delta, target_residual.detach(), supervision_mask)
    effective_flow = slat_flow_matching_loss(effective_delta, target_residual.detach(), supervision_mask)
    flow_loss = raw_residual_weight * raw_flow["loss"] + effective_residual_weight * effective_flow["loss"]
    base_valid_mask = batch.get("v_slat_base_valid_mask")
    joint_confidence = out["debug"]["joint_confidence"]
    if base_valid_mask is not None:
        assert base_valid_mask.shape == joint_confidence.shape, (
            f"v_slat_base_valid_mask shape {tuple(base_valid_mask.shape)} "
            f"must match joint_confidence {tuple(joint_confidence.shape)}"
        )
        # The prior loss uses weight=(1-confidence). Invalid frozen-teacher
        # tokens must therefore be assigned confidence=1 so they contribute
        # zero preservation pressure instead of preserving the placeholder base.
        joint_confidence = torch.where(base_valid_mask.bool(), joint_confidence, torch.ones_like(joint_confidence))
    base_invalid_ratio = batch.get("v_slat_base_invalid_ratio", torch.zeros((), device=out["v_slat_geo"].device))
    token_valid_mask = batch.get("slat_token_valid_mask")
    factorized = factorized_control_loss(
        out["correction_demand"],
        out["residual_variance"],
        effective_delta,
        target_residual,
        token_mask=supervision_mask,
    )
    return {
        "slat_flow": {
            "loss": flow_loss,
            "slat_flow_mse": flow_loss,
            "raw_residual_mse": raw_flow["loss"],
            "effective_residual_mse": effective_flow["loss"],
        },
        "base_velocity": {"invalid_ratio": base_invalid_ratio.detach()},
        "view": view_consistency_loss(out["sampled_features"], out["visibility"], token_valid_mask),
        "appearance": appearance_feature_loss(
            out["slat_cond_tokens"], out["sampled_features"], out["visibility"], out["view_weights"], token_valid_mask
        ),
        "visibility_confidence": visibility_confidence_loss(
            out["slat_confidence"],
            out["visibility"],
            out["depth_residual"],
            appearance_conflict=out["appearance_conflict"],
            occlusion_score=out["occlusion_score"],
            token_valid_mask=token_valid_mask,
        ),
        "factorized_control": factorized,
        "velocity": slat_velocity_regularization_loss(out["delta_v_slat"], batch["timestep"], token_valid_mask),
        "prior": slat_prior_preservation_loss(out["v_slat_geo"], batch["v_slat_base"], joint_confidence),
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
        "slat_base_invalid_ratio": float(terms["base_velocity"]["invalid_ratio"].detach().cpu()),
        "loss_prior": float(terms["prior"]["loss"].detach().cpu()),
        "loss_velocity": float(terms["velocity"]["loss"].detach().cpu()),
        "loss_decoded_asset": float(terms["decoded_asset"]["loss"].detach().cpu()) if "decoded_asset" in terms else 0.0,
        "loss_decoded_render": float(terms["decoded_asset"].get("render_loss", terms["decoded_asset"]["loss"]).detach().cpu()) if "decoded_asset" in terms else 0.0,
        "loss_decoded_geometry": float(terms["decoded_asset"].get("geometry_loss", terms["decoded_asset"]["loss"]).detach().cpu()) if "decoded_asset" in terms else 0.0,
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
    base_valid_mask = batch.get("v_slat_base_valid_mask")
    if base_valid_mask is not None:
        assert base_valid_mask.shape == (*x_t.shape[:2], 1), (
            f"v_slat_base_valid_mask must be [B,L,1], got {tuple(base_valid_mask.shape)}"
        )
        assert base_valid_mask.dtype == torch.float32, f"v_slat_base_valid_mask must be float32, got {base_valid_mask.dtype}"
        assert torch.isfinite(base_valid_mask).all().item(), "v_slat_base_valid_mask contains NaN or Inf."
        assert base_valid_mask.min().item() >= 0.0 and base_valid_mask.max().item() <= 1.0, "v_slat_base_valid_mask must be in [0, 1]."
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
        assert tensor.dtype in (torch.float16, torch.bfloat16, torch.float32), f"{name} must be a floating point training dtype, got {tensor.dtype}"
        assert torch.isfinite(tensor).all().item(), f"{name} contains NaN or Inf."
    assert raw_delta.requires_grad, "raw SLAT residual must depend on GeoVisSLATAdapter parameters."
    assert effective_delta.requires_grad, "effective SLAT residual must depend on GeoVisSLATAdapter parameters."
    assert not target_residual.requires_grad, "target_residual must be a frozen flow-matching target."
    assert out["slat_confidence"].shape == (*x_t.shape[:2], 1), f"slat_confidence must be [B,L,1], got {tuple(out['slat_confidence'].shape)}"
    assert out["visibility"].shape[:2] == x_t.shape[:2], f"visibility must align with SLAT tokens, got {tuple(out['visibility'].shape)}"
    assert out["slat_confidence"].min().item() >= 0.0 and out["slat_confidence"].max().item() <= 1.0, "slat_confidence must be in [0, 1]."
    assert out["visibility"].min().item() >= 0.0 and out["visibility"].max().item() <= 1.0, "visibility must be in [0, 1]."


def _inspect_geovis_slat_gradients(model: GeoVisSLATAdapter, step: int) -> tuple[dict[str, float], list[str]]:
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
    norms = {
        name: float(param.grad.detach().norm().cpu()) if name not in nonfinite else float("nan")
        for name, param in critical.items()
    }
    return norms, nonfinite


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
    batch["v_slat_base_valid_mask"] = torch.ones(B, L, 1, device=device, dtype=torch.float32)
    batch["v_slat_base_invalid_ratio"] = torch.zeros((), device=device, dtype=torch.float32)
    batch["target_residual"] = batch["target_velocity"].detach()
    return batch


def prepare_batch(
    raw: dict,
    cfg: dict,
    args: argparse.Namespace,
    device: torch.device,
    *,
    trellis_pipeline=None,
    vggt_geometry: VGGTGeometryWrapper | None = None,
) -> dict:
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in raw.items()}
    if vggt_geometry is None:
        raise RuntimeError("real_train requires a frozen VGGT geometry extractor; mock VGGT is not permitted.")
    with torch.inference_mode():
        # Refresh frozen VGGT evidence from the exact camera views used by this
        # batch; only the adapter remains trainable and no mock context leaks in.
        vggt_context = vggt_geometry(batch["images"])
    for key in ("vggt_features", "vggt_depth", "vggt_pointmap", "vggt_confidence", "vggt_camera"):
        value = vggt_context.get(key)
        if isinstance(value, (torch.Tensor, dict)):
            batch[key] = value
    batch = align_vggt_batch(batch)
    model_cfg = cfg.get("model", {})
    slat_dim = int(model_cfg.get("slat_dim", 8))
    resolution = int(model_cfg.get("resolution", 64))
    if "trellis_slat_feats" in batch and "trellis_slat_indices" in batch:
        feats, indices, token_valid_mask = _pad_latents(
            batch["trellis_slat_feats"], batch["trellis_slat_indices"], args.active_tokens, device
        )
        x0_raw = feats[..., :slat_dim]
        if x0_raw.shape[-1] < slat_dim:
            x0_raw = torch.cat([x0_raw, x0_raw.new_zeros(*x0_raw.shape[:-1], slat_dim - x0_raw.shape[-1])], dim=-1)
        if trellis_pipeline is None:
            raise RuntimeError("Real SLAT training requires a TRELLIS pipeline with its published latent normalization.")
        x0 = normalize_slat(x0_raw, trellis_pipeline.slat_normalization)
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
    batch["slat_clean_tokens"] = x0
    batch["slat_raw_tokens"] = x0_raw
    batch["slat_indices"] = indices
    batch["flow_sigma_min"] = sigma_min
    batch["target_velocity"] = (1 - sigma_min) * noise - x0
    base = batch.get("trellis_slat_base_velocity")
    if base is None:
        base = _compute_trellis_slat_base_velocity(
            batch, x_t, indices, token_valid_mask, t, trellis_pipeline, device,
            use_multiview=bool(cfg.get("training", {}).get("multiview_teacher", True)),
        )
    base, base_valid_mask, base_invalid_ratio = _sanitize_slat_base_velocity(base, expected_shape=x_t.shape, device=device, dtype=x_t.dtype)
    batch["v_slat_base"] = base
    batch["slat_token_valid_mask"] = token_valid_mask
    batch["v_slat_base_valid_mask"] = base_valid_mask
    batch["v_slat_base_invalid_ratio"] = base_invalid_ratio
    sigma = sigma_min + (1.0 - sigma_min) * t.view(B, 1, 1)
    base_reference = ((1.0 - sigma_min) * x_t - sigma * base).detach()
    # Match test time: the evidence aggregator receives a prediction from the
    # frozen TRELLIS prior, never the clean target latent. Invalid teacher rows
    # retain the observable noisy state and are excluded from teacher losses.
    batch["slat_reference_tokens"] = torch.where(
        base_valid_mask.bool(), base_reference, x_t.detach()
    )
    batch["target_residual"] = (batch["target_velocity"] - batch["v_slat_base"]).detach()
    has_gt = batch.get("has_gt", batch.get("gt_available", True))
    has_gt = torch.as_tensor(has_gt, device=device, dtype=torch.float32).reshape(-1, 1, 1)
    if has_gt.shape[0] == 1 and B > 1:
        has_gt = has_gt.expand(B, -1, -1)
    batch["slat_supervision_mask"] = (
        has_gt.expand(B, x_t.shape[1], 1)
        * torch.isfinite(batch["target_residual"]).all(dim=-1, keepdim=True).float()
        * token_valid_mask
    )
    batch["timestep"] = t
    batch["slat_target_source"] = source
    return batch


def _sanitize_slat_base_velocity(
    base: torch.Tensor,
    *,
    expected_shape: torch.Size,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a finite frozen-teacher velocity and a mask of valid teacher tokens.

    Stage 3 trains a residual velocity:
        delta_theta(x_t, t, c) ~= v_target - v_base.
    When the frozen TRELLIS teacher produces NaN/Inf for a sparse token, v_base is
    undefined for that token. The correct objective is to remove the teacher prior
    for that token and train against the direct target velocity, which is exactly
    implemented by setting the invalid teacher contribution to zero and carrying
    a validity mask into the prior-preservation term.
    """
    assert isinstance(base, torch.Tensor), f"v_slat_base must be a tensor, got {type(base)!r}"
    assert len(expected_shape) == 3, f"expected_shape must describe [B,L,C], got {tuple(expected_shape)}"
    B, L, C = expected_shape
    assert base.ndim == 3, f"v_slat_base must be [B,L,C], got {tuple(base.shape)}"
    assert base.shape[0] == B, f"v_slat_base batch {base.shape[0]} != expected {B}"
    assert base.shape[1] >= L, f"v_slat_base tokens {base.shape[1]} < expected {L}"
    assert base.shape[2] >= C, f"v_slat_base channels {base.shape[2]} < expected {C}"
    sliced = base[:, :L, :C].to(device=device, dtype=dtype)
    assert sliced.dtype == torch.float32, f"SLAT base velocity must be float32 after conversion, got {sliced.dtype}"
    valid_mask_bool = torch.isfinite(sliced).all(dim=-1, keepdim=True)
    invalid_ratio = 1.0 - valid_mask_bool.float().mean()
    safe_base = torch.where(valid_mask_bool, torch.nan_to_num(sliced, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(sliced))
    valid_mask = valid_mask_bool.to(dtype=torch.float32)
    assert safe_base.shape == expected_shape, f"sanitized v_slat_base shape {tuple(safe_base.shape)} != {tuple(expected_shape)}"
    assert valid_mask.shape == (B, L, 1), f"v_slat_base_valid_mask shape {tuple(valid_mask.shape)} != {(B, L, 1)}"
    assert torch.isfinite(safe_base).all().item(), "sanitized v_slat_base still contains NaN or Inf."
    assert torch.isfinite(invalid_ratio).all().item(), "v_slat_base_invalid_ratio is non-finite."
    return safe_base, valid_mask, invalid_ratio


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
                require_voxels=True,
                uid_manifest=args.train_manifest,
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
        max_length = max(f.shape[0] for f in feats)
        L = min(limit, max_length) if limit > 0 else max_length
        C = feats[0].shape[-1]
        out_feats = torch.zeros(B, L, C, device=device)
        out_idx = torch.zeros(B, L, 3, dtype=torch.long, device=device)
        valid = torch.zeros(B, L, 1, dtype=torch.float32, device=device)
        for b, (f, idx) in enumerate(zip(feats, indices)):
            take = min(L, f.shape[0])
            out_feats[b, :take] = f[:take].to(device)
            out_idx[b, :take] = idx[:take].to(device)
            valid[b, :take] = 1.0
        return out_feats, out_idx, valid
    length = min(limit, feats.shape[1]) if limit > 0 else feats.shape[1]
    valid = torch.ones(feats.shape[0], length, 1, dtype=torch.float32, device=device)
    return feats[:, :length].to(device), indices[:, :length].to(device), valid


def _pad_indices(indices, limit: int, device: torch.device):
    if isinstance(indices, list):
        B = len(indices)
        max_length = max(max(1, idx.shape[0]) for idx in indices)
        L = min(limit, max_length) if limit > 0 else max_length
        out = torch.zeros(B, L, 3, dtype=torch.long, device=device)
        for b, idx in enumerate(indices):
            take = min(L, idx.shape[0])
            if take:
                out[b, :take] = idx[:take].to(device)
        return out
    length = min(limit, indices.shape[1]) if limit > 0 else indices.shape[1]
    return indices[:, :length].to(device)


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
        tensor_contract={
            "version": SLAT_TENSOR_CONTRACT_VERSION,
            "flow_state": "trellis_normalized_slat",
            "decoder_state": "trellis_raw_vae_slat",
            "control_variables": ["evidence_reliability", "correction_demand", "residual_variance"],
            "teacher_velocity": "multiview_cfg_matched",
            "sampler_correction": "cfg_invariant_unit_residual",
            "decoded_supervision": bool(cfg.get("decoded_supervision", {}).get("enabled", False)),
        },
        early_stop=early_status.as_dict() if early_status is not None else None,
        early_stopper=early_stopper.state_dict(),
    )


def _make_grad_scaler(*, enabled: bool):
    """Use the current AMP API while retaining support for older torch builds."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _validate_checkpoint_tensor_contract(state: dict, path: str | Path) -> None:
    contract = state.get("tensor_contract") if isinstance(state, dict) else None
    version = contract.get("version") if isinstance(contract, dict) else None
    if version != SLAT_TENSOR_CONTRACT_VERSION:
        raise RuntimeError(
            f"SLAT checkpoint {path} has tensor contract {version!r}; "
            f"expected {SLAT_TENSOR_CONTRACT_VERSION!r}. Retrain instead of partially loading an incompatible model."
        )


def _validate_checkpoint_model_config(state: dict, model_cfg: dict, *, context: str) -> None:
    checkpoint_cfg = state.get("config", {}).get("model", {}) if isinstance(state, dict) else {}
    keys = (
        "slat_dim", "resolution", "evidence_dim", "hidden_dim", "feature_dim", "num_heads",
        "fusion_mode", "confidence_floor", "trust_region", "beta_mode", "beta_strength",
        "factorized_control", "use_geovis_slat",
    )
    for key in keys:
        if key in checkpoint_cfg and key in model_cfg and checkpoint_cfg[key] != model_cfg[key]:
            raise RuntimeError(
                f"{context} model mismatch for {key}: checkpoint={checkpoint_cfg[key]!r}, "
                f"config={model_cfg[key]!r}"
            )


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--active_tokens", type=int, default=0, help="Maximum SLAT tokens per object; 0 keeps every active voxel.")
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
    parser.add_argument("--init_checkpoint", type=str, default=None, help="Weights-only initialization; optimizer and step restart from zero.")
    parser.add_argument("--meshfleet_root", type=str, default=None)
    parser.add_argument("--meshfleet_split", type=str, default="train")
    parser.add_argument("--train_manifest", type=str, default=None, help="UID manifest generated by inspect_meshfleet_dataset.py.")
    parser.add_argument("--meshfleet_category", type=str, default=None)
    parser.add_argument("--meshfleet_slat_latent_model", type=str, default="dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16")
    parser.add_argument("--srn_root", type=str, default=None)
    parser.add_argument("--objaverse_rendered_root", type=str, default=None)
    parser.add_argument("--trellis_root", type=str, default=None)
    parser.add_argument("--trellis_model_path", type=str, default=None)
    parser.add_argument("--vggt_root", type=str, default=None)
    parser.add_argument("--vggt_checkpoint", type=str, default=None)
    parser.add_argument("--vggt_pretrained", type=str, default=None)
    parser.add_argument("--torch_hub_dir", type=str, default=None)
    parser.add_argument("--dinov2_repo", type=str, default=None)
    parser.add_argument("--real_train", action="store_true")
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)
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


def _load_trellis_pipeline(
    args: argparse.Namespace,
    device: torch.device,
    ctx,
    *,
    required_models: list[str],
):
    if args.trellis_root:
        sys.path.insert(0, args.trellis_root)
    if not args.trellis_model_path:
        raise FileNotFoundError("real_train requires --trellis_model_path or trellis.pipeline in config.")
    _configure_trellis_hub(args)

    if ctx.distributed and not ctx.is_main:
        dist.barrier()
    pipeline = _load_trellis_pipeline_impl(args, device, required_models=required_models)
    if ctx.distributed and ctx.is_main:
        dist.barrier()
    return pipeline


def _load_vggt_geometry(args: argparse.Namespace, device: torch.device, ctx) -> VGGTGeometryWrapper:
    """Load one frozen real VGGT replica per DDP rank before training starts."""
    if ctx.distributed and not ctx.is_main:
        dist.barrier()
    geometry = VGGTGeometryWrapper(
        vggt_root=args.vggt_root,
        checkpoint=args.vggt_checkpoint,
        pretrained_name=args.vggt_pretrained,
        mock=False,
    ).to(device)
    geometry.eval()
    if ctx.distributed and ctx.is_main:
        dist.barrier()
    return geometry


def _load_trellis_pipeline_impl(
    args: argparse.Namespace,
    device: torch.device,
    *,
    required_models: list[str],
):
    from trellis.pipelines import TrellisImageTo3DPipeline

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.trellis_model_path)
    pipeline.training_residency = configure_trellis_training_residency(
        pipeline,
        required_models=required_models,
        device=device,
    )
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
def _compute_trellis_slat_base_velocity(
    batch: dict,
    x_t: torch.Tensor,
    indices: torch.Tensor,
    token_valid_mask: torch.Tensor,
    t: torch.Tensor,
    pipeline,
    device: torch.device,
    *,
    use_multiview: bool = True,
) -> torch.Tensor:
    if pipeline is None:
        raise KeyError("real_train requires trellis_slat_base_velocity or a real TRELLIS pipeline to compute frozen SLAT base velocity.")
    from trellis.modules import sparse as sp

    B, L, C = x_t.shape
    if token_valid_mask.shape != (B, L, 1):
        raise ValueError(
            f"token_valid_mask must be [B,L,1], got {tuple(token_valid_mask.shape)} for {tuple(x_t.shape)}"
        )
    valid = token_valid_mask.squeeze(-1).bool()
    sparse_feats, sparse_coords = [], []
    for batch_index in range(B):
        selected = valid[batch_index]
        if not selected.any():
            raise RuntimeError(f"Object {batch_index} has no valid SLAT tokens.")
        sparse_feats.append(x_t[batch_index, selected])
        batch_column = torch.full(
            (int(selected.sum()), 1), batch_index, device=device, dtype=indices.dtype
        )
        sparse_coords.append(torch.cat([batch_column, indices[batch_index, selected]], dim=-1))
    coords = torch.cat(sparse_coords, dim=0).int().contiguous()
    sparse = sp.SparseTensor(feats=torch.cat(sparse_feats, dim=0).contiguous(), coords=coords)
    conditions = _real_slat_conditions(batch, device, pipeline)
    if not use_multiview:
        conditions = conditions[:1]
    predictions = [pipeline.models["slat_flow_model"](sparse, t * 1000.0, cond) for cond in conditions]
    if not predictions or any(not hasattr(item, "feats") for item in predictions):
        raise TypeError("TRELLIS slat_flow_model must return SparseTensor predictions with feats.")
    if any(not torch.equal(item.coords, predictions[0].coords) for item in predictions[1:]):
        raise RuntimeError("TRELLIS multi-view teacher returned inconsistent sparse coordinate ordering.")
    # Match the positive branch of inference-time MultiDiffusion: every view
    # predicts a velocity for the same sparse state, then velocities are averaged.
    positive_feats = torch.stack([item.feats for item in predictions], dim=0).mean(dim=0)
    cfg_strength, cfg_interval = _trellis_slat_cfg_parameters(pipeline)
    if cfg_strength != 0.0:
        negative = pipeline.models["slat_flow_model"](
            sparse, t * 1000.0, torch.zeros_like(conditions[0])
        )
        if not hasattr(negative, "feats") or not torch.equal(negative.coords, predictions[0].coords):
            raise RuntimeError("TRELLIS negative CFG teacher changed the sparse coordinate contract.")
        cfg_active = ((t >= cfg_interval[0]) & (t <= cfg_interval[1])).to(positive_feats.dtype)
        per_row_strength = cfg_active[predictions[0].coords[:, 0].long()].unsqueeze(-1) * cfg_strength
        feats = positive_feats + per_row_strength * (positive_feats - negative.feats)
    else:
        feats = positive_feats
    if feats.ndim != 2:
        raise ValueError(f"TRELLIS slat_flow_model feats must be [B*L,C], got {tuple(feats.shape)}")
    if feats.shape[0] != int(valid.sum()):
        raise ValueError(
            f"TRELLIS slat_flow_model returned {feats.shape[0]} tokens, expected {int(valid.sum())} valid tokens."
        )
    if feats.shape[1] < C:
        raise ValueError(f"TRELLIS slat_flow_model returned {feats.shape[1]} channels, expected at least {C}.")
    return _repad_sparse_prediction(
        feats[:, :C], predictions[0].coords, indices, valid, dtype=x_t.dtype
    ).detach()


def _trellis_slat_cfg_parameters(pipeline) -> tuple[float, tuple[float, float]]:
    """Read the exact CFG defaults/overrides used by TRELLIS' configured sampler."""
    parameters = inspect.signature(pipeline.slat_sampler.sample).parameters
    if "cfg_strength" not in parameters:
        return 0.0, (0.0, 1.0)
    overrides = dict(getattr(pipeline, "slat_sampler_params", {}) or {})
    strength = float(overrides.get("cfg_strength", parameters["cfg_strength"].default))
    interval_default = parameters["cfg_interval"].default if "cfg_interval" in parameters else (0.0, 1.0)
    interval = overrides.get("cfg_interval", interval_default)
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise ValueError(f"TRELLIS SLAT cfg_interval must have two values, got {interval!r}")
    interval_pair = (float(interval[0]), float(interval[1]))
    if strength <= -1.0 or not 0.0 <= interval_pair[0] <= interval_pair[1] <= 1.0:
        raise ValueError(f"Invalid TRELLIS SLAT CFG contract: strength={strength}, interval={interval_pair}")
    return strength, interval_pair


def _repad_sparse_prediction(
    feats: torch.Tensor,
    coords: torch.Tensor,
    target_indices: torch.Tensor,
    valid: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Repad teacher output by exact coordinate identity, even if sparse ops reorder rows."""
    B, L = valid.shape
    out = feats.new_zeros(B, L, feats.shape[-1], dtype=dtype)
    for batch_index in range(B):
        target_positions = torch.nonzero(valid[batch_index], as_tuple=False).flatten()
        target = target_indices[batch_index, target_positions].long()
        source_rows = torch.nonzero(coords[:, 0].long() == batch_index, as_tuple=False).flatten()
        source = coords[source_rows, 1:].long()
        if source.shape[0] != target.shape[0]:
            raise RuntimeError(
                f"TRELLIS teacher changed sparse token count for object {batch_index}: "
                f"input={target.shape[0]}, output={source.shape[0]}."
            )
        resolution = int(torch.cat([source, target], dim=0).max().item()) + 1
        source_keys = source[:, 0] * resolution * resolution + source[:, 1] * resolution + source[:, 2]
        target_keys = target[:, 0] * resolution * resolution + target[:, 1] * resolution + target[:, 2]
        sorted_keys, order = source_keys.sort()
        lookup = torch.searchsorted(sorted_keys, target_keys)
        if (lookup >= sorted_keys.numel()).any() or not torch.equal(sorted_keys[lookup], target_keys):
            raise RuntimeError(f"TRELLIS teacher changed sparse coordinates for object {batch_index}.")
        out[batch_index, target_positions] = feats[source_rows[order[lookup]]].to(dtype=dtype)
    return out


@torch.no_grad()
def _real_slat_conditions(batch: dict, device: torch.device, pipeline) -> list[torch.Tensor]:
    for key in ("trellis_cond", "trellis_cond_tokens", "image_cond", "cond"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            return [value.to(device=device, dtype=torch.float32)]
    images = batch.get("images")
    if isinstance(images, torch.Tensor) and images.ndim == 5:
        return [
            pipeline.encode_image(
                F.interpolate(
                    images[:, view].to(device=device, dtype=torch.float32),
                    size=(518, 518), mode="bicubic", align_corners=False, antialias=True,
                )
            )
            for view in range(images.shape[1])
        ]
    cond_image = batch.get("trellis_cond_image")
    if isinstance(cond_image, torch.Tensor):
        return [pipeline.encode_image(cond_image.to(device=device, dtype=torch.float32))]
    images = batch.get("images")
    if isinstance(images, torch.Tensor):
        first_view = F.interpolate(images[:, 0].to(device=device, dtype=torch.float32), size=(518, 518), mode="bilinear", align_corners=False)
        return [pipeline.encode_image(first_view)]
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
