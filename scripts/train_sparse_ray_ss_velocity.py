from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from geoss.datasets.meshfleet_trellis_dataset import MeshFleetTrellisDataset
from geoss.datasets.vehicle_multiview_dataset import VehicleMultiViewDataset
from geoss.integration.vggt_geometry_wrapper import VGGTGeometryWrapper
from geoss.integration.trellis_ss_hook import GeoSSTrellisSSWrapper, ss_grid_to_tokens, tokens_to_ss_grid
from geoss.integration.trellis_residency import configure_trellis_training_residency
from geoss.integration.trellis_hub import configure_trellis_hub
from geoss.losses.prior_preservation_loss import prior_preservation_loss
from geoss.losses.velocity_loss import velocity_regularization_loss
from geoss.models.sparse_ray_geoss_adapter import SparseRayGeoSSAdapter
from geoss.models.ss_velocity_adapter import SSVelocityAdapter
from geoss.ops.flow_matching import flow_matching_pair
from geoss.utils.adaptive_batch import AdaptiveBatchController, adaptive_config_defaults, add_adaptive_batch_args
from geoss.utils.checkpoint import save_checkpoint
from geoss.utils.config import add_common_args, apply_config_mappings, load_config, str2bool
from geoss.utils.run_mode import validate_real_mode
from geoss.utils.training_budget import (
    compute_training_budget,
    defer_nonfatal_early_stop_until_minimum_exposure,
    enforce_minimum_dataset_passes,
    minimum_updates_for_dataset_passes,
)
from geoss.utils.distributed import (
    build_dataloader,
    cleanup_distributed,
    init_distributed,
    maybe_wrap_ddp,
    next_from_loader,
    sync_should_stop,
    unwrap_model,
)
from geoss.utils.early_stopping import (
    EarlyStopper,
    apply_early_stop_action,
    distributed_early_stop_update,
    handoff_contract,
    quarantine_legacy_best_checkpoint,
)
from geoss.utils.elastic_engine import cuda_memory_watermark, slice_batch_to_size, train_step_with_oom_retry


OPTIMIZATION_CONTRACT_VERSION = "stage2_gate_consistent_flow_v2"
REQUIRED_STAGE1_OPTIMIZATION_CONTRACT = "stage1_stationary_control_v2"


class MockSSFlowModel(nn.Module):
    resolution = 8
    in_channels = 8
    out_channels = 8

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.tanh(x) * 0.1


def run_dry_run(cfg: dict, device: str) -> dict:
    B = cfg.get("batch_size", 2)
    C = cfg.get("latent_dim", 8)
    R = cfg.get("resolution", 8)
    M = cfg.get("num_anchors", 2048)
    geo_dim = cfg.get("geo_dim", 256)
    x = torch.randn(B, C, R, R, R, device=device)
    t = torch.rand(B, device=device) * 1000
    synthetic_condition = torch.randn(B, 16, geo_dim, device=device)
    geo_context = {
        "geo_tokens": torch.randn(B, M, geo_dim, device=device),
        "geo_confidence": torch.rand(B, M, 1, device=device),
        # The production adapter uses spatially local attention.  A dry run
        # must exercise that same interface instead of omitting anchor geometry
        # and failing before the adapter forward.
        "anchor_xyz": torch.rand(B, M, 3, device=device) * 2.0 - 1.0,
    }
    adapter = SSVelocityAdapter(latent_dim=C, geo_dim=geo_dim).to(device)
    wrapper = GeoSSTrellisSSWrapper(MockSSFlowModel().to(device), adapter)
    v_geo = wrapper(x, t, synthetic_condition, geoss_context=geo_context)
    enabled_debug = dict(wrapper.last_debug)
    v_base = wrapper(x, t, synthetic_condition, geoss_context=geo_context, use_geoss_adapter=False)
    summary = {
        "mode": "dry_run",
        "mock_trellis_base_velocity": True,
        "not_for_paper_metrics": True,
        "v_geo": list(v_geo.shape),
        "v_base": list(v_base.shape),
        "disabled_exact_base": bool(torch.allclose(v_base, torch.tanh(x) * 0.1)),
        "token_confidence": list(enabled_debug.get("token_confidence", torch.empty(0)).shape),
    }
    return summary


def run_training(cfg: dict, args: argparse.Namespace) -> dict:
    run_modes = validate_real_mode(cfg=cfg, args=args, mode="real_train", required=("vggt", "trellis", "dataset"))
    ctx = init_distributed(args)
    device = ctx.device
    execution_mode = str(args.execution_mode)
    is_probe = execution_mode in {"probe", "smoke"}
    if is_probe:
        if float(args.minimum_dataset_passes) != 0.0:
            raise ValueError(
                f"execution_mode={execution_mode} requires --minimum_dataset_passes 0; "
                "probe exposure is not training evidence."
            )
        if args.resume:
            raise ValueError(f"execution_mode={execution_mode} cannot resume or mutate a training checkpoint.")
        # Enforce artifact isolation in the trainer as well as the launcher so
        # a hand-written direct probe command cannot accidentally promote data.
        args.early_stop = False
        args.save_best = False
        args.fault_tolerant_save_every = 0
    elif float(args.minimum_dataset_passes) < 1.0:
        raise ValueError("execution_mode=train requires minimum_dataset_passes >= 1.")
    batch_controller = AdaptiveBatchController.from_args(args)
    args.batch_size = batch_controller.batch_size
    args.grad_accum_steps = max(1, int(args.grad_accum_steps))
    if args.steps is None:
        raise ValueError("real_train requires an explicit --steps value or a steps entry in the config.")
    loader, sampler = _build_meshfleet_loader(args, ctx)
    if loader is None:
        raise FileNotFoundError("real_train requires a non-empty MeshFleet dataset loader.")
    training_budget = compute_training_budget(
        dataset_objects=len(loader.dataset),
        world_size=ctx.world_size,
        microbatch_per_rank=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        planned_optimizer_updates=int(args.steps),
        drop_last=bool(getattr(loader, "drop_last", False)),
    )
    required_updates = minimum_updates_for_dataset_passes(
        training_budget, args.minimum_dataset_passes
    )
    configured_steps = int(args.steps)
    if (
        execution_mode == "train"
        and args.auto_expand_training_budget
        and configured_steps < required_updates
    ):
        args.steps = required_updates
        training_budget = compute_training_budget(
            dataset_objects=len(loader.dataset),
            world_size=ctx.world_size,
            microbatch_per_rank=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            planned_optimizer_updates=int(args.steps),
            drop_last=bool(getattr(loader, "drop_last", False)),
        )
    enforce_minimum_dataset_passes(
        training_budget,
        minimum_dataset_passes=args.minimum_dataset_passes,
        stage="SS",
    )
    support_provenance = {
        "ss_value_source": "cached_trellis_ss_teacher",
        "downstream_slat_training_connection": "absent",
        "cross_stage_gradient": "absent_discrete_support_boundary",
    }
    if ctx.is_main:
        preflight_dir = Path(args.output_dir)
        preflight_dir.mkdir(parents=True, exist_ok=True)
        (preflight_dir / "training_preflight.json").write_text(
            json.dumps(
                {
                    "stage": "SS",
                    "execution_mode": execution_mode,
                    "is_probe": is_probe,
                    "promotable_checkpoint": False if is_probe else True,
                    "training_budget": training_budget.as_dict(),
                    "minimum_dataset_passes": args.minimum_dataset_passes,
                    "configured_steps": configured_steps,
                    "resolved_steps": int(args.steps),
                    "budget_auto_expanded": int(args.steps) != configured_steps,
                    "support_provenance": support_provenance,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    trellis_hub = configure_trellis_hub(args)
    base, trellis_pipeline = _load_trellis_or_mock(args, cfg, allow_mock=False)
    if trellis_pipeline is not None:
        trellis_residency = configure_trellis_training_residency(
            trellis_pipeline,
            required_models=("sparse_structure_flow_model", "image_cond_model"),
            device=device,
        )
        base = trellis_pipeline.models["sparse_structure_flow_model"]
    else:
        base = base.to(device)
        trellis_residency = {
            "policy": "direct_sparse_structure_flow_model",
            "required_models": ["sparse_structure_flow_model"],
            "removed_models": [],
            "device": str(device),
        }
    spconv_status = _force_spconv_algo(base, args.spconv_algo)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    latent_dim = getattr(base, "out_channels", getattr(base, "in_channels", cfg.get("latent_dim", 8)))
    resolution = getattr(base, "resolution", cfg.get("resolution", 8))
    geo_dim = cfg.get("geo_dim", 256)
    num_anchors = cfg.get("num_anchors", 4096)
    adapter = SSVelocityAdapter(
        latent_dim=latent_dim,
        geo_dim=geo_dim,
        attention_chunk_size=args.attention_chunk_size,
        activation_checkpointing=args.activation_checkpointing,
    ).to(device)
    _repair_zero_terminal_delta_head(adapter)
    start_step = 0
    resume_state = None
    reset_legacy_optimization_state = False
    if args.resume and Path(args.resume).exists():
        resume_state = torch.load(args.resume, map_location="cpu")
        adapter.load_state_dict(resume_state.get("velocity_adapter", resume_state), strict=True)
        _repair_zero_terminal_delta_head(adapter)
        start_step = int(resume_state.get("step", 0))
        reset_legacy_optimization_state = (
            resume_state.get("optimization_contract_version")
            != OPTIMIZATION_CONTRACT_VERSION
        )
    _assert_terminal_delta_head_is_trainable(adapter)
    adapter_model = maybe_wrap_ddp(adapter, ctx, find_unused_parameters=False)
    opt = torch.optim.AdamW(adapter_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = _build_lr_scheduler(opt, total_updates=int(args.steps), warmup_updates=args.warmup_updates, min_lr_ratio=args.min_lr_ratio)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = _make_grad_scaler(enabled=args.amp and args.amp_dtype == "fp16" and device.type == "cuda")
    if resume_state is not None and not reset_legacy_optimization_state and "optimizer" in resume_state:
        opt.load_state_dict(resume_state["optimizer"])
    if resume_state is not None and not reset_legacy_optimization_state and "scheduler" in resume_state:
        scheduler.load_state_dict(resume_state["scheduler"])
    iterator = iter(loader) if loader is not None else None
    data_epoch = 0
    geoss_model = None
    geoss_handoff = None
    vggt = None
    if iterator is not None:
        geoss_cfg = _geoss_model_cfg_from_velocity_cfg(cfg)
        geoss_model = SparseRayGeoSSAdapter(**geoss_cfg).to(device).eval()
        if not args.geoss_checkpoint:
            raise FileNotFoundError("Stage 2 requires --geoss_checkpoint selected from Stage 1.")
        geoss_checkpoint = Path(args.geoss_checkpoint)
        if not geoss_checkpoint.is_file():
            raise FileNotFoundError(f"Stage 1 GeoSS checkpoint does not exist: {geoss_checkpoint}")
        state = torch.load(geoss_checkpoint, map_location="cpu")
        if state.get("optimization_contract_version") != REQUIRED_STAGE1_OPTIMIZATION_CONTRACT:
            raise RuntimeError(
                "Stage 2 refuses a Stage-1 checkpoint from the collapsed optimization regime: "
                f"expected contract {REQUIRED_STAGE1_OPTIMIZATION_CONTRACT!r}, got "
                f"{state.get('optimization_contract_version')!r}. Resume Stage 1 once under the "
                "repaired trainer before Stage-2 handoff."
            )
        geoss_state = state.get("model", state)
        incompatible = geoss_model.load_state_dict(geoss_state, strict=True)
        geoss_handoff = _checkpoint_handoff_report(
            geoss_checkpoint, geoss_state, incompatible
        )
        for p in geoss_model.parameters():
            p.requires_grad_(False)
        vggt = VGGTGeometryWrapper(
            vggt_root=args.vggt_root,
            checkpoint=args.vggt_checkpoint,
            pretrained_name=args.vggt_pretrained,
            mock=False,
            cache_features=False,
        ).to(device).eval()
        for p in vggt.parameters():
            p.requires_grad_(False)
    parameter_inventory = {
        "trellis_sparse_structure_flow": _module_parameter_report(base),
        "stage1_geoss": _module_parameter_report(geoss_model),
        "vggt": _module_parameter_report(vggt),
        "stage2_velocity_adapter": _module_parameter_report(unwrap_model(adapter_model)),
    }
    out_dir = Path(args.output_dir)
    if ctx.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_sparse_ray_ss_velocity.jsonl"
    last = {}
    sigma_min = args.sigma_min
    early_stopper = EarlyStopper.from_args(
        args,
        default_metric="normalized_effective_residual",
        min_control_updates=max(
            int(args.early_stop_min_steps or 0),
            (
                start_step + int(scheduler.resolved_warmup_updates)
                if reset_legacy_optimization_state
                else int(scheduler.resolved_warmup_updates)
            ),
        ),
    )
    if resume_state is not None and not reset_legacy_optimization_state:
        early_stopper.load_state_dict(resume_state.get("early_stopper"))
    if resume_state is not None and ctx.is_main:
        quarantine_legacy_best_checkpoint(
            out_dir / "ss_velocity_adapter_best.pt",
            resume_state,
        )
    end_step = int(args.steps) if args.steps_are_total else start_step + int(args.steps)
    update_contract = {
        "configured_steps": configured_steps,
        "resolved_steps": int(args.steps),
        "budget_auto_expanded": int(args.steps) != configured_steps,
        "steps_are_total": bool(args.steps_are_total),
        "resume_start_step": start_step,
        "target_step": end_step,
        "remaining_update_attempts_at_start": max(0, end_step - start_step),
        "execution_mode": execution_mode,
        "is_probe": is_probe,
        "promotable_checkpoint": not is_probe,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_global_batch_size": args.batch_size * ctx.world_size * args.grad_accum_steps,
        "learning_rate_scaling": "none; AdamW uses mean loss over accumulation and the launcher preserves target effective batch",
        "optimization_contract_version": OPTIMIZATION_CONTRACT_VERSION,
        "legacy_weights_warm_started_with_control_reset": reset_legacy_optimization_state,
    }
    if start_step >= end_step:
        return {
            "step": start_step,
            "target_step": end_step,
            "mode": "already_complete",
            "rank": ctx.rank,
            "world_size": ctx.world_size,
            "training_budget": training_budget.as_dict(),
            "update_contract": update_contract,
            "support_provenance": support_provenance,
        }
    step = start_step
    seen_object_ids: set[str] = set()
    actual_sample_presentations = 0
    while step < end_step:
        step_wall_start = time.perf_counter()
        data_fetch_start = step_wall_start
        step += 1
        data_batches = []
        for _ in range(args.grad_accum_steps):
            if iterator is None:
                data_batches.append(None)
                continue
            raw_batch, iterator, data_epoch = next_from_loader(iterator, loader, sampler, data_epoch)
            object_ids = raw_batch.get("object_id", []) if isinstance(raw_batch, dict) else []
            if isinstance(object_ids, str):
                object_ids = [object_ids]
            seen_object_ids.update(str(uid) for uid in object_ids)
            data_batches.append(_move_batch(raw_batch, device))
        data_batch = data_batches[-1] if data_batches else None
        data_fetch_seconds = time.perf_counter() - data_fetch_start

        def rebuild_after_adjustment(adjustment):
            nonlocal data_batch, data_batches, loader, sampler, iterator
            args.batch_size = adjustment.new_batch_size
            data_batches = [
                slice_batch_to_size(batch, args.batch_size) if batch is not None else None
                for batch in data_batches
            ]
            data_batch = data_batches[-1] if data_batches else None
            loader, sampler = _build_meshfleet_loader(args, ctx)
            iterator = iter(loader) if loader is not None else None

        def log_oom(record):
            if ctx.is_main:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, **record}) + "\n")

        def step_fn():
            memory_start = cuda_memory_watermark(device, reset_peak=True)
            opt.zero_grad(set_to_none=True)
            accumulated = []
            for micro_step, micro_batch in enumerate(data_batches):
                if micro_batch is not None and "ss_latent_grid" in micro_batch:
                    x0 = micro_batch["ss_latent_grid"].to(device=device, dtype=torch.float32)
                    B = x0.shape[0]
                else:
                    object_ids = micro_batch.get("object_id") if isinstance(micro_batch, dict) else None
                    keys = sorted(micro_batch.keys()) if isinstance(micro_batch, dict) else []
                    raise KeyError(
                        "real_train requires ss_latent_grid from MeshFleet_TRELLIS ss_latents. "
                        f"batch_keys={keys}, object_id={object_ids}"
                    )
                noise = torch.randn_like(x0)
                t = _sample_flow_timesteps(
                    B,
                    device,
                    mode=args.timestep_sampling,
                    mean=args.timestep_logit_mean,
                    std=args.timestep_logit_std,
                )
                x_t, target_v, flow_matching_backend = flow_matching_pair(
                    x0,
                    noise,
                    t,
                    sigma_min,
                    backend=args.fused_flow_matching,
                )
                voxel_valid_mask = _exact_zero_voxel_mask(x0, args.voxel_prune_epsilon) if args.adaptive_voxel_pruning else None
                cond = _real_condition_or_fail(micro_batch, device, cfg, trellis_pipeline)
                if geoss_model is not None and vggt is not None:
                    geoss_context = _compute_geoss_context(micro_batch, geoss_model, vggt)
                else:
                    raise RuntimeError("real_train requires real dataset GeoSS context; use --dry_run true for synthetic context.")
                _assert_stage2_batch_contract(micro_batch, x0, device)
                t_model = t * 1000.0
                with torch.inference_mode(), torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp and device.type == "cuda"):
                    v_base = base(x_t, t_model, cond)
                allocator_reclamation = _release_frozen_forward_cache(
                    device,
                    enabled=args.release_frozen_cache,
                    reserved_fraction=args.release_cache_reserved_fraction,
                    inactive_gib=args.release_cache_inactive_gib,
                )
                ss_tokens = ss_grid_to_tokens(x_t)
                v_base_tokens = ss_grid_to_tokens(v_base).detach()
                target_residual_tokens = ss_grid_to_tokens(target_v - v_base).detach()
                _assert_stage2_geoss_context(geoss_context, ss_tokens)
                sync_context = (
                    adapter_model.no_sync()
                    if ctx.distributed and hasattr(adapter_model, "no_sync") and micro_step < len(data_batches) - 1
                    else contextlib.nullcontext()
                )
                with sync_context:
                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp and device.type == "cuda"):
                        vel = adapter_model(
                            ss_latent_tokens=ss_tokens,
                            geo_tokens=geoss_context["geo_tokens"],
                            geo_confidence=geoss_context["geo_confidence"],
                            timestep=t_model,
                            v_base=v_base_tokens,
                            voxel_xyz=geoss_context.get("ss_voxel_xyz", _ss_grid_xyz(x_t, ss_tokens.dtype)),
                            anchor_xyz=geoss_context["anchor_xyz"],
                            anchor_metadata=geoss_context.get("anchor_metadata"),
                            voxel_valid_mask=voxel_valid_mask,
                        )
                    _assert_velocity_adapter_output(vel, ss_tokens, v_base_tokens)
                    v_geo_tokens = vel["v_geo"]
                    effective_delta_tokens = v_geo_tokens - v_base_tokens
                    if not effective_delta_tokens.requires_grad:
                        raise RuntimeError(
                            "Stage 2 graph invariant failed: effective residual is detached. "
                            "The loss must consume SSVelocityAdapter's DDP forward output."
                        )
                    delta_tokens = vel["delta_v_geo"]
                    raw_delta_tokens = vel["debug"]["delta_raw"]
                    _assert_residual_training_contract(raw_delta_tokens, effective_delta_tokens, target_residual_tokens)
                    token_mask = voxel_valid_mask[..., None] if voxel_valid_mask is not None else None
                    frozen_base_mse = _masked_mse(
                        torch.zeros_like(effective_delta_tokens, dtype=torch.float32),
                        target_residual_tokens.float(),
                        token_mask,
                    )
                    residual_terms = _stage2_residual_objective(
                        raw_delta_tokens.float(),
                        effective_delta_tokens.float(),
                        target_residual_tokens.float(),
                        alpha_t=vel["alpha_t"],
                        token_confidence=vel["token_confidence"],
                        token_mask=token_mask,
                        frozen_base_mse=frozen_base_mse,
                        auxiliary_weight=args.raw_residual_weight,
                        normalize=args.normalize_residual_loss,
                    )
                    effective_mse = residual_terms["effective_mse"]
                    raw_mse = residual_terms["unclipped_effective_mse"]
                    mse = residual_terms["optimization_residual"]
                    vel_reg = velocity_regularization_loss(delta_tokens, t)
                    prior = prior_preservation_loss(v_geo_tokens, v_base_tokens, vel["token_confidence"].detach())
                    loss = mse + args.velocity_reg_weight * vel_reg + args.prior_weight * prior
                    scaler.scale(loss / len(data_batches)).backward()
                accumulated.append((loss, mse, effective_mse, raw_mse, frozen_base_mse, vel_reg, prior))
            scaler.unscale_(opt)
            grad_norms = _assert_adapter_gradients(unwrap_model(adapter_model), step)
            measure_update = is_probe or step == 1 or step % 100 == 0
            parameters_before = (
                [parameter.detach().clone() for parameter in unwrap_model(adapter_model).parameters()]
                if measure_update
                else None
            )
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            grad_norms["update_to_weight_ratio"] = _update_to_weight_ratio(
                unwrap_model(adapter_model), parameters_before
            )
            loss, mse, effective_mse, raw_mse, frozen_base_mse, vel_reg, prior = [
                torch.stack([values[index].detach() for values in accumulated]).mean()
                for index in range(7)
            ]
            debug = vel["debug"]
            effective_delta_grid = tokens_to_ss_grid(effective_delta_tokens, tuple(x_t.shape[-3:]))
            target_residual_grid = tokens_to_ss_grid(target_residual_tokens, tuple(x_t.shape[-3:]))
            tensor_shapes = {
                "ss_latent_grid": list(x0.shape),
                "noised_ss_grid": list(x_t.shape),
                "ss_tokens": list(ss_tokens.shape),
                "trellis_condition": list(cond.shape),
                "geo_tokens": list(geoss_context["geo_tokens"].shape),
                "geo_confidence": list(geoss_context["geo_confidence"].shape),
                "anchor_xyz": list(geoss_context["anchor_xyz"].shape),
                "base_velocity_tokens": list(v_base_tokens.shape),
                "target_residual_tokens": list(target_residual_tokens.shape),
                "corrected_velocity_tokens": list(v_geo_tokens.shape),
            }
            return loss, mse, effective_mse, raw_mse, frozen_base_mse, target_residual_grid, debug, effective_delta_grid, vel_reg, prior, grad_norms, tensor_shapes, memory_start, cuda_memory_watermark(device), flow_matching_backend, allocator_reclamation, t.detach().mean()

        compute_start = time.perf_counter()
        retry = train_step_with_oom_retry(
            step_fn,
            model=adapter_model,
            optimizer=opt,
            scaler=scaler,
            sampler=sampler,
            device=device,
            batch_controller=batch_controller,
            rebuild_after_adjustment=rebuild_after_adjustment,
            max_retries=getattr(args, "adaptive_oom_retries", 8),
            log_oom=log_oom,
        )
        batch_adjustment = retry.adjustment
        loss, mse, effective_mse, raw_mse, frozen_base_mse, target_residual, debug, effective_delta, vel_reg, prior, grad_norms, tensor_shapes, memory_start, memory_end, flow_matching_backend, allocator_reclamation, timestep_mean = retry.value
        if is_probe and device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_seconds = time.perf_counter() - compute_start
        step_latency_seconds = time.perf_counter() - step_wall_start
        local_presentations = sum(
            int(batch["ss_latent_grid"].shape[0])
            for batch in data_batches
            if isinstance(batch, dict) and isinstance(batch.get("ss_latent_grid"), torch.Tensor)
        )
        global_presentations = local_presentations * ctx.world_size
        actual_sample_presentations += global_presentations
        if batch_adjustment.changed:
            args.batch_size = batch_adjustment.new_batch_size
            loader, sampler = _build_meshfleet_loader(args, ctx)
            iterator = iter(loader) if loader is not None else None
        last = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "cfm_mse": float(effective_mse.detach().cpu()),
            "loss_residual_optimization": float(mse.detach().cpu()),
            "loss_effective_residual": float(effective_mse.detach().cpu()),
            "loss_raw_residual": float(raw_mse.detach().cpu()),
            "loss_unclipped_effective_residual": float(raw_mse.detach().cpu()),
            "loss_frozen_base_residual": float(frozen_base_mse.detach().cpu()),
            "normalized_effective_residual": float((effective_mse / frozen_base_mse.clamp_min(1.0e-6)).detach().cpu()),
            "normalized_unclipped_effective_residual": float((raw_mse / frozen_base_mse.clamp_min(1.0e-6)).detach().cpu()),
            "causal_residual_gain": float((frozen_base_mse - effective_mse).detach().cpu()),
            "residual_target_norm": float(target_residual.norm(dim=1).mean().detach().cpu()),
            "residual_base_ratio": float((effective_delta.norm(dim=1).mean() / debug["velocity_base_norm"].clamp_min(1e-6)).detach().cpu()),
            "velocity_regularization": float(vel_reg.detach().cpu()),
            "prior_preservation": float(prior.detach().cpu()),
            "base_velocity_contract": "direct frozen TRELLIS sparse_structure_flow_model output; adapter disabled path is not invoked",
            "velocity_norm": float(debug["velocity_norm"].detach().cpu()),
            "delta_norm": float(effective_delta.norm(dim=1).mean().detach().cpu()),
            "clipping_ratio": float(debug["clipping_ratio"].detach().cpu()),
            "confidence_mean": float(debug["confidence_mean"].detach().cpu()),
            "confidence_std": float(debug["confidence_std"].detach().cpu()),
            "confidence_all_zero": bool(debug["confidence_all_zero"].detach().cpu()),
            "confidence_all_one": bool(debug["confidence_all_one"].detach().cpu()),
            "effective_gate_mean": float(debug["effective_gate_mean"].detach().cpu()),
            "voxel_prune_ratio": float(debug.get("voxel_prune_ratio", torch.zeros((), device=device)).detach().cpu()),
            "mode": _training_mode(args, data_batch is not None),
            "execution_mode": execution_mode,
            "is_probe": is_probe,
            "promotable_checkpoint": not is_probe,
            **run_modes,
            "rank": ctx.rank,
            "world_size": ctx.world_size,
            "per_gpu_batch_size": args.batch_size,
            "global_batch_size": args.batch_size * ctx.world_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_global_batch_size": args.batch_size * ctx.world_size * args.grad_accum_steps,
            "actual_global_sample_presentations_this_update": global_presentations,
            "actual_sample_presentations_cumulative": actual_sample_presentations,
            "actual_dataset_passes_cumulative": actual_sample_presentations / len(loader.dataset),
            "learning_rate": float(opt.param_groups[0]["lr"]),
            "step_latency_seconds": step_latency_seconds,
            "compute_seconds": compute_seconds,
            "dataloader_seconds": data_fetch_seconds,
            "dataloader_stall_fraction": data_fetch_seconds / max(step_latency_seconds, 1.0e-12),
            "samples_per_second": global_presentations / max(step_latency_seconds, 1.0e-12),
            "gpu_utilization_percent": _cuda_utilization_percent(device),
            "communication_fraction": None,
            "communication_fraction_status": "requires training-server profiler trace; not inferred from wall time",
            "timestep_sampling": args.timestep_sampling,
            "timestep_mean": float(timestep_mean.cpu()),
            "flow_matching_backend": flow_matching_backend,
            "allocator_reclamation": allocator_reclamation,
            "adapter_grad_norms": grad_norms,
            "tensor_shapes": tensor_shapes,
            "stage1_handoff": geoss_handoff,
            "parameter_inventory": parameter_inventory,
            "spconv": spconv_status,
            "trellis_residency": trellis_residency,
            "trellis_hub": trellis_hub,
            "memory_start": memory_start,
            "memory": memory_end,
            "adaptive_batch": {**batch_controller.state_dict(), "last_adjustment": batch_adjustment.as_dict()},
            "training_budget": training_budget.as_dict(),
            "update_contract": update_contract,
            "support_provenance": support_provenance,
        }
        early_status = distributed_early_stop_update(
            early_stopper,
            last,
            rank=ctx.rank,
        )
        last["minimum_exposure_gate"] = defer_nonfatal_early_stop_until_minimum_exposure(
            early_status,
            budget=training_budget,
            step=step,
            minimum_dataset_passes=args.minimum_dataset_passes,
        )
        if early_status.lr_multiplier is not None:
            early_status.lr_multiplier = _bound_scheduler_intervention(
                scheduler,
                float(early_status.lr_multiplier),
            )
        last["early_stop_action"] = apply_early_stop_action(opt, early_status)
        if last["early_stop_action"].get("applied"):
            multiplier = float(last["early_stop_action"]["multiplier"])
            scheduler.base_lrs = [float(value) * multiplier for value in scheduler.base_lrs]
            scheduler.control_multiplier *= multiplier
            last["early_stop_action"]["scheduler_base_lrs"] = list(scheduler.base_lrs)
        last["early_stop"] = early_status.as_dict()
        if ctx.is_main:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(last) + "\n")
        if ctx.is_main and not is_probe and args.save_best and early_status.is_best:
            _save_velocity_checkpoint(out_dir / "ss_velocity_adapter_best.pt", unwrap_model(adapter_model), opt, scheduler, step, cfg, early_stopper, early_status, training_budget=training_budget.as_dict(), support_provenance=support_provenance, execution_mode=execution_mode, stage1_handoff=geoss_handoff)
        if ctx.is_main and not is_probe and early_status.is_candidate:
            _save_velocity_checkpoint(out_dir / "ss_velocity_adapter_candidate.pt", unwrap_model(adapter_model), opt, scheduler, step, cfg, early_stopper, early_status, training_budget=training_budget.as_dict(), support_provenance=support_provenance, execution_mode=execution_mode, stage1_handoff=geoss_handoff)
        should_fault_save = args.fault_tolerant_save_every > 0 and step % args.fault_tolerant_save_every == 0
        if ctx.is_main and not is_probe and (should_fault_save or step % args.save_every == 0 or step == end_step):
            _save_velocity_checkpoint(out_dir / "ss_velocity_adapter_last.pt", unwrap_model(adapter_model), opt, scheduler, step, cfg, early_stopper, early_status, training_budget=training_budget.as_dict(), support_provenance=support_provenance, execution_mode=execution_mode, stage1_handoff=geoss_handoff)
        if sync_should_stop(early_status.should_stop, device):
            if ctx.is_main and not is_probe:
                _save_velocity_checkpoint(out_dir / "ss_velocity_adapter_last.pt", unwrap_model(adapter_model), opt, scheduler, step, cfg, early_stopper, early_status, training_budget=training_budget.as_dict(), support_provenance=support_provenance, execution_mode=execution_mode, stage1_handoff=geoss_handoff)
            break
    coverage = _distributed_unique_coverage(seen_object_ids, len(loader.dataset))
    last["observed_dataset_coverage"] = coverage
    last["per_rank_hardware"] = _distributed_rank_telemetry(
        {
            "rank": ctx.rank,
            "memory": last.get("memory"),
            "step_latency_seconds": last.get("step_latency_seconds"),
            "samples_per_second": last.get("samples_per_second"),
            "dataloader_stall_fraction": last.get("dataloader_stall_fraction"),
            "gpu_utilization_percent": last.get("gpu_utilization_percent"),
            "communication_fraction": last.get("communication_fraction"),
        }
    )
    if ctx.is_main and last:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "final_coverage", **coverage}) + "\n")
    return last


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--steps", type=int, default=None, help="Required for real training; dry runs do not consume an update budget.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_updates", type=int, default=0, help="0 derives a 3% warmup from the resolved real horizon.")
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--sigma_min", type=float, default=1e-5)
    parser.add_argument("--velocity_reg_weight", type=float, default=1e-3)
    parser.add_argument("--prior_weight", type=float, default=1e-2)
    parser.add_argument(
        "--raw_residual_weight",
        type=float,
        default=0.25,
        help="Weight of the gate-consistent, unclipped residual auxiliary.",
    )
    parser.add_argument("--normalize_residual_loss", type=str2bool, default=True)
    parser.add_argument("--timestep_sampling", choices=("logit_normal", "uniform"), default="logit_normal")
    parser.add_argument("--timestep_logit_mean", type=float, default=0.0)
    parser.add_argument("--timestep_logit_std", type=float, default=1.0)
    parser.add_argument("--fused_flow_matching", choices=("auto", "torch", "triton"), default="auto")
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--trellis_root", type=str, default=None)
    parser.add_argument("--trellis_model_path", type=str, default=None)
    parser.add_argument("--torch_hub_dir", type=str, default=None)
    parser.add_argument("--dinov2_repo", type=str, default=None)
    parser.add_argument("--meshfleet_root", type=str, default=None)
    parser.add_argument("--meshfleet_split", type=str, default="train")
    parser.add_argument("--train_manifest", type=str, default=None, help="UID manifest generated by inspect_meshfleet_dataset.py.")
    parser.add_argument("--meshfleet_category", type=str, default=None)
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--meshfleet_occ_resolution", type=int, default=64)
    parser.add_argument("--meshfleet_prefer_cond_render", action="store_true")
    parser.add_argument("--geoss_checkpoint", type=str, default=None)
    parser.add_argument("--vggt_root", type=str, default=None)
    parser.add_argument("--vggt_checkpoint", type=str, default=None)
    parser.add_argument("--vggt_pretrained", type=str, default=None)
    parser.add_argument("--real_train", action="store_true")
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--activation_checkpointing", type=str2bool, default=True)
    parser.add_argument("--attention_chunk_size", type=int, default=8192)
    # Avoid the backend's unrestricted auto-selection; MaskImplicitGemm keeps
    # sparse-convolution index/workspace storage bounded without changing weights.
    parser.add_argument("--spconv_algo", choices=("native", "mask_implicit_gemm"), default="mask_implicit_gemm")
    parser.add_argument("--adaptive_voxel_pruning", type=str2bool, default=True)
    parser.add_argument("--voxel_prune_epsilon", type=float, default=0.0)
    parser.add_argument("--release_frozen_cache", type=str2bool, default=True)
    parser.add_argument("--release_cache_reserved_fraction", type=float, default=0.90)
    parser.add_argument("--release_cache_inactive_gib", type=float, default=8.0)
    add_adaptive_batch_args(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    _apply_config_defaults(args, cfg, parser)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _interrupt(_signum, _frame):
        raise KeyboardInterrupt("received SIGTERM")

    signal.signal(signal.SIGTERM, _interrupt)
    summary = None
    rank = 0
    try:
        summary = run_training(cfg, args) if not args.dry_run else run_dry_run(cfg, args.device)
        rank = getattr(args, "rank", 0)
        if rank == 0:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            if args.dry_run:
                summary_name = "train_sparse_ray_ss_velocity_dry_run.json"
            elif args.execution_mode in {"probe", "smoke"}:
                summary_name = "probe_summary.json"
            else:
                summary_name = "training_summary.json"
            (output_dir / summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary, indent=2))
    except KeyboardInterrupt:
        rank = getattr(args, "rank", 0)
        if rank == 0:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "interrupted.json").write_text(
                json.dumps(
                    {
                        "status": "interrupted",
                        "execution_mode": args.execution_mode,
                        "partial_checkpoint_promoted": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        raise
    finally:
        cleanup_distributed()
        signal.signal(signal.SIGTERM, previous_sigterm)
    if not args.dry_run and rank == 0 and summary is not None:
        _maybe_launch_stage2(summary, cfg)


def _load_trellis_or_mock(args: argparse.Namespace, cfg: dict, allow_mock: bool = True) -> tuple[nn.Module, object | None]:
    if args.trellis_model_path:
        if args.trellis_root:
            sys.path.insert(0, args.trellis_root)
        return _load_trellis_ss_flow(args.trellis_model_path)
    if not allow_mock:
        raise FileNotFoundError("real_train requires --trellis_model_path; mock TRELLIS flow is only allowed in --dry_run.")
    mock = MockSSFlowModel()
    mock.resolution = cfg.get("resolution", mock.resolution)
    mock.in_channels = cfg.get("latent_dim", mock.in_channels)
    mock.out_channels = cfg.get("latent_dim", mock.out_channels)
    return mock, None


def _load_trellis_ss_flow(path: str) -> tuple[nn.Module, object | None]:
    """Load either a TRELLIS model checkpoint path or a full TRELLIS pipeline repo."""
    from trellis import models

    try:
        return models.from_pretrained(path), None
    except Exception as model_exc:
        try:
            from trellis.pipelines import TrellisImageTo3DPipeline

            pipeline = TrellisImageTo3DPipeline.from_pretrained(path)
            return pipeline.models["sparse_structure_flow_model"], pipeline
        except Exception as pipe_exc:
            raise RuntimeError(
                f"Could not load TRELLIS sparse structure flow from {path}. "
                "Pass either a concrete ckpt path under microsoft/TRELLIS-image-large/ckpts "
                "or the full pipeline repo/path microsoft/TRELLIS-image-large."
            ) from pipe_exc


def _force_spconv_algo(model: nn.Module, requested: str) -> dict[str, object]:
    """Force a bounded-workspace spconv algorithm before the first SS forward."""
    try:
        from spconv.core import ConvAlgo
    except ImportError:
        return {"requested": requested, "applied": 0, "available": False}
    candidates = {
        "native": ("Native",),
        "mask_implicit_gemm": ("MaskImplicitGemm", "MaskSplitImplicitGemm"),
    }[requested]
    algo = next((getattr(ConvAlgo, name) for name in candidates if hasattr(ConvAlgo, name)), None)
    if algo is None:
        return {"requested": requested, "applied": 0, "available": False}
    applied = 0
    for module in model.modules():
        current = getattr(module, "algo", None)
        if current is not None and (module.__class__.__module__.startswith("spconv") or isinstance(current, type(algo))):
            module.algo = algo
            applied += 1
    # TRELLIS reads this during lazy sparse-module construction in supported builds.
    os.environ["SPCONV_ALGO"] = requested
    return {"requested": requested, "applied": applied, "available": True}


def _make_grad_scaler(*, enabled: bool):
    """Support both current and older PyTorch AMP namespaces without changing checkpoints."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _sample_flow_timesteps(
    batch_size: int,
    device: torch.device,
    *,
    mode: str,
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    """Match the frozen TRELLIS trainer's timestep distribution by default."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if mode == "uniform":
        return torch.rand(batch_size, device=device)
    if mode != "logit_normal":
        raise ValueError(f"Unsupported timestep sampling mode {mode!r}.")
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError(f"timestep logit-normal std must be positive and finite, got {std}.")
    logits = torch.randn(batch_size, device=device) * float(std) + float(mean)
    return torch.sigmoid(logits)


def _stage2_residual_objective(
    raw_delta: torch.Tensor,
    effective_delta: torch.Tensor,
    target_residual: torch.Tensor,
    *,
    alpha_t: torch.Tensor,
    token_confidence: torch.Tensor,
    token_mask: torch.Tensor | None,
    frozen_base_mse: torch.Tensor,
    auxiliary_weight: float,
    normalize: bool,
) -> dict[str, torch.Tensor]:
    """Build a gate-consistent residual objective with a stationary baseline.

    The old auxiliary compared the *ungated* head directly with the desired
    effective correction.  For a gate ``g < 1`` it therefore asked the same
    parameter to approach both ``target`` and ``target / g``.  The auxiliary
    below bypasses only the smooth trust projection while retaining the exact
    inference gate, so both terms have one optimum and clipped heads keep a
    useful analytic gradient.
    """

    if auxiliary_weight < 0.0:
        raise ValueError(f"auxiliary_weight must be non-negative, got {auxiliary_weight}")
    effective_mse = _masked_mse(effective_delta, target_residual, token_mask)
    detached_gate = (alpha_t.float() * token_confidence.float()).detach()
    if detached_gate.shape != (*raw_delta.shape[:2], 1):
        raise ValueError(
            f"effective gate must be [B,L,1], got {tuple(detached_gate.shape)} "
            f"for residual {tuple(raw_delta.shape)}"
        )
    unclipped_effective = detached_gate * raw_delta
    unclipped_effective_mse = _masked_mse(unclipped_effective, target_residual, token_mask)
    denominator = frozen_base_mse.detach().float().clamp_min(1.0e-6) if normalize else raw_delta.new_ones(())
    # The derivative of a gated residual carries an unavoidable factor ``g``.
    # Divide the auxiliary by E[g²] (with a conservative floor) to condition
    # the residual-head gradient without changing the auxiliary's minimizer.
    gate_energy = detached_gate.square().mean().clamp_min(1.0e-2)
    optimization_residual = (
        effective_mse / denominator
        + float(auxiliary_weight) * unclipped_effective_mse / (denominator * gate_energy)
    )
    return {
        "effective_mse": effective_mse,
        "unclipped_effective_mse": unclipped_effective_mse,
        "optimization_residual": optimization_residual,
        "detached_gate_mean": detached_gate.mean(),
        "detached_gate_energy": gate_energy,
    }


def _release_frozen_forward_cache(
    device: torch.device,
    *,
    enabled: bool,
    reserved_fraction: float,
    inactive_gib: float,
) -> dict[str, float | bool | str]:
    """Release only allocator cache left by sequential frozen giant models."""

    if not enabled or device.type != "cuda" or not torch.cuda.is_available():
        return {"released": False, "reason": "disabled_or_non_cuda"}
    if not 0.0 < float(reserved_fraction) <= 1.0:
        raise ValueError("release_cache_reserved_fraction must be in (0,1].")
    if float(inactive_gib) < 0.0:
        raise ValueError("release_cache_inactive_gib must be non-negative.")
    gib = 1024.0 ** 3
    allocated = float(torch.cuda.memory_allocated(device))
    reserved = float(torch.cuda.memory_reserved(device))
    total = float(torch.cuda.get_device_properties(device).total_memory)
    inactive = max(0.0, reserved - allocated)
    should_release = reserved / max(total, 1.0) >= float(reserved_fraction) and inactive / gib >= float(inactive_gib)
    if not should_release:
        return {
            "released": False,
            "reason": "below_threshold",
            "reserved_gib_before": reserved / gib,
            "allocated_gib": allocated / gib,
            "inactive_gib_before": inactive / gib,
        }
    torch.cuda.empty_cache()
    reserved_after = float(torch.cuda.memory_reserved(device))
    return {
        "released": True,
        "reason": "frozen_forward_cache_pressure",
        "reserved_gib_before": reserved / gib,
        "reserved_gib_after": reserved_after / gib,
        "allocated_gib": allocated / gib,
        "reclaimed_gib": max(0.0, reserved - reserved_after) / gib,
    }


def _build_lr_scheduler(optimizer, *, total_updates: int, warmup_updates: int, min_lr_ratio: float):
    total_updates = max(1, int(total_updates))
    if int(warmup_updates) <= 0:
        warmup_updates = max(1, min(1000, math.ceil(0.03 * total_updates)))
    warmup_updates = min(int(warmup_updates), max(0, total_updates - 1))
    min_lr_ratio = float(min_lr_ratio)
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0,1], got {min_lr_ratio}")

    def multiplier(update: int) -> float:
        if warmup_updates > 0 and update < warmup_updates:
            return max(1.0 / warmup_updates, (update + 1) / warmup_updates)
        decay_updates = max(1, total_updates - warmup_updates)
        progress = min(1.0, max(0.0, (update - warmup_updates) / decay_updates))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)
    scheduler.resolved_warmup_updates = warmup_updates
    scheduler.resolved_total_updates = total_updates
    scheduler.min_lr_ratio = min_lr_ratio
    scheduler.control_multiplier = 1.0
    return scheduler


def _bound_scheduler_intervention(
    scheduler,
    requested: float,
    *,
    floor: float = 0.1,
    ceiling: float = 2.0,
) -> float:
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError(f"LR intervention must be positive and finite, got {requested}.")
    current = float(getattr(scheduler, "control_multiplier", 1.0))
    target = min(float(ceiling), max(float(floor), current * float(requested)))
    return target / current


def _checkpoint_handoff_report(path: Path, state_dict: dict, incompatible) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    tensors = [value for value in state_dict.values() if isinstance(value, torch.Tensor)]
    return {
        "checkpoint_path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "missing_keys": list(getattr(incompatible, "missing_keys", [])),
        "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
        "tensor_count": len(tensors),
        "parameter_numel": sum(int(value.numel()) for value in tensors),
        "strict_load": True,
        "loaded_module": "SparseRayGeoSSAdapter",
        "frozen_in_stage2": True,
    }


def _distributed_unique_coverage(local_uids: set[str], dataset_objects: int) -> dict:
    gathered: list[list[str]] = [list(local_uids)]
    if dist.is_available() and dist.is_initialized():
        gathered = [[] for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, sorted(local_uids))
    unique = set().union(*(set(items) for items in gathered))
    return {
        "unique_objects_seen": len(unique),
        "dataset_objects": int(dataset_objects),
        "unique_object_fraction": len(unique) / max(1, int(dataset_objects)),
        "distributed_sampler_padding_counted_as_unique": False,
    }


def _distributed_rank_telemetry(local: dict) -> list[dict]:
    if not (dist.is_available() and dist.is_initialized()):
        return [local]
    gathered: list[dict] = [{} for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local)
    return gathered


def _cuda_utilization_percent(device: torch.device) -> float | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    utilization = getattr(torch.cuda, "utilization", None)
    if utilization is None:
        return None
    try:
        return float(utilization(device.index))
    except Exception:
        return None


def _module_parameter_report(module: nn.Module | None) -> dict:
    if module is None:
        return {"present": False, "parameters": 0, "trainable_parameters": 0, "frozen_parameters": 0}
    parameters = list(module.parameters())
    total = sum(int(parameter.numel()) for parameter in parameters)
    trainable = sum(int(parameter.numel()) for parameter in parameters if parameter.requires_grad)
    return {
        "present": True,
        "parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
    }


def _build_meshfleet_loader(args: argparse.Namespace, ctx):
    if not args.meshfleet_root:
        return None, None
    root = Path(args.meshfleet_root)
    if not root.exists():
        return None, None
    dataset = MeshFleetTrellisDataset(
        root,
        split=args.meshfleet_split,
        category=args.meshfleet_category,
        num_views=args.num_views,
        image_size=args.image_size,
        occ_resolution=args.meshfleet_occ_resolution,
        prefer_cond_render=args.meshfleet_prefer_cond_render,
        require_ss_latents=True,
        require_voxels=True,
        uid_manifest=args.train_manifest,
    )
    if len(dataset) == 0:
        raise FileNotFoundError(
            "MeshFleet_TRELLIS split has no reconstructed samples. "
            f"Checked root={root}, split={args.meshfleet_split}, category={args.meshfleet_category}. "
            "If this path contains webdataset shards, reconstruct them first with the dataset card's reconstruct_data.py; "
            "for the current local sample use --meshfleet_split test --meshfleet_category sdvas."
        )
    return build_dataloader(dataset, args=args, ctx=ctx, collate_fn=VehicleMultiViewDataset.collate_fn, shuffle=True)


def _move_batch(batch, device):
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def _assert_stage2_batch_contract(batch: dict, x0: torch.Tensor, device: torch.device) -> None:
    assert isinstance(batch, dict), "Stage 2 batch must be a dict."
    assert x0.ndim == 5, f"ss_latent_grid must be [B,C,D,H,W], got {tuple(x0.shape)}"
    assert x0.device == device, f"ss_latent_grid is on {x0.device}, expected {device}"
    assert x0.dtype == torch.float32, f"ss_latent_grid must be float32 after load, got {x0.dtype}"
    assert x0.shape[0] > 0 and x0.shape[1] > 0, f"invalid ss_latent_grid shape {tuple(x0.shape)}"
    assert torch.isfinite(x0).all().item(), "ss_latent_grid contains NaN or Inf."
    for key in ("images", "K", "c2w", "w2c", "masks"):
        assert key in batch, f"Stage 2 requires batch['{key}'] for GeoSS context construction."
        value = batch[key]
        assert isinstance(value, torch.Tensor), f"batch['{key}'] must be a tensor."
        assert value.shape[0] == x0.shape[0], f"batch['{key}'] batch size {value.shape[0]} != ss_latent_grid batch size {x0.shape[0]}"


def _assert_stage2_geoss_context(context: dict, ss_tokens: torch.Tensor) -> None:
    assert isinstance(context, dict), "GeoSS context must be a dict."
    for key in ("geo_tokens", "geo_confidence", "anchor_xyz"):
        assert key in context, f"GeoSS context missing '{key}'."
        assert isinstance(context[key], torch.Tensor), f"GeoSS context '{key}' must be a tensor."
    geo_tokens = context["geo_tokens"]
    geo_confidence = context["geo_confidence"]
    anchor_xyz = context["anchor_xyz"]
    B = ss_tokens.shape[0]
    assert ss_tokens.ndim == 3, f"ss_tokens must be [B,L,C], got {tuple(ss_tokens.shape)}"
    assert geo_tokens.ndim == 3, f"geo_tokens must be [B,M,G], got {tuple(geo_tokens.shape)}"
    assert geo_tokens.shape[0] == B, f"geo_tokens batch {geo_tokens.shape[0]} != ss_tokens batch {B}"
    assert geo_tokens.device == ss_tokens.device, f"geo_tokens is on {geo_tokens.device}, expected {ss_tokens.device}"
    assert geo_tokens.dtype == torch.float32, f"geo_tokens must be float32, got {geo_tokens.dtype}"
    assert torch.isfinite(geo_tokens).all().item(), "geo_tokens contains NaN or Inf."
    assert geo_confidence.shape == (*geo_tokens.shape[:2], 1), (
        f"geo_confidence must be [B,M,1], got {tuple(geo_confidence.shape)} for geo_tokens {tuple(geo_tokens.shape)}"
    )
    assert geo_confidence.device == ss_tokens.device, f"geo_confidence is on {geo_confidence.device}, expected {ss_tokens.device}"
    assert geo_confidence.dtype == torch.float32, f"geo_confidence must be float32, got {geo_confidence.dtype}"
    assert torch.isfinite(geo_confidence).all().item(), "geo_confidence contains NaN or Inf."
    assert geo_confidence.min().item() >= 0.0 and geo_confidence.max().item() <= 1.0, "geo_confidence must be in [0, 1]."
    assert anchor_xyz.shape == (*geo_tokens.shape[:2], 3), (
        f"anchor_xyz must be [B,M,3], got {tuple(anchor_xyz.shape)} for geo_tokens {tuple(geo_tokens.shape)}"
    )
    assert anchor_xyz.device == ss_tokens.device, f"anchor_xyz is on {anchor_xyz.device}, expected {ss_tokens.device}"
    assert anchor_xyz.dtype == torch.float32, f"anchor_xyz must be float32, got {anchor_xyz.dtype}"
    assert torch.isfinite(anchor_xyz).all().item(), "anchor_xyz contains NaN or Inf."
    anchor_metadata = context.get("anchor_metadata")
    if isinstance(anchor_metadata, torch.Tensor):
        assert anchor_metadata.shape[:2] == geo_tokens.shape[:2], (
            f"anchor_metadata must align with anchors, got {tuple(anchor_metadata.shape)} vs {tuple(geo_tokens.shape[:2])}"
        )
        assert anchor_metadata.device == ss_tokens.device, f"anchor_metadata is on {anchor_metadata.device}, expected {ss_tokens.device}"


def _assert_velocity_adapter_output(output: dict, ss_tokens: torch.Tensor, v_base_tokens: torch.Tensor) -> None:
    assert isinstance(output, dict), "SSVelocityAdapter must return a dict."
    required = ("v_geo", "delta_v_geo", "token_confidence", "debug")
    for key in required:
        assert key in output, f"SSVelocityAdapter output missing '{key}'."
    assert output["v_geo"].shape == v_base_tokens.shape, f"v_geo shape {tuple(output['v_geo'].shape)} != {tuple(v_base_tokens.shape)}"
    assert output["delta_v_geo"].shape == ss_tokens.shape, f"delta_v_geo shape {tuple(output['delta_v_geo'].shape)} != {tuple(ss_tokens.shape)}"
    expected_conf = (*ss_tokens.shape[:2], 1)
    assert output["token_confidence"].shape == expected_conf, f"token_confidence shape {tuple(output['token_confidence'].shape)} != {expected_conf}"
    allowed_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    assert output["v_geo"].dtype in allowed_dtypes, f"v_geo must be floating point, got {output['v_geo'].dtype}"
    assert output["delta_v_geo"].dtype in allowed_dtypes, f"delta_v_geo must be floating point, got {output['delta_v_geo'].dtype}"
    assert output["token_confidence"].dtype in allowed_dtypes, f"token_confidence must be floating point, got {output['token_confidence'].dtype}"
    assert torch.isfinite(output["v_geo"]).all().item(), "v_geo contains NaN or Inf."
    assert torch.isfinite(output["delta_v_geo"]).all().item(), "delta_v_geo contains NaN or Inf."
    assert torch.isfinite(output["token_confidence"]).all().item(), "token_confidence contains NaN or Inf."
    assert output["token_confidence"].min().item() >= 0.0 and output["token_confidence"].max().item() <= 1.0, (
        "token_confidence must be in [0, 1]."
    )
    assert output["delta_v_geo"].requires_grad, "delta_v_geo must carry gradients from SSVelocityAdapter parameters."
    assert output["v_geo"].requires_grad, "v_geo must carry gradients from SSVelocityAdapter parameters."
    debug = output["debug"]
    assert isinstance(debug, dict), "SSVelocityAdapter debug output must be a dict."
    assert "delta_raw" in debug, "SSVelocityAdapter debug output missing pre-clipped 'delta_raw'."
    assert debug["delta_raw"].shape == ss_tokens.shape, f"delta_raw shape {tuple(debug['delta_raw'].shape)} != {tuple(ss_tokens.shape)}"
    assert debug["delta_raw"].dtype in allowed_dtypes, f"delta_raw must be floating point, got {debug['delta_raw'].dtype}"
    assert torch.isfinite(debug["delta_raw"]).all().item(), "delta_raw contains NaN or Inf."
    assert debug["delta_raw"].requires_grad, "delta_raw must carry gradients from SSVelocityAdapter parameters."


def _assert_residual_training_contract(
    delta_tokens: torch.Tensor,
    effective_delta_tokens: torch.Tensor,
    target_residual_tokens: torch.Tensor,
) -> None:
    assert delta_tokens.shape == target_residual_tokens.shape, (
        f"raw residual shape {tuple(delta_tokens.shape)} != target residual {tuple(target_residual_tokens.shape)}"
    )
    assert effective_delta_tokens.shape == target_residual_tokens.shape, (
        f"effective residual shape {tuple(effective_delta_tokens.shape)} != target residual {tuple(target_residual_tokens.shape)}"
    )
    allowed_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    assert delta_tokens.dtype in allowed_dtypes, f"raw residual must be floating point, got {delta_tokens.dtype}"
    assert effective_delta_tokens.dtype in allowed_dtypes, f"effective residual must be floating point, got {effective_delta_tokens.dtype}"
    assert target_residual_tokens.dtype in allowed_dtypes, f"target residual must be floating point, got {target_residual_tokens.dtype}"
    assert torch.isfinite(delta_tokens).all().item(), "raw residual contains NaN or Inf."
    assert torch.isfinite(effective_delta_tokens).all().item(), "effective residual contains NaN or Inf."
    assert torch.isfinite(target_residual_tokens).all().item(), "target residual contains NaN or Inf."
    assert delta_tokens.requires_grad, "raw residual must depend on SSVelocityAdapter parameters."
    assert effective_delta_tokens.requires_grad, "effective residual must depend on SSVelocityAdapter parameters."
    assert not target_residual_tokens.requires_grad, "target residual must be a frozen flow-matching target."


def _assert_adapter_gradients(adapter: SSVelocityAdapter, step: int) -> dict[str, float]:
    critical = {
        "latent_proj.weight": adapter.latent_proj.weight,
        "latent_proj.bias": adapter.latent_proj.bias,
        "geo_proj.weight": adapter.geo_proj.weight,
        "geo_proj.bias": adapter.geo_proj.bias,
        "delta_head.2.weight": adapter.delta_head[-1].weight,
        "delta_head.2.bias": adapter.delta_head[-1].bias,
    }
    missing = [name for name, param in critical.items() if param.grad is None]
    assert not missing, f"Stage 2 DDP graph break at step={step}; missing gradients for {missing}"
    nonfinite = [name for name, param in critical.items() if param.grad is not None and not torch.isfinite(param.grad).all().item()]
    assert not nonfinite, f"Stage 2 non-finite gradients at step={step}: {nonfinite}"
    grad_norms = {name: float(param.grad.detach().norm().cpu()) for name, param in critical.items()}
    zero_weights = [name for name in ("latent_proj.weight", "geo_proj.weight", "delta_head.2.weight") if grad_norms[name] == 0.0]
    assert not zero_weights, (
        f"Stage 2 degenerate residual graph at step={step}; zero gradient norms for {zero_weights}. "
        "The raw residual estimator must receive supervised gradients independent of the confidence gate."
    )
    all_gradients = [
        parameter.grad.detach().float()
        for parameter in adapter.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    total_elements = sum(int(gradient.numel()) for gradient in all_gradients)
    grad_norms["total"] = math.sqrt(
        sum(float(gradient.square().sum().cpu()) for gradient in all_gradients)
    )
    grad_norms["zero_fraction"] = (
        sum(int((gradient == 0).sum().cpu()) for gradient in all_gradients) / max(1, total_elements)
    )
    grad_norms["nonfinite_fraction"] = (
        sum(int((~torch.isfinite(gradient)).sum().cpu()) for gradient in all_gradients) / max(1, total_elements)
    )
    return grad_norms


def _update_to_weight_ratio(
    adapter: SSVelocityAdapter,
    parameters_before: list[torch.Tensor] | None,
) -> float | None:
    if parameters_before is None:
        return None
    parameters_after = [parameter.detach() for parameter in adapter.parameters()]
    update_squared = sum(
        float((after.float() - before.float()).square().sum().cpu())
        for before, after in zip(parameters_before, parameters_after)
    )
    weight_squared = sum(
        float(after.float().square().sum().cpu()) for after in parameters_after
    )
    return math.sqrt(update_squared) / max(math.sqrt(weight_squared), 1.0e-12)


def _assert_terminal_delta_head_is_trainable(adapter: SSVelocityAdapter) -> None:
    final = adapter.delta_head[-1]
    assert isinstance(final, nn.Linear), "SSVelocityAdapter.delta_head terminal layer must be nn.Linear."
    assert final.weight.requires_grad, "terminal delta head weight must be trainable."
    assert final.weight.abs().max().item() > 0.0, (
        "terminal delta head was initialized to exact zero. "
        "That blocks first-step gradients into latent_proj/geo_proj under the raw residual objective."
    )


def _exact_zero_voxel_mask(x0: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Identify only encoded padding/inactive SS sites; never rank-prune learned voxels."""
    if x0.ndim != 5:
        raise ValueError(f"expected [B,C,D,H,W] SS latents, got {tuple(x0.shape)}")
    return (x0.abs().amax(dim=1) > max(0.0, float(epsilon))).flatten(1)


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
    if token_mask is None:
        return F.mse_loss(prediction, target)
    mask = token_mask.to(device=prediction.device, dtype=prediction.dtype)
    if mask.shape != (*prediction.shape[:2], 1):
        raise ValueError(f"SS token mask must be [B,L,1], got {tuple(mask.shape)}")
    return ((prediction - target).square() * mask).sum() / (mask.sum() * prediction.shape[-1]).clamp_min(1.0)


def _ss_grid_xyz(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert x.ndim == 5, f"SS grid must be [B,C,D,H,W], got {tuple(x.shape)}"
    B, _, D, H, W = x.shape
    # Dense latent sites represent voxel centers. This must be bitwise aligned
    # with geoss.integration.trellis_ss_hook._ss_grid_xyz; endpoint coordinates
    # shift local anchor neighborhoods between training and inference.
    z = (torch.arange(D, device=x.device, dtype=dtype) + 0.5) / D * 2.0 - 1.0
    y = (torch.arange(H, device=x.device, dtype=dtype) + 0.5) / H * 2.0 - 1.0
    x_coord = (torch.arange(W, device=x.device, dtype=dtype) + 0.5) / W * 2.0 - 1.0
    zz, yy, xx = torch.meshgrid(
        z,
        y,
        x_coord,
        indexing="ij",
    )
    xyz = torch.stack([xx, yy, zz], dim=-1).reshape(1, D * H * W, 3)
    return xyz.expand(B, -1, -1).contiguous()


@torch.no_grad()
def _compute_geoss_context(batch: dict, geoss_model: SparseRayGeoSSAdapter, vggt: VGGTGeometryWrapper) -> dict:
    batch = dict(batch)
    batch.update(vggt(batch["images"], use_cache=False))
    # Dataset samples include cached SS tokens. Removing them selects the
    # context-only branch of the exact frozen Stage 1 operator, avoiding a
    # second hand-maintained copy of alignment, camera/world scaling, dynamic
    # anchor construction, ray evidence, and confidence aggregation.
    batch.pop("ss_latent_tokens", None)
    batch.pop("v_base", None)
    context = geoss_model(batch)
    return {
        "geo_tokens": context["geo_tokens"].detach(),
        "geo_confidence": context["geo_confidence"].detach(),
        "anchor_xyz": context["anchor_xyz"].detach(),
        "anchor_metadata": context["anchor_metadata"].detach(),
    }


def _real_condition_or_fail(batch: dict | None, device: torch.device, cfg: dict, trellis_pipeline=None) -> torch.Tensor:
    if batch is None:
        raise RuntimeError("real_train requires real TRELLIS image condition tokens.")
    for key in ("trellis_cond", "trellis_cond_tokens", "image_cond", "cond"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.float32)
    cond_image = batch.get("trellis_cond_image")
    if isinstance(cond_image, torch.Tensor) and trellis_pipeline is not None and hasattr(trellis_pipeline, "encode_image"):
        with torch.no_grad():
            return trellis_pipeline.encode_image(cond_image.to(device=device, dtype=torch.float32))
    images = batch.get("images")
    if isinstance(images, torch.Tensor) and trellis_pipeline is not None and hasattr(trellis_pipeline, "encode_image"):
        first_view = F.interpolate(images[:, 0].to(device=device, dtype=torch.float32), size=(518, 518), mode="bilinear", align_corners=False)
        with torch.no_grad():
            return trellis_pipeline.encode_image(first_view)
    raise KeyError(
        "real_train requires image condition tokens from the real TRELLIS image encoder "
        "or a TRELLIS pipeline that can encode trellis_cond_image/images."
    )


def _training_mode(args: argparse.Namespace, real_dataset: bool) -> str:
    trellis = "real_trellis" if args.trellis_model_path else "mock_trellis"
    data = "meshfleet_trellis" if real_dataset else "synthetic_context"
    return f"{trellis}+{data}"


def _geoss_model_cfg_from_velocity_cfg(cfg: dict) -> dict:
    if "model" in cfg and isinstance(cfg["model"], dict):
        source = cfg["model"]
    else:
        source = cfg
    allowed = {"num_anchors", "anchor_dim", "evidence_dim", "geo_dim", "latent_dim"}
    return {key: source[key] for key in allowed if key in source}


def _save_velocity_checkpoint(
    path: Path,
    adapter: SSVelocityAdapter,
    optimizer,
    scheduler,
    step: int,
    cfg: dict,
    early_stopper: EarlyStopper,
    early_status,
    *,
    training_budget: dict | None = None,
    support_provenance: dict | None = None,
    execution_mode: str = "train",
    stage1_handoff: dict | None = None,
) -> None:
    if execution_mode != "train":
        raise RuntimeError(
            f"Refusing to write a promotable Stage 2 checkpoint in execution_mode={execution_mode}."
        )
    save_checkpoint(
        path,
        velocity_adapter=adapter.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        step=step,
        config=cfg,
        optimization_contract_version=OPTIMIZATION_CONTRACT_VERSION,
        early_stop=early_status.as_dict() if early_status is not None else None,
        early_stopper=early_stopper.state_dict(),
        training_budget=training_budget,
        support_provenance=support_provenance,
        execution_mode=execution_mode,
        is_probe=False,
        promotable_checkpoint=True,
        stage1_handoff=stage1_handoff,
    )


def _repair_zero_terminal_delta_head(adapter: SSVelocityAdapter) -> None:
    final = adapter.delta_head[-1]
    if not isinstance(final, nn.Linear):
        return
    with torch.no_grad():
        if final.weight.abs().max().item() == 0.0:
            nn.init.normal_(final.weight, mean=0.0, std=1e-5)


def _apply_config_defaults(args: argparse.Namespace, cfg: dict, parser: argparse.ArgumentParser) -> None:
    if not cfg:
        return
    dataset = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else {}
    trellis = cfg.get("trellis") if isinstance(cfg.get("trellis"), dict) else {}
    vggt = cfg.get("vggt") if isinstance(cfg.get("vggt"), dict) else {}
    mappings = {
        "meshfleet_root": cfg.get("meshfleet_root") or cfg.get("dataset_root") or dataset.get("root"),
        "meshfleet_split": cfg.get("meshfleet_split") or dataset.get("train_split") or dataset.get("split"),
        "meshfleet_category": cfg.get("meshfleet_category") or dataset.get("category"),
        "num_views": cfg.get("num_views") or dataset.get("num_views"),
        "image_size": cfg.get("image_size") or dataset.get("image_size"),
        "meshfleet_occ_resolution": cfg.get("meshfleet_occ_resolution") or dataset.get("occ_resolution"),
        "trellis_root": cfg.get("trellis_root") or trellis.get("root"),
        "trellis_model_path": cfg.get("trellis_model_path") or cfg.get("trellis_pipeline") or cfg.get("trellis_checkpoint") or trellis.get("model_path") or trellis.get("pipeline") or trellis.get("checkpoint"),
        "torch_hub_dir": cfg.get("torch_hub_dir") or trellis.get("torch_hub_dir"),
        "dinov2_repo": cfg.get("dinov2_repo") or trellis.get("dinov2_repo"),
        "vggt_root": cfg.get("vggt_root") or vggt.get("root"),
        "vggt_checkpoint": cfg.get("vggt_checkpoint") or vggt.get("checkpoint"),
        "vggt_pretrained": cfg.get("vggt_pretrained") or vggt.get("pretrained"),
        "geoss_checkpoint": cfg.get("geoss_checkpoint"),
        "steps": cfg.get("steps"),
        "steps_are_total": cfg.get("steps_are_total"),
        "execution_mode": cfg.get("execution_mode"),
        "auto_expand_training_budget": cfg.get("auto_expand_training_budget"),
        "minimum_dataset_passes": cfg.get("minimum_dataset_passes"),
        "early_stop": cfg.get("early_stop"),
        "early_stop_metric": cfg.get("early_stop_metric"),
        "early_stop_mode": cfg.get("early_stop_mode"),
        "max_train_hours": cfg.get("max_train_hours"),
        "save_best": cfg.get("save_best"),
        "train_manifest": cfg.get("train_manifest") or dataset.get("train_manifest"),
        "batch_size": cfg.get("batch_size"),
        "grad_accum_steps": cfg.get("grad_accum_steps"),
        "lr": cfg.get("lr"),
        "warmup_updates": cfg.get("warmup_updates"),
        "min_lr_ratio": cfg.get("min_lr_ratio"),
        "weight_decay": cfg.get("weight_decay"),
        "sigma_min": cfg.get("sigma_min"),
        "velocity_reg_weight": cfg.get("velocity_reg_weight"),
        "prior_weight": cfg.get("prior_weight"),
        "raw_residual_weight": cfg.get("raw_residual_weight"),
        "normalize_residual_loss": cfg.get("normalize_residual_loss"),
        "timestep_sampling": cfg.get("timestep_sampling"),
        "timestep_logit_mean": cfg.get("timestep_logit_mean"),
        "timestep_logit_std": cfg.get("timestep_logit_std"),
        "fused_flow_matching": cfg.get("fused_flow_matching"),
        "release_frozen_cache": cfg.get("release_frozen_cache"),
        "release_cache_reserved_fraction": cfg.get("release_cache_reserved_fraction"),
        "release_cache_inactive_gib": cfg.get("release_cache_inactive_gib"),
        "save_every": cfg.get("save_every"),
        "output_dir": cfg.get("output_dir"),
        "device": cfg.get("device"),
        **adaptive_config_defaults(cfg),
    }
    apply_config_mappings(args, parser, mappings)


def _maybe_launch_stage2(summary: dict, cfg: dict) -> None:
    workflow = cfg.get("workflow", {}) if isinstance(cfg.get("workflow"), dict) else {}
    if not workflow.get("auto_stage2_on_convergence", False):
        return
    early = summary.get("early_stop", {}) if isinstance(summary, dict) else {}
    ready, _ = handoff_contract(early)
    if not early.get("should_stop") or not ready:
        return
    command = workflow.get("stage2_command")
    if command:
        subprocess.run(str(command), shell=True, check=True)
        return
    script = workflow.get("stage2_script", "scripts/train_geovis_slat.py")
    config = workflow.get("stage2_config", "configs/real_train_slat_only.yaml")
    cmd = [sys.executable, str(script), "--config", str(config)]
    extra_args = workflow.get("stage2_extra_args", [])
    if isinstance(extra_args, list):
        cmd.extend(str(x) for x in extra_args)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
