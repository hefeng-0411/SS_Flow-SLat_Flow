#!/usr/bin/env python3
"""Production DDP trainer for stochastic multi-view, voxel-conditioned SS Flow."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geoss.datasets.dataset_stochastic_meshfleet import (  # noqa: E402
    StochasticMeshFleetDataset,
    seed_meshfleet_worker,
    stochastic_meshfleet_collate,
)
from geoss.integration.trellis_hub import configure_trellis_hub  # noqa: E402
from geoss.integration.trellis_ss_hook import ss_grid_to_tokens, tokens_to_ss_grid  # noqa: E402
from geoss.integration.vggt_geometry_wrapper import VGGTGeometryWrapper  # noqa: E402
from geoss.losses.geometric_loss import FlowMatchingLossBuilder  # noqa: E402
from geoss.models.ss_flow_adapter import SSFlowAdapter  # noqa: E402
from geoss.models.voxel_fusion_engine import ConfidenceSparseVoxelFusion  # noqa: E402
from geoss.ops.flow_matching import construct_flow_training_pair  # noqa: E402


class TrainableVoxelSSBranch(nn.Module):
    """The only DDP-replicated branch: voxel projection/refinement + adapter."""

    def __init__(self, fusion: ConfidenceSparseVoxelFusion, adapter: SSFlowAdapter) -> None:
        super().__init__()
        self.fusion = fusion
        self.adapter = adapter

    def forward(
        self,
        geometry,
        batch: Dict[str, torch.Tensor],
        x_t_tokens: torch.Tensor,
        timestep_model: torch.Tensor,
        v_base_tokens: torch.Tensor,
        *,
        profile: bool = False,
    ):
        fusion_timer = CudaStageTimer(x_t_tokens.device, profile)
        fusion_timer.start()
        fused = self.fusion(
            geometry,
            foreground_masks=batch["masks"],
            dataset_c2w=batch["c2w_dataset"],
            canonical_center=batch["canonical_center"],
            canonical_half_extent=batch["canonical_half_extent"],
            profile=profile,
        )
        fusion_ms = fusion_timer.stop()
        adapter_timer = CudaStageTimer(x_t_tokens.device, profile)
        adapter_timer.start()
        adapted = self.adapter(
            x_t_tokens,
            fused.dense_tokens,
            timestep_model,
            fused.observation_mask,
            fused.voxel_confidence,
            v_base=v_base_tokens,
        )
        adapter_ms = adapter_timer.stop()
        return fused, adapted, {
            "voxel_fusion": fusion_ms,
            **fused.timings_ms,
            "adapter_forward": adapter_ms,
        }


def main() -> None:
    parser = build_parser()
    apply_yaml_defaults(parser)
    args = parser.parse_args()
    context = init_distributed(args)
    try:
        run_training(args, context)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_training(args: argparse.Namespace, context: Dict[str, Any]) -> None:
    rank, world_size, local_rank, device = (
        context["rank"], context["world_size"], context["local_rank"], context["device"]
    )
    seed_everything(args.seed, rank)
    if args.batch_size != 1:
        raise ValueError(
            "VGGT has no padded-view attention mask; production stochastic 1-8 view training therefore requires --batch-size 1 per rank."
        )
    precision = resolve_precision(args.precision, device)
    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(json.dumps({"event": "startup", "world_size": world_size, "precision": str(precision), "args": vars(args)}))

    dataset = StochasticMeshFleetDataset(
        args.dataset_root,
        split=args.train_split,
        min_views=args.min_views,
        max_views=args.max_views,
        seed=args.seed,
        rank=rank,
        use_stochastic_views=args.use_stochastic_views,
        image_size=args.image_size,
        load_gt_occupancy=args.use_occupancy_loss,
        occupancy_resolution=64,
        meshfleet_root=args.meshfleet_root,
        require_ss_latents=True,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        worker_init_fn=seed_meshfleet_worker,
        collate_fn=stochastic_meshfleet_collate,
        drop_last=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    if len(loader) == 0:
        raise RuntimeError("Distributed MeshFleet loader is empty")
    validation_loader = build_validation_loader(args, rank, world_size, device)

    vggt, trellis_pipeline, base_flow, decoder = load_frozen_foundations(args, device)
    fusion = ConfidenceSparseVoxelFusion(
        grid_resolution=args.grid_resolution,
        input_feature_dim=2048,
        projected_feature_dim=args.voxel_feature_dim,
        positional_frequencies=args.positional_frequencies,
        fusion_mode=args.fusion_mode,
        use_vggt_depth=args.use_vggt_depth,
        use_vggt_pointmap=args.use_vggt_pointmap,
        use_confidence_weighting=args.use_confidence_weighting,
        use_visibility_weighting=args.use_visibility_weighting,
        use_3d_positional_encoding=args.use_3d_positional_encoding,
        require_spconv=True,
        use_spconv_refinement=args.use_spconv_refinement,
    ).to(device)
    adapter = SSFlowAdapter(
        latent_dim=8,
        condition_dim=fusion.condition_dim,
        hidden_dim=args.adapter_hidden_dim,
        num_heads=args.adapter_heads,
        num_blocks=args.adapter_blocks,
        g_observed=args.g_observed,
        g_unobserved=args.g_unobserved,
        trust_region=args.trust_region,
        gradient_checkpointing=args.gradient_checkpointing,
        use_observation_gate=args.use_observation_gate,
    ).to(device)
    trainable: nn.Module = TrainableVoxelSSBranch(fusion, adapter).to(device)
    optimizer = torch.optim.AdamW(trainable.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.max_steps, 1), eta_min=args.min_learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == torch.float16)
    start_step, epoch = maybe_resume(args.resume, trainable, optimizer, scheduler, scaler, device, rank)
    if world_size > 1:
        trainable = DDP(
            trainable,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    loss_builder = FlowMatchingLossBuilder(
        lambda_cfm=args.lambda_cfm,
        lambda_depth=args.lambda_depth,
        lambda_silhouette=args.lambda_silhouette,
        lambda_occupancy=args.lambda_occupancy,
        lambda_surface=args.lambda_surface,
        lambda_prior=args.lambda_prior,
        use_depth_loss=args.use_depth_loss,
        use_silhouette_loss=args.use_silhouette_loss,
        use_surface_loss=args.use_surface_loss,
        use_prior_preservation=args.use_prior_preservation,
        surface_backend=args.surface_backend,
        extensions_root=args.extensions_root,
        render_resolution=args.loss_render_resolution,
        ray_samples=args.loss_ray_samples,
    ).to(device)
    report_parameter_counts(rank, trainable, (vggt, trellis_pipeline))

    metrics_path = Path(args.output_dir) / "metrics.jsonl"
    csv_path = Path(args.output_dir) / "metrics.csv"
    sampler.set_epoch(epoch)
    dataset.set_epoch(epoch)
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, args.max_steps + 1):
        step_wall = time.perf_counter()
        accumulated_loss = torch.zeros((), device=device)
        last_payload: Dict[str, Any] = {}
        dataloader_seconds = 0.0
        profile_timings: Dict[str, Any] = {}
        for accumulation_index in range(args.gradient_accumulation_steps):
            fetch_start = time.perf_counter()
            try:
                cpu_batch = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                dataset.set_epoch(epoch)
                iterator = iter(loader)
                cpu_batch = next(iterator)
            dataloader_seconds += time.perf_counter() - fetch_start
            h2d_timer = CudaStageTimer(device, args.profile)
            h2d_timer.start()
            batch = move_batch(cpu_batch, device)
            profile_timings["h2d"] = h2d_timer.stop()

            sync_context = (
                trainable.no_sync()
                if isinstance(trainable, DDP) and accumulation_index < args.gradient_accumulation_steps - 1
                else contextlib.nullcontext()
            )
            with sync_context:
                payload = forward_training_batch(
                    args,
                    batch,
                    vggt,
                    trellis_pipeline,
                    base_flow,
                    decoder,
                    trainable,
                    loss_builder,
                    precision,
                    device,
                )
                loss = payload["losses"]["loss_total"] / args.gradient_accumulation_steps
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step={step}, uid={batch['uid']}")
                scaler.scale(loss).backward()
            accumulated_loss += loss.detach()
            last_payload = payload
            profile_timings.update(payload["timings"])

        backward_timer = CudaStageTimer(device, args.profile)
        backward_timer.start()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable.parameters(), args.gradient_clip_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm at step={step}")
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        profile_timings["optimizer"] = backward_timer.stop()

        reduced_loss = distributed_mean(accumulated_loss)
        elapsed = time.perf_counter() - step_wall
        metrics = build_metrics(
            step,
            epoch,
            reduced_loss,
            last_payload,
            profile_timings,
            dataloader_seconds,
            elapsed,
            grad_norm,
            optimizer,
            batch,
            world_size,
            device,
        )
        if rank == 0:
            append_metrics(metrics_path, csv_path, metrics)
            if step == 1 or step % args.log_every == 0:
                print(json.dumps(metrics, default=json_default))
        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(
                Path(args.output_dir) / "stochastic_voxel_ss_flow_last.pt",
                trainable,
                optimizer,
                scheduler,
                scaler,
                step,
                epoch,
                args,
                rank,
            )
        if validation_loader is not None and step % args.validate_every == 0:
            validation_metrics = run_validation(
                args,
                validation_loader,
                vggt,
                trellis_pipeline,
                base_flow,
                decoder,
                trainable,
                loss_builder,
                precision,
                device,
                rank,
                step,
            )
            if rank == 0:
                validation_path = Path(args.output_dir) / "validation.jsonl"
                with validation_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(validation_metrics, default=json_default) + "\n")
                print(json.dumps(validation_metrics, default=json_default))
        if world_size > 1 and step % args.barrier_every == 0:
            dist.barrier()


def forward_training_batch(
    args,
    batch,
    vggt,
    pipeline,
    base_flow,
    decoder,
    trainable,
    loss_builder,
    precision,
    device,
) -> Dict[str, Any]:
    vggt_timer = CudaStageTimer(device, args.profile)
    vggt_timer.start()
    geometry = vggt.extract(batch["images"], valid_view_mask=batch["view_valid_mask"])
    vggt_ms = vggt_timer.stop()
    x0 = batch["ss_latent_grid"].float()
    noise = torch.randn_like(x0)
    timestep = sample_timestep(x0.shape[0], device, args)
    x_t, v_target_grid, flow_backend = construct_flow_training_pair(
        x0, noise, timestep, args.sigma_min, backend=args.flow_backend
    )
    trellis_timer = CudaStageTimer(device, args.profile)
    trellis_timer.start()
    with torch.inference_mode():
        condition = encode_real_trellis_condition(pipeline, batch["images"], precision)
        timestep_model = timestep * 1000.0
        v_base_grid = base_flow(x_t, timestep_model, condition)
    trellis_ms = trellis_timer.stop()
    x_tokens = ss_grid_to_tokens(x_t)
    base_tokens = ss_grid_to_tokens(v_base_grid)
    target_tokens = ss_grid_to_tokens(v_target_grid)
    with torch.autocast(device_type=device.type, dtype=precision, enabled=device.type == "cuda"):
        try:
            fused, adapted, branch_timings = trainable(
                geometry,
                batch,
                x_tokens,
                timestep_model,
                base_tokens,
                profile=args.profile,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"SS voxel fusion failed for uid={batch.get('uid')}, "
                f"view_ids={batch.get('view_ids')}: {exc}"
            ) from exc
        v_final_grid = tokens_to_ss_grid(adapted.v_final, (16, 16, 16))
        predicted_clean = velocity_to_clean_state(x_t, v_final_grid, timestep, args.sigma_min)
        target_surface_points, target_surface_weights = (
            surface_targets_from_fusion(fused) if args.use_surface_loss else (None, None)
        )
        loss_timer = CudaStageTimer(device, args.profile)
        loss_timer.start()
        losses = loss_builder(
            v_final=adapted.v_final,
            v_target=target_tokens,
            v_base=base_tokens,
            gate=adapted.gate,
            valid_mask=torch.ones_like(fused.valid_mask),
            predicted_clean_grid=predicted_clean,
            ss_decoder=(
                decoder
                if args.use_occupancy_loss or args.use_depth_loss or args.use_silhouette_loss or args.use_surface_loss
                else None
            ),
            gt_occ=batch.get("gt_occ") if args.use_occupancy_loss else None,
            masks=batch.get("masks"),
            K_dataset=batch.get("K_dataset"),
            c2w_dataset=batch.get("c2w_dataset"),
            canonical_center=batch.get("canonical_center"),
            canonical_half_extent=batch.get("canonical_half_extent"),
            vggt_depth=geometry.depth,
            vggt_depth_confidence=geometry.depth_confidence,
            alignment_scale=fused.alignment_scale,
            target_surface_points=target_surface_points,
            target_surface_weights=target_surface_weights,
        )
        losses_ms = loss_timer.stop()
    return {
        "losses": losses,
        "adapter_diagnostics": adapted.diagnostics,
        "flow_backend": flow_backend,
        "observed_fraction": fused.observation_mask.float().mean(),
        "alignment_plausible_fraction": fused.alignment_plausible_fraction.mean(),
        "alignment_scale": fused.alignment_scale.mean(),
        "timings": {"vggt_forward": vggt_ms, "frozen_trellis_forward": trellis_ms, "losses": losses_ms, **branch_timings},
    }


def surface_targets_from_fusion(fused) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose confidence-weighted VGGT fused support to GeomLoss/FRNN."""
    batch_ids = fused.voxel_indices[:, 0].long()
    if batch_ids.numel() == 0 or int(batch_ids.max().item()) != 0:
        raise RuntimeError("Surface supervision currently requires the enforced per-rank batch_size=1 contract")
    indices = fused.voxel_indices.long()
    resolution = 16
    local_keys = indices[:, 3] + resolution * indices[:, 2] + resolution**2 * indices[:, 1]
    weights = fused.voxel_confidence[batch_ids, local_keys, 0].float().clamp_min(1e-6)
    return fused.voxel_xyz.unsqueeze(0), weights.unsqueeze(0)


def build_validation_loader(args, rank: int, world_size: int, device: torch.device):
    if args.validate_every <= 0 or args.validation_batches <= 0:
        return None
    view_counts = parse_view_counts(args.validation_view_counts)
    validation_dataset = StochasticMeshFleetDataset(
        args.dataset_root,
        split=args.validation_split,
        min_views=max(view_counts),
        max_views=max(view_counts),
        seed=args.seed + 17,
        rank=rank,
        use_stochastic_views=False,
        image_size=args.image_size,
        load_gt_occupancy=args.use_occupancy_loss,
        occupancy_resolution=64,
        meshfleet_root=args.meshfleet_root,
        require_ss_latents=True,
    )
    validation_sampler = DistributedSampler(
        validation_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    validation_workers = max(0, int(args.validation_workers))
    return DataLoader(
        validation_dataset,
        batch_size=1,
        sampler=validation_sampler,
        num_workers=validation_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
        persistent_workers=args.persistent_workers and validation_workers > 0,
        worker_init_fn=seed_meshfleet_worker,
        collate_fn=stochastic_meshfleet_collate,
        drop_last=False,
        prefetch_factor=args.prefetch_factor if validation_workers > 0 else None,
    )


def run_validation(
    args,
    loader,
    vggt,
    pipeline,
    base_flow,
    decoder,
    trainable,
    loss_builder,
    precision,
    device,
    rank: int,
    step: int,
) -> Dict[str, Any]:
    view_counts = parse_view_counts(args.validation_view_counts)
    totals = {count: torch.zeros((), device=device) for count in view_counts}
    batches = torch.zeros((), device=device)
    was_training = trainable.training
    trainable.eval()
    cuda_devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
        iterator = iter(loader)
        for batch_index in range(args.validation_batches):
            try:
                cpu_batch = next(iterator)
            except StopIteration:
                break
            full_batch = move_batch(cpu_batch, device)
            for view_count in view_counts:
                batch = slice_validation_views(full_batch, view_count)
                validation_seed = args.seed + 1_000_003 * step + 10_007 * batch_index + 101 * view_count + rank
                torch.manual_seed(validation_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed(validation_seed)
                payload = forward_training_batch(
                    args,
                    batch,
                    vggt,
                    pipeline,
                    base_flow,
                    decoder,
                    trainable,
                    loss_builder,
                    precision,
                    device,
                )
                totals[view_count] += payload["losses"]["loss_total"].detach().float()
            batches += 1
    if was_training:
        trainable.train()
    if dist.is_initialized():
        dist.all_reduce(batches, op=dist.ReduceOp.SUM)
        for value in totals.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    if batches.item() <= 0:
        raise RuntimeError("Validation loader produced no batches")
    return {
        "event": "validation",
        "step": step,
        "view_counts": view_counts,
        "loss_by_view_count": {str(count): float((totals[count] / batches).cpu()) for count in view_counts},
        "global_batches": int(batches.item()),
    }


def slice_validation_views(batch: Dict[str, Any], requested_views: int) -> Dict[str, Any]:
    available = int(batch["view_valid_mask"][0].sum().item())
    count = min(int(requested_views), available)
    if count < 1:
        raise RuntimeError("Validation object has no usable views")
    view_keys = {
        "images",
        "masks",
        "K_dataset",
        "c2w_dataset",
        "w2c_dataset",
        "view_valid_mask",
        "view_ids",
        "view_metadata_indices",
    }
    result = {
        key: value[:, :count] if key in view_keys and isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    result["num_views"] = torch.full_like(batch["num_views"], count)
    return result


def load_frozen_foundations(args, device):
    if not Path(args.trellis_root).is_dir():
        raise FileNotFoundError(f"TRELLIS root not found: {args.trellis_root}")
    if not Path(args.vggt_root).is_dir():
        raise FileNotFoundError(f"VGGT root not found: {args.vggt_root}")
    sys.path.insert(0, args.trellis_root)
    configure_trellis_hub(args)
    try:
        from trellis.pipelines import TrellisImageTo3DPipeline
    except Exception as exc:
        raise RuntimeError("Could not import the real local TRELLIS image-to-3D pipeline") from exc
    trellis_source = resolve_local_hf_snapshot(
        args.trellis_model,
        args.hf_cache_root,
        required_file="pipeline.json",
    )
    try:
        pipeline = TrellisImageTo3DPipeline.from_pretrained(trellis_source)
    except Exception as exc:
        raise RuntimeError(f"Could not load real local TRELLIS pipeline {trellis_source!r}") from exc
    pipeline.to(device)
    required = ("sparse_structure_flow_model", "sparse_structure_decoder", "image_cond_model")
    missing = [name for name in required if name not in pipeline.models]
    if missing:
        raise RuntimeError(f"TRELLIS pipeline lacks production components: {missing}")
    for model in pipeline.models.values():
        if isinstance(model, nn.Module):
            model.eval().requires_grad_(False)
    base_flow = pipeline.models["sparse_structure_flow_model"]
    decoder = pipeline.models["sparse_structure_decoder"]
    contract = (base_flow.in_channels, base_flow.out_channels, base_flow.resolution, base_flow.cond_channels)
    if contract != (8, 8, 16, 1024):
        raise RuntimeError(f"Incompatible TRELLIS SS contract {contract}; expected (8,8,16,1024)")
    vggt_source = None
    if args.vggt_checkpoint is None:
        vggt_source = resolve_local_hf_snapshot(
            args.vggt_pretrained,
            args.hf_cache_root,
            required_file="model.safetensors",
        )
    vggt = VGGTGeometryWrapper(
        vggt_root=args.vggt_root,
        checkpoint=args.vggt_checkpoint,
        pretrained_name=vggt_source,
        mock=False,
        require_real=True,
        vggt_image_size=518,
    ).to(device)
    return vggt, pipeline, base_flow, decoder


def resolve_local_hf_snapshot(model_or_path: str, cache_root: str, *, required_file: str) -> str:
    """Resolve a model to a complete local snapshot without implicit network I/O."""
    explicit = Path(model_or_path).expanduser()
    if explicit.is_dir():
        if not (explicit / required_file).is_file():
            raise FileNotFoundError(f"Local model path lacks {required_file}: {explicit}")
        return str(explicit.resolve())
    if "/" not in model_or_path:
        raise FileNotFoundError(f"Expected a local model directory or Hugging Face repo id, got {model_or_path!r}")
    repository = Path(cache_root).expanduser() / f"models--{model_or_path.replace('/', '--')}" / "snapshots"
    if not repository.is_dir():
        raise FileNotFoundError(
            f"No local snapshots for {model_or_path!r} under {repository}; production training will not download implicitly"
        )
    candidates = [path for path in repository.iterdir() if path.is_dir() and (path / required_file).is_file()]
    if not candidates:
        raise FileNotFoundError(f"No complete local {model_or_path!r} snapshot contains {required_file} under {repository}")
    selected = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    return str(selected.resolve())


@torch.inference_mode()
def encode_real_trellis_condition(pipeline, images: torch.Tensor, precision: torch.dtype) -> torch.Tensor:
    if images.ndim != 5 or images.shape[0] < 1:
        raise ValueError(f"Images must be real [B,V,3,H,W], got {tuple(images.shape)}")
    first_view = torch.nn.functional.interpolate(
        images[:, 0].float(), (518, 518), mode="bicubic", align_corners=False, antialias=True
    ).clamp(0, 1)
    with torch.autocast(device_type=images.device.type, dtype=precision, enabled=images.device.type == "cuda"):
        condition = pipeline.encode_image(first_view)
    if condition.ndim != 3 or condition.shape[-1] != 1024 or not torch.isfinite(condition).all():
        raise RuntimeError(f"Real TRELLIS DINO condition contract failed: {tuple(condition.shape)}")
    return condition


def velocity_to_clean_state(x_t: torch.Tensor, velocity: torch.Tensor, t: torch.Tensor, sigma_min: float) -> torch.Tensor:
    t_view = t.view(-1, *([1] * (x_t.ndim - 1))).float()
    # Exact inverse used by local TRELLIS FlowEulerSampler._v_to_xstart_eps.
    return (1.0 - sigma_min) * x_t - (sigma_min + (1.0 - sigma_min) * t_view) * velocity


def sample_timestep(batch_size: int, device: torch.device, args) -> torch.Tensor:
    if args.timestep_sampling == "uniform":
        return torch.rand(batch_size, device=device)
    return torch.sigmoid(
        torch.randn(batch_size, device=device) * args.timestep_logit_std + args.timestep_logit_mean
    )


def init_distributed(args) -> Dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"LOCAL_RANK={local_rank} exceeds visible CUDA devices={torch.cuda.device_count()}")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1:
        dist.init_process_group(backend=backend, init_method="env://")
    args.rank, args.world_size, args.local_rank = rank, world_size, local_rank
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank, "device": device}


def seed_everything(seed: int, rank: int) -> None:
    rank_seed = int(seed) + 100_003 * int(rank)
    random.seed(rank_seed)
    np.random.seed(rank_seed % (2**32))
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)


def resolve_precision(name: str, device: torch.device) -> torch.dtype:
    if name == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if name in {"bf16", "fp16"} and device.type == "cuda":
        return torch.float16
    return torch.float32


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def distributed_mean(value: torch.Tensor) -> torch.Tensor:
    result = value.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= dist.get_world_size()
    return result


def maybe_resume(path, model, optimizer, scheduler, scaler, device, rank: int) -> tuple[int, int]:
    if path is None:
        return 0, 0
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    rng_by_rank = payload.get("rng_by_rank")
    if rng_by_rank is not None:
        if rank >= len(rng_by_rank):
            raise RuntimeError(f"Checkpoint contains {len(rng_by_rank)} RNG streams but resume requested rank={rank}")
        rng = rng_by_rank[rank]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if device.type == "cuda" and rng.get("cuda") is not None:
            torch.cuda.set_rng_state(rng["cuda"].cpu(), device)
    else:  # backward compatibility with earlier rank-0-only checkpoints
        torch.set_rng_state(payload["torch_rng"].cpu())
        if device.type == "cuda" and payload.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(payload["cuda_rng"].cpu(), device)
        if rank != 0:
            seed_everything(int(payload.get("args", {}).get("seed", 42)) + 1_000_003 * int(payload["step"]), rank)
    return int(payload["step"]), int(payload["epoch"])


def save_checkpoint(path, model, optimizer, scheduler, scaler, step, epoch, args, rank: int) -> None:
    local_rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    if dist.is_initialized():
        rng_by_rank = [None] * dist.get_world_size() if rank == 0 else None
        dist.gather_object(local_rng, rng_by_rank, dst=0)
    else:
        rng_by_rank = [local_rng]
    if rank != 0:
        return
    unwrapped = model.module if isinstance(model, DDP) else model
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": unwrapped.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "epoch": epoch,
            "args": vars(args),
            "rng_by_rank": rng_by_rank,
            # Retain legacy fields for older readers.
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        },
        temporary,
    )
    os.replace(temporary, path)


def report_parameter_counts(rank: int, trainable: nn.Module, frozen_modules: Iterable[Any]) -> None:
    if rank != 0:
        return
    train_count = sum(parameter.numel() for parameter in trainable.parameters() if parameter.requires_grad)
    frozen_count = 0
    seen = set()
    for module in frozen_modules:
        modules = module.models.values() if hasattr(module, "models") else (module,)
        for item in modules:
            if not isinstance(item, nn.Module):
                continue
            for parameter in item.parameters():
                if id(parameter) not in seen:
                    frozen_count += parameter.numel()
                    seen.add(id(parameter))
    if train_count >= frozen_count:
        raise RuntimeError(f"Adapter parameter invariant failed: trainable={train_count}, frozen={frozen_count}")
    print(json.dumps({"trainable_parameters": train_count, "frozen_parameters": frozen_count}))


def build_metrics(step, epoch, loss, payload, timings, dataloader, elapsed, grad_norm, optimizer, batch, world_size, device):
    adapter = payload["adapter_diagnostics"]
    losses = payload["losses"]
    views = int(batch["view_valid_mask"].sum().item())
    uid = batch.get("uid", [""])
    uid = uid[0] if isinstance(uid, (list, tuple)) else uid
    selected_view_ids = batch["view_ids"][batch["view_valid_mask"]].detach().cpu().tolist()
    metrics = {
        "step": step,
        "epoch": epoch,
        "uid": uid,
        "view_ids": selected_view_ids,
        "loss_total": float(loss.cpu()),
        "loss_cfm": float(losses["loss_cfm"].detach().float().cpu()),
        "loss_depth": float(losses["loss_depth"].detach().float().cpu()),
        "loss_silhouette": float(losses["loss_silhouette"].detach().float().cpu()),
        "loss_occupancy": float(losses["loss_occupancy"].detach().float().cpu()),
        "loss_surface": float(losses["loss_surface"].detach().float().cpu()),
        "loss_prior": float(losses["loss_prior"].detach().float().cpu()),
        "adapter_norm": float(adapter["mean_adapter_norm"].detach().float().cpu()),
        "residual_base_ratio": float(adapter["residual_base_ratio"].detach().float().cpu()),
        "observed_percent": float(adapter["percentage_observed"].detach().float().cpu()),
        "alignment_plausible_percent": float(
            100.0 * payload["alignment_plausible_fraction"].detach().float().cpu()
        ),
        "alignment_scale": float(payload["alignment_scale"].detach().float().cpu()),
        "grad_norm": float(grad_norm.detach().float().cpu()),
        "learning_rate": optimizer.param_groups[0]["lr"],
        "flow_backend": payload["flow_backend"],
        "dataloader_wait": dataloader,
        "step_seconds": elapsed,
        "samples_per_second": world_size / max(elapsed, 1e-8),
        "views_per_second": views * world_size / max(elapsed, 1e-8),
        "max_allocated_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "max_reserved_cuda_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        "timings_ms": timings,
    }
    return metrics


def append_metrics(jsonl_path: Path, csv_path: Path, metrics: Dict[str, Any]) -> None:
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, default=json_default) + "\n")
    flat = {key: value for key, value in metrics.items() if isinstance(value, (str, int, float, bool))}
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    fieldnames = list(flat)
    if not new_file:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            existing_header = next(csv.reader(handle), [])
        if existing_header:
            # Resumed runs may add JSONL diagnostics without corrupting the
            # fixed schema of an already-created CSV metrics file.
            fieldnames = existing_header
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(flat)


def json_default(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().item() if value.numel() == 1 else value.detach().float().cpu().tolist()
    return str(value)


class CudaStageTimer:
    def __init__(self, device: torch.device, enabled: bool) -> None:
        self.enabled = bool(enabled and device.type == "cuda")
        self.start_event = torch.cuda.Event(enable_timing=True) if self.enabled else None
        self.end_event = torch.cuda.Event(enable_timing=True) if self.enabled else None

    def start(self) -> None:
        if self.start_event is not None:
            self.start_event.record()

    def stop(self) -> Optional[float]:
        if self.start_event is None or self.end_event is None:
            return None
        self.end_event.record()
        self.end_event.synchronize()  # profiler boundary only
        return float(self.start_event.elapsed_time(self.end_event))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stochastic_voxel_ss_flow.yaml")
    parser.add_argument("--trellis-root", required=False, default="/mnt/sda/hf/MVG/Base/TRELLIS")
    parser.add_argument("--trellis-model", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--vggt-root", default="/mnt/sda/hf/MVG/Base/vggt")
    parser.add_argument("--vggt-pretrained", default="facebook/VGGT-1B")
    parser.add_argument("--vggt-checkpoint")
    parser.add_argument("--hf-cache-root", default="/mnt/sda/hf/.cache/huggingface/hub")
    parser.add_argument("--meshfleet-root", default="/mnt/sda/hf/MVG/Base/MeshFleet")
    parser.add_argument("--dataset-root", default="/mnt/sda/hf/MVG/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97")
    parser.add_argument("--extensions-root", default="/mnt/sda/hf/MVG/Base/extensions")
    parser.add_argument("--torch-hub-dir", default="/mnt/sda/hf/.cache/torch/hub")
    parser.add_argument("--dinov2-repo", default="/mnt/sda/hf/.cache/torch/hub/facebookresearch_dinov2_main")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="test")
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--validation-batches", type=int, default=1)
    parser.add_argument("--validation-workers", type=int, default=2)
    parser.add_argument("--validation-view-counts", type=parse_view_counts, default=(1, 2, 4, 6, 8))
    parser.add_argument("--output-dir", default="outputs/stochastic_voxel_ss_flow")
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-views", type=int, default=1)
    parser.add_argument("--max-views", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--grid-resolution", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--sigma-min", type=float, default=1e-5)
    parser.add_argument("--timestep-sampling", choices=("uniform", "logit_normal"), default="logit_normal")
    parser.add_argument("--timestep-logit-mean", type=float, default=0.0)
    parser.add_argument("--timestep-logit-std", type=float, default=1.0)
    parser.add_argument("--flow-backend", choices=("auto", "torch", "triton"), default="auto")
    parser.add_argument("--voxel-feature-dim", type=int, default=256)
    parser.add_argument("--positional-frequencies", type=int, default=4)
    parser.add_argument("--adapter-hidden-dim", type=int, default=256)
    parser.add_argument("--adapter-heads", type=int, default=8)
    parser.add_argument("--adapter-blocks", type=int, default=2)
    parser.add_argument("--trust-region", type=float, default=0.25)
    parser.add_argument("--g-observed", type=float, default=1.0)
    parser.add_argument("--g-unobserved", type=float, default=0.0)
    parser.add_argument("--fusion-mode", choices=("confidence", "average_ablation"), default="confidence")
    for name, default in (
        ("use-stochastic-views", True),
        ("use-vggt-depth", True),
        ("use-vggt-pointmap", True),
        ("use-confidence-weighting", True),
        ("use-visibility-weighting", True),
        ("use-3d-positional-encoding", True),
        ("use-observation-gate", True),
        ("use-prior-preservation", True),
        ("use-occupancy-loss", True),
        ("use-depth-loss", True),
        ("use-silhouette-loss", True),
        ("use-surface-loss", False),
        ("use-spconv-refinement", True),
    ):
        parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=default)
    parser.add_argument("--surface-backend", choices=("geomloss", "frnn"), default="geomloss")
    parser.add_argument("--lambda-cfm", type=float, default=1.0)
    parser.add_argument("--lambda-depth", type=float, default=0.1)
    parser.add_argument("--lambda-silhouette", type=float, default=0.1)
    parser.add_argument("--lambda-occupancy", type=float, default=0.5)
    parser.add_argument("--lambda-surface", type=float, default=0.05)
    parser.add_argument("--lambda-prior", type=float, default=0.01)
    parser.add_argument("--loss-render-resolution", type=int, default=64)
    parser.add_argument("--loss-ray-samples", type=int, default=48)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--barrier-every", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    return parser


def apply_yaml_defaults(parser: argparse.ArgumentParser) -> None:
    preliminary, _ = argparse.ArgumentParser(add_help=False).parse_known_args()
    config_arg = next((sys.argv[index + 1] for index, value in enumerate(sys.argv[:-1]) if value == "--config"), None)
    config_path = Path(config_arg or "configs/stochastic_voxel_ss_flow.yaml")
    if not config_path.is_file():
        return
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    flattened = {}
    for section in (
        config,
        config.get("dataset", {}),
        config.get("model", {}),
        config.get("loss", {}),
        config.get("training", {}),
        config.get("evaluation", {}),
    ):
        for key, value in section.items():
            if not isinstance(value, dict):
                flattened[key.replace("-", "_")] = value
    valid_destinations = {action.dest for action in parser._actions}
    parser.set_defaults(**{key: value for key, value in flattened.items() if key in valid_destinations})


def parse_view_counts(value) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        counts = tuple(int(item) for item in value)
    else:
        counts = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not counts or any(count not in {1, 2, 4, 6, 8} for count in counts):
        raise argparse.ArgumentTypeError("validation view counts must be a comma-separated subset of 1,2,4,6,8")
    return tuple(dict.fromkeys(counts))


if __name__ == "__main__":
    main()
