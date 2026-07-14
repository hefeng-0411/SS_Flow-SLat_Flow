from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoss.datasets.meshfleet_trellis_dataset import MeshFleetTrellisDataset
from geoss.datasets.vehicle_multiview_dataset import VehicleMultiViewDataset
from geoss.integration.vggt_geometry_wrapper import VGGTGeometryWrapper
from geoss.integration.trellis_ss_hook import GeoSSTrellisSSWrapper, ss_grid_to_tokens, tokens_to_ss_grid
from geoss.losses.prior_preservation_loss import prior_preservation_loss
from geoss.losses.velocity_loss import velocity_regularization_loss
from geoss.models.sparse_ray_geoss_adapter import SparseRayGeoSSAdapter
from geoss.models.ss_velocity_adapter import SSVelocityAdapter
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
from geoss.utils.elastic_engine import cuda_memory_watermark, slice_batch_to_size, train_step_with_oom_retry


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
    cond = torch.randn(B, 16, geo_dim, device=device)
    geo_context = {
        "geo_tokens": torch.randn(B, M, geo_dim, device=device),
        "geo_confidence": torch.rand(B, M, 1, device=device),
    }
    adapter = SSVelocityAdapter(latent_dim=C, geo_dim=geo_dim).to(device)
    wrapper = GeoSSTrellisSSWrapper(MockSSFlowModel().to(device), adapter)
    v_geo = wrapper(x, t, cond, geoss_context=geo_context)
    enabled_debug = dict(wrapper.last_debug)
    v_base = wrapper(x, t, cond, geoss_context=geo_context, use_geoss_adapter=False)
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
    batch_controller = AdaptiveBatchController.from_args(args)
    args.batch_size = batch_controller.batch_size
    base, trellis_pipeline = _load_trellis_or_mock(args, cfg, allow_mock=False)
    spconv_status = _force_spconv_algo(base, args.spconv_algo)
    base = base.to(device)
    if trellis_pipeline is not None:
        trellis_pipeline.to(device)
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
    if args.resume and Path(args.resume).exists():
        resume_state = torch.load(args.resume, map_location="cpu")
        adapter.load_state_dict(resume_state.get("velocity_adapter", resume_state), strict=False)
        _repair_zero_terminal_delta_head(adapter)
        start_step = int(resume_state.get("step", 0))
    _assert_terminal_delta_head_is_trainable(adapter)
    adapter_model = maybe_wrap_ddp(adapter, ctx, find_unused_parameters=False)
    opt = torch.optim.AdamW(adapter_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = _make_grad_scaler(enabled=args.amp and args.amp_dtype == "fp16" and device.type == "cuda")
    if resume_state is not None and "optimizer" in resume_state:
        opt.load_state_dict(resume_state["optimizer"])
    loader, sampler = _build_meshfleet_loader(args, ctx)
    iterator = iter(loader) if loader is not None else None
    data_epoch = 0
    geoss_model = None
    vggt = None
    if iterator is not None:
        geoss_cfg = _geoss_model_cfg_from_velocity_cfg(cfg)
        geoss_model = SparseRayGeoSSAdapter(**geoss_cfg).to(device).eval()
        if args.geoss_checkpoint and Path(args.geoss_checkpoint).exists():
            state = torch.load(args.geoss_checkpoint, map_location="cpu")
            geoss_model.load_state_dict(state.get("model", state), strict=False)
        for p in geoss_model.parameters():
            p.requires_grad_(False)
        vggt = VGGTGeometryWrapper(
            vggt_root=args.vggt_root,
            checkpoint=args.vggt_checkpoint,
            pretrained_name=args.vggt_pretrained,
            mock=False,
            cache_features=False,
        ).to(device)
    out_dir = Path(args.output_dir)
    if ctx.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_sparse_ray_ss_velocity.jsonl"
    last = {}
    sigma_min = args.sigma_min
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
            data_batch = _move_batch(raw_batch, device)
        else:
            data_batch = None

        def rebuild_after_adjustment(adjustment):
            nonlocal data_batch, loader, sampler, iterator
            args.batch_size = adjustment.new_batch_size
            data_batch = slice_batch_to_size(data_batch, args.batch_size)
            loader, sampler = _build_meshfleet_loader(args, ctx)
            iterator = iter(loader) if loader is not None else None

        def log_oom(record):
            if ctx.is_main:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, **record}) + "\n")

        def step_fn():
            memory_start = cuda_memory_watermark(device, reset_peak=True)
            if data_batch is not None and "ss_latent_grid" in data_batch:
                x0 = data_batch["ss_latent_grid"].to(device=device, dtype=torch.float32)
                B = x0.shape[0]
            else:
                object_ids = data_batch.get("object_id") if isinstance(data_batch, dict) else None
                keys = sorted(data_batch.keys()) if isinstance(data_batch, dict) else []
                raise KeyError(
                    "real_train requires ss_latent_grid from MeshFleet_TRELLIS ss_latents. "
                    f"batch_keys={keys}, object_id={object_ids}"
                )
            noise = torch.randn_like(x0)
            t = torch.rand(B, device=device)
            t_view = t.view(B, 1, 1, 1, 1)
            x_t = (1 - t_view) * x0 + (sigma_min + (1 - sigma_min) * t_view) * noise
            target_v = (1 - sigma_min) * noise - x0
            voxel_valid_mask = _exact_zero_voxel_mask(x0, args.voxel_prune_epsilon) if args.adaptive_voxel_pruning else None
            cond = _real_condition_or_fail(data_batch, device, cfg, trellis_pipeline)
            if data_batch is not None and geoss_model is not None and vggt is not None:
                geoss_context = _compute_geoss_context(data_batch, geoss_model, vggt)
            else:
                raise RuntimeError("real_train requires real dataset GeoSS context; use --dry_run true for synthetic context.")
            _assert_stage2_batch_contract(data_batch, x0, device)
            t_model = t * 1000.0
            with torch.inference_mode(), torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp and device.type == "cuda"):
                v_base = base(x_t, t_model, cond)
                direct_base = v_base
                identity_error = (v_base - direct_base).abs().max()
            ss_tokens = ss_grid_to_tokens(x_t)
            v_base_tokens = ss_grid_to_tokens(v_base).detach()
            target_residual_tokens = ss_grid_to_tokens(target_v - v_base).detach()
            _assert_stage2_geoss_context(geoss_context, ss_tokens)
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
            # Keep reductions in FP32 while activations stay BF16/FP16.
            token_mask = voxel_valid_mask[..., None] if voxel_valid_mask is not None else None
            effective_mse = _masked_mse(effective_delta_tokens.float(), target_residual_tokens.float(), token_mask)
            raw_mse = _masked_mse(raw_delta_tokens.float(), target_residual_tokens.float(), token_mask)
            mse = effective_mse + args.raw_residual_weight * raw_mse
            vel_reg = velocity_regularization_loss(delta_tokens, t)
            prior = prior_preservation_loss(v_geo_tokens, v_base_tokens, vel["token_confidence"].detach())
            loss = mse + args.velocity_reg_weight * vel_reg + args.prior_weight * prior
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norms = _assert_adapter_gradients(unwrap_model(adapter_model), step)
            scaler.step(opt)
            scaler.update()
            debug = vel["debug"]
            effective_delta_grid = tokens_to_ss_grid(effective_delta_tokens, tuple(x_t.shape[-3:]))
            target_residual_grid = tokens_to_ss_grid(target_residual_tokens, tuple(x_t.shape[-3:]))
            return loss, mse, effective_mse, raw_mse, target_residual_grid, debug, effective_delta_grid, vel_reg, prior, identity_error, grad_norms, memory_start, cuda_memory_watermark(device)

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
        loss, mse, effective_mse, raw_mse, target_residual, debug, effective_delta, vel_reg, prior, identity_error, grad_norms, memory_start, memory_end = retry.value
        if batch_adjustment.changed:
            args.batch_size = batch_adjustment.new_batch_size
            loader, sampler = _build_meshfleet_loader(args, ctx)
            iterator = iter(loader) if loader is not None else None
        last = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "cfm_mse": float(mse.detach().cpu()),
            "loss_effective_residual": float(effective_mse.detach().cpu()),
            "loss_raw_residual": float(raw_mse.detach().cpu()),
            "residual_target_norm": float(target_residual.norm(dim=1).mean().detach().cpu()),
            "residual_base_ratio": float((effective_delta.norm(dim=1).mean() / debug["velocity_base_norm"].clamp_min(1e-6)).detach().cpu()),
            "velocity_regularization": float(vel_reg.detach().cpu()),
            "prior_preservation": float(prior.detach().cpu()),
            "identity_error": float(identity_error.detach().cpu()),
            "velocity_norm": float(debug["velocity_norm"].detach().cpu()),
            "delta_norm": float(effective_delta.norm(dim=1).mean().detach().cpu()),
            "clipping_ratio": float(debug["clipping_ratio"].detach().cpu()),
            "voxel_prune_ratio": float(debug.get("voxel_prune_ratio", torch.zeros((), device=device)).detach().cpu()),
            "mode": _training_mode(args, data_batch is not None),
            **run_modes,
            "rank": ctx.rank,
            "world_size": ctx.world_size,
            "per_gpu_batch_size": args.batch_size,
            "global_batch_size": args.batch_size * ctx.world_size,
            "adapter_grad_norms": grad_norms,
            "spconv": spconv_status,
            "memory_start": memory_start,
            "memory": memory_end,
            "adaptive_batch": {**batch_controller.state_dict(), "last_adjustment": batch_adjustment.as_dict()},
        }
        early_status = early_stopper.update(last)
        last["early_stop"] = early_status.as_dict()
        if ctx.is_main:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(last) + "\n")
        if ctx.is_main and args.save_best and early_status.is_best:
            _save_velocity_checkpoint(out_dir / "ss_velocity_adapter_best.pt", unwrap_model(adapter_model), opt, step, cfg, early_stopper, early_status)
        should_fault_save = args.fault_tolerant_save_every > 0 and step % args.fault_tolerant_save_every == 0
        if ctx.is_main and (should_fault_save or step % args.save_every == 0 or step == end_step):
            _save_velocity_checkpoint(out_dir / "ss_velocity_adapter_last.pt", unwrap_model(adapter_model), opt, step, cfg, early_stopper, early_status)
        if sync_should_stop(early_status.should_stop, device):
            if ctx.is_main:
                _save_velocity_checkpoint(out_dir / "ss_velocity_adapter_last.pt", unwrap_model(adapter_model), opt, step, cfg, early_stopper, early_status)
            break
    return last


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--sigma_min", type=float, default=1e-5)
    parser.add_argument("--velocity_reg_weight", type=float, default=1e-3)
    parser.add_argument("--prior_weight", type=float, default=1e-2)
    parser.add_argument("--raw_residual_weight", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--trellis_root", type=str, default=None)
    parser.add_argument("--trellis_model_path", type=str, default=None)
    parser.add_argument("--meshfleet_root", type=str, default=None)
    parser.add_argument("--meshfleet_split", type=str, default="train")
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
    add_adaptive_batch_args(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    _apply_config_defaults(args, cfg, parser)
    if not args.dry_run:
        summary = run_training(cfg, args)
    else:
        summary = run_dry_run(cfg, args.device)
    rank = getattr(args, "rank", 0)
    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "train_sparse_ray_ss_velocity_dry_run.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    cleanup_distributed()
    if not args.dry_run and rank == 0:
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
    return grad_norms


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
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, D, device=x.device, dtype=dtype),
        torch.linspace(-1, 1, H, device=x.device, dtype=dtype),
        torch.linspace(-1, 1, W, device=x.device, dtype=dtype),
        indexing="ij",
    )
    xyz = torch.stack([xx, yy, zz], dim=-1).reshape(1, D * H * W, 3)
    return xyz.expand(B, -1, -1).contiguous()


@torch.no_grad()
def _compute_geoss_context(batch: dict, geoss_model: SparseRayGeoSSAdapter, vggt: VGGTGeometryWrapper) -> dict:
    batch = dict(batch)
    batch.update(vggt(batch["images"], use_cache=False))
    if getattr(geoss_model, "alignment_enabled", True):
        alignment = geoss_model.geometry_alignment(
            vggt_depth=batch.get("vggt_depth"),
            vggt_pointmap=batch.get("vggt_pointmap"),
            K=batch["K"],
            c2w=batch["c2w"],
            w2c=batch["w2c"],
            masks=batch.get("masks"),
            vggt_confidence=batch.get("vggt_confidence"),
            dataset_depth=batch.get("depths"),
            vggt_camera=batch.get("vggt_camera"),
        )
        batch["aligned_pointmap"] = alignment["aligned_pointmap"]
        batch["aligned_depth"] = alignment["aligned_depth"]
        batch["alignment_confidence"] = alignment["alignment_confidence"]
    B = batch["images"].shape[0]
    device = batch["images"].device
    anchors = geoss_model.anchor_queries.forward_dynamic(
        B,
        device=device,
        aligned_pointmap=batch.get("aligned_pointmap", batch.get("vggt_pointmap")),
        masks=batch.get("masks"),
        confidence=batch.get("alignment_confidence", batch.get("vggt_confidence")),
    )
    anchor_xyz, anchor_feat = anchors["anchor_xyz"], anchors["anchor_feat"]
    ray = geoss_model.ray_sampler(
        anchor_xyz=anchor_xyz,
        K=batch["K"],
        c2w=batch["c2w"],
        w2c=batch["w2c"],
        masks=batch["masks"],
        depths=batch.get("aligned_depth", batch.get("depths")),
        vggt_depth=batch.get("aligned_depth", batch.get("vggt_depth")),
        vggt_pointmap=batch.get("aligned_pointmap", batch.get("vggt_pointmap")),
        vggt_features=batch.get("vggt_features"),
    )
    agg = geoss_model.aggregator(ray["view_tokens"], anchor_feat, ray["ray_valid"], conflict_score=ray.get("conflict_score"))
    return {
        "geo_tokens": agg["geo_tokens"].detach(),
        "geo_confidence": agg["geo_confidence"].detach(),
        "anchor_xyz": anchor_xyz.detach(),
        "anchor_metadata": anchors["anchor_metadata"].detach(),
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


def _save_velocity_checkpoint(path: Path, adapter: SSVelocityAdapter, optimizer, step: int, cfg: dict, early_stopper: EarlyStopper, early_status) -> None:
    save_checkpoint(
        path,
        velocity_adapter=adapter.state_dict(),
        optimizer=optimizer.state_dict(),
        step=step,
        config=cfg,
        early_stop=early_status.as_dict() if early_status is not None else None,
        early_stopper=early_stopper.state_dict(),
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
        "vggt_root": cfg.get("vggt_root") or vggt.get("root"),
        "vggt_checkpoint": cfg.get("vggt_checkpoint") or vggt.get("checkpoint"),
        "vggt_pretrained": cfg.get("vggt_pretrained") or vggt.get("pretrained"),
        "geoss_checkpoint": cfg.get("geoss_checkpoint"),
        "steps": cfg.get("steps"),
        "batch_size": cfg.get("batch_size"),
        "lr": cfg.get("lr"),
        "weight_decay": cfg.get("weight_decay"),
        "sigma_min": cfg.get("sigma_min"),
        "velocity_reg_weight": cfg.get("velocity_reg_weight"),
        "prior_weight": cfg.get("prior_weight"),
        "raw_residual_weight": cfg.get("raw_residual_weight"),
        "save_every": cfg.get("save_every"),
        "output_dir": cfg.get("output_dir"),
        "device": cfg.get("device"),
        **adaptive_config_defaults(cfg),
    }
    for name, value in mappings.items():
        if value is None or not hasattr(args, name):
            continue
        if getattr(args, name) == parser.get_default(name):
            setattr(args, name, value)


def _maybe_launch_stage2(summary: dict, cfg: dict) -> None:
    workflow = cfg.get("workflow", {}) if isinstance(cfg.get("workflow"), dict) else {}
    if not workflow.get("auto_stage2_on_convergence", False):
        return
    early = summary.get("early_stop", {}) if isinstance(summary, dict) else {}
    reason = str(early.get("reason", ""))
    if not early.get("should_stop") or "plateau" not in reason:
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
