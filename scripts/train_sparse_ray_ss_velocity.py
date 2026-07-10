from __future__ import annotations

import argparse
import json
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
from geoss.integration.trellis_ss_hook import GeoSSTrellisSSWrapper
from geoss.losses.prior_preservation_loss import prior_preservation_loss
from geoss.losses.velocity_loss import velocity_regularization_loss
from geoss.models.sparse_ray_geoss_adapter import SparseRayGeoSSAdapter
from geoss.models.ss_velocity_adapter import SSVelocityAdapter
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
    adapter = SSVelocityAdapter(latent_dim=latent_dim, geo_dim=geo_dim).to(device)
    wrapper = GeoSSTrellisSSWrapper(base, adapter, use_geoss_adapter=True).to(device)
    start_step = 0
    resume_state = None
    if args.resume and Path(args.resume).exists():
        resume_state = torch.load(args.resume, map_location="cpu")
        adapter.load_state_dict(resume_state.get("velocity_adapter", resume_state), strict=False)
        start_step = int(resume_state.get("step", 0))
    wrapper = maybe_wrap_ddp(wrapper, ctx, find_unused_parameters=args.ddp_find_unused_parameters)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
            cond = _real_condition_or_fail(data_batch, device, cfg, trellis_pipeline)
            if data_batch is not None and geoss_model is not None and vggt is not None:
                geoss_context = _compute_geoss_context(data_batch, geoss_model, vggt)
            else:
                raise RuntimeError("real_train requires real dataset GeoSS context; use --dry_run true for synthetic context.")
            t_model = t * 1000.0
            with torch.no_grad():
                v_base = base(x_t, t_model, cond)
                direct_base = v_base
                identity_error = (v_base - direct_base).abs().max()
            v_geo = wrapper(x_t, t_model, cond, geoss_context=geoss_context, use_geoss_adapter=True)
            debug = unwrap_model(wrapper).last_debug
            target_residual = (target_v - v_base).detach()
            mse = F.mse_loss(debug["delta_v_geo"], target_residual)
            delta_tokens = debug["delta_v_geo"].flatten(2).transpose(1, 2)
            v_geo_tokens = v_geo.flatten(2).transpose(1, 2)
            v_base_tokens = v_base.flatten(2).transpose(1, 2)
            vel_reg = velocity_regularization_loss(delta_tokens, t)
            prior = prior_preservation_loss(v_geo_tokens, v_base_tokens, debug["token_confidence"])
            loss = mse + args.velocity_reg_weight * vel_reg + args.prior_weight * prior
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            return loss, mse, target_residual, debug, vel_reg, prior, identity_error

        retry = train_step_with_oom_retry(
            step_fn,
            model=adapter,
            optimizer=opt,
            sampler=sampler,
            device=device,
            batch_controller=batch_controller,
            rebuild_after_adjustment=rebuild_after_adjustment,
            max_retries=getattr(args, "adaptive_oom_retries", 8),
            log_oom=log_oom,
        )
        batch_adjustment = retry.adjustment
        loss, mse, target_residual, debug, vel_reg, prior, identity_error = retry.value
        if batch_adjustment.changed:
            args.batch_size = batch_adjustment.new_batch_size
            loader, sampler = _build_meshfleet_loader(args, ctx)
            iterator = iter(loader) if loader is not None else None
        last = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "cfm_mse": float(mse.detach().cpu()),
            "residual_target_norm": float(target_residual.norm(dim=1).mean().detach().cpu()),
            "residual_base_ratio": float((debug["velocity_delta_norm"] / debug["velocity_base_norm"].clamp_min(1e-6)).detach().cpu()),
            "velocity_regularization": float(vel_reg.detach().cpu()),
            "prior_preservation": float(prior.detach().cpu()),
            "identity_error": float(identity_error.detach().cpu()),
            "velocity_norm": float(debug["velocity_norm"].detach().cpu()),
            "delta_norm": float(debug["velocity_delta_norm"].detach().cpu()),
            "clipping_ratio": float(debug["clipping_ratio"].detach().cpu()),
            "mode": _training_mode(args, data_batch is not None),
            **run_modes,
            "rank": ctx.rank,
            "world_size": ctx.world_size,
            "per_gpu_batch_size": args.batch_size,
            "global_batch_size": args.batch_size * ctx.world_size,
            "adaptive_batch": {**batch_controller.state_dict(), "last_adjustment": batch_adjustment.as_dict()},
        }
        early_status = early_stopper.update(last)
        last["early_stop"] = early_status.as_dict()
        if ctx.is_main:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(last) + "\n")
        if ctx.is_main and args.save_best and early_status.is_best:
            _save_velocity_checkpoint(out_dir / "ss_velocity_adapter_best.pt", adapter, opt, step, cfg, early_stopper, early_status)
        should_fault_save = args.fault_tolerant_save_every > 0 and step % args.fault_tolerant_save_every == 0
        if ctx.is_main and (should_fault_save or step % args.save_every == 0 or step == end_step):
            _save_velocity_checkpoint(out_dir / "ss_velocity_adapter_last.pt", adapter, opt, step, cfg, early_stopper, early_status)
        if sync_should_stop(early_status.should_stop, device):
            if ctx.is_main:
                _save_velocity_checkpoint(out_dir / "ss_velocity_adapter_last.pt", adapter, opt, step, cfg, early_stopper, early_status)
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
