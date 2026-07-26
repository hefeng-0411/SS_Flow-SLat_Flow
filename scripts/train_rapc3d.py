from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoss.datasets.meshfleet_trellis_dataset import MeshFleetTrellisDataset
from geoss.integration.real_trellis_pipeline import RealTrellisGeoPipeline
from geoss.integration.vggt_geometry_wrapper import VGGTGeometryWrapper
from geoss.losses.render_losses import build_lpips
from geoss.reconstruction.models import RAPC3D, RAPC3DConfig
from geoss.reconstruction.objectives import RAPC3DObjective
from geoss.reconstruction.priors import ConditioningTrellisPriorProvider
from geoss.reconstruction.training import RAPCTrainer, RAPCTrainerConfig, collate_rapc_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train observation-authoritative RAPC-3D.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--validation_manifest")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--field_resolution", type=int, default=32)
    parser.add_argument("--posterior_iterations", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--render_pixel_stride", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--vggt_root", required=True)
    parser.add_argument("--vggt_checkpoint")
    parser.add_argument("--vggt_pretrained_name", default="facebook/VGGT-1B")
    parser.add_argument("--trellis_root")
    parser.add_argument("--trellis_model_path")
    parser.add_argument("--trellis_prior_cache")
    parser.add_argument("--trellis_prior_cache_only", action="store_true")
    parser.add_argument("--disable_trellis_prior", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--save_every_steps", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = _initialize_distributed()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "rapc3d_train_config.json").write_text(
            json.dumps(vars(args), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    train_dataset = _make_dataset(args, args.train_manifest)
    if args.max_train_samples > 0:
        train_dataset.samples = train_dataset.samples[: args.max_train_samples]
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        if world_size > 1
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_rapc_samples,
        drop_last=False,
    )
    validation_loader = _make_validation_loader(args, world_size, rank)

    core_model = RAPC3D(
        RAPC3DConfig(
            field_resolution=args.field_resolution,
            posterior_iterations=args.posterior_iterations,
        )
    ).to(device)
    if args.disable_trellis_prior:
        # The no-prior ablation uses the analytic weak fallback, so none of
        # the adapter parameters participate in forward.  Freezing them keeps
        # DDP's reducer contract exact without enabling the costly global
        # unused-parameter graph traversal.
        core_model.prior_adapter.requires_grad_(False)
    model = (
        DistributedDataParallel(
            core_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        if world_size > 1
        else core_model
    )
    vggt = VGGTGeometryWrapper(
        vggt_root=args.vggt_root,
        checkpoint=args.vggt_checkpoint,
        pretrained_name=args.vggt_pretrained_name if args.vggt_checkpoint is None else None,
        mock=False,
        cache_features=False,
    ).to(device)
    lpips_model = build_lpips(real_mode=True).to(device).eval()
    for parameter in lpips_model.parameters():
        parameter.requires_grad_(False)
    trainer = RAPCTrainer(
        model,
        vggt,
        RAPC3DObjective(perceptual_metric=lpips_model),
        config=RAPCTrainerConfig(
            learning_rate=args.learning_rate,
            render_pixel_stride=args.render_pixel_stride,
        ),
        prior_provider=_make_prior_provider(args, core_model, device),
    )
    start_epoch, global_step, best_validation = _resume(args, core_model, trainer, device)

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            metrics = trainer.train_step(_to_device(batch, device))
            global_step += 1
            if rank == 0:
                _append_metrics(output_dir / "train_metrics.jsonl", epoch, global_step, metrics)
                if global_step % args.save_every_steps == 0:
                    _save_checkpoint(
                        output_dir / f"rapc3d_step_{global_step:08d}.pt",
                        core_model,
                        trainer,
                        epoch,
                        global_step,
                        best_validation,
                    )
        if validation_loader is not None:
            validation = _distributed_mean(
                _validate(trainer, validation_loader, device),
                device,
                world_size,
            )
            if rank == 0:
                _append_metrics(
                    output_dir / "validation_metrics.jsonl",
                    epoch,
                    global_step,
                    {"total": torch.tensor(validation)},
                )
                if validation < best_validation:
                    best_validation = validation
                    _save_checkpoint(
                        output_dir / "rapc3d_best.pt",
                        core_model,
                        trainer,
                        epoch + 1,
                        global_step,
                        best_validation,
                    )
        if rank == 0:
            _save_checkpoint(
                output_dir / "rapc3d_latest.pt",
                core_model,
                trainer,
                epoch + 1,
                global_step,
                best_validation,
            )
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _make_dataset(args, manifest: str) -> MeshFleetTrellisDataset:
    return MeshFleetTrellisDataset(
        args.data_root,
        split=args.split,
        num_views=args.num_views,
        image_size=args.image_size,
        uid_manifest=manifest,
        require_voxels=False,
        require_ss_latents=False,
        require_slat_latents=False,
        require_features=False,
        strict_uid_manifest=True,
        load_3d_modalities=True,
    )


def _make_validation_loader(args, world_size: int, rank: int):
    if not args.validation_manifest:
        return None
    dataset = _make_dataset(args, args.validation_manifest)
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_rapc_samples,
    )


def _make_prior_provider(args, model: RAPC3D, device):
    if args.disable_trellis_prior:
        return None
    if not args.trellis_root or not args.trellis_model_path:
        raise ValueError(
            "Final training requires --trellis_root and --trellis_model_path; "
            "--disable_trellis_prior is reserved for the no-prior ablation."
        )
    pipeline = RealTrellisGeoPipeline(
        args.trellis_root,
        args.trellis_model_path,
        device=str(device),
    )
    return ConditioningTrellisPriorProvider(
        pipeline,
        cache_directory=args.trellis_prior_cache,
        cache_only=args.trellis_prior_cache_only,
        seed=args.seed,
        dense_resolution=model.config.field_resolution,
    )


def _resume(args, model, trainer, device):
    if not args.resume:
        return 0, 0, float("inf")
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = int(checkpoint.get("global_step", 0))
    trainer.step_index = global_step
    return (
        int(checkpoint.get("epoch", 0)),
        global_step,
        float(checkpoint.get("best_validation", float("inf"))),
    )


def _initialize_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


@torch.no_grad()
def _validate(trainer, loader, device):
    total = 0.0
    count = 0
    for batch in loader:
        metrics = trainer.validation_step(_to_device(batch, device))
        total += float(metrics["total"].detach())
        count += 1
    return total / max(count, 1)


def _distributed_mean(value: float, device, world_size: int) -> float:
    tensor = torch.tensor([value, 1.0], device=device)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor[0] / tensor[1].clamp_min(1.0))


def _append_metrics(path: Path, epoch: int, step: int, metrics: Dict[str, torch.Tensor]) -> None:
    row = {"epoch": epoch, "step": step}
    row.update(
        {
            name: float(value.detach().float().mean())
            for name, value in metrics.items()
            if isinstance(value, torch.Tensor) and value.numel() > 0
        }
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _save_checkpoint(path, model, trainer, epoch, global_step, best_validation):
    torch.save(
        {
            "architecture": "RAPC-3D-v1",
            "model": model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_validation": best_validation,
        },
        path,
    )


if __name__ == "__main__":
    main()
