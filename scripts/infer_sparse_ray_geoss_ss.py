from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from geoss.datasets.meshfleet_trellis_dataset import MeshFleetTrellisDataset
from geoss.datasets.objaverse_cars_rendered_dataset import ObjaverseCarsRenderedDataset
from geoss.datasets.vehicle_multiview_dataset import VehicleMultiViewDataset
from geoss.datasets.vehicle_multiview_dataset import make_dry_run_batch
from geoss.integration.vggt_geometry_wrapper import VGGTGeometryWrapper
from geoss.integration.real_trellis_pipeline import RealTrellisGeoPipeline
from geoss.models.sparse_ray_geoss_adapter import SparseRayGeoSSAdapter
from geoss.utils.config import add_common_args, load_config, str2bool
from geoss.utils.run_mode import validate_real_mode
from geoss.utils.visualization import save_npz, write_point_cloud_ply


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--continue_to_slat", type=str, default="false")
    parser.add_argument("--meshfleet_root", type=str, default=None)
    parser.add_argument("--meshfleet_split", type=str, default="test")
    parser.add_argument("--meshfleet_category", type=str, default=None)
    parser.add_argument("--meshfleet_index", type=int, default=0)
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--meshfleet_occ_resolution", type=int, default=64)
    parser.add_argument("--geoss_checkpoint", type=str, default=None)
    parser.add_argument("--ss_adapter_checkpoint", type=str, default=None)
    parser.add_argument("--vggt_root", type=str, default=None)
    parser.add_argument("--vggt_checkpoint", type=str, default=None)
    parser.add_argument("--vggt_pretrained", type=str, default=None)
    parser.add_argument("--trellis_root", type=str, default=None)
    parser.add_argument("--trellis_model_path", type=str, default=None)
    parser.add_argument("--decode", type=str2bool, default=True)
    parser.add_argument("--render_eval", type=str2bool, default=True)
    parser.add_argument("--save_context", type=str2bool, default=True)
    parser.add_argument("--disable_ss_adapter", type=str2bool, default=False)
    parser.add_argument("--real_infer", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    _apply_config_defaults(args, cfg, parser)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        batch = make_dry_run_batch(batch_size=1, num_views=3, image_size=64, latent_tokens=512, device=args.device)
        vggt = VGGTGeometryWrapper(mock=True).to(args.device)
        mode = "dry_run"
    else:
        required = ("vggt", "trellis", "dataset", "decoder") if args.decode else ("vggt", "trellis", "dataset")
        run_modes = validate_real_mode(cfg=cfg, args=args, mode="real_infer", required=required)
        batch = _load_real_batch(args)
        batch = _move_batch(batch, torch.device(args.device))
        vggt = VGGTGeometryWrapper(
            vggt_root=args.vggt_root,
            checkpoint=args.vggt_checkpoint,
            pretrained_name=args.vggt_pretrained,
            mock=False,
            cache_features=True,
        ).to(args.device)
        mode = batch.get("dataset_name", ["real"])[0] if isinstance(batch.get("dataset_name"), list) else "real"
    batch.update(vggt(batch["images"], use_cache=False))
    model = SparseRayGeoSSAdapter(**cfg.get("model", {})).to(args.device)
    _load_optional_checkpoints(model, args)
    model.eval()
    with torch.no_grad():
        out = model(_context_only_batch(batch))
    geoss_context = {
        "geo_tokens": out["geo_tokens"].detach().cpu(),
        "geo_confidence": out["geo_confidence"].detach().cpu(),
        "anchor_xyz": out["anchor_xyz"].detach().cpu(),
        "anchor_metadata": out["anchor_metadata"].detach().cpu(),
    }
    if args.save_context:
        torch.save(geoss_context, output_dir / "geoss_context.pt")
    if not args.dry_run and args.decode:
        pipe = RealTrellisGeoPipeline(args.trellis_root, args.trellis_model_path, device=args.device)
        if not args.disable_ss_adapter:
            pipe.install_ss_adapter(model.velocity_adapter)
        cond_input = batch.get("trellis_cond_image")
        if not isinstance(cond_input, torch.Tensor):
            cond_input = batch["images"][:, 0]
        decode_context = None if args.disable_ss_adapter else {k: v.to(args.device) for k, v in geoss_context.items()}
        decoded = pipe.run(cond_input, geoss_context=decode_context, formats=("gaussian", "mesh"))
        saved_assets = pipe.save_outputs(decoded, output_dir)
    else:
        run_modes = {"vggt_mode": "mock", "trellis_mode": "mock", "data_mode": "synthetic", "decoder_enabled": False, "render_eval_enabled": False, "official_metrics": False}
        saved_assets = {}
    if "ss_latent_tokens" in batch and "v_base" in batch:
        save_npz(output_dir / "original_trellis_ss.npz", ss_latent_tokens=batch["ss_latent_tokens"], v_base=batch["v_base"])
    if "v_geo" in out:
        save_npz(
            output_dir / "geoss_controlled_ss.npz",
            v_geo=out["v_geo"],
            delta_v_geo=out["delta_v_geo"],
            token_confidence=out["token_confidence"],
        )
    save_npz(output_dir / "occ_evidence.npz", occ_evidence=out["occ_evidence"])
    save_npz(output_dir / "free_evidence.npz", free_evidence=out["free_evidence"])
    if "delta_v_geo" in out:
        save_npz(output_dir / "velocity_debug.npz", delta_v_geo=out["delta_v_geo"], token_confidence=out["token_confidence"])
    write_point_cloud_ply(output_dir / "anchor_xyz.ply", out["anchor_xyz"][0])
    write_point_cloud_ply(output_dir / "anchor_confidence.ply", out["anchor_xyz"][0], out["geo_confidence"][0])
    metrics = {
        "num_anchors": int(out["anchor_xyz"].shape[1]),
        "mean_confidence": float(out["geo_confidence"].mean().detach().cpu()),
        "token_confidence_mean": float(out["token_confidence"].mean().detach().cpu()) if "token_confidence" in out else None,
        "delta_norm": float(out["delta_v_geo"].norm(dim=-1).mean().detach().cpu()) if "delta_v_geo" in out else None,
        "continue_to_slat": False,
        "mode": mode,
        **run_modes,
        "decode_enabled": bool(args.decode),
        "ss_adapter_enabled": bool(args.decode and not args.disable_ss_adapter),
        "geoss_context": str(output_dir / "geoss_context.pt") if args.save_context else None,
        "saved_assets": saved_assets,
    }
    if "gt_occ" in batch:
        metrics["gt_occ_voxels"] = int(batch["gt_occ"].sum().detach().cpu())
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "projection_debug").mkdir(exist_ok=True)
    print(json.dumps(metrics, indent=2))

def _load_real_batch(args: argparse.Namespace) -> dict:
    if args.meshfleet_root:
        dataset = MeshFleetTrellisDataset(
            args.meshfleet_root,
            split=args.meshfleet_split,
            category=args.meshfleet_category,
            num_views=args.num_views,
            image_size=args.image_size,
            occ_resolution=args.meshfleet_occ_resolution,
        )
        if len(dataset) == 0:
            raise FileNotFoundError(
                "No reconstructed MeshFleet_TRELLIS samples found. "
                f"Checked root={args.meshfleet_root}, split={args.meshfleet_split}, category={args.meshfleet_category}."
            )
        return VehicleMultiViewDataset.collate_fn([dataset[min(args.meshfleet_index, len(dataset) - 1)]])
    if args.input:
        obj_dir = Path(args.input)
        if (obj_dir / "transforms.json").exists():
            dataset = ObjaverseCarsRenderedDataset(str(obj_dir.parent), split="inference", num_views=args.num_views, image_size=args.image_size)
            dataset.uids = [obj_dir.name]
            return VehicleMultiViewDataset.collate_fn([dataset[0]])
    raise ValueError("Real inference requires --meshfleet_root or --input pointing to a rendered object folder with transforms.json.")


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def _load_optional_checkpoints(model: SparseRayGeoSSAdapter, args: argparse.Namespace) -> None:
    if args.geoss_checkpoint and Path(args.geoss_checkpoint).exists():
        state = torch.load(args.geoss_checkpoint, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)
    if args.ss_adapter_checkpoint and Path(args.ss_adapter_checkpoint).exists():
        state = torch.load(args.ss_adapter_checkpoint, map_location="cpu")
        adapter_state = state.get("velocity_adapter", state)
        model.velocity_adapter.load_state_dict(adapter_state, strict=False)


def _context_only_batch(batch: dict) -> dict:
    blocked = {"ss_latent_grid", "ss_latent_tokens", "v_base", "timestep"}
    return {key: value for key, value in batch.items() if key not in blocked}


def _apply_config_defaults(args: argparse.Namespace, cfg: dict, parser: argparse.ArgumentParser) -> None:
    if not cfg:
        return
    dataset = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else {}
    trellis = cfg.get("trellis") if isinstance(cfg.get("trellis"), dict) else {}
    vggt = cfg.get("vggt") if isinstance(cfg.get("vggt"), dict) else {}
    mappings = {
        "meshfleet_root": cfg.get("meshfleet_root") or cfg.get("dataset_root") or dataset.get("root"),
        "meshfleet_split": cfg.get("meshfleet_split") or dataset.get("test_split") or dataset.get("split"),
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
        "ss_adapter_checkpoint": cfg.get("ss_adapter_checkpoint"),
        "output_dir": cfg.get("output_dir"),
        "device": cfg.get("device"),
    }
    for name, value in mappings.items():
        if value is None or not hasattr(args, name):
            continue
        if getattr(args, name) == parser.get_default(name):
            setattr(args, name, value)


if __name__ == "__main__":
    main()
