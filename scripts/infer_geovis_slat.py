from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from PIL import Image

from geoss.datasets.meshfleet_trellis_dataset import MeshFleetTrellisDataset
from geoss.datasets.vehicle_multiview_dataset import VehicleMultiViewDataset
from geoss.integration.real_trellis_pipeline import RealTrellisGeoPipeline
from geoss.integration.vggt_geometry_wrapper import VGGTGeometryWrapper
from geoss.slat.utils.slat_visualization import save_slat_debug_npz, write_active_voxels_ply
from geoss.utils.config import add_common_args, load_config, str2bool
from geoss.utils.run_mode import validate_real_mode
from scripts.train_geovis_slat import make_synthetic_slat_batch, prepare_batch
from geoss.slat.models.geovis_slat_adapter import GeoVisSLATAdapter


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--active_tokens", type=int, default=512)
    parser.add_argument("--continue_to_decoder", type=str, default="false")
    parser.add_argument("--trellis_root", type=str, default=None)
    parser.add_argument("--trellis_model_path", type=str, default=None)
    parser.add_argument("--vggt_root", type=str, default=None)
    parser.add_argument("--vggt_checkpoint", type=str, default=None)
    parser.add_argument("--vggt_pretrained", type=str, default=None)
    parser.add_argument("--slat_adapter_checkpoint", type=str, default=None)
    parser.add_argument("--meshfleet_root", type=str, default=None)
    parser.add_argument("--meshfleet_split", type=str, default="test")
    parser.add_argument("--meshfleet_category", type=str, default=None)
    parser.add_argument("--meshfleet_index", type=int, default=0)
    parser.add_argument("--meshfleet_slat_latent_model", type=str, default="dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16")
    parser.add_argument("--geovis_context", type=str, default=None)
    parser.add_argument("--geoss_context", type=str, default=None)
    parser.add_argument("--trellis_latents", type=str, default=None, help="Stage-2 trellis_latents.pt; preserves its sparse structure for SLAT.")
    parser.add_argument("--decode", type=str2bool, default=True)
    parser.add_argument("--render_eval", type=str2bool, default=True)
    parser.add_argument("--save_context", type=str2bool, default=True)
    parser.add_argument("--real_infer", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    # Legacy Stage-2 artifacts may contain trellis.modules.sparse.SparseTensor.
    # Make that package importable *before* torch.load resolves its class.
    _configure_trellis_import_path(args.trellis_root)
    try:
        stage2_latents, stage2_status = _load_stage2_latents(args.trellis_latents, device)
    except Exception as exc:
        _write_blocked_metrics(args.output_dir, f"Could not load Stage-2 TRELLIS latents: {exc}", {"used": False})
        return
    model_cfg = dict(cfg.get("model", {}))
    if stage2_latents is not None:
        actual_dim = int(stage2_latents["slat"].shape[-1])
        configured_dim = int(model_cfg.get("slat_dim", actual_dim))
        if configured_dim != actual_dim:
            _write_blocked_metrics(args.output_dir, f"SLAT channel mismatch: checkpoint/config expects {configured_dim}, Stage 2 emitted {actual_dim}.", stage2_status)
            return
    model = GeoVisSLATAdapter(**model_cfg).to(device).eval()
    if args.slat_adapter_checkpoint:
        if not Path(args.slat_adapter_checkpoint).is_file():
            _write_blocked_metrics(args.output_dir, f"SLAT adapter checkpoint not found: {args.slat_adapter_checkpoint}", stage2_status)
            return
        state = torch.load(args.slat_adapter_checkpoint, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)

    if not args.dry_run:
        required = ("vggt", "trellis", "dataset", "decoder") if args.decode else ("vggt", "trellis", "dataset")
        run_modes = validate_real_mode(cfg=cfg, args=args, mode="real_infer", required=required)
        pipe = RealTrellisGeoPipeline(args.trellis_root, args.trellis_model_path, device=args.device)
        vggt_geometry = _load_vggt_geometry(args, device)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        context = _load_context(args.geovis_context, device) if args.geovis_context and Path(args.geovis_context).exists() else None
        batch = None
        if context is None:
            batch = _load_meshfleet_batch(args)
            batch = _move_batch(batch, device)
            batch = prepare_batch(
                batch, cfg, args, device,
                trellis_pipeline=pipe.pipeline,
                vggt_geometry=vggt_geometry,
            )
            with torch.no_grad():
                out = model(batch)
            context = {
                "slat_cond_tokens": out["slat_cond_tokens"].detach().cpu(),
                "slat_confidence": out["slat_confidence"].detach().cpu(),
                "ss_confidence": out["ss_confidence"].detach().cpu(),
                "active_xyz": out["active_xyz"].detach().cpu(),
            }
            torch.save(context, out_dir / "geovis_slat_context.pt")
            save_slat_debug_npz(out_dir / "geovis_controlled_slat.npz", feats=out["v_slat_geo"], indices=batch["ss_active_indices"])
            write_active_voxels_ply(out_dir / "ss_active_voxels.ply", out["active_xyz"][0], out["ss_confidence"][0])
            write_active_voxels_ply(out_dir / "slat_confidence.ply", out["active_xyz"][0], out["slat_confidence"][0])
            save_slat_debug_npz(out_dir / "slat_visibility_debug.npz", visibility=out["visibility"])
            save_slat_debug_npz(out_dir / "view_weights.npz", view_weights=out["view_weights"])
            save_slat_debug_npz(out_dir / "slat_velocity_debug.npz", delta_v_slat=out["delta_v_slat"], clipping_ratio=out["debug"]["clipping_ratio"])
        elif args.save_context:
            torch.save({k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in context.items()}, out_dir / "geovis_slat_context.pt")
        saved_assets = {}
        if args.decode:
            images = _images_from_batch_or_path(batch, args)
            pipe.install_slat_adapter(model.velocity_adapter)
            geoss_context = _load_context(args.geoss_context, device) if args.geoss_context and Path(args.geoss_context).exists() else None
            decoded = pipe.run(
                images,
                geoss_context=geoss_context,
                geovis_slat_context={k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in context.items()},
                # Reuse Stage 2 coordinates so the adapter sees the exact sparse
                # structure that produced trellis_latents.pt instead of resampling it.
                coords_override=stage2_latents["coords"] if stage2_latents is not None else None,
                formats=("gaussian", "mesh"),
            )
            saved_assets = pipe.save_outputs(decoded, out_dir)
        metrics = {
            **run_modes,
            "mode": "real_infer",
            "saved_assets": saved_assets,
            "continue_to_decoder": bool(args.decode),
            "stage2_latents": stage2_status,
            "geovis_context": str(out_dir / "geovis_slat_context.pt") if args.save_context or batch is not None else args.geovis_context,
        }
        if batch is not None:
            metrics.update(
                {
                    "uid": _first_scalar(batch.get("uid")),
                    "slat_confidence_mean": float(out["slat_confidence"].mean().detach().cpu()),
                    "visibility_mean": float(out["visibility"].mean().detach().cpu()),
                    "delta_norm": float(out["delta_v_slat"].norm(dim=-1).mean().detach().cpu()),
                    "slat_base_invalid_ratio": float(batch.get("v_slat_base_invalid_ratio", torch.zeros(())).detach().cpu()),
                }
            )
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return

    batch = make_synthetic_slat_batch(cfg, args, device)
    with torch.no_grad():
        out = model(batch)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_slat_debug_npz(out_dir / "original_slat.npz", feats=batch["slat_latent_tokens"], indices=batch["ss_active_indices"])
    save_slat_debug_npz(out_dir / "geovis_controlled_slat.npz", feats=out["v_slat_geo"], indices=batch["ss_active_indices"])
    write_active_voxels_ply(out_dir / "ss_active_voxels.ply", out["active_xyz"][0], out["ss_confidence"][0])
    write_active_voxels_ply(out_dir / "slat_confidence.ply", out["active_xyz"][0], out["slat_confidence"][0])
    save_slat_debug_npz(out_dir / "slat_visibility_debug.npz", visibility=out["visibility"])
    save_slat_debug_npz(out_dir / "view_weights.npz", view_weights=out["view_weights"])
    save_slat_debug_npz(out_dir / "slat_velocity_debug.npz", delta_v_slat=out["delta_v_slat"], clipping_ratio=out["debug"]["clipping_ratio"])
    metrics = {
        "mode": "dry_run",
        "slat_confidence_mean": float(out["slat_confidence"].mean().cpu()),
        "visibility_mean": float(out["visibility"].mean().cpu()),
        "delta_norm": float(out["delta_v_slat"].norm(dim=-1).mean().cpu()),
        "continue_to_decoder": False,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def _load_images(path: str | None) -> list[Image.Image]:
    if path is None:
        raise FileNotFoundError("real_infer requires --input image file or directory.")
    p = Path(path)
    if p.is_file():
        return [Image.open(p).convert("RGB")]
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"No input images found in {p}")
    return [Image.open(f).convert("RGB") for f in files]


def _load_meshfleet_batch(args: argparse.Namespace) -> dict:
    if not args.meshfleet_root:
        raise FileNotFoundError("real_infer requires --geovis_context or --meshfleet_root to build SLAT context.")
    dataset = MeshFleetTrellisDataset(
        args.meshfleet_root,
        split=args.meshfleet_split,
        category=args.meshfleet_category,
        num_views=args.num_views,
        image_size=args.image_size,
        slat_latent_model=args.meshfleet_slat_latent_model,
        require_slat_latents=True,
    )
    if len(dataset) == 0:
        raise FileNotFoundError(f"No MeshFleet samples found at root={args.meshfleet_root}, split={args.meshfleet_split}.")
    return VehicleMultiViewDataset.collate_fn([dataset[min(args.meshfleet_index, len(dataset) - 1)]])


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _images_from_batch_or_path(batch: dict | None, args: argparse.Namespace) -> list[Image.Image] | torch.Tensor:
    if batch is not None:
        cond = batch.get("trellis_cond_image")
        if isinstance(cond, torch.Tensor):
            return cond
        return batch["images"][:, 0]
    return _load_images(args.input)


def _first_scalar(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _configure_trellis_import_path(trellis_root: str | None) -> None:
    """Expose the local TRELLIS package before unpickling legacy sparse latents."""
    if not trellis_root:
        return
    root = Path(trellis_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"TRELLIS root does not exist: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def _load_vggt_geometry(args: argparse.Namespace, device: torch.device) -> VGGTGeometryWrapper:
    return VGGTGeometryWrapper(
        vggt_root=args.vggt_root,
        checkpoint=args.vggt_checkpoint,
        pretrained_name=args.vggt_pretrained,
        mock=False,
    ).to(device).eval()


def _load_context(path: str, device: torch.device) -> dict:
    p = Path(path)
    if p.suffix.lower() == ".pt":
        obj = torch.load(p, map_location=device)
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obj.items()}
    data = np.load(p)
    required = {"slat_cond_tokens", "slat_confidence", "ss_confidence"}
    missing = required - set(data.keys())
    if missing:
        raise KeyError(f"geovis_context is missing required arrays: {sorted(missing)}")
    return {k: torch.tensor(data[k], device=device).float() for k in required}


def _load_stage2_latents(path: str | None, device: torch.device) -> tuple[dict | None, dict]:
    """Validate the explicit Stage-2→3 contract before TRELLIS allocation."""
    if not path:
        return None, {"used": False, "reason": "not_requested"}
    source = Path(path)
    if not source.is_file():
        return None, {"used": False, "reason": "missing", "path": str(source)}
    try:
        # New artifacts are tensor-only and can use the restricted loader.
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception:
        # Backward compatibility for trusted legacy artifacts containing a
        # TRELLIS SparseTensor; _configure_trellis_import_path ran first.
        payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("coords"), torch.Tensor):
        return None, {"used": False, "reason": "invalid_payload", "path": str(source)}
    coords = payload["coords"]
    slat = payload.get("slat")
    feats = slat.feats if hasattr(slat, "feats") else slat
    if coords.ndim != 2 or coords.shape[1] != 4 or coords.numel() == 0:
        return None, {"used": False, "reason": "invalid_coords", "shape": list(coords.shape), "path": str(source)}
    if not isinstance(feats, torch.Tensor) or feats.ndim != 2 or feats.shape[0] != coords.shape[0] or not torch.isfinite(feats).all():
        return None, {"used": False, "reason": "invalid_slat", "path": str(source)}
    return {
        "coords": coords.to(device=device, dtype=torch.int32).contiguous(),
        "slat": feats.to(device=device, dtype=torch.float32).contiguous(),
    }, {"used": True, "path": str(source), "coords_shape": list(coords.shape), "slat_shape": list(feats.shape)}


def _write_blocked_metrics(output_dir: str, error: str, stage2_status: dict) -> None:
    """Publish actionable diagnostics instead of failing a worker with an opaque returncode=1."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"status": "blocked", "mode": "real_infer", "error": error, "stage2_latents": stage2_status}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
