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
from geoss.geometry.alignment import align_vggt_batch
from geoss.slat.integration.ss_slat_context import build_ss_slat_context
from geoss.slat.utils.normalization import SLAT_TENSOR_CONTRACT_VERSION, normalize_slat
from geoss.slat.utils.slat_visualization import save_slat_debug_npz, write_active_voxels_ply
from geoss.utils.config import add_common_args, load_config, str2bool
from geoss.utils.run_mode import validate_real_mode
from scripts.train_geovis_slat import make_synthetic_slat_batch
from geoss.slat.models.geovis_slat_adapter import GeoVisSLATAdapter


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--active_tokens", type=int, default=0, help="Maximum SLAT tokens; 0 keeps every active voxel.")
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
    parser.add_argument("--meshfleet_uid", type=str, default=None, help="Exact UID; preferred over layout-dependent --meshfleet_index.")
    parser.add_argument("--meshfleet_slat_latent_model", type=str, default="dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16")
    parser.add_argument("--geovis_context", type=str, default=None)
    parser.add_argument("--geoss_context", type=str, default=None)
    parser.add_argument("--trellis_latents", type=str, default=None, help="Stage-2 trellis_latents.pt; preserves its sparse structure for SLAT.")
    parser.add_argument("--decode", type=str2bool, default=True)
    parser.add_argument("--export_textured_glb", type=str2bool, default=False)
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
    checkpoint_status = {"used": False}
    if args.slat_adapter_checkpoint:
        if not Path(args.slat_adapter_checkpoint).is_file():
            _write_blocked_metrics(args.output_dir, f"SLAT adapter checkpoint not found: {args.slat_adapter_checkpoint}", stage2_status)
            return
        try:
            checkpoint_status = _load_adapter_checkpoint(model, args.slat_adapter_checkpoint, model_cfg)
        except Exception as exc:
            _write_blocked_metrics(args.output_dir, f"SLAT adapter checkpoint contract failed: {exc}", stage2_status)
            return

    if not args.dry_run:
        required = ("vggt", "trellis", "dataset", "decoder") if args.decode else ("vggt", "trellis", "dataset")
        run_modes = validate_real_mode(cfg=cfg, args=args, mode="real_infer", required=required)
        pipe = RealTrellisGeoPipeline(args.trellis_root, args.trellis_model_path, device=args.device)
        vggt_geometry = _load_vggt_geometry(args, device)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        context = _load_context(args.geovis_context, device) if args.geovis_context and Path(args.geovis_context).exists() else None
        geoss_context = _load_context(args.geoss_context, device) if args.geoss_context and Path(args.geoss_context).exists() else None
        context_coords = _context_sparse_coords(context, device) if context is not None else None
        batch = None
        if context is None:
            if stage2_latents is None:
                _write_blocked_metrics(
                    args.output_dir,
                    "Leakage-free GeoVis inference requires --trellis_latents from Stage 2; ground-truth MeshFleet latents are forbidden.",
                    stage2_status,
                )
                return
            batch = _inference_only_batch(_load_meshfleet_batch(args))
            batch = _move_batch(batch, device)
            batch = _prepare_inference_context_batch(
                batch,
                cfg,
                stage2_latents,
                geoss_context,
                vggt_geometry=vggt_geometry,
                slat_normalization=pipe.pipeline.slat_normalization,
            )
            with torch.no_grad():
                out = model(batch)
            context = {
                "slat_cond_tokens": out["slat_cond_tokens"].detach().cpu(),
                "slat_confidence": out["slat_confidence"].detach().cpu(),
                "correction_demand": out["correction_demand"].detach().cpu(),
                "residual_variance": out["residual_variance"].detach().cpu(),
                "ss_confidence": out["ss_confidence"].detach().cpu(),
                "active_xyz": out["active_xyz"].detach().cpu(),
                "ss_active_indices": batch["ss_active_indices"].detach().cpu(),
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
            coords_override = stage2_latents["coords"] if stage2_latents is not None else context_coords
            if coords_override is None:
                _write_blocked_metrics(
                    args.output_dir,
                    "Decoded GeoVis inference requires Stage-2 coordinates or an index-bearing GeoVis context.",
                    stage2_status,
                )
                return
            images = _images_from_batch_or_path(batch, args)
            pipe.install_slat_adapter(model.velocity_adapter)
            decoded = pipe.run(
                images,
                geoss_context=geoss_context,
                geovis_slat_context={k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in context.items()},
                # Reuse Stage 2 coordinates so the adapter sees the exact sparse
                # structure that produced trellis_latents.pt instead of resampling it.
                coords_override=coords_override,
                formats=("gaussian", "mesh"),
            )
            saved_assets = pipe.save_outputs(decoded, out_dir, export_textured_glb=args.export_textured_glb)
            _require_decoded_assets(saved_assets, out_dir)
        metrics = {
            **run_modes,
            "mode": "real_infer",
            "saved_assets": saved_assets,
            "continue_to_decoder": bool(args.decode),
            "stage2_latents": stage2_status,
            "adapter_checkpoint": checkpoint_status,
            "geovis_context": str(out_dir / "geovis_slat_context.pt") if args.save_context or batch is not None else args.geovis_context,
            "test_time_ground_truth_latents_used": False if batch is not None else None,
            "inference_context_source": batch.get("inference_context_source") if batch is not None else "external_context",
        }
        if batch is not None:
            visibility_mean = float(out["visibility"].mean().detach().cpu())
            confidence_mean = float(out["slat_confidence"].mean().detach().cpu())
            projection_valid_ratio = float(out["debug"]["projection"]["valid_ratio"].detach().cpu())
            evidence_valid = visibility_mean > 1e-6 and confidence_mean > 1e-6 and projection_valid_ratio > 1e-6
            metrics.update(
                {
                    "uid": _first_scalar(batch.get("uid")),
                    "slat_confidence_mean": confidence_mean,
                    "visibility_mean": visibility_mean,
                    "projection_valid_ratio": projection_valid_ratio,
                    # A decoded asset can be valid while the learned guidance is
                    # inactive. Publish that distinction for scientific ablation
                    # analysis instead of treating a zero-evidence pass as proof
                    # that the adapter improved TRELLIS.
                    "adapter_evidence_valid": evidence_valid,
                    "evaluation_warnings": [] if evidence_valid else ["zero_or_invalid_geovis_evidence"],
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
        return [Image.open(p).convert("RGBA")]
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"No input images found in {p}")
    return [Image.open(f).convert("RGBA") for f in files]


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
        require_slat_latents=False,
        uid_manifest=[args.meshfleet_uid] if args.meshfleet_uid else None,
    )
    if len(dataset) == 0:
        raise FileNotFoundError(f"No MeshFleet samples found at root={args.meshfleet_root}, split={args.meshfleet_split}.")
    if args.meshfleet_uid:
        sample = dataset.get_by_uid(args.meshfleet_uid)
    else:
        if args.meshfleet_index < 0 or args.meshfleet_index >= len(dataset):
            raise IndexError(f"meshfleet_index={args.meshfleet_index} outside [0, {len(dataset) - 1}]")
        sample = dataset[args.meshfleet_index]
    return VehicleMultiViewDataset.collate_fn([sample])


def _inference_only_batch(batch: dict) -> dict:
    """Remove MeshFleet 3D supervision before any inference component runs."""
    blocked = {
        "gt_occ",
        "mesh_path",
        "ss_latent_grid",
        "ss_latent_tokens",
        "trellis_slat_feats",
        "trellis_slat_indices",
        "v_base",
        "v_slat_base",
        "timestep",
    }
    return {key: value for key, value in batch.items() if key not in blocked}


def _prepare_inference_context_batch(
    batch: dict,
    cfg: dict,
    stage2_latents: dict,
    geoss_context: dict | None,
    *,
    vggt_geometry: VGGTGeometryWrapper,
    slat_normalization: dict,
) -> dict:
    """Build test-time SLAT evidence without reading ground-truth 3D latents."""
    with torch.inference_mode():
        vggt_context = vggt_geometry(batch["images"])
    for key in ("vggt_features", "vggt_depth", "vggt_pointmap", "vggt_confidence", "vggt_camera"):
        value = vggt_context.get(key)
        if isinstance(value, (torch.Tensor, dict)):
            batch[key] = value
    batch = align_vggt_batch(batch)
    coords = stage2_latents["coords"]
    feats_raw = stage2_latents["slat"]
    if coords.ndim != 2 or coords.shape[-1] != 4 or not torch.all(coords[:, 0] == 0):
        raise ValueError("Single-object inference requires Stage-2 coords [T,4] with batch column equal to zero.")
    if feats_raw.ndim != 2 or feats_raw.shape[0] != coords.shape[0]:
        raise ValueError("Stage-2 SLAT features must align one-to-one with predicted sparse coordinates.")
    slat_dim = int(cfg.get("model", {}).get("slat_dim", feats_raw.shape[-1]))
    if feats_raw.shape[-1] != slat_dim:
        raise ValueError(f"Stage-2 SLAT width {feats_raw.shape[-1]} does not match adapter width {slat_dim}.")
    feats = normalize_slat(feats_raw, slat_normalization)
    active_indices = coords[:, 1:].long().unsqueeze(0)
    context = build_ss_slat_context(
        ss_active_indices=active_indices,
        # Stage-3 training does not expose Stage-1 anchor embeddings to this
        # aggregator. Injecting 256-D GeoSS tokens into its 8-D projection at
        # test time is both a shape error and a train/inference distribution
        # shift; geometry already enters through Stage-2 predicted coords.
        geoss_output=None,
        resolution=int(cfg.get("model", {}).get("resolution", 64)),
        target_dim=slat_dim,
    )
    batch.update(context)
    batch["slat_latent_tokens"] = feats.float().unsqueeze(0)
    batch["slat_reference_tokens"] = batch["slat_latent_tokens"]
    batch["v_slat_base"] = torch.zeros_like(batch["slat_latent_tokens"])
    batch["timestep"] = torch.zeros(1, device=feats.device, dtype=torch.float32)
    batch["slat_token_valid_mask"] = torch.ones(1, feats.shape[0], 1, device=feats.device, dtype=torch.float32)
    batch["inference_context_source"] = "stage2_predicted_coords_and_slat"
    batch["geoss_context_available"] = geoss_context is not None
    return batch


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _images_from_batch_or_path(batch: dict | None, args: argparse.Namespace) -> list[Image.Image] | torch.Tensor:
    if batch is not None:
        images = batch["images"]
        if not isinstance(images, torch.Tensor) or images.ndim != 5 or images.shape[0] != 1:
            raise ValueError(f"Expected one multi-view object [1,N,3,H,W], got {type(images)!r} {getattr(images, 'shape', None)}")
        return images[0]
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
        obj = _trusted_tensor_load(p, map_location=device)
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obj.items()}
    data = np.load(p)
    required = {"slat_cond_tokens", "slat_confidence", "ss_confidence"}
    missing = required - set(data.keys())
    if missing:
        raise KeyError(f"geovis_context is missing required arrays: {sorted(missing)}")
    keys = required | ({"ss_active_indices"} if "ss_active_indices" in data else set())
    return {
        key: torch.tensor(data[key], device=device).long() if key == "ss_active_indices" else torch.tensor(data[key], device=device).float()
        for key in keys
    }


def _context_sparse_coords(context: dict | None, device: torch.device) -> torch.Tensor | None:
    if not isinstance(context, dict) or not isinstance(context.get("ss_active_indices"), torch.Tensor):
        return None
    indices = context["ss_active_indices"].to(device=device)
    if indices.ndim != 3 or indices.shape[0] != 1 or indices.shape[-1] != 3:
        raise ValueError(f"GeoVis context ss_active_indices must be [1,L,3], got {tuple(indices.shape)}")
    batch_column = torch.zeros(indices.shape[1], 1, device=device, dtype=torch.int32)
    return torch.cat([batch_column, indices[0].to(torch.int32)], dim=-1).contiguous()


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


def _trusted_tensor_load(path: Path, *, map_location):
    """Prefer PyTorch's restricted loader for local tensor-only evaluation artifacts."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch versions before the weights_only keyword.
        return torch.load(path, map_location=map_location)


def _load_adapter_checkpoint(model: GeoVisSLATAdapter, checkpoint_path: str, model_cfg: dict) -> dict:
    """Load exactly the trained adapter architecture; never evaluate partial weights."""
    path = Path(checkpoint_path)
    payload = _trusted_tensor_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"expected checkpoint dictionary, got {type(payload).__name__}")
    state = payload.get("model", payload)
    if not isinstance(state, dict):
        raise TypeError("checkpoint 'model' entry is not a state dictionary")
    contract = payload.get("tensor_contract")
    version = contract.get("version") if isinstance(contract, dict) else None
    if version != SLAT_TENSOR_CONTRACT_VERSION:
        raise RuntimeError(
            f"checkpoint tensor contract is {version!r}; expected {SLAT_TENSOR_CONTRACT_VERSION!r}"
        )
    checkpoint_model_cfg = payload.get("config", {}).get("model", {}) if isinstance(payload.get("config", {}), dict) else {}
    for key in (
        "slat_dim", "resolution", "evidence_dim", "hidden_dim", "feature_dim", "num_heads",
        "fusion_mode", "confidence_floor", "trust_region", "beta_mode", "beta_strength",
        "factorized_control", "use_geovis_slat",
    ):
        if key in checkpoint_model_cfg and key in model_cfg and checkpoint_model_cfg[key] != model_cfg[key]:
            raise RuntimeError(f"{key}: checkpoint={checkpoint_model_cfg[key]!r}, inference_config={model_cfg[key]!r}")
    # ``strict=True`` is essential: strict=False can turn an architecture drift
    # into a plausible-looking but scientifically invalid evaluation result.
    model.load_state_dict(state, strict=True)
    return {
        "used": True,
        "path": str(path),
        "step": payload.get("step"),
        "slat_dim": int(model.velocity_adapter.slat_dim),
        "tensor_contract": payload.get("tensor_contract"),
    }


def _require_decoded_assets(saved_assets: dict, out_dir: Path) -> None:
    """A successful Stage-3/4 evaluation must produce a renderable Gaussian asset."""
    gaussian = saved_assets.get("gaussian_ply")
    if not gaussian or not Path(gaussian).is_file():
        raise RuntimeError(f"TRELLIS decode completed without asset_gaussian.ply in {out_dir}")


if __name__ == "__main__":
    main()
