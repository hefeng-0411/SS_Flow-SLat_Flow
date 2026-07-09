from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from PIL import Image

from geoss.integration.real_trellis_pipeline import RealTrellisGeoPipeline
from geoss.slat.utils.slat_visualization import save_slat_debug_npz, write_active_voxels_ply
from geoss.utils.config import add_common_args, load_config
from geoss.utils.run_mode import validate_real_mode
from scripts.train_geovis_slat import make_synthetic_slat_batch
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
    parser.add_argument("--slat_adapter_checkpoint", type=str, default=None)
    parser.add_argument("--geovis_context", type=str, default=None)
    parser.add_argument("--real_infer", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    model = GeoVisSLATAdapter(**cfg.get("model", {})).to(device).eval()
    if args.slat_adapter_checkpoint:
        state = torch.load(args.slat_adapter_checkpoint, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)

    if not args.dry_run:
        run_modes = validate_real_mode(cfg=cfg, args=args, mode="real_infer", required=("trellis", "dataset", "decoder"))
        if not args.geovis_context or not Path(args.geovis_context).exists():
            raise FileNotFoundError("real_infer requires --geovis_context produced by the SS/GeoVis context stage; no zero-context fallback is allowed.")
        images = _load_images(args.input)
        context = _load_context(args.geovis_context, device)
        pipe = RealTrellisGeoPipeline(args.trellis_root, args.trellis_model_path, device=args.device)
        pipe.install_slat_adapter(model.velocity_adapter)
        decoded = pipe.run(images, geovis_slat_context=context, formats=("gaussian", "mesh"))
        out_dir = Path(args.output_dir)
        saved_assets = pipe.save_outputs(decoded, out_dir)
        metrics = {**run_modes, "mode": "real_infer", "saved_assets": saved_assets, "continue_to_decoder": True}
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


if __name__ == "__main__":
    main()
