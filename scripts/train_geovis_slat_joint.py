from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoss.utils.config import add_common_args, load_config
from geoss.utils.distributed import cleanup_distributed
from geoss.utils.adaptive_batch import add_adaptive_batch_args
from scripts.train_geovis_slat import _apply_config_defaults, run_dry_run, run_training


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--active_tokens", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--velocity_weight", type=float, default=1e-3)
    parser.add_argument("--prior_weight", type=float, default=1e-2)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--visualize_every", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--joint_enabled", type=str, default="true")
    parser.add_argument("--meshfleet_root", type=str, default=None)
    parser.add_argument("--meshfleet_split", type=str, default="train")
    parser.add_argument("--meshfleet_category", type=str, default=None)
    parser.add_argument("--meshfleet_slat_latent_model", type=str, default="dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16")
    parser.add_argument("--srn_root", type=str, default=None)
    parser.add_argument("--objaverse_rendered_root", type=str, default=None)
    parser.add_argument("--trellis_root", type=str, default=None)
    parser.add_argument("--trellis_model_path", type=str, default=None)
    add_adaptive_batch_args(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    _apply_config_defaults(args, cfg, parser)
    summary = run_dry_run(cfg, args) if args.dry_run else run_training(cfg, args)
    summary["joint_mode"] = "adapter_only_joint_finetune_sanity"
    if getattr(args, "rank", 0) == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "train_geovis_slat_joint_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    cleanup_distributed()


if __name__ == "__main__":
    main()
