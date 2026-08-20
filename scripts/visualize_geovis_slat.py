from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from geoss.slat.utils.slat_visualization import write_active_voxels_ply


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="outputs/geovis_slat_infer_dry")
    parser.add_argument("--output_dir", type=str, default="outputs/geovis_slat_vis")
    args = parser.parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    conf_path = in_dir / "slat_visibility_debug.npz"
    slat_path = in_dir / "original_slat.npz"
    if conf_path.exists() and slat_path.exists():
        visibility = torch.from_numpy(np.load(conf_path)["visibility"]).float()
        indices = torch.from_numpy(np.load(slat_path)["indices"]).long()
        from geoss.slat.utils.active_voxel_utils import indices_to_active_xyz

        xyz = indices_to_active_xyz(indices[0], 64)
        write_active_voxels_ply(out_dir / "visibility_mean.ply", xyz, visibility[0].mean(dim=1))
        summary["visibility_mean_ply"] = str(out_dir / "visibility_mean.ply")
    else:
        summary["warning"] = "missing infer outputs"
    (out_dir / "visualize_geovis_slat_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
