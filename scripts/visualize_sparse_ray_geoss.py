from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from geoss.utils.config import add_common_args
from geoss.utils.visualization import write_point_cloud_ply


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    xyz = torch.rand(512, 3) * 2 - 1
    conf = torch.rand(512)
    write_point_cloud_ply(out / "visualize_sparse_ray_geoss_demo.ply", xyz, conf)
    print(str(out / "visualize_sparse_ray_geoss_demo.ply"))


if __name__ == "__main__":
    main()
