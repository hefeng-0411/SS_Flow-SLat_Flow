from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from geoss.utils.voxelization import voxelize_mesh_to_occ


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--output", type=str, default="outputs/gt_occ.pt")
    args = parser.parse_args()
    occ = voxelize_mesh_to_occ(args.mesh, resolution=args.resolution)
    if occ is None:
        raise RuntimeError("Could not voxelize mesh; install trimesh and verify mesh path.")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"gt_occ": occ}, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
