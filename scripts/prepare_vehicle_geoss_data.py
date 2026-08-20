from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--srn_root", type=str, default=None)
    parser.add_argument("--objaverse_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/prepared_vehicle_geoss")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"srn_root": args.srn_root, "objaverse_root": args.objaverse_root, "status": "manifest_only"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
