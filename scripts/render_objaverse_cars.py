from __future__ import annotations

import argparse
import shutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    args = parser.parse_args()
    if shutil.which("blender") is None:
        raise SystemExit("Objaverse rendering requires Blender on PATH; this preprocessing script is not part of the VGGT+TRELLIS runtime.")
    raise SystemExit("Blender rendering CLI wiring is external preprocessing; use an existing rendered dataset for real_train/real_eval.")


if __name__ == "__main__":
    main()
