from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_record(path: Path, stage: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{stage} checkpoint does not exist: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "step" not in state:
        raise RuntimeError(f"{stage} checkpoint lacks a top-level step: {path}")
    expected_weights = "velocity_adapter" if stage == "SS" else "model"
    if expected_weights not in state:
        raise RuntimeError(
            f"{stage} checkpoint lacks {expected_weights!r} weights: {path}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "step": int(state["step"]),
        "training_budget": state.get("training_budget"),
        "support_provenance": state.get("support_provenance"),
        "tensor_contract": state.get("tensor_contract"),
    }


def build_pair_manifest(ss_checkpoint: Path, slat_checkpoint: Path) -> dict:
    ss = _checkpoint_record(ss_checkpoint, "SS")
    slat = _checkpoint_record(slat_checkpoint, "SLAT")
    slat_support = slat.get("support_provenance") or {}
    declared_upstream = slat_support.get("upstream_ss_checkpoint")
    connected = bool(
        declared_upstream
        and Path(str(declared_upstream)).expanduser().resolve()
        == ss_checkpoint.expanduser().resolve()
    )
    return {
        "protocol_version": "ss_slat_checkpoint_pair_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "topology": "sequential_discrete_support_boundary",
        "ss": ss,
        "slat": slat,
        "coupling": {
            "slat_declares_this_ss_as_upstream": connected,
            "cross_stage_gradient": "absent",
            "status": (
                "connected_sequential_pair"
                if connected
                else "diagnostic_pair_only_slat_was_teacher_support_trained"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a hash-addressed SS/SLAT checkpoint-pair manifest."
    )
    parser.add_argument("--ss_checkpoint", type=Path, required=True)
    parser.add_argument("--slat_checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require_connected",
        action="store_true",
        help="Fail unless the SLAT checkpoint provenance names this exact SS checkpoint.",
    )
    args = parser.parse_args()
    manifest = build_pair_manifest(args.ss_checkpoint, args.slat_checkpoint)
    if args.require_connected and not manifest["coupling"]["slat_declares_this_ss_as_upstream"]:
        raise RuntimeError(manifest["coupling"]["status"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
