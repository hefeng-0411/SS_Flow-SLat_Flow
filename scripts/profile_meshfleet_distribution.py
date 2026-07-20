from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.inspect_meshfleet_dataset import RENDER_SETS, _dataset_roots, _modality_maps, _resolve_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile MeshFleet view, camera, latent, voxel, and duplicate distributions.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default="outputs/dataset_profile")
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--uid_manifest", default=None)
    parser.add_argument("--images_per_set", type=int, default=8)
    parser.add_argument("--hash_meshes", action="store_true")
    parser.add_argument("--ss_latent_model", default="ss_enc_conv3d_16l8_fp16")
    parser.add_argument("--slat_latent_model", default="dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16")
    parser.add_argument("--feature_model", default="dinov2_vitl14_reg")
    args = parser.parse_args()
    root = Path(args.data_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    allowed = set(_manifest_uids(args.uid_manifest)) if args.uid_manifest else None
    rows: list[dict[str, Any]] = []
    mesh_hash_to_uids: dict[str, list[str]] = defaultdict(list)
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        for category, layout in _dataset_roots(root / split):
            maps = _modality_maps(
                layout,
                ss_latent_model=args.ss_latent_model,
                slat_latent_model=args.slat_latent_model,
                feature_model=args.feature_model,
            )
            uids = sorted(set().union(*(set(mapping) for mapping in maps.values())))
            for uid in uids:
                if allowed is not None and uid not in allowed:
                    continue
                row = _profile_uid(uid, split, category, maps, args.images_per_set)
                mesh = maps["mesh"].get(uid)
                if args.hash_meshes and mesh is not None:
                    digest = _sha256(mesh)
                    row["mesh_sha256"] = digest
                    mesh_hash_to_uids[digest].append(f"{split}:{uid}")
                rows.append(row)
    duplicates = {digest: values for digest, values in mesh_hash_to_uids.items() if len(values) > 1}
    summary = {
        "protocol": "meshfleet_distribution_profile_v1",
        "data_root": str(root),
        "objects": len(rows),
        "splits": dict(Counter(row["split"] for row in rows)),
        "preprocessing_models": {
            "ss_latents": args.ss_latent_model,
            "latents": args.slat_latent_model,
            "features": args.feature_model,
        },
        "numeric": _numeric_summary(rows),
        "shape_histograms": _shape_histograms(rows),
        "mesh_duplicate_groups": duplicates,
        "mesh_duplicate_object_count": sum(len(values) for values in duplicates.values()),
        "failed_profiles": [row for row in rows if row.get("profile_errors")],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "objects.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    _write_csv(output / "objects.csv", rows)
    print(json.dumps({"objects": len(rows), "failed": len(summary["failed_profiles"]), "duplicate_groups": len(duplicates)}, indent=2))


def _profile_uid(uid: str, split: str, category: str, maps: dict[str, dict[str, Path]], images_per_set: int) -> dict[str, Any]:
    row: dict[str, Any] = {"uid": uid, "split": split, "category": category, "profile_errors": []}
    for render_set in RENDER_SETS:
        folder = maps[render_set].get(uid)
        try:
            row.update(_render_stats(folder, render_set, images_per_set))
        except Exception as exc:
            row["profile_errors"].append(f"{render_set}: {exc}")
    for name, prefix in (("latents", "slat"), ("ss_latents", "ss"), ("features", "feature")):
        path = maps[name].get(uid)
        if path is None:
            continue
        try:
            with np.load(path) as data:
                for key in data.files:
                    array = data[key]
                    row[f"{prefix}_{key}_shape"] = list(array.shape)
                    row[f"{prefix}_{key}_count"] = int(array.shape[0]) if array.ndim else 1
                    if np.issubdtype(array.dtype, np.number) and array.size:
                        sample = array.reshape(-1)[:: max(1, array.size // 100000)].astype(np.float64, copy=False)
                        row[f"{prefix}_{key}_mean"] = float(np.mean(sample))
                        row[f"{prefix}_{key}_std"] = float(np.std(sample))
        except Exception as exc:
            row["profile_errors"].append(f"{name}: {exc}")
    voxel = maps["voxels"].get(uid)
    if voxel is not None:
        row["voxel_vertex_count"] = _ply_vertex_count(voxel)
    return row


def _render_stats(folder: Path | None, prefix: str, image_limit: int) -> dict[str, Any]:
    result: dict[str, Any] = {f"{prefix}_views": 0}
    if folder is None:
        return result
    transforms_path = folder / "transforms.json"
    payload = json.loads(transforms_path.read_text(encoding="utf-8")) if transforms_path.is_file() else {}
    frames = payload.get("frames", []) if isinstance(payload, dict) else []
    centers, foreground, widths, heights = [], [], [], []
    available = []
    for frame in frames:
        path = _resolve_image(folder, frame)
        if path is not None:
            available.append((frame, path))
            matrix = frame.get("transform_matrix") or frame.get("c2w")
            if matrix is not None:
                value = np.asarray(matrix, dtype=np.float64)
                if value.shape == (4, 4):
                    centers.append(value[:3, 3])
    for _, path in available[: max(0, image_limit)]:
        with Image.open(path) as image:
            widths.append(image.width)
            heights.append(image.height)
            if "A" in image.getbands():
                alpha = np.asarray(image.getchannel("A"), dtype=np.float32) / 255.0
                foreground.append(float(np.mean(alpha > 0.5)))
    result[f"{prefix}_views"] = len(available)
    if widths:
        result[f"{prefix}_width"] = float(np.median(widths))
        result[f"{prefix}_height"] = float(np.median(heights))
    if foreground:
        result[f"{prefix}_foreground_fraction"] = float(np.mean(foreground))
    if centers:
        c = np.stack(centers)
        radius = np.linalg.norm(c, axis=1)
        azimuth = np.degrees(np.arctan2(c[:, 1], c[:, 0]))
        elevation = np.degrees(np.arcsin(np.clip(c[:, 2] / np.maximum(radius, 1e-9), -1, 1)))
        result[f"{prefix}_camera_radius_mean"] = float(radius.mean())
        result[f"{prefix}_camera_radius_std"] = float(radius.std())
        result[f"{prefix}_azimuth_coverage_deg"] = float(_circular_coverage(azimuth))
        result[f"{prefix}_elevation_min_deg"] = float(elevation.min())
        result[f"{prefix}_elevation_max_deg"] = float(elevation.max())
    return result


def _circular_coverage(angles: np.ndarray) -> float:
    if angles.size < 2:
        return 0.0
    values = np.sort(np.mod(angles, 360.0))
    gaps = np.diff(np.concatenate([values, values[:1] + 360.0]))
    return 360.0 - float(gaps.max())


def _numeric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float)) and not isinstance(value, bool)})
    summary = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))], dtype=np.float64)
        if values.size:
            summary[key] = {
                "n": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
                "min": float(values.min()), "p10": float(np.percentile(values, 10)),
                "median": float(np.median(values)), "p90": float(np.percentile(values, 90)), "max": float(values.max()),
            }
    return summary


def _shape_histograms(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result = {}
    for key in sorted({key for row in rows for key in row if key.endswith("_shape")}):
        result[key] = dict(Counter("x".join(str(v) for v in row[key]) for row in rows if isinstance(row.get(key), list)))
    return result


def _manifest_uids(path: str) -> Iterable[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("uids", payload) if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError("UID manifest must be a JSON list or contain a uids list.")
    return [str(value) for value in values]


def _ply_vertex_count(path: Path) -> int | None:
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


if __name__ == "__main__":
    main()
