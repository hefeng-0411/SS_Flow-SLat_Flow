from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image


RENDER_SETS = ("renders", "renders_cond", "renders_eval_70", "renders_eval_90")
FILE_MODALITIES = {
    "features": ".npz",
    "latents": ".npz",
    "ss_latents": ".npz",
    "voxels": ".ply",
}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MeshFleet/TRELLIS layouts and create leakage-free UID manifests.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default="outputs/dataset_audit")
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--validation_percent", type=float, default=10.0)
    parser.add_argument("--split_seed", type=int, default=20260720)
    parser.add_argument("--required_modalities", default="renders,renders_cond,renders_eval_70,renders_eval_90,voxels,ss_latents,latents,features,mesh")
    parser.add_argument("--ss_latent_model", default="ss_enc_conv3d_16l8_fp16")
    parser.add_argument("--slat_latent_model", default="dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16")
    parser.add_argument("--feature_model", default="dinov2_vitl14_reg")
    parser.add_argument("--min_train_views", type=int, default=8, help="Minimum usable primary renders for stage-aware training manifests.")
    parser.add_argument("--min_eval_views", type=int, default=12, help="Minimum usable views in each held-out evaluation render set.")
    parser.add_argument("--hash_files", action="store_true")
    parser.add_argument("--validate_payloads", action="store_true", help="Open images and validate required NPZ/PLY payload structure.")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--strict_scope",
        choices=("all_discovered", "manifests"),
        default="all_discovered",
        help="With --strict, fail on any raw-object defect or only when generated strict manifests are invalid.",
    )
    args = parser.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    required = tuple(item.strip() for item in args.required_modalities.split(",") if item.strip())
    split_names = tuple(item.strip() for item in args.splits.split(",") if item.strip())
    rows: List[Dict[str, Any]] = []
    split_uids: Dict[str, set[str]] = {}
    discovery_failures: List[Dict[str, str]] = []

    for split in split_names:
        split_root = root / split
        layouts = _dataset_roots(split_root)
        if not layouts:
            discovery_failures.append({"split": split, "error": f"No flat/category MeshFleet layout under {split_root}"})
            split_uids[split] = set()
            continue
        split_uids[split] = set()
        for category, layout_root in layouts:
            modality_maps = _modality_maps(
                layout_root,
                ss_latent_model=args.ss_latent_model,
                slat_latent_model=args.slat_latent_model,
                feature_model=args.feature_model,
            )
            uids = sorted(set().union(*(set(values) for values in modality_maps.values())))
            for uid in uids:
                split_uids[split].add(uid)
                rows.append(
                    _audit_uid(
                        root=root,
                        split=split,
                        category=category,
                        layout_root=layout_root,
                        uid=uid,
                        maps=modality_maps,
                        required=required,
                        hash_files=args.hash_files,
                        min_train_views=args.min_train_views,
                        min_eval_views=args.min_eval_views,
                        validate_payloads=args.validate_payloads,
                    )
                )

    uid_location_counts = Counter((row["split"], row["uid"]) for row in rows)
    ambiguous_pairs = {pair for pair, count in uid_location_counts.items() if count > 1}
    ambiguous_uids = [
        {"split": split, "uid": uid, "count": uid_location_counts[(split, uid)]}
        for split, uid in sorted(ambiguous_pairs)
    ]
    overlap = sorted(split_uids.get("train", set()) & split_uids.get("test", set()))
    train_population, validation_population = _deterministic_validation_split(
        (uid for uid in split_uids.get("train", set()) if ("train", uid) not in ambiguous_pairs),
        args.validation_percent,
        args.split_seed,
    )
    rows_by_split_uid = {(row["split"], row["uid"]): row for row in rows}
    train_manifest = _eligible_uids(train_population, "train", rows_by_split_uid, "valid")
    val_manifest = _eligible_uids(validation_population, "train", rows_by_split_uid, "valid")
    test_population = [uid for uid in split_uids.get("test", set()) if ("test", uid) not in ambiguous_pairs]
    test_manifest = _eligible_uids(test_population, "test", rows_by_split_uid, "valid")
    stage_manifests = {
        "stage1_train_uids.json": _eligible_uids(train_population, "train", rows_by_split_uid, "stage1"),
        "stage2_train_uids.json": _eligible_uids(train_population, "train", rows_by_split_uid, "stage2"),
        "stage3_train_uids.json": _eligible_uids(train_population, "train", rows_by_split_uid, "stage3"),
        "stage4_train_uids.json": _eligible_uids(train_population, "train", rows_by_split_uid, "stage4"),
        "validation_evaluation_uids.json": _eligible_uids(validation_population, "train", rows_by_split_uid, "evaluation"),
        "test_evaluation_uids.json": _eligible_uids(test_population, "test", rows_by_split_uid, "evaluation"),
    }
    failures = [row for row in rows if not row["valid"]]
    invalid_by_split = Counter(row["split"] for row in failures)
    invalid_reasons = Counter(reason for row in failures for reason in _failure_reasons(row))
    render_issue_counts = _render_issue_counts(rows)
    requested_manifests_nonempty = (
        ("train" not in split_names or bool(train_manifest or val_manifest))
        and ("test" not in split_names or bool(test_manifest))
    )
    manifests_valid = requested_manifests_nonempty and not overlap and not discovery_failures and not ambiguous_uids
    summary = {
        "protocol_version": "meshfleet_dataset_audit_v3",
        "data_root": str(root),
        "split_seed": args.split_seed,
        "validation_percent": args.validation_percent,
        "payload_validation_enabled": bool(args.validate_payloads),
        "required_modalities": list(required),
        "preprocessing_models": {
            "ss_latents": args.ss_latent_model,
            "latents": args.slat_latent_model,
            "features": args.feature_model,
        },
        "counts": {split: len(uids) for split, uids in split_uids.items()},
        "strict_valid_counts": {
            split: sum(1 for row in rows if row["split"] == split and row["valid"])
            for split in split_names
        },
        "train_manifest_count": len(train_manifest),
        "validation_manifest_count": len(val_manifest),
        "test_manifest_count": len(test_manifest),
        "stage_manifest_counts": {name: len(uids) for name, uids in stage_manifests.items()},
        "invalid_object_count": len(failures),
        "invalid_object_count_by_split": dict(sorted(invalid_by_split.items())),
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
        "render_issue_counts": dict(sorted(render_issue_counts.items())),
        "train_test_overlap_count": len(overlap),
        "train_test_overlap_uids": overlap,
        "ambiguous_uid_count": len(ambiguous_uids),
        "ambiguous_uids": ambiguous_uids,
        "discovery_failures": discovery_failures,
        "modality_coverage": _coverage(rows),
        "all_discovered_valid": not overlap and not discovery_failures and not ambiguous_uids and not failures,
        "manifests_valid": manifests_valid,
        "valid": not overlap and not discovery_failures and not ambiguous_uids and not failures,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "objects.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_csv(output_dir / "objects.csv", rows)
    _write_manifest(output_dir / "train_uids.json", "train", train_manifest, args)
    _write_manifest(output_dir / "validation_uids.json", "validation", val_manifest, args)
    _write_manifest(output_dir / "test_uids.json", "test", test_manifest, args)
    for filename, uids in stage_manifests.items():
        _write_manifest(output_dir / filename, filename.removesuffix("_uids.json"), uids, args)
    (output_dir / "failures.json").write_text(json.dumps(failures + discovery_failures, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    strict_valid = summary["valid"] if args.strict_scope == "all_discovered" else summary["manifests_valid"]
    if args.strict and not strict_valid:
        raise SystemExit(2)


def _dataset_roots(split_root: Path) -> List[Tuple[str, Path]]:
    if _is_layout(split_root):
        return [("meshfleet", split_root)]
    if not split_root.is_dir():
        return []
    return [(path.name, path) for path in sorted(split_root.iterdir()) if path.is_dir() and _is_layout(path)]


def _is_layout(path: Path) -> bool:
    return path.is_dir() and any((path / name).is_dir() for name in RENDER_SETS) and any(
        (path / name).exists() for name in (*FILE_MODALITIES, "mesh_normalized")
    )


def _modality_maps(
    root: Path,
    *,
    ss_latent_model: str,
    slat_latent_model: str,
    feature_model: str,
) -> Dict[str, Dict[str, Path]]:
    maps: Dict[str, Dict[str, Path]] = {}
    for name in RENDER_SETS:
        base = root / name
        maps[name] = {path.name: path for path in base.iterdir() if path.is_dir()} if base.is_dir() else {}
    model_dirs = {
        "features": feature_model,
        "latents": slat_latent_model,
        "ss_latents": ss_latent_model,
        "voxels": None,
    }
    for name, suffix in FILE_MODALITIES.items():
        bases = [root / name]
        if model_dirs[name]:
            bases.append(root / name / str(model_dirs[name]))
        mapping: Dict[str, Path] = {}
        for base in bases:
            if not base.is_dir():
                continue
            for path in base.glob(f"*{suffix}"):
                mapping.setdefault(path.stem, path)
        maps[name] = mapping
    mesh_root = root / "mesh_normalized"
    maps["mesh"] = {}
    if mesh_root.is_dir():
        for path in mesh_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".glb", ".gltf", ".obj", ".ply"}:
                maps["mesh"].setdefault(path.parent.name, path)
    return maps


def _audit_uid(
    *,
    root: Path,
    split: str,
    category: str,
    layout_root: Path,
    uid: str,
    maps: Dict[str, Dict[str, Path]],
    required: Tuple[str, ...],
    hash_files: bool,
    min_train_views: int,
    min_eval_views: int,
    validate_payloads: bool,
) -> Dict[str, Any]:
    available = {name: uid in mapping for name, mapping in maps.items()}
    missing = [name for name in required if not available.get(name, False)]
    render_info = {name: _audit_render_dir(maps[name].get(uid), validate_images=validate_payloads) for name in RENDER_SETS}
    camera_sets = {name: set(info["camera_signatures"]) for name, info in render_info.items()}
    cond_signatures = camera_sets["renders"] | camera_sets["renders_cond"]
    overlap_70 = len(cond_signatures & camera_sets["renders_eval_70"])
    overlap_90 = len(cond_signatures & camera_sets["renders_eval_90"])
    render_failures = {
        name: info["errors"]
        for name, info in render_info.items()
        if available.get(name) and info["errors"] and name != "renders_cond"
    }
    render_warnings = {
        "renders_cond": render_info["renders_cond"]["errors"]
    } if available.get("renders_cond") and render_info["renders_cond"]["errors"] else {}
    if available.get("renders_cond") and render_info["renders_cond"]["available_images"] == 0:
        render_failures["renders_cond"] = ["no conditioning image is available"]
    paths = {name: _relative(mapping.get(uid), root) for name, mapping in maps.items()}
    checksums = {}
    if hash_files:
        for name, mapping in maps.items():
            path = mapping.get(uid)
            if path is not None and path.is_file():
                checksums[name] = _sha256(path)
            elif path is not None and path.is_dir() and (path / "transforms.json").is_file():
                checksums[f"{name}_transforms"] = _sha256(path / "transforms.json")
    payload_failures = _validate_payloads(maps, uid) if validate_payloads else {}
    valid = (
        not missing
        and not render_failures
        and not payload_failures
        and overlap_70 == 0
        and overlap_90 == 0
    )
    primary_usable = _render_usable(render_info["renders"], min_train_views)
    stage_eligibility = {
        # Stage 1 needs image/camera evidence and voxel supervision, but not
        # precomputed TRELLIS features or either latent family.
        "stage1": primary_usable and available.get("voxels", False) and "voxels" not in payload_failures,
        # Stage 2 targets the sparse-structure latent and also keeps geometry
        # supervision available for the GeoSS context.
        "stage2": (
            primary_usable
            and available.get("voxels", False)
            and available.get("ss_latents", False)
            and "voxels" not in payload_failures
            and "ss_latents" not in payload_failures
        ),
        # Stages 3/4 target SLAT features and use voxels for the decoded
        # geometry branch; precomputed image features are not consumed.
        "stage3": (
            primary_usable
            and available.get("voxels", False)
            and available.get("latents", False)
            and "voxels" not in payload_failures
            and "latents" not in payload_failures
        ),
        "stage4": (
            primary_usable
            and available.get("voxels", False)
            and available.get("latents", False)
            and "voxels" not in payload_failures
            and "latents" not in payload_failures
        ),
        # Protocol validity is based only on actual inference inputs and
        # evaluation GT. Training-only cached latents/features are irrelevant.
        "evaluation": (
            primary_usable
            and _render_usable(render_info["renders_eval_70"], min_eval_views)
            and _render_usable(render_info["renders_eval_90"], min_eval_views)
            and available.get("mesh", False)
            and "mesh" not in payload_failures
            and overlap_70 == 0
            and overlap_90 == 0
        ),
    }
    return {
        "uid": uid,
        "split": split,
        "category": category,
        "layout_root": _relative(layout_root, root),
        "valid": valid,
        "stage_eligibility": stage_eligibility,
        "missing_modalities": missing,
        "render_failures": render_failures,
        "render_warnings": render_warnings,
        "payload_failures": payload_failures,
        "conditioning_overlap_eval_70": overlap_70,
        "conditioning_overlap_eval_90": overlap_90,
        "available": available,
        "paths": paths,
        "render_counts": {name: info["available_images"] for name, info in render_info.items()},
        "usable_render_counts": {name: info["usable_frames"] for name, info in render_info.items()},
        "declared_frame_counts": {name: info["declared_frames"] for name, info in render_info.items()},
        "checksums": checksums,
    }


def _audit_render_dir(path: Optional[Path], *, validate_images: bool = False) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "declared_frames": 0,
        "available_images": 0,
        "usable_frames": 0,
        "camera_signatures": [],
        "errors": [],
    }
    if path is None:
        return result
    transforms_path = path / "transforms.json"
    if not transforms_path.is_file():
        result["errors"].append("missing transforms.json")
        return result
    try:
        transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["errors"].append(f"invalid transforms.json: {exc}")
        return result
    frames = transforms.get("frames") or []
    result["declared_frames"] = len(frames)
    for index, frame in enumerate(frames):
        image = _resolve_image(path, frame)
        if image is None:
            result["errors"].append(f"frame[{index}] missing image")
        else:
            if validate_images:
                try:
                    with Image.open(image) as handle:
                        handle.verify()
                except Exception as exc:
                    result["errors"].append(f"frame[{index}] invalid image: {exc}")
                    image = None
            if image is not None:
                result["available_images"] += 1
        matrix = frame.get("transform_matrix") or frame.get("camera_to_world") or frame.get("c2w")
        if matrix is None:
            result["errors"].append(f"frame[{index}] missing camera matrix")
        else:
            try:
                rows = [[float(value) for value in row] for row in matrix]
                if len(rows) != 4 or any(len(row) != 4 for row in rows):
                    raise ValueError("camera matrix must be 4x4")
                flat = [round(value, 6) for row in rows for value in row]
                if any(value != value or abs(value) == float("inf") for value in flat):
                    raise ValueError("camera matrix contains non-finite values")
                if image is not None:
                    result["camera_signatures"].append(hashlib.sha256(json.dumps(flat).encode("utf-8")).hexdigest())
                    result["usable_frames"] += 1
            except (TypeError, ValueError):
                result["errors"].append(f"frame[{index}] invalid camera matrix")
    if not frames:
        result["errors"].append("no frames")
    return result


def _render_usable(info: Dict[str, Any], minimum_views: int) -> bool:
    if info["usable_frames"] < max(1, int(minimum_views)):
        return False
    fatal_markers = (
        "missing transforms.json",
        "invalid transforms.json",
        "no frames",
    )
    return not any(any(marker in error for marker in fatal_markers) for error in info["errors"])


def _eligible_uids(
    uids: Iterable[str],
    split: str,
    rows_by_split_uid: Dict[Tuple[str, str], Dict[str, Any]],
    policy: str,
) -> List[str]:
    selected = []
    for uid in sorted(uids):
        row = rows_by_split_uid.get((split, uid))
        if row is None:
            continue
        eligible = row["valid"] if policy == "valid" else row["stage_eligibility"].get(policy, False)
        if eligible:
            selected.append(uid)
    return selected


def _failure_reasons(row: Dict[str, Any]) -> List[str]:
    reasons = [f"missing:{name}" for name in row["missing_modalities"]]
    reasons.extend(f"render:{name}" for name in row["render_failures"])
    reasons.extend(f"render_warning:{name}" for name in row["render_warnings"])
    reasons.extend(f"payload:{name}" for name in row["payload_failures"])
    if row["conditioning_overlap_eval_70"]:
        reasons.append("camera_overlap:renders_eval_70")
    if row["conditioning_overlap_eval_90"]:
        reasons.append("camera_overlap:renders_eval_90")
    return reasons or ["unknown"]


def _render_issue_counts(rows: List[Dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        for severity, field in (("error", "render_failures"), ("warning", "render_warnings")):
            for render_set, messages in row[field].items():
                for message in messages:
                    if "missing image" in message:
                        kind = "missing_image"
                    elif "invalid image" in message:
                        kind = "invalid_image"
                    elif "missing camera matrix" in message:
                        kind = "missing_camera_matrix"
                    elif "invalid camera matrix" in message:
                        kind = "invalid_camera_matrix"
                    elif "missing transforms.json" in message:
                        kind = "missing_transforms"
                    elif "invalid transforms.json" in message:
                        kind = "invalid_transforms"
                    elif "no frames" in message:
                        kind = "no_frames"
                    else:
                        kind = "other"
                    counts[f"{severity}:{render_set}:{kind}"] += 1
    return counts


def _validate_payloads(maps: Dict[str, Dict[str, Path]], uid: str) -> Dict[str, str]:
    failures: Dict[str, str] = {}
    required_npz_keys = {
        "features": set(),
        "latents": {"feats.npy", "coords.npy"},
        "ss_latents": {"mean.npy"},
    }
    for modality, required_keys in required_npz_keys.items():
        path = maps[modality].get(uid)
        if path is None:
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                missing_keys = sorted(required_keys - names)
                corrupt_member = archive.testzip()
            if missing_keys:
                failures[modality] = f"missing NPZ arrays: {missing_keys}"
            elif corrupt_member is not None:
                failures[modality] = f"corrupt NPZ member: {corrupt_member}"
            elif not names:
                failures[modality] = "empty NPZ archive"
        except (OSError, zipfile.BadZipFile) as exc:
            failures[modality] = f"invalid NPZ: {exc}"

    voxel = maps["voxels"].get(uid)
    if voxel is not None:
        try:
            with voxel.open("rb") as handle:
                header_lines = []
                for _ in range(256):
                    line = handle.readline()
                    if not line:
                        break
                    decoded = line.decode("ascii", errors="strict").strip()
                    header_lines.append(decoded)
                    if decoded == "end_header":
                        break
            vertex_lines = [line for line in header_lines if line.startswith("element vertex ")]
            properties = {line.split()[-1] for line in header_lines if line.startswith("property ")}
            if not header_lines or header_lines[0] != "ply" or "end_header" not in header_lines:
                raise ValueError("missing PLY header/end_header")
            if not vertex_lines or int(vertex_lines[0].split()[-1]) <= 0:
                raise ValueError("PLY has no vertices")
            if not {"x", "y", "z"}.issubset(properties):
                raise ValueError("PLY lacks x/y/z vertex properties")
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            failures["voxels"] = f"invalid voxel PLY: {exc}"

    mesh = maps["mesh"].get(uid)
    if mesh is not None:
        try:
            if mesh.stat().st_size <= 0:
                failures["mesh"] = "empty mesh file"
        except OSError as exc:
            failures["mesh"] = f"unreadable mesh file: {exc}"
    return failures


def _resolve_image(render_dir: Path, frame: Dict[str, Any]) -> Optional[Path]:
    raw_value = frame.get("file_path") or frame.get("image_path") or frame.get("filename")
    if not raw_value:
        return None
    raw = Path(str(raw_value))
    candidates = [raw] if raw.is_absolute() else [render_dir / raw, render_dir / raw.name]
    for candidate in candidates:
        variants = [candidate]
        if candidate.suffix:
            variants.extend(candidate.with_suffix(suffix) for suffix in IMAGE_SUFFIXES if suffix != candidate.suffix.lower())
        else:
            variants.extend(candidate.with_suffix(suffix) for suffix in IMAGE_SUFFIXES)
        for path in variants:
            if path.is_file():
                return path
    return None


def _deterministic_validation_split(uids: Iterable[str], percent: float, seed: int) -> Tuple[List[str], List[str]]:
    if not 0.0 <= percent < 100.0:
        raise ValueError("validation_percent must be in [0, 100).")
    train, validation = [], []
    threshold = int(round(percent * 100))
    for uid in sorted(uids):
        bucket = int(hashlib.sha256(f"{seed}:{uid}".encode("utf-8")).hexdigest()[:8], 16) % 10000
        (validation if bucket < threshold else train).append(uid)
    return train, validation


def _coverage(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    total = max(1, len(rows))
    counts = Counter(name for row in rows for name, present in row["available"].items() if present)
    return {name: {"count": counts[name], "fraction": counts[name] / total} for name in sorted({*RENDER_SETS, *FILE_MODALITIES, "mesh"})}


def _write_manifest(path: Path, role: str, uids: List[str], args: argparse.Namespace) -> None:
    payload = {
        "role": role,
        "uids": uids,
        "count": len(uids),
        "split_seed": args.split_seed,
        "validation_percent": args.validation_percent,
        "audit_protocol_version": "meshfleet_dataset_audit_v3",
        "eligibility_policy": role,
        "payload_validation_enabled": bool(args.validate_payloads),
        "preprocessing_models": {
            "ss_latents": args.ss_latent_model,
            "latents": args.slat_latent_model,
            "features": args.feature_model,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    flat_rows = []
    for row in rows:
        flat_rows.append({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    fields = sorted({key for row in flat_rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)


def _relative(path: Optional[Path], root: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
