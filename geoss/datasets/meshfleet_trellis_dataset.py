from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from geoss.utils.coordinates import c2w_to_w2c, parse_objaverse_camera
from geoss.utils.coordinates import anchor_to_occ_index
from geoss.utils.voxelization import points_to_occupancy


class MeshFleetTrellisDataset(Dataset):
    """Dataset adapter for MeshFleet_TRELLIS TRELLIS-toolkit layout.

    Supported layouts:

    1. Category layout, as in the local single-object sample:

    root/split/category/
      renders/<uid>/{000.png,...,transforms.json}
      renders_cond/<uid>/{*.png,transforms.json} optional
      voxels/<uid>.ply
      ss_latents/<uid>.npz or ss_latents/<latent_model>/<uid>.npz
      latents/<uid>.npz or latents/<latent_model>/<uid>.npz optional
      features/<uid>.npz or features/<feature_model>/<uid>.npz optional
      mesh_normalized/<uid>/mesh.glb optional

    2. Flat split layout, as in the full server dataset:

    root/split/
      renders/<uid>/{000.png,...,transforms.json}
      renders_cond/<uid>/{*.png,transforms.json} optional
      voxels/<uid>.ply
      ss_latents/<uid>.npz or ss_latents/<latent_model>/<uid>.npz
      latents/<uid>.npz or latents/<latent_model>/<uid>.npz optional
      features/<uid>.npz or features/<feature_model>/<uid>.npz optional
      mesh_normalized/<uid>/mesh.glb optional

    3. Direct split root layout:

    root/
      renders/
      latents/
      ...
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        category: Optional[str] = None,
        num_views: int = 8,
        image_size: int = 128,
        ss_latent_model: str = "ss_enc_conv3d_16l8_fp16",
        slat_latent_model: str = "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16",
        feature_model: str = "dinov2_vitl14_reg",
        occ_resolution: int = 64,
        prefer_cond_render: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.category = category
        self.num_views = num_views
        self.image_size = image_size
        self.ss_latent_model = ss_latent_model
        self.slat_latent_model = slat_latent_model
        self.feature_model = feature_model
        self.occ_resolution = occ_resolution
        self.prefer_cond_render = prefer_cond_render
        self.samples = self._discover_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        last_error: Optional[Exception] = None
        for offset in range(max(1, len(self.samples))):
            sample = self.samples[(idx + offset) % len(self.samples)]
            try:
                return self._load_sample(sample)
            except FileNotFoundError as exc:
                last_error = exc
                continue
        raise FileNotFoundError(f"No loadable MeshFleet sample found after scanning {len(self.samples)} samples. Last error: {last_error}")

    def _load_sample(self, sample: Dict) -> Dict:
        uid = sample["uid"]
        render_dir = sample["render_dir"]
        transforms = json.loads((render_dir / "transforms.json").read_text(encoding="utf-8"))
        frames = transforms.get("frames", [])
        if not frames:
            raise FileNotFoundError(f"No frames in {render_dir / 'transforms.json'}")
        available = _available_render_frames(render_dir, frames)
        if not available:
            raise FileNotFoundError(f"No valid render images for uid={uid} in {render_dir}; all missing frames are skipped")
        chosen = self._choose_frames(available, self.num_views)
        images, masks, K_list, c2w_list = [], [], [], []
        missing_view_count = len(frames) - len(available)
        for frame, image_path in chosen:
            image = Image.open(image_path).convert("RGBA")
            rgb, mask = _rgba_to_rgb_mask(image, self.image_size)
            camera_data = {**{k: v for k, v in transforms.items() if k != "frames"}, **frame}
            c2w, K = parse_objaverse_camera(
                camera_data,
                image_size=(self.image_size, self.image_size),
                assume_opengl=True,
            )
            images.append(rgb)
            masks.append(mask)
            K_list.append(K)
            c2w_list.append(c2w)
        c2w = torch.stack(c2w_list)
        pack = {
            "images": torch.stack(images),
            "masks": torch.stack(masks),
            "K": torch.stack(K_list),
            "c2w": c2w,
            "w2c": c2w_to_w2c(c2w),
            "uid": uid,
            "object_id": uid,
            "category": sample["category"],
            "dataset_name": "meshfleet_trellis",
            "split": sample["split"],
            "mesh_path": str(sample["mesh_path"]) if sample.get("mesh_path") else None,
            "metadata": {
                "aabb": transforms.get("aabb"),
                "scale": transforms.get("scale"),
                "offset": transforms.get("offset"),
                "split": sample["split"],
                "layout": sample["layout"],
                "layout_root": str(sample["layout_root"]),
                "split_root": str(sample["split_root"]),
                "render_dir": str(render_dir),
                "paths": {key: str(value) for key, value in sample["paths"].items() if value is not None},
                "num_frames_total": len(frames),
                "num_frames_available": len(available),
                "num_frames_missing": missing_view_count,
                "missing_frames_skipped": missing_view_count,
                "num_views_requested": self.num_views,
                "num_views_returned": len(chosen),
            },
        }
        cond_image = _load_condition_image(sample.get("cond_render_dir"), chosen[0][1])
        if cond_image is not None:
            pack["trellis_cond_image"] = cond_image
        if sample.get("voxel_path"):
            points = read_ply_xyz(sample["voxel_path"])
            points_canonical, voxel_meta = _meshfleet_voxels_to_canonical(points)
            pack["gt_occ"] = points_to_occupancy(points_canonical, resolution=self.occ_resolution)
            pack["gt_sparse_indices"] = anchor_to_occ_index(points_canonical, self.occ_resolution)
            pack["gt_sparse_xyz"] = points_canonical
            pack["metadata"]["voxel_coordinate"] = voxel_meta
        if sample.get("ss_latent_path"):
            latent = np.load(sample["ss_latent_path"])
            pack["ss_latent_grid"] = torch.tensor(latent["mean"]).float()
            pack["ss_latent_tokens"] = pack["ss_latent_grid"].flatten(1).transpose(0, 1).contiguous()
        if sample.get("slat_latent_path"):
            latent = np.load(sample["slat_latent_path"])
            pack["trellis_slat_feats"] = torch.tensor(latent["feats"]).float()
            pack["trellis_slat_indices"] = torch.tensor(latent["coords"]).long()
        if sample.get("feature_path"):
            feats = np.load(sample["feature_path"])
            if "patchtokens" in feats:
                pack["trellis_patchtokens"] = torch.tensor(feats["patchtokens"]).float()
            if "indices" in feats:
                pack["trellis_feature_indices"] = torch.tensor(feats["indices"]).long()
        return pack

    def _discover_samples(self) -> List[Dict]:
        split_roots = self._resolve_split_roots()
        samples: List[Dict] = []
        for split_name, split_root in split_roots:
            dataset_roots = self._discover_dataset_roots(split_root)
            for category, cat_root, layout_name in dataset_roots:
                for uid in _discover_uids(cat_root):
                    render_dir = _select_render_dir(cat_root, uid, prefer_cond_render=self.prefer_cond_render)
                    if render_dir is None:
                        continue
                    paths = _meshfleet_uid_paths(
                        cat_root,
                        uid,
                        self.ss_latent_model,
                        self.slat_latent_model,
                        self.feature_model,
                    )
                    sample = {
                        "uid": uid,
                        "split": split_name,
                        "category": category,
                        "layout": layout_name,
                        "layout_root": cat_root,
                        "split_root": split_root,
                        "render_dir": render_dir,
                        "cond_render_dir": paths["renders_cond"],
                        "voxel_path": paths["voxels"],
                        "ss_latent_path": paths["ss_latents"],
                        "slat_latent_path": paths["latents"],
                        "feature_path": paths["features"],
                        "mesh_path": _first_existing([paths["mesh_normalized_mesh"], render_dir / "mesh.ply"]),
                        "paths": paths,
                    }
                    samples.append(sample)
        return samples

    def _resolve_split_roots(self) -> List[Tuple[str, Path]]:
        """Resolve one or more split roots.

        `split` may be a single split (`train`, `test`), a comma-separated
        list (`train,test`), or `all`/`all/train,test` for both standard
        MeshFleet_TRELLIS splits. Passing a direct split root still works.
        """
        split_names = _parse_split_spec(self.split)
        roots: List[Tuple[str, Path]] = []
        checked: List[Path] = []
        for split_name in split_names:
            candidates = [self.root] if split_name in {"", "."} else [self.root / split_name]
            if len(split_names) == 1 and self.root not in candidates and not _has_standard_split_dirs(self.root):
                candidates.append(self.root)
            for candidate in candidates:
                checked.append(candidate)
                if candidate.exists() and (_is_meshfleet_layout(candidate) or _has_category_layout(candidate)):
                    roots.append((split_name if candidate != self.root else _direct_split_name(split_name), candidate))
                    break
        if roots:
            return roots
        existing = [str(c) for c in checked if c.exists()]
        raise FileNotFoundError(
            "Could not find a MeshFleet_TRELLIS split layout. "
            f"root={self.root}, split={self.split}, existing_candidates={existing}. "
            "Expected root/{train,test}/{renders,latents,...}, root/{train,test}/<category>/{renders,latents,...}, "
            "or a direct split root containing those folders."
        )

    def _discover_dataset_roots(self, split_root: Path) -> List[Tuple[str, Path, str]]:
        """Return `(category, dataset_root, layout_name)` entries."""
        if self.category:
            category_root = split_root / self.category
            if _is_meshfleet_layout(category_root):
                return [(self.category, category_root, "category")]
            if _is_meshfleet_layout(split_root):
                # Server flat layout has no category directory; keep the user label as metadata only.
                return [(self.category, split_root, "flat_split")]
            raise FileNotFoundError(
                f"Requested MeshFleet category '{self.category}', but neither {category_root} nor {split_root} "
                "contains the required renders/latents layout."
            )
        if _is_meshfleet_layout(split_root):
            return [("meshfleet", split_root, "flat_split")]
        roots = []
        for candidate in sorted([p for p in split_root.iterdir() if p.is_dir()]):
            if _is_meshfleet_layout(candidate):
                roots.append((candidate.name, candidate, "category"))
        if not roots:
            raise FileNotFoundError(
                f"No MeshFleet samples found under {split_root}. "
                "Expected flat split folders {renders,features,latents,ss_latents,voxels} "
                "or category folders that contain those subfolders."
            )
        return roots

    @staticmethod
    def _choose_frames(frames: List[Tuple[Dict, Path]], num_views: int) -> List[Tuple[Dict, Path]]:
        if len(frames) == 0:
            raise ValueError("Cannot choose frames from an empty list")
        indices = torch.linspace(0, len(frames) - 1, num_views).round().long().tolist()
        return [frames[i] for i in indices]


def read_ply_xyz(path: str | Path) -> torch.Tensor:
    """Read ASCII or binary_little_endian PLY vertices with x/y/z float fields."""
    path = Path(path)
    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header.append(line)
            if line == "end_header":
                break
        fmt = next((line for line in header if line.startswith("format ")), "")
        vertex_line = next(line for line in header if line.startswith("element vertex"))
        count = int(vertex_line.split()[-1])
        props = [line.split()[-1] for line in header if line.startswith("property ")]
        xyz_idx = [props.index(axis) for axis in ("x", "y", "z")]
        if "binary_little_endian" in fmt:
            # MeshFleet voxel PLY uses float properties only.
            row_fmt = "<" + "f" * len(props)
            row_size = struct.calcsize(row_fmt)
            data = []
            for _ in range(count):
                row = struct.unpack(row_fmt, f.read(row_size))
                data.append([row[i] for i in xyz_idx])
            return torch.tensor(data, dtype=torch.float32)
        data = []
        for _ in range(count):
            row = f.readline().decode("ascii").split()
            data.append([float(row[i]) for i in xyz_idx])
    return torch.tensor(data, dtype=torch.float32)


def _rgba_to_rgb_mask(image: Image.Image, image_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    arr = np.array(image).astype("float32") / 255.0
    rgb = torch.from_numpy(arr[..., :3]).permute(2, 0, 1).contiguous()
    alpha = torch.from_numpy(arr[..., 3:4]).permute(2, 0, 1).contiguous()
    return rgb, alpha.clamp(0, 1)


def _load_condition_image(cond_render_dir: Optional[Path], fallback_image_path: Path) -> Optional[torch.Tensor]:
    image_path = None
    if cond_render_dir is not None and cond_render_dir.exists():
        candidates = sorted(
            p for p in cond_render_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if candidates:
            image_path = candidates[0]
    if image_path is None:
        image_path = fallback_image_path
    image = Image.open(image_path).convert("RGB").resize((518, 518), Image.Resampling.BICUBIC)
    arr = np.array(image).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _first_existing(paths: List[Optional[Path]]) -> Optional[Path]:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def _parse_split_spec(split: str | None) -> List[str]:
    if split is None:
        return ["train"]
    text = str(split).strip()
    if text in {"", "."}:
        return ["."]
    lower = text.lower()
    if lower == "all":
        return ["train", "test"]
    if lower.startswith("all/"):
        text = text.split("/", 1)[1]
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return parts or ["train"]


def _direct_split_name(requested: str) -> str:
    return "direct" if requested in {"", "."} else requested


def _meshfleet_uid_paths(
    root: Path,
    uid: str,
    ss_latent_model: str,
    slat_latent_model: str,
    feature_model: str,
) -> Dict[str, Optional[Path]]:
    return {
        "features": _first_existing([
            root / "features" / f"{uid}.npz",
            root / "features" / feature_model / f"{uid}.npz",
        ]),
        "latents": _first_existing([
            root / "latents" / f"{uid}.npz",
            root / "latents" / slat_latent_model / f"{uid}.npz",
        ]),
        "mesh_normalized_dir": _first_existing([root / "mesh_normalized" / uid]),
        "mesh_normalized_mesh": _first_existing([root / "mesh_normalized" / uid / "mesh.glb"]),
        "renders": _first_existing([root / "renders" / uid]),
        "renders_cond": _first_existing([root / "renders_cond" / uid]),
        "renders_eval_70": _first_existing([root / "renders_eval_70" / uid]),
        "renders_eval_90": _first_existing([root / "renders_eval_90" / uid]),
        "ss_latents": _first_existing([
            root / "ss_latents" / f"{uid}.npz",
            root / "ss_latents" / ss_latent_model / f"{uid}.npz",
        ]),
        "voxels": _first_existing([root / "voxels" / f"{uid}.ply"]),
    }


def _discover_uids(root: Path) -> List[str]:
    """Discover object IDs from all MeshFleet_TRELLIS per-object folders/files."""
    uids = set()
    for folder in ("renders", "renders_cond", "renders_eval_70", "renders_eval_90", "mesh_normalized"):
        base = root / folder
        if base.is_dir():
            uids.update(path.name for path in base.iterdir() if path.is_dir())
    for folder, suffix in (("features", ".npz"), ("latents", ".npz"), ("ss_latents", ".npz"), ("voxels", ".ply")):
        base = root / folder
        if not base.is_dir():
            continue
        for path in base.iterdir():
            if path.is_file() and path.suffix.lower() == suffix:
                uids.add(path.stem)
            elif path.is_dir():
                uids.update(child.stem for child in path.iterdir() if child.is_file() and child.suffix.lower() == suffix)
    return sorted(uids)


def _select_render_dir(root: Path, uid: str, *, prefer_cond_render: bool) -> Optional[Path]:
    ordered = ["renders_cond", "renders"] if prefer_cond_render else ["renders", "renders_cond"]
    return _first_existing([root / name / uid for name in ordered])


def _available_render_frames(render_dir: Path, frames: List[Dict]) -> List[Tuple[Dict, Path]]:
    available: List[Tuple[Dict, Path]] = []
    for frame in frames:
        image_path = _resolve_frame_image_path(render_dir, frame)
        if image_path is not None:
            available.append((frame, image_path))
    return available


def _resolve_frame_image_path(render_dir: Path, frame: Dict) -> Optional[Path]:
    file_path = frame.get("file_path") or frame.get("image_path") or frame.get("filename")
    if not file_path:
        return None
    raw = Path(str(file_path))
    candidates: List[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(render_dir / raw)
        candidates.append(render_dir / raw.name)
    expanded: List[Path] = []
    suffixes = [".png", ".jpg", ".jpeg", ".webp"]
    for candidate in candidates:
        expanded.append(candidate)
        if candidate.suffix == "":
            expanded.extend(candidate.with_suffix(suffix) for suffix in suffixes)
        else:
            expanded.extend(candidate.with_suffix(suffix) for suffix in suffixes if suffix != candidate.suffix.lower())
    seen = set()
    for candidate in expanded:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _is_meshfleet_layout(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    has_render = (path / "renders").is_dir() or (path / "renders_cond").is_dir()
    has_asset_data = any((path / name).exists() for name in ("voxels", "ss_latents", "latents", "features", "mesh_normalized"))
    return has_render and has_asset_data


def _has_category_layout(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(_is_meshfleet_layout(child) for child in path.iterdir() if child.is_dir())


def _has_standard_split_dirs(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any((path / split).is_dir() for split in ("train", "test"))


def _meshfleet_voxels_to_canonical(points_xyz: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Convert MeshFleet/TRELLIS voxel PLY coordinates to GeoSS canonical [-1, 1].

    The provided voxel PLY files in `MeshFleet_TRELLIS` store centers in a
    TRELLIS-normalized cube close to [-0.5, 0.5]. GeoSS anchors use [-1, 1],
    so these points are scaled by 2 before occupancy sampling. If a future
    dump is already in [-1, 1], it is only clamped.
    """
    if points_xyz.numel() == 0:
        return points_xyz, {"source": "empty", "transform": "identity"}
    min_xyz = points_xyz.min(dim=0).values
    max_xyz = points_xyz.max(dim=0).values
    max_abs = points_xyz.abs().max()
    if bool((max_abs <= 0.5 + 1e-4).item()):
        return (points_xyz * 2.0).clamp(-1.0, 1.0), {
            "source": "trellis_centered_half_cube",
            "source_min": min_xyz.tolist(),
            "source_max": max_xyz.tolist(),
            "transform": "xyz_canonical = clamp(xyz * 2, -1, 1)",
        }
    return points_xyz.clamp(-1.0, 1.0), {
        "source": "canonical_or_clamped",
        "source_min": min_xyz.tolist(),
        "source_max": max_xyz.tolist(),
        "transform": "xyz_canonical = clamp(xyz, -1, 1)",
    }
