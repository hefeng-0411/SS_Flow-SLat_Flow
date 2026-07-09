from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from geoss.utils.coordinates import c2w_to_w2c, parse_objaverse_camera


class ObjaverseCarsRenderedDataset(Dataset):
    """Objaverse Cars rendered multi-view dataset with per-view json or transforms.json."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_views: int = 4,
        image_size: int = 128,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.num_views = num_views
        self.image_size = image_size
        split_file = self.root / "dataset_split.json"
        if split_file.exists():
            data = json.loads(split_file.read_text())
            ids = data.get(split, data if isinstance(data, list) else [])
        else:
            ids = [p.name for p in sorted(self.root.iterdir()) if p.is_dir()]
        self.uids = ids

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, idx: int) -> Dict:
        uid = self.uids[idx]
        obj_dir = self.root / uid
        frames = self._collect_frames(obj_dir)[: self.num_views]
        if not frames:
            raise FileNotFoundError(f"No rendered frames found in {obj_dir}")
        images, masks, c2w_list, K_list, depths, normals = [], [], [], [], [], []
        transforms = self._load_transforms(obj_dir)
        for image_path, camera_data in frames:
            img = Image.open(image_path).convert("RGBA")
            rgb_raw = _pil_to_tensor(img.convert("RGB"))
            alpha_raw = _pil_to_tensor(img.getchannel("A"))
            if alpha_raw.max() <= 0:
                alpha_raw = (rgb_raw.mean(dim=0, keepdim=True) < 0.98).float()
            rgb = _resize_tensor_image(rgb_raw, self.image_size)
            alpha = _resize_tensor_image(alpha_raw, self.image_size)
            if transforms:
                camera_data = {**transforms, **camera_data}
            c2w, K = parse_objaverse_camera(camera_data, image_size=(self.image_size, self.image_size), assume_opengl=True)
            images.append(rgb)
            masks.append(alpha)
            c2w_list.append(c2w)
            K_list.append(K)
            depth_path = image_path.with_name(f"{image_path.stem}_depth.png")
            if depth_path.exists():
                depths.append(_resize_tensor_image(_pil_to_tensor(Image.open(depth_path))[:1], self.image_size))
            normal_path = image_path.with_name(f"{image_path.stem}_normal.png")
            if normal_path.exists():
                normals.append(_resize_tensor_image(_pil_to_tensor(Image.open(normal_path).convert("RGB")), self.image_size))
        c2w = torch.stack(c2w_list)
        pack = {
            "images": torch.stack(images),
            "masks": torch.stack(masks),
            "K": torch.stack(K_list),
            "c2w": c2w,
            "w2c": c2w_to_w2c(c2w),
            "uid": uid,
            "object_id": uid,
            "dataset_name": "objaverse_cars",
            "split": self.split,
        }
        if len(depths) == len(images):
            pack["depths"] = torch.stack(depths)
        if len(normals) == len(images):
            pack["normals"] = torch.stack(normals)
        mesh_candidates = list(obj_dir.glob("*.glb")) + list(obj_dir.glob("*.obj"))
        if mesh_candidates:
            pack["mesh_path"] = str(mesh_candidates[0])
        return pack

    def _load_transforms(self, obj_dir: Path) -> Dict:
        transforms_path = obj_dir / "transforms.json"
        if not transforms_path.exists():
            return {}
        data = json.loads(transforms_path.read_text())
        return {k: v for k, v in data.items() if k != "frames"}

    def _collect_frames(self, obj_dir: Path) -> List[Tuple[Path, Dict]]:
        transforms_path = obj_dir / "transforms.json"
        if transforms_path.exists():
            data = json.loads(transforms_path.read_text())
            out = []
            for frame in data.get("frames", []):
                fp = frame.get("file_path") or frame.get("image_path")
                if fp is None:
                    continue
                image_path = obj_dir / fp
                if image_path.suffix == "":
                    image_path = image_path.with_suffix(".png")
                if image_path.exists():
                    out.append((image_path, frame))
            return out
        images = sorted([p for p in obj_dir.glob("*.png") if not p.name.endswith("_normal.png") and "_depth" not in p.name])
        out = []
        for image_path in images:
            json_path = image_path.with_suffix(".json")
            if json_path.exists():
                out.append((image_path, json.loads(json_path.read_text())))
        return out


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype("float32") / 255.0
    if arr.ndim == 2:
        return torch.from_numpy(arr).unsqueeze(0)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _resize_tensor_image(tensor: torch.Tensor, size: int) -> torch.Tensor:
    import torch.nn.functional as F

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return F.interpolate(tensor.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)
