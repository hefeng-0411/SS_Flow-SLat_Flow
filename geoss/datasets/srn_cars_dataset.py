from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from geoss.utils.coordinates import c2w_to_w2c, parse_srn_intrinsics, parse_srn_pose
from geoss.utils.mesh import find_shapenet_mesh
from geoss.utils.voxelization import voxelize_mesh_to_occ


class SRNCarsDataset(Dataset):
    """ShapeNet-SRN Cars multi-view dataset in OpenCV canonical coordinates."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_views: int = 4,
        image_size: int = 128,
        shapenet_mesh_root: Optional[str] = None,
        occ_resolution: int = 64,
        view_sampling: str = "fixed",
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.num_views = num_views
        self.image_size = image_size
        self.shapenet_mesh_root = Path(shapenet_mesh_root) if shapenet_mesh_root else None
        self.occ_resolution = occ_resolution
        self.view_sampling = view_sampling
        list_file = self.root / f"list_{split}.txt"
        split_dir = self.root / f"cars_{split}"
        if list_file.exists():
            ids = [line.strip() for line in list_file.read_text().splitlines() if line.strip()]
        elif split_dir.exists():
            ids = [p.name for p in sorted(split_dir.iterdir()) if p.is_dir()]
        else:
            ids = []
        self.instances = ids
        self.split_dir = split_dir

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict:
        object_id = self.instances[idx]
        obj_dir = self.split_dir / object_id
        rgb_files = self._select_views(sorted((obj_dir / "rgb").glob("*.png")))
        if not rgb_files:
            raise FileNotFoundError(f"No SRN rgb frames found in {obj_dir / 'rgb'}")
        images, masks, c2w_list = [], [], []
        for rgb_file in rgb_files:
            img = Image.open(rgb_file)
            has_alpha = img.mode in {"RGBA", "LA"} or ("transparency" in img.info)
            rgba = img.convert("RGBA")
            rgb_pil = rgba.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
            rgb = _pil_to_tensor(rgb_pil)
            if has_alpha:
                alpha_pil = rgba.getchannel("A").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
                alpha = _pil_to_tensor(alpha_pil).unsqueeze(0)
            else:
                alpha = (rgb.mean(dim=0, keepdim=True) < 0.98).float()
            images.append(rgb)
            masks.append(alpha.clamp(0, 1))
            pose_file = obj_dir / "pose" / f"{rgb_file.stem}.txt"
            c2w_list.append(parse_srn_pose(pose_file))
        images_t = torch.stack(images, dim=0)
        masks_t = torch.stack(masks, dim=0)
        K0 = parse_srn_intrinsics(obj_dir / "intrinsics.txt", image_size=(self.image_size, self.image_size))
        K = K0.unsqueeze(0).expand(len(rgb_files), -1, -1).clone()
        c2w = torch.stack(c2w_list, dim=0)
        w2c = c2w_to_w2c(c2w)
        pack = {
            "images": images_t,
            "masks": masks_t,
            "K": K,
            "c2w": c2w,
            "w2c": w2c,
            "object_id": object_id,
            "dataset_name": "srn_cars",
            "split": self.split,
        }
        if self.shapenet_mesh_root is not None:
            mesh = find_shapenet_mesh(self.shapenet_mesh_root, object_id)
            if mesh is not None:
                pack["mesh_path"] = str(mesh)
                gt_occ = voxelize_mesh_to_occ(mesh, resolution=self.occ_resolution)
                if gt_occ is not None:
                    pack["gt_occ"] = gt_occ
        return pack

    def _select_views(self, rgb_files: List[Path]) -> List[Path]:
        if self.view_sampling == "all":
            return rgb_files
        if len(rgb_files) <= self.num_views:
            return rgb_files
        if self.view_sampling == "random":
            indices = torch.randperm(len(rgb_files))[: self.num_views].sort().values.tolist()
            return [rgb_files[i] for i in indices]
        if self.view_sampling != "fixed":
            raise ValueError(f"Unknown SRN view_sampling={self.view_sampling}; expected fixed/random/all")
        return rgb_files[: self.num_views]


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype("float32") / 255.0
    if arr.ndim == 2:
        return torch.from_numpy(arr)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
