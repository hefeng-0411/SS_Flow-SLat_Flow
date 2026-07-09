from __future__ import annotations

from typing import Dict, Iterable, List

import torch
from torch.utils.data import ConcatDataset, Dataset

from geoss.utils.coordinates import c2w_to_w2c


class VehicleMultiViewDataset(Dataset):
    """Unified dataset wrapper for SRN Cars and Objaverse Cars."""

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        self.dataset = ConcatDataset(list(datasets))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        return self.dataset[idx]

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        tensor_keys = [
            "images",
            "masks",
            "depths",
            "normals",
            "K",
            "c2w",
            "w2c",
            "gt_occ",
            "gt_sparse_indices",
            "gt_sparse_xyz",
            "ss_latent_grid",
            "ss_latent_tokens",
            "trellis_patchtokens",
            "trellis_cond_image",
            "trellis_feature_indices",
            "trellis_slat_feats",
            "trellis_slat_indices",
        ]
        out: Dict = {}
        for key in tensor_keys:
            vals = [b[key] for b in batch if key in b]
            if len(vals) == len(batch):
                out[key] = _stack_if_same_shape(vals)
        out["dataset_name"] = [b.get("dataset_name", "") for b in batch]
        out["object_id"] = [b.get("object_id", b.get("uid", "")) for b in batch]
        if any("mesh_path" in b for b in batch):
            out["mesh_path"] = [b.get("mesh_path") for b in batch]
        if any("category" in b for b in batch):
            out["category"] = [b.get("category") for b in batch]
        if any("metadata" in b for b in batch):
            out["metadata"] = [b.get("metadata") for b in batch]
        return out


def _stack_if_same_shape(vals: List[torch.Tensor]):
    """Stack fixed-shape tensors and keep variable-length tensors as a list."""
    shape0 = tuple(vals[0].shape)
    if all(tuple(v.shape) == shape0 for v in vals):
        return torch.stack(vals, dim=0)
    return vals


def make_dry_run_batch(
    batch_size: int = 2,
    num_views: int = 3,
    image_size: int = 64,
    latent_tokens: int = 512,
    latent_dim: int = 8,
    device: str | torch.device = "cpu",
) -> Dict[str, torch.Tensor]:
    """Synthetic RGB-only vehicle batch for shape and dry-run checks."""
    device = torch.device(device)
    images = torch.rand(batch_size, num_views, 3, image_size, image_size, device=device)
    masks = torch.zeros(batch_size, num_views, 1, image_size, image_size, device=device)
    yy, xx = torch.meshgrid(torch.arange(image_size, device=device), torch.arange(image_size, device=device), indexing="ij")
    circle = ((xx - image_size / 2) ** 2 / (image_size * 0.32) ** 2 + (yy - image_size / 2) ** 2 / (image_size * 0.22) ** 2) < 1
    masks[:] = circle.float().view(1, 1, 1, image_size, image_size)
    depths = torch.ones(batch_size, num_views, 1, image_size, image_size, device=device) * 2.0
    K = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(batch_size, num_views, 1, 1)
    K[..., 0, 0] = image_size
    K[..., 1, 1] = image_size
    K[..., 0, 2] = image_size / 2
    K[..., 1, 2] = image_size / 2
    c2w = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(batch_size, num_views, 1, 1)
    angles = torch.linspace(0, 2 * torch.pi, num_views + 1, device=device)[:-1]
    for i, a in enumerate(angles):
        cam_pos = torch.tensor([torch.sin(a), 0.0, -torch.cos(a)], device=device) * 2.0
        c2w[:, i, :3, 3] = cam_pos
        c2w[:, i, :3, :3] = torch.eye(3, device=device)
    w2c = c2w_to_w2c(c2w)
    return {
        "images": images,
        "masks": masks,
        "depths": depths,
        "K": K,
        "c2w": c2w,
        "w2c": w2c,
        "ss_latent_tokens": torch.randn(batch_size, latent_tokens, latent_dim, device=device),
        "v_base": torch.randn(batch_size, latent_tokens, latent_dim, device=device) * 0.1,
        "timestep": torch.rand(batch_size, device=device),
        "dataset_name": "dry_run",
        "object_id": "synthetic_car",
    }
