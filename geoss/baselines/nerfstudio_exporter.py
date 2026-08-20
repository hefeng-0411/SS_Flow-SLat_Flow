from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch


def export_nerfstudio_dataset(batch: Dict[str, torch.Tensor], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    images = batch["images"].detach().cpu().clamp(0, 1)
    B, N = images.shape[:2]
    if B != 1:
        raise ValueError("Nerfstudio exporter expects one object per export.")
    from PIL import Image
    import numpy as np

    for i in range(N):
        path = image_dir / f"{i:03d}.png"
        arr = (images[0, i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(path)
        frames.append({"file_path": f"images/{path.name}", "transform_matrix": batch["c2w"][0, i].detach().cpu().tolist()})
    meta = {"camera_model": "OPENCV", "frames": frames}
    (output_dir / "transforms.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return output_dir / "transforms.json"
