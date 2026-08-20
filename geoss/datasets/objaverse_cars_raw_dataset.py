from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from torch.utils.data import Dataset


class ObjaverseCarsRawDataset(Dataset):
    """Metadata loader for raw Objaverse Cars GLB assets."""

    def __init__(self, root: str, filter_vehicle: bool = True) -> None:
        self.root = Path(root)
        ann = self.root / "annotations.json"
        self.annotations = json.loads(ann.read_text()) if ann.exists() else {}
        glbs: List[Path] = sorted((self.root / "glbs").rglob("*.glb"))
        if filter_vehicle and self.annotations:
            glbs = [path for path in glbs if _is_vehicle_annotation(self.annotations.get(path.stem, {}))]
        self.glbs = glbs

    def __len__(self) -> int:
        return len(self.glbs)

    def __getitem__(self, idx: int) -> Dict:
        mesh_path = self.glbs[idx]
        uid = mesh_path.stem
        return {
            "mesh_path": str(mesh_path),
            "uid": uid,
            "annotation": self.annotations.get(uid, {}),
            "dataset_name": "objaverse_cars",
        }


def _is_vehicle_annotation(annotation: Dict) -> bool:
    text = json.dumps(annotation, ensure_ascii=False).lower()
    return any(token in text for token in ("car", "vehicle", "truck", "bus", "van", "automobile"))
