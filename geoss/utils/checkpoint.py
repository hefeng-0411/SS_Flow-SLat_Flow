from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def load_state_dict_flexible(module: torch.nn.Module, path: str | Path, strict: bool = False) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = module.load_state_dict(state, strict=strict)
    return {"missing_keys": missing, "unexpected_keys": unexpected}


def save_checkpoint(path: str | Path, **payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
