from __future__ import annotations

import atexit
from pathlib import Path
from typing import Any, Dict

import torch

from geoss.utils.elastic_engine import AsyncArtifactManager, write_checkpoint_atomic


_ASYNC_MANAGER: AsyncArtifactManager | None = None


def load_state_dict_flexible(module: torch.nn.Module, path: str | Path, strict: bool = False) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = module.load_state_dict(state, strict=strict)
    return {"missing_keys": missing, "unexpected_keys": unexpected}


def save_checkpoint(path: str | Path, **payload: Any) -> None:
    if _sync_checkpoints_enabled():
        write_checkpoint_atomic(Path(path), payload)
        return
    save_checkpoint_async(path, **payload)


def save_checkpoint_async(path: str | Path, **payload: Any):
    global _ASYNC_MANAGER
    if _ASYNC_MANAGER is None:
        _ASYNC_MANAGER = AsyncArtifactManager(max_workers=1)
    return _ASYNC_MANAGER.submit_checkpoint(path, **payload)


def wait_for_async_checkpoints() -> None:
    if _ASYNC_MANAGER is not None:
        _ASYNC_MANAGER.drain()


def _sync_checkpoints_enabled() -> bool:
    import os

    return os.environ.get("GEOSS_SYNC_CHECKPOINTS", "").lower() in {"1", "true", "yes", "on"}


atexit.register(wait_for_async_checkpoints)
