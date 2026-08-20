from __future__ import annotations

import os
from pathlib import Path

import torch


def configure_trellis_hub(args) -> dict[str, str | None]:
    """Resolve TRELLIS' DINO dependency locally before model construction.

    Native TRELLIS calls ``torch.hub.load('facebookresearch/dinov2', ...)``.
    Even with a populated torch cache that API can query GitHub to resolve the
    default branch.  Rewriting only that repository call to ``source='local'``
    makes training reproducible when the verified checkout is already present.
    """

    hub_dir = resolve_torch_hub_dir(args)
    if hub_dir is not None:
        torch.hub.set_dir(str(hub_dir))
        os.environ["TORCH_HUB_DIR"] = str(hub_dir)
        os.environ.setdefault(
            "TORCH_HOME", str(hub_dir.parent if hub_dir.name == "hub" else hub_dir)
        )
    dinov2_repo = resolve_dinov2_repo(args, hub_dir)
    if dinov2_repo is not None:
        patch_torch_hub_for_local_dinov2(dinov2_repo)
    return {
        "torch_hub_dir": str(hub_dir) if hub_dir is not None else None,
        "dinov2_repo": str(dinov2_repo) if dinov2_repo is not None else None,
    }


def resolve_torch_hub_dir(args) -> Path | None:
    candidates: list[Path] = []
    for value in (
        getattr(args, "torch_hub_dir", None),
        os.environ.get("TORCH_HUB_DIR"),
    ):
        if value:
            candidates.append(Path(value).expanduser())
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        home = Path(torch_home).expanduser()
        candidates.extend([home / "hub", home])
    candidates.extend(
        [
            Path.home() / ".cache" / "torch" / "hub",
            Path("/mnt/sda/hf/.cache/torch/hub"),
            Path("/mnt/sda3/yu/checkpoints/hub/hub"),
            Path("/mnt/sda3/yu/checkpoints/hub"),
        ]
    )
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if (candidate / "facebookresearch_dinov2_main" / "hubconf.py").exists():
            return candidate
        nested = candidate / "hub"
        if (nested / "facebookresearch_dinov2_main" / "hubconf.py").exists():
            return nested
        if candidate.exists():
            return candidate
    return None


def resolve_dinov2_repo(args, hub_dir: Path | None) -> Path | None:
    candidates: list[Path] = []
    explicit = getattr(args, "dinov2_repo", None)
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if hub_dir is not None:
        candidates.append(hub_dir / "facebookresearch_dinov2_main")
        candidates.extend(sorted(hub_dir.glob("facebookresearch_dinov2*")))
    for candidate in candidates:
        if (candidate / "hubconf.py").exists():
            return candidate
    return None


def patch_torch_hub_for_local_dinov2(local_repo: Path) -> None:
    if getattr(torch.hub, "_geoss_local_dinov2_patch", False):
        return
    original_load = torch.hub.load

    def load(repo_or_dir, model, *args, **kwargs):
        if str(repo_or_dir).rstrip("/") == "facebookresearch/dinov2":
            local_kwargs = dict(kwargs)
            local_kwargs.pop("trust_repo", None)
            local_kwargs.pop("force_reload", None)
            return original_load(
                str(local_repo), model, *args, source="local", **local_kwargs
            )
        return original_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = load
    torch.hub._geoss_local_dinov2_patch = True
