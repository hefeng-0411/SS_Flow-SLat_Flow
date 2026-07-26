from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Optional

import torch

from geoss.integration.real_trellis_pipeline import RealTrellisGeoPipeline
from geoss.reconstruction.priors.trellis_completion import (
    TrellisPriorInput,
)


class ConditioningTrellisPriorProvider:
    """Generate or load conditioning-only TRELLIS completion priors by exact UID."""

    protocol_version = "rapc3d_conditioning_trellis_prior_v2"

    def __init__(
        self,
        pipeline: RealTrellisGeoPipeline,
        *,
        cache_directory: Optional[str | Path] = None,
        cache_only: bool = False,
        seed: int = 20260720,
        mask_aware_crop: bool = True,
        crop_padding: float = 1.2,
        include_slat_appearance: bool = True,
        slat_resolution: int = 64,
        dense_resolution: int = 32,
    ) -> None:
        self.pipeline = pipeline
        self.cache_directory = Path(cache_directory) if cache_directory is not None else None
        self.cache_only = bool(cache_only)
        self.seed = int(seed)
        self.mask_aware_crop = bool(mask_aware_crop)
        self.crop_padding = float(crop_padding)
        self.include_slat_appearance = bool(include_slat_appearance)
        self.slat_resolution = int(slat_resolution)
        self.dense_resolution = int(dense_resolution)
        if self.cache_directory is not None:
            self.cache_directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, batch: Dict[str, torch.Tensor]) -> TrellisPriorInput:
        images = batch["images"]
        masks = batch["masks"]
        uids = batch.get("uid")
        if not isinstance(uids, list) or len(uids) != images.shape[0]:
            raise ValueError("TRELLIS prior generation requires the exact UID list for every batch item")
        latents = []
        slat_grids = []
        slat_masks = []
        for index, uid in enumerate(uids):
            payload = self._load(uid)
            if payload is None:
                if self.cache_only:
                    raise FileNotFoundError(f"No conditioning TRELLIS prior cache for uid={uid!r}")
                generator = (
                    self.pipeline.generate_completion_prior
                    if self.include_slat_appearance
                    else self.pipeline.generate_sparse_structure_prior
                )
                result = generator(
                    images[index : index + 1],
                    masks=masks[index],
                    mask_aware_crop=self.mask_aware_crop,
                    crop_padding=self.crop_padding,
                    seed=_uid_seed(uid, self.seed),
                )
                payload = {
                    "ss_latent_grid": result["ss_latent_grid"].detach().float().cpu(),
                }
                if self.include_slat_appearance:
                    payload.update(
                        {
                            "slat_feats": result["slat_feats"].detach().float().cpu(),
                            "slat_coords": result["coords"].detach().long().cpu(),
                        }
                    )
                self._save(uid, payload)
            latents.append(payload["ss_latent_grid"])
            if self.include_slat_appearance:
                if "slat_feats" not in payload or "slat_coords" not in payload:
                    raise ValueError(
                        f"TRELLIS completion prior for uid={uid!r} has no generated SLAT appearance"
                    )
                grid, occupancy = _sparse_slat_to_dense(
                    payload["slat_feats"],
                    payload["slat_coords"],
                    resolution=self.dense_resolution,
                    source_resolution=self.slat_resolution,
                )
                slat_grids.append(grid)
                slat_masks.append(occupancy)
        latent_batch = torch.cat(latents, dim=0).to(device=images.device, dtype=torch.float32)
        return TrellisPriorInput(
            ss_latent_grid=latent_batch,
            slat_dense_grid=(
                torch.cat(slat_grids, dim=0).to(device=images.device, dtype=torch.float32)
                if slat_grids
                else None
            ),
            slat_occupancy=(
                torch.cat(slat_masks, dim=0).to(device=images.device, dtype=torch.float32)
                if slat_masks
                else None
            ),
        )

    def _path(self, uid: str) -> Optional[Path]:
        if self.cache_directory is None:
            return None
        safe_uid = "".join(character for character in uid if character.isalnum() or character in "-_")
        if not safe_uid or safe_uid != uid:
            raise ValueError(f"Unsafe or non-canonical UID {uid!r}")
        return self.cache_directory / f"{safe_uid}.pt"

    def _load(self, uid: str) -> Optional[Dict[str, torch.Tensor]]:
        path = self._path(uid)
        if path is None or not path.exists():
            return None
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("protocol_version") != self.protocol_version:
            raise ValueError(f"Invalid RAPC TRELLIS prior cache provenance for uid={uid!r}: {path}")
        if payload.get("uid") != uid or not isinstance(payload.get("ss_latent_grid"), torch.Tensor):
            raise ValueError(f"RAPC TRELLIS prior cache identity/payload mismatch for uid={uid!r}: {path}")
        result = {"ss_latent_grid": payload["ss_latent_grid"].float()}
        for name, dtype in (("slat_feats", torch.float32), ("slat_coords", torch.long)):
            value = payload.get(name)
            if isinstance(value, torch.Tensor):
                result[name] = value.to(dtype=dtype)
        return result

    def _save(self, uid: str, prior: Dict[str, torch.Tensor]) -> None:
        path = self._path(uid)
        if path is None:
            return
        temporary = path.with_suffix(f".pt.tmp.{os.getpid()}")
        torch.save(
            {
                "protocol_version": self.protocol_version,
                "uid": uid,
                "conditioning_only": True,
                "test_time_ground_truth_latents_used": False,
                "ss_latent_grid": prior["ss_latent_grid"],
                "slat_feats": prior.get("slat_feats"),
                "slat_coords": prior.get("slat_coords"),
            },
            temporary,
        )
        temporary.replace(path)


def _uid_seed(uid: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{uid}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % (2**31 - 1)


def _sparse_slat_to_dense(
    features: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    resolution: int,
    source_resolution: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter one TRELLIS sparse latent into its canonical z-y-x lattice."""

    if features.ndim != 2:
        raise ValueError(f"SLAT features must be [N,C], got {tuple(features.shape)}")
    if coordinates.ndim != 2 or coordinates.shape != (features.shape[0], 4):
        raise ValueError("SLAT coordinates must be [N,4] and align with features")
    if resolution < 2:
        raise ValueError("SLAT resolution must be at least two")
    source_resolution = int(source_resolution or resolution)
    spatial = coordinates[:, 1:4].long()
    if bool(((spatial < 0) | (spatial >= source_resolution)).any()):
        raise ValueError(
            f"SLAT coordinates fall outside the declared {source_resolution}^3 source lattice"
        )
    if source_resolution != resolution:
        spatial = torch.div(
            spatial * resolution,
            source_resolution,
            rounding_mode="floor",
        ).clamp_max(resolution - 1)
    linear = (
        spatial[:, 0] * resolution * resolution
        + spatial[:, 1] * resolution
        + spatial[:, 2]
    )
    channels = features.shape[1]
    flat = torch.zeros(
        resolution**3,
        channels,
        device=features.device,
        dtype=features.dtype,
    )
    count = torch.zeros(
        resolution**3,
        1,
        device=features.device,
        dtype=features.dtype,
    )
    flat.index_add_(0, linear, features)
    count.index_add_(0, linear, torch.ones_like(features[:, :1]))
    flat = flat / count.clamp_min(1.0)
    grid = flat.reshape(resolution, resolution, resolution, channels).permute(
        3, 0, 1, 2
    )[None]
    occupancy = (count > 0).to(features.dtype).reshape(
        1, 1, resolution, resolution, resolution
    )
    return grid, occupancy
