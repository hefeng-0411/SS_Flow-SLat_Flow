from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch

def indices_to_active_xyz(active_indices: torch.Tensor, resolution: int = 64) -> torch.Tensor:
    """Map TRELLIS indices to decoder-world voxel centers in ``[-0.5,0.5]^3``.

    TRELLIS' Gaussian/RF decoders first compute ``(coord + 0.5) / R`` and
    instantiate representations with ``aabb=[-0.5,-0.5,-0.5,1,1,1]``.  Using
    the GeoSS ``[-1,1]`` occupancy convention here doubles every projected
    SLAT location and destroys image-feature correspondence.
    """
    if active_indices.shape[-1] == 4:
        active_indices = active_indices[..., 1:]
    if active_indices.shape[-1] != 3:
        raise ValueError(f"active_indices must end in 3 or 4 dims, got {tuple(active_indices.shape)}")
    return (active_indices.to(torch.float32) + 0.5) / float(resolution) - 0.5


def active_xyz_to_indices(active_xyz: torch.Tensor, resolution: int = 64) -> torch.Tensor:
    """Map TRELLIS decoder-world coordinates back to sparse grid indices."""
    return ((active_xyz + 0.5) * float(resolution)).floor().clamp(0, resolution - 1).long()


def sparse_coords_to_padded_indices(
    coords: torch.Tensor,
    batch_size: Optional[int] = None,
    pad_value: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, List[slice]]:
    """Convert sparse coords `[T,4]` into padded `[B,L,3]` indices and mask."""
    if coords.ndim != 2 or coords.shape[-1] != 4:
        raise ValueError(f"coords must be [T,4], got {tuple(coords.shape)}")
    if batch_size is None:
        batch_size = int(coords[:, 0].max().item()) + 1 if coords.numel() else 1
    lengths = [(coords[:, 0] == b).sum().item() for b in range(batch_size)]
    max_len = max(1, max(lengths))
    padded = torch.full((batch_size, max_len, 3), pad_value, dtype=coords.dtype, device=coords.device)
    mask = torch.zeros(batch_size, max_len, 1, dtype=torch.bool, device=coords.device)
    layout: List[slice] = []
    start = 0
    for b, length in enumerate(lengths):
        idx = torch.nonzero(coords[:, 0] == b, as_tuple=False).flatten()
        if length:
            padded[b, :length] = coords[idx, 1:]
            mask[b, :length] = True
        layout.append(slice(start, start + length))
        start += length
    return padded, mask, layout


def _infer_layout(sparse_tensor) -> List[slice]:
    layout = getattr(sparse_tensor, "layout", None)
    if layout is not None:
        return list(layout)
    coords = getattr(sparse_tensor, "coords")
    batch_size = int(getattr(sparse_tensor, "shape", [int(coords[:, 0].max().item()) + 1])[0])
    _, _, layout = sparse_coords_to_padded_indices(coords, batch_size=batch_size)
    return layout


def pad_sparse_tensor_tokens(sparse_tensor) -> Tuple[torch.Tensor, torch.Tensor, List[slice], torch.Tensor]:
    """Pad TRELLIS SparseTensor feats into `[B,L,C]` tokens without changing order."""
    feats = getattr(sparse_tensor, "feats")
    coords = getattr(sparse_tensor, "coords")
    layout = _infer_layout(sparse_tensor)
    batch_size = len(layout)
    max_len = max(1, max((sl.stop - sl.start for sl in layout), default=1))
    padded = feats.new_zeros(batch_size, max_len, feats.shape[-1])
    mask = torch.zeros(batch_size, max_len, 1, dtype=torch.bool, device=feats.device)
    active_indices = torch.zeros(batch_size, max_len, 3, dtype=coords.dtype, device=coords.device)
    for b, sl in enumerate(layout):
        length = sl.stop - sl.start
        if length:
            padded[b, :length] = feats[sl]
            active_indices[b, :length] = coords[sl, 1:]
            mask[b, :length] = True
    return padded, mask, layout, active_indices


def unpad_sparse_tensor_tokens(padded_tokens: torch.Tensor, layout: Iterable[slice]) -> torch.Tensor:
    """Restore `[sum L,C]` token order from padded `[B,L,C]` using TRELLIS layout."""
    chunks = []
    for b, sl in enumerate(layout):
        length = sl.stop - sl.start
        if length:
            chunks.append(padded_tokens[b, :length])
    if not chunks:
        return padded_tokens.new_zeros(0, padded_tokens.shape[-1])
    return torch.cat(chunks, dim=0)
