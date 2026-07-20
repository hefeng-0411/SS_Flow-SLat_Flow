from __future__ import annotations

from typing import Dict, Optional

import torch

def geometry_metrics(
    pred_points: torch.Tensor,
    gt_points: Optional[torch.Tensor] = None,
    *,
    threshold: float = 0.01,
    real_mode: bool = False,
    chunk_size: int = 4096,
) -> Dict[str, float]:
    """Compute deterministic symmetric point-set metrics in canonical units.

    ``Chamfer Distance`` is the mean bidirectional Euclidean distance,
    ``Chamfer-L2`` is the mean bidirectional squared distance, and F-score uses
    the strict Euclidean threshold supplied by the caller. Inputs must already
    share one object coordinate frame; this function never independently
    recenters or rescales predictions and references.
    """
    if gt_points is None:
        centered = pred_points - pred_points.mean(dim=0, keepdim=True)
        radius = centered.norm(dim=-1)
        return {
            "outlier ratio": float((radius > radius.median() * 4.0).float().mean().detach().cpu()),
            "surface coverage": float((radius <= radius.quantile(0.95)).float().mean().detach().cpu()),
        }
    pred = pred_points.detach().float()
    gt = gt_points.detach().to(device=pred.device, dtype=torch.float32)
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape[1] != 3 or gt.shape[1] != 3:
        raise ValueError(f"geometry point sets must be [N,3], got {tuple(pred.shape)} and {tuple(gt.shape)}")
    if pred.numel() == 0 or gt.numel() == 0:
        raise ValueError("geometry point sets must be non-empty")
    if not torch.isfinite(pred).all() or not torch.isfinite(gt).all():
        raise ValueError("geometry point sets contain NaN or Inf")
    d_pg = _nearest_distances(pred, gt, chunk_size=chunk_size)
    d_gp = _nearest_distances(gt, pred, chunk_size=chunk_size)
    precision = (d_pg < threshold).float().mean()
    recall = (d_gp < threshold).float().mean()
    fscore = 2 * precision * recall / (precision + recall).clamp_min(1e-6)
    return {
        "Chamfer Distance": float((0.5 * (d_pg.mean() + d_gp.mean())).detach().cpu()),
        "Chamfer-L2": float((0.5 * (d_pg.square().mean() + d_gp.square().mean())).detach().cpu()),
        "F-score": float(fscore.detach().cpu()),
        "F-score precision": float(precision.detach().cpu()),
        "F-score recall": float(recall.detach().cpu()),
        "F-score threshold": float(threshold),
        "surface coverage": float(recall.detach().cpu()),
        "point-to-surface distance": float(d_pg.mean().detach().cpu()),
    }


def _nearest_distances(query: torch.Tensor, reference: torch.Tensor, *, chunk_size: int) -> torch.Tensor:
    try:
        from scipy.spatial import cKDTree

        query_np = query.detach().cpu().numpy()
        reference_np = reference.detach().cpu().numpy()
        distances, _ = cKDTree(reference_np).query(query_np, k=1, workers=-1)
        return torch.from_numpy(distances).to(dtype=torch.float32)
    except (ImportError, TypeError):
        # Older/minimal environments retain a bounded-memory exact fallback.
        pass
    chunk_size = max(1, int(chunk_size))
    distances = []
    # Chunk both axes so 100k-point asset evaluation has bounded memory.
    reference_chunk = max(chunk_size, 1024)
    for q_start in range(0, query.shape[0], chunk_size):
        q = query[q_start : q_start + chunk_size]
        best = torch.full((q.shape[0],), float("inf"), device=q.device, dtype=q.dtype)
        for r_start in range(0, reference.shape[0], reference_chunk):
            r = reference[r_start : r_start + reference_chunk]
            best = torch.minimum(best, torch.cdist(q, r).amin(dim=1))
        distances.append(best)
    return torch.cat(distances, dim=0)
