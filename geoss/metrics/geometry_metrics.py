from __future__ import annotations

from typing import Dict, Optional

import torch

from geoss.utils.optional_deps import require_dependency


def geometry_metrics(pred_points: torch.Tensor, gt_points: Optional[torch.Tensor] = None, *, threshold: float = 0.01, real_mode: bool = False) -> Dict[str, float]:
    if gt_points is None:
        centered = pred_points - pred_points.mean(dim=0, keepdim=True)
        radius = centered.norm(dim=-1)
        return {
            "outlier ratio": float((radius > radius.median() * 4.0).float().mean().detach().cpu()),
            "surface coverage": float((radius <= radius.quantile(0.95)).float().mean().detach().cpu()),
        }
    if require_dependency("point_cloud_utils", real_mode=real_mode, feature="Chamfer/F-score geometry metrics"):
        import point_cloud_utils as pcu

        p = pred_points.detach().cpu().numpy()
        g = gt_points.detach().cpu().numpy()
        d_pg, _ = pcu.k_nearest_neighbors(p, g, k=1)
        d_gp, _ = pcu.k_nearest_neighbors(g, p, k=1)
        d_pg = torch.tensor(d_pg).float().sqrt()
        d_gp = torch.tensor(d_gp).float().sqrt()
    else:
        d = torch.cdist(pred_points.float(), gt_points.float())
        d_pg = d.min(dim=1).values
        d_gp = d.min(dim=0).values
    precision = (d_pg < threshold).float().mean()
    recall = (d_gp < threshold).float().mean()
    fscore = 2 * precision * recall / (precision + recall).clamp_min(1e-6)
    return {
        "Chamfer Distance": float((d_pg.mean() + d_gp.mean()).detach().cpu()),
        "F-score": float(fscore.detach().cpu()),
        "surface coverage": float(recall.detach().cpu()),
        "point-to-surface distance": float(d_pg.mean().detach().cpu()),
    }
