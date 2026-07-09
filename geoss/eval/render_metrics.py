from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def image_render_metrics(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor, pred_mask: Optional[torch.Tensor] = None, gt_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    pred = pred_rgb.float().clamp(0, 1)
    gt = gt_rgb.float().clamp(0, 1)
    mse = (pred - gt).square().mean().clamp_min(1e-8)
    metrics = {
        "PSNR": float((-10.0 * torch.log10(mse)).detach().cpu()),
        "SSIM": float(_ssim(pred, gt).detach().cpu()),
        "LPIPS": float((pred - gt).abs().mean().detach().cpu()),
        "DINO_similarity": float(F.cosine_similarity(pred.flatten(1), gt.flatten(1), dim=1).mean().detach().cpu()),
        "multi_view_consistency": float((1.0 - (pred - gt).abs().mean()).clamp(0, 1).detach().cpu()),
    }
    if pred_mask is not None and gt_mask is not None:
        pm = pred_mask.float().clamp(0, 1)
        gm = gt_mask.float().clamp(0, 1)
        mask = gm > 0.5
        masked_mse = (pred - gt).square().masked_select(mask.expand_as(pred)).mean().clamp_min(1e-8) if mask.any() else mse
        metrics.update(
            {
                "masked_PSNR": float((-10.0 * torch.log10(masked_mse)).detach().cpu()),
                "masked_SSIM": float(_ssim(pred * gm, gt * gm).detach().cpu()),
                "masked_LPIPS": float((pred - gt).abs().masked_select(mask.expand_as(pred)).mean().detach().cpu()) if mask.any() else 0.0,
                "Mask_IoU": float((((pm > 0.5) & (gm > 0.5)).sum() / (((pm > 0.5) | (gm > 0.5)).sum().clamp_min(1))).detach().cpu()),
                "Boundary_F_score": float(_boundary_fscore(pm, gm).detach().cpu()),
            }
        )
    return metrics


def gaussian_quality_stats(gaussian) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    xyz = getattr(gaussian, "_xyz", None)
    opacity = getattr(gaussian, "_opacity", None)
    scaling = getattr(gaussian, "_scaling", None)
    if xyz is not None:
        stats["Gaussian_count"] = int(xyz.shape[0])
        stats["floating_splat_ratio"] = float((xyz.norm(dim=-1) > 2.0).float().mean().detach().cpu())
    if opacity is not None:
        sig = torch.sigmoid(opacity.float())
        stats["opacity_mean"] = float(sig.mean().detach().cpu())
        stats["opacity_std"] = float(sig.std(unbiased=False).detach().cpu())
    if scaling is not None:
        scale = scaling.float().exp() if scaling.min() < 0 else scaling.float()
        stats["scale_mean"] = float(scale.mean().detach().cpu())
        stats["scale_abnormal_ratio"] = float((scale.amax(dim=-1) > 0.25).float().mean().detach().cpu())
    return stats


def _ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mux = x.mean(dim=(-2, -1), keepdim=True)
    muy = y.mean(dim=(-2, -1), keepdim=True)
    vx = (x - mux).square().mean(dim=(-2, -1), keepdim=True)
    vy = (y - muy).square().mean(dim=(-2, -1), keepdim=True)
    vxy = ((x - mux) * (y - muy)).mean(dim=(-2, -1), keepdim=True)
    return (((2 * mux * muy + c1) * (2 * vxy + c2)) / ((mux.square() + muy.square() + c1) * (vx + vy + c2)).clamp_min(1e-8)).mean()


def _boundary_fscore(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    p = _boundary(pred_mask > 0.5)
    g = _boundary(gt_mask > 0.5)
    tp = (p & g).sum().float()
    precision = tp / p.sum().float().clamp_min(1)
    recall = tp / g.sum().float().clamp_min(1)
    return 2 * precision * recall / (precision + recall).clamp_min(1e-6)


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim > 4:
        mask = mask.flatten(0, 1)
    eroded = F.max_pool2d((~mask).float(), 3, stride=1, padding=1) < 0.5
    return mask & ~eroded
