from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F


_LPIPS_MODELS: dict[str, torch.nn.Module] = {}


def image_render_metrics(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor, pred_mask: Optional[torch.Tensor] = None, gt_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Compute full-reference metrics in sRGB [0, 1].

    PSNR uses per-image RGB MSE, SSIM uses the conventional 11x11 Gaussian
    window (sigma 1.5), and LPIPS uses the learned VGG network on [-1, 1].
    Callers are responsible for compositing RGBA references onto the declared
    evaluation background before invoking this function.
    """
    pred = _as_nchw(pred_rgb).float().clamp(0, 1)
    gt = _as_nchw(gt_rgb).to(device=pred.device, dtype=torch.float32).clamp(0, 1)
    if pred.shape != gt.shape:
        raise ValueError(f"render/GT shape mismatch: predicted {tuple(pred.shape)}, GT {tuple(gt.shape)}")
    mse = (pred - gt).square().mean().clamp_min(1e-8)
    metrics = {
        "PSNR": float((-10.0 * torch.log10(mse)).detach().cpu()),
        "SSIM": float(_ssim(pred, gt).detach().cpu()),
        "LPIPS": float(_lpips(pred, gt).detach().cpu()),
    }
    if pred_mask is not None and gt_mask is not None:
        pm = _as_mask_nchw(pred_mask).to(device=pred.device, dtype=torch.float32).clamp(0, 1)
        gm = _as_mask_nchw(gt_mask).to(device=pred.device, dtype=torch.float32).clamp(0, 1)
        if pm.shape != gm.shape or pm.shape[0] != pred.shape[0] or pm.shape[-2:] != pred.shape[-2:]:
            raise ValueError(f"render mask shape mismatch: predicted {tuple(pm.shape)}, GT {tuple(gm.shape)}, RGB {tuple(pred.shape)}")
        mask = gm > 0.5
        masked_mse = (pred - gt).square().masked_select(mask.expand_as(pred)).mean().clamp_min(1e-8) if mask.any() else mse
        pred_fg = pred * gm
        gt_fg = gt * gm
        metrics.update(
            {
                "masked_PSNR": float((-10.0 * torch.log10(masked_mse)).detach().cpu()),
                "masked_SSIM": float(_ssim(pred_fg, gt_fg).detach().cpu()),
                "masked_LPIPS": float(_lpips(pred_fg, gt_fg).detach().cpu()),
                "foreground_L1": float((pred - gt).abs().masked_select(mask.expand_as(pred)).mean().detach().cpu()) if mask.any() else 0.0,
                "Mask_IoU": float((((pm > 0.5) & (gm > 0.5)).sum() / (((pm > 0.5) | (gm > 0.5)).sum().clamp_min(1))).detach().cpu()),
                "Boundary_F_score": float(_boundary_fscore(pm, gm).detach().cpu()),
            }
        )
    return metrics


def _as_nchw(images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4:
        raise ValueError(f"render tensors must be [N,3,H,W] or [N,H,W,3], got {tuple(images.shape)}")
    if images.shape[1] == 3:
        return images
    if images.shape[-1] == 3:
        return images.permute(0, 3, 1, 2)
    raise ValueError(f"render tensors need three RGB channels, got {tuple(images.shape)}")


def _as_mask_nchw(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None] if mask.shape[-1] != 1 else mask.permute(2, 0, 1)[None]
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask.permute(0, 3, 1, 2)
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"render masks must be [N,1,H,W] or [N,H,W,1], got {tuple(mask.shape)}")
    return mask


def _lpips(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Use the learned LPIPS metric; never substitute pixel L1 under its name."""
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("LPIPS render evaluation requires the 'lpips' package; install requirements/gsplat.txt or requirements/all.txt.") from exc
    key = str(pred.device)
    model = _LPIPS_MODELS.get(key)
    if model is None:
        model = lpips.LPIPS(net="vgg").to(pred.device).eval()
        _LPIPS_MODELS[key] = model
    with torch.no_grad():
        return model(pred * 2.0 - 1.0, gt * 2.0 - 1.0).mean()


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


def _ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError(f"SSIM inputs must have identical shape, got {tuple(x.shape)} and {tuple(y.shape)}")
    # Match the standard Wang et al. local-window formulation. For very small
    # unit-test images, use the largest odd window that fits.
    max_window = min(int(x.shape[-2]), int(x.shape[-1]), int(window_size))
    window_size = max_window if max_window % 2 == 1 else max_window - 1
    window_size = max(1, window_size)
    coords = torch.arange(window_size, device=x.device, dtype=x.dtype) - (window_size - 1) / 2.0
    kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    channels = x.shape[1]
    kernel = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    padding = window_size // 2
    pad_mode = "reflect" if min(x.shape[-2:]) > padding else "replicate"
    x_pad = F.pad(x, (padding, padding, padding, padding), mode=pad_mode)
    y_pad = F.pad(y, (padding, padding, padding, padding), mode=pad_mode)
    mu_x = F.conv2d(x_pad, kernel, groups=channels)
    mu_y = F.conv2d(y_pad, kernel, groups=channels)
    mu_x_sq, mu_y_sq, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x = (F.conv2d(x_pad * x_pad, kernel, groups=channels) - mu_x_sq).clamp_min(0.0)
    sigma_y = (F.conv2d(y_pad * y_pad, kernel, groups=channels) - mu_y_sq).clamp_min(0.0)
    sigma_xy = F.conv2d(x_pad * y_pad, kernel, groups=channels) - mu_xy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        ((mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)).clamp_min(1e-12)
    )
    return score.mean()


def _boundary_fscore(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    p = _boundary(pred_mask > 0.5)
    g = _boundary(gt_mask > 0.5)
    h, w = int(p.shape[-2]), int(p.shape[-1])
    tolerance = max(1, int(round(0.0075 * math.sqrt(h * h + w * w))))
    kernel = 2 * tolerance + 1
    p_near = F.max_pool2d(p.float(), kernel, stride=1, padding=tolerance) > 0.5
    g_near = F.max_pool2d(g.float(), kernel, stride=1, padding=tolerance) > 0.5
    precision = (p & g_near).sum().float() / p.sum().float().clamp_min(1)
    recall = (g & p_near).sum().float() / g.sum().float().clamp_min(1)
    return 2 * precision * recall / (precision + recall).clamp_min(1e-6)


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim > 4:
        mask = mask.flatten(0, 1)
    eroded = F.max_pool2d((~mask).float(), 3, stride=1, padding=1) < 0.5
    return mask & ~eroded
