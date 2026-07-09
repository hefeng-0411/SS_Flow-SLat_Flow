from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Sim3:
    scale: torch.Tensor
    rotation: torch.Tensor
    translation: torch.Tensor


def estimate_sim3_umeyama(src: torch.Tensor, dst: torch.Tensor, weights: Optional[torch.Tensor] = None, eps: float = 1e-6) -> Sim3:
    """Weighted Umeyama Sim(3) alignment for batched point sets.

    Args:
        src: `[B,P,3]` VGGT/world points.
        dst: `[B,P,3]` canonical target points.
        weights: optional `[B,P]` robust foreground/confidence weights.
    """
    if src.ndim != 3 or dst.ndim != 3 or src.shape != dst.shape or src.shape[-1] != 3:
        raise ValueError(f"src/dst must both be [B,P,3], got {tuple(src.shape)} {tuple(dst.shape)}")
    B, P, _ = src.shape
    if weights is None:
        weights = torch.ones(B, P, device=src.device, dtype=src.dtype)
    weights = weights.to(device=src.device, dtype=src.dtype).clamp_min(0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)
    mu_src = (src * weights[..., None]).sum(dim=1, keepdim=True)
    mu_dst = (dst * weights[..., None]).sum(dim=1, keepdim=True)
    xs = src - mu_src
    xd = dst - mu_dst
    cov = torch.matmul((weights[..., None] * xd).transpose(1, 2), xs)
    U, S, Vh = torch.linalg.svd(cov)
    det = torch.det(torch.matmul(U, Vh))
    D = torch.eye(3, device=src.device, dtype=src.dtype).expand(B, 3, 3).clone()
    D[:, -1, -1] = torch.where(det < 0, -1.0, 1.0)
    R = torch.matmul(torch.matmul(U, D), Vh)
    var_src = (weights * (xs.square().sum(dim=-1))).sum(dim=1).clamp_min(eps)
    scale = (S * torch.diagonal(D, dim1=-2, dim2=-1)).sum(dim=1) / var_src
    t = mu_dst[:, 0] - scale[:, None] * torch.matmul(mu_src[:, 0:1], R.transpose(1, 2))[:, 0]
    return Sim3(scale=scale, rotation=R, translation=t)


class GeometryAlignment(nn.Module):
    """Align VGGT geometry to the TRELLIS canonical cube before ray evidence use."""

    def __init__(self, robust_percentile: float = 0.95, canonical_extent: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.robust_percentile = robust_percentile
        self.canonical_extent = canonical_extent
        self.eps = eps

    def forward(
        self,
        *,
        vggt_depth: Optional[torch.Tensor],
        vggt_pointmap: Optional[torch.Tensor],
        K: torch.Tensor,
        c2w: torch.Tensor,
        w2c: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        vggt_confidence: Optional[torch.Tensor] = None,
        dataset_depth: Optional[torch.Tensor] = None,
        vggt_camera: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        if vggt_pointmap is None:
            raise ValueError("GeometryAlignment requires vggt_pointmap in real geometry mode.")
        if vggt_pointmap.ndim != 5 or vggt_pointmap.shape[2] != 3:
            raise ValueError(f"vggt_pointmap must be [B,N,3,H,W], got {tuple(vggt_pointmap.shape)}")
        B, N, _, H, W = vggt_pointmap.shape
        device, dtype = vggt_pointmap.device, vggt_pointmap.dtype
        mask = _mask_to_weight(masks, B, N, H, W, device, dtype)
        if vggt_confidence is not None:
            conf = vggt_confidence
            if conf.ndim == 4:
                conf = conf.unsqueeze(2)
            conf = F.interpolate(conf.reshape(B * N, 1, conf.shape[-2], conf.shape[-1]).float(), size=(H, W), mode="bilinear", align_corners=False)
            conf = conf.reshape(B, N, 1, H, W).to(dtype)
            mask = mask * conf.clamp(0, 1)

        points = vggt_pointmap.permute(0, 1, 3, 4, 2).reshape(B, N * H * W, 3)
        weights = mask.reshape(B, N * H * W)
        target = _canonicalize_by_bbox(points, weights, self.canonical_extent, self.robust_percentile, self.eps)
        sim3 = estimate_sim3_umeyama(points, target, weights)
        aligned_points = _apply_sim3(points, sim3).reshape(B, N, H, W, 3).permute(0, 1, 4, 2, 3).contiguous()

        aligned_depth, depth_scale, depth_shift = _align_depth(
            vggt_depth,
            dataset_depth,
            aligned_points,
            K,
            w2c,
            mask,
            eps=self.eps,
        )
        residual = ((aligned_points.permute(0, 1, 3, 4, 2).reshape(B, -1, 3) - target).norm(dim=-1) * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(self.eps)
        bbox_size = _weighted_extent(aligned_points.permute(0, 1, 3, 4, 2).reshape(B, -1, 3), weights, self.robust_percentile)
        alignment_confidence = torch.exp(-residual).view(B, 1, 1, 1, 1).clamp(0, 1) * (bbox_size.mean(dim=-1).view(B, 1, 1, 1, 1) > 0.05).to(dtype)
        pose_stats = _pose_consistency(vggt_camera, K, c2w, w2c)

        return {
            "aligned_pointmap": aligned_points,
            "aligned_depth": aligned_depth,
            "aligned_camera": {"K": K, "c2w": c2w, "w2c": w2c},
            "alignment_confidence": alignment_confidence.expand(B, N, 1, H, W).contiguous(),
            "sim3_scale": sim3.scale,
            "sim3_rotation": sim3.rotation,
            "sim3_translation": sim3.translation,
            "depth_scale": depth_scale,
            "depth_shift": depth_shift,
            "alignment_residual": residual,
            "pose_stats": pose_stats,
        }


def align_vggt_batch(batch: Dict[str, torch.Tensor], aligner: Optional[GeometryAlignment] = None) -> Dict[str, torch.Tensor]:
    aligner = aligner or GeometryAlignment()
    out = aligner(
        vggt_depth=batch.get("vggt_depth"),
        vggt_pointmap=batch.get("vggt_pointmap"),
        K=batch["K"],
        c2w=batch["c2w"],
        w2c=batch["w2c"],
        masks=batch.get("masks"),
        vggt_confidence=batch.get("vggt_confidence"),
        dataset_depth=batch.get("depths"),
        vggt_camera=batch.get("vggt_camera"),
    )
    batch = dict(batch)
    batch["aligned_pointmap"] = out["aligned_pointmap"]
    batch["aligned_depth"] = out["aligned_depth"]
    batch["alignment_confidence"] = out["alignment_confidence"]
    batch["alignment_debug"] = {k: v for k, v in out.items() if k not in {"aligned_pointmap", "aligned_depth", "alignment_confidence"}}
    return batch


def _apply_sim3(points: torch.Tensor, sim3: Sim3) -> torch.Tensor:
    return sim3.scale[:, None, None] * torch.matmul(points, sim3.rotation.transpose(1, 2)) + sim3.translation[:, None]


def _mask_to_weight(masks, B, N, H, W, device, dtype) -> torch.Tensor:
    if masks is None:
        return torch.ones(B, N, 1, H, W, device=device, dtype=dtype)
    if masks.ndim != 5:
        raise ValueError(f"masks must be [B,N,1,H,W], got {tuple(masks.shape)}")
    if masks.shape[-2:] != (H, W):
        masks = F.interpolate(masks.reshape(B * N, 1, masks.shape[-2], masks.shape[-1]).float(), size=(H, W), mode="bilinear", align_corners=False)
        masks = masks.reshape(B, N, 1, H, W)
    return masks.to(device=device, dtype=dtype).clamp(0, 1)


def _canonicalize_by_bbox(points: torch.Tensor, weights: torch.Tensor, extent: float, q: float, eps: float) -> torch.Tensor:
    lo = _weighted_quantile(points, weights, (1.0 - q) * 0.5)
    hi = _weighted_quantile(points, weights, 1.0 - (1.0 - q) * 0.5)
    center = (lo + hi) * 0.5
    scale = (hi - lo).amax(dim=-1, keepdim=True).clamp_min(eps) / (2.0 * extent)
    return ((points - center[:, None]) / scale[:, None]).clamp(-extent, extent)


def _weighted_quantile(points: torch.Tensor, weights: torch.Tensor, q: float) -> torch.Tensor:
    # Robust percentile filtering; weights are used as a foreground selector.
    out = []
    for d in range(3):
        vals = points[..., d]
        fill = torch.full_like(vals, float("nan"))
        masked = torch.where(weights > 1e-6, vals, fill)
        out.append(torch.nanquantile(masked, q, dim=1))
    return torch.stack(out, dim=-1)


def _weighted_extent(points: torch.Tensor, weights: torch.Tensor, q: float) -> torch.Tensor:
    return (_weighted_quantile(points, weights, q) - _weighted_quantile(points, weights, 1.0 - q)).abs()


def _align_depth(vggt_depth, dataset_depth, aligned_points, K, w2c, mask, eps: float):
    B, N, _, H, W = aligned_points.shape
    if vggt_depth is None:
        depth = aligned_points[:, :, 2:3].abs()
        return depth, torch.ones(B, N, 1, device=aligned_points.device, dtype=aligned_points.dtype), torch.zeros(B, N, 1, device=aligned_points.device, dtype=aligned_points.dtype)
    vggt_depth = vggt_depth.to(device=aligned_points.device, dtype=aligned_points.dtype)
    if dataset_depth is None:
        target_depth = aligned_points[:, :, 2:3].abs()
    else:
        target_depth = dataset_depth.to(device=aligned_points.device, dtype=aligned_points.dtype)
        if target_depth.shape[-2:] != (H, W):
            target_depth = F.interpolate(target_depth.reshape(B * N, 1, target_depth.shape[-2], target_depth.shape[-1]).float(), size=(H, W), mode="bilinear", align_corners=False).reshape(B, N, 1, H, W).to(aligned_points.dtype)
    w = mask
    x = vggt_depth.reshape(B, N, -1)
    y = target_depth.reshape(B, N, -1)
    ww = w.reshape(B, N, -1)
    mx = (x * ww).sum(dim=-1, keepdim=True) / ww.sum(dim=-1, keepdim=True).clamp_min(eps)
    my = (y * ww).sum(dim=-1, keepdim=True) / ww.sum(dim=-1, keepdim=True).clamp_min(eps)
    var = ((x - mx).square() * ww).sum(dim=-1, keepdim=True).clamp_min(eps)
    scale = (((x - mx) * (y - my) * ww).sum(dim=-1, keepdim=True) / var).clamp(1e-3, 1e3)
    shift = my - scale * mx
    aligned = scale.view(B, N, 1, 1, 1) * vggt_depth + shift.view(B, N, 1, 1, 1)
    return aligned, scale, shift


def _pose_consistency(vggt_camera, K, c2w, w2c) -> Dict[str, torch.Tensor]:
    if not isinstance(vggt_camera, dict) or "w2c" not in vggt_camera:
        zero = torch.zeros(K.shape[:2], device=K.device, dtype=K.dtype)
        return {"available": torch.tensor(False, device=K.device), "pose_residual": zero}
    pred_w2c = vggt_camera["w2c"].to(device=w2c.device, dtype=w2c.dtype)
    pose_residual = (pred_w2c[..., :3, 3] - w2c[..., :3, 3]).norm(dim=-1)
    return {"available": torch.tensor(True, device=K.device), "pose_residual": pose_residual}
