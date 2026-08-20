from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from geoss.eval.render_metrics import _as_mask_nchw, _as_nchw, _ssim
from geoss.losses.stable_bce import probability_binary_cross_entropy
from geoss.utils.optional_deps import require_dependency


def render_level_losses(
    rendered_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    rendered_alpha: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
    rendered_depth: Optional[torch.Tensor] = None,
    target_depth: Optional[torch.Tensor] = None,
    *,
    lpips_model=None,
    dino_model=None,
) -> Dict[str, torch.Tensor]:
    rendered_rgb = _as_nchw(rendered_rgb).float().clamp(0, 1)
    target_rgb = _as_nchw(target_rgb).to(device=rendered_rgb.device, dtype=rendered_rgb.dtype).clamp(0, 1)
    losses: Dict[str, torch.Tensor] = {}
    losses["L_rgb"] = F.l1_loss(rendered_rgb, target_rgb)
    if rendered_alpha is not None and target_mask is not None:
        rendered_alpha = _as_mask_nchw(rendered_alpha).to(rendered_rgb)
        target_mask = _as_mask_nchw(target_mask).to(rendered_rgb)
        losses["L_mask"] = probability_binary_cross_entropy(
            rendered_alpha.clamp(1e-4, 1 - 1e-4),
            target_mask.float().clamp(0, 1),
        )
        foreground = target_mask.expand_as(rendered_rgb).clamp(0, 1)
        losses["L_rgb_foreground"] = (
            (rendered_rgb - target_rgb).abs() * foreground
        ).sum() / foreground.sum().clamp_min(1.0)
    if rendered_depth is not None and target_depth is not None:
        rendered_depth = _as_mask_nchw(rendered_depth).to(rendered_rgb)
        target_depth = _as_mask_nchw(target_depth).to(rendered_rgb)
        valid = target_depth > 1e-6
        losses["L_depth"] = (rendered_depth - target_depth).abs().masked_select(valid).mean() if valid.any() else rendered_rgb.new_zeros(())
    # Keep SSIM inside autograd. Calling the scalar evaluation helper here used
    # to detach the graph and made the configured SSIM loss a constant.
    losses["L_ssim"] = 1.0 - _ssim(rendered_rgb, target_rgb)
    if lpips_model is not None:
        losses["L_lpips"] = lpips_model(rendered_rgb * 2 - 1, target_rgb * 2 - 1).mean()
    if dino_model is not None:
        losses["L_dino"] = 1.0 - F.cosine_similarity(dino_model(rendered_rgb), dino_model(target_rgb), dim=-1).mean()
    return losses


def build_lpips(real_mode: bool = False):
    if not require_dependency("lpips", real_mode=real_mode, feature="LPIPS render loss"):
        return None
    import lpips

    return lpips.LPIPS(net="vgg")
