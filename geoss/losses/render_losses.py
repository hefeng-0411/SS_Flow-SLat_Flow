from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from geoss.eval.render_metrics import image_render_metrics
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
    losses: Dict[str, torch.Tensor] = {}
    losses["L_rgb"] = F.l1_loss(rendered_rgb, target_rgb)
    if rendered_alpha is not None and target_mask is not None:
        losses["L_mask"] = F.binary_cross_entropy(rendered_alpha.clamp(1e-4, 1 - 1e-4), target_mask.float().clamp(0, 1))
    if rendered_depth is not None and target_depth is not None:
        valid = target_depth > 1e-6
        losses["L_depth"] = (rendered_depth - target_depth).abs().masked_select(valid).mean() if valid.any() else rendered_rgb.new_zeros(())
    metrics = image_render_metrics(rendered_rgb, target_rgb, rendered_alpha, target_mask)
    losses["L_ssim"] = rendered_rgb.new_tensor(1.0 - metrics["SSIM"])
    if lpips_model is not None:
        losses["L_lpips"] = lpips_model(rendered_rgb * 2 - 1, target_rgb * 2 - 1).mean()
    if dino_model is not None:
        losses["L_dino"] = 1.0 - F.cosine_similarity(dino_model(rendered_rgb), dino_model(target_rgb), dim=-1).mean()
    losses["L_multiview_consistency"] = rendered_rgb.var(dim=0).mean() if rendered_rgb.ndim >= 4 and rendered_rgb.shape[0] > 1 else rendered_rgb.new_zeros(())
    return losses


def build_lpips(real_mode: bool = False):
    if not require_dependency("lpips", real_mode=real_mode, feature="LPIPS render loss"):
        return None
    import lpips

    return lpips.LPIPS(net="vgg")
