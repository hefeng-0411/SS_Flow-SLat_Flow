from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from geoss.utils.voxelization import sample_occupancy_at_points
from .dice_loss import dice_loss


def occupancy_bce_loss(
    occ_evidence: torch.Tensor,
    free_evidence: torch.Tensor,
    anchor_xyz: torch.Tensor,
    gt_occ: torch.Tensor,
    dice_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    target = sample_occupancy_at_points(gt_occ, anchor_xyz)
    target = torch.nan_to_num(target.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    logits = torch.nan_to_num(occ_evidence - free_evidence, nan=0.0, posinf=30.0, neginf=-30.0)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dice = dice_loss(prob, target)
    return {"occupancy_bce": bce, "occupancy_dice": dice, "loss": bce + dice_weight * dice}
