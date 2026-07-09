from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _feats(x):
    return x.feats if hasattr(x, "feats") else x


def slat_flow_matching_loss(pred_velocity, target_velocity) -> Dict[str, torch.Tensor]:
    """TRELLIS-compatible SLAT velocity MSE over SparseTensor.feats or dense tokens."""
    pred = _feats(pred_velocity)
    target = _feats(target_velocity)
    loss = F.mse_loss(pred, target)
    return {"loss": loss, "slat_flow_mse": loss}
