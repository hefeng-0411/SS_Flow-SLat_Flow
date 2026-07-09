from __future__ import annotations

from typing import Dict

import torch


def slat_prior_preservation_loss(
    v_slat_geo: torch.Tensor,
    v_slat_base: torch.Tensor,
    joint_confidence: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    weight = (1.0 - joint_confidence.clamp(0, 1)).detach()
    loss = ((v_slat_geo - v_slat_base).abs() * weight).mean()
    return {"loss": loss, "slat_prior_preservation": loss}
