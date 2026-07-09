from __future__ import annotations

from typing import Dict

import torch


def slat_velocity_regularization_loss(delta_v_slat: torch.Tensor, timestep: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
    loss = delta_v_slat.pow(2).mean()
    if timestep is not None:
        t = timestep.flatten().float().to(delta_v_slat.device)
        if t.max() > 1.5:
            t = t / 1000.0
        if t.numel() == delta_v_slat.shape[0]:
            loss = (delta_v_slat.pow(2).mean(dim=(1, 2)) * (0.25 + 0.75 * t.clamp(0, 1))).mean()
    return {"loss": loss, "slat_velocity_l2": loss}
