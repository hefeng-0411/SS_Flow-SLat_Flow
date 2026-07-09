from __future__ import annotations

import torch


def velocity_regularization_loss(delta_v_geo: torch.Tensor, timestep: torch.Tensor | None = None) -> torch.Tensor:
    loss = delta_v_geo.pow(2).mean(dim=-1)
    if timestep is not None:
        t = timestep.flatten().float().to(delta_v_geo.device)
        if t.max() > 1.5:
            t = t / 1000.0
        weight = (0.25 + 0.75 * t).view(-1, 1)
        loss = loss * weight
    return loss.mean()
