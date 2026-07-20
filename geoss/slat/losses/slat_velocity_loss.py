from __future__ import annotations

from typing import Dict

import torch


def slat_velocity_regularization_loss(
    delta_v_slat: torch.Tensor,
    timestep: torch.Tensor | None = None,
    token_valid_mask: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    per_token = delta_v_slat.pow(2).mean(dim=-1, keepdim=True)
    if token_valid_mask is None:
        loss = per_token.mean()
    else:
        valid = token_valid_mask.to(device=delta_v_slat.device, dtype=delta_v_slat.dtype)
        loss = (per_token * valid).sum() / valid.sum().clamp_min(1e-6)
    if timestep is not None:
        t = timestep.flatten().float().to(delta_v_slat.device)
        if t.max() > 1.5:
            t = t / 1000.0
        if t.numel() == delta_v_slat.shape[0]:
            weighted = per_token * (0.25 + 0.75 * t.clamp(0, 1)).view(-1, 1, 1)
            if token_valid_mask is None:
                loss = weighted.mean()
            else:
                valid = token_valid_mask.to(device=weighted.device, dtype=weighted.dtype)
                loss = (weighted * valid).sum() / valid.sum().clamp_min(1e-6)
    return {"loss": loss, "slat_velocity_l2": loss}
