from __future__ import annotations

from typing import Dict

import torch


def render_proxy_loss(*args, **kwargs) -> Dict[str, torch.Tensor | bool | str]:
    """Optional decoder-dependent proxy loss; disabled by default to keep SLAT-only scope."""
    device = None
    for value in list(args) + list(kwargs.values()):
        if isinstance(value, torch.Tensor):
            device = value.device
            break
    loss = torch.zeros((), device=device) if device is not None else torch.tensor(0.0)
    return {"loss": loss, "enabled": False, "reason": "decoder/render proxy is optional and not invoked in SLAT-only training"}
