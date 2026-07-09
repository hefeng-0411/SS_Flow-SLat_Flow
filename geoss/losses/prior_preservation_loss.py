from __future__ import annotations

import torch


def prior_preservation_loss(v_geo: torch.Tensor, v_base: torch.Tensor, token_confidence: torch.Tensor) -> torch.Tensor:
    return ((1.0 - token_confidence).clamp(0, 1) * (v_geo - v_base).pow(2).mean(dim=-1, keepdim=True)).mean()
