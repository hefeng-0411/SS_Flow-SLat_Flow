from __future__ import annotations

from typing import Literal

import math
import torch
import torch.nn as nn


class GuidanceGate(nn.Module):
    """Timestep-adaptive alpha(t) for geometry guidance."""

    def __init__(
        self,
        mode: Literal["fixed", "cosine", "learned"] = "cosine",
        strength: float = 1.0,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.strength = strength
        if mode == "learned":
            self.net = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, timestep: torch.Tensor, batch_size: int) -> torch.Tensor:
        t = normalize_timestep(timestep, batch_size)
        if self.mode == "fixed":
            alpha = torch.full_like(t, self.strength)
        elif self.mode == "cosine":
            alpha = self.strength * 0.5 * (1.0 - torch.cos(math.pi * t))
        elif self.mode == "learned":
            alpha = self.strength * self.net(t[:, None]).squeeze(-1)
        else:
            raise ValueError(f"Unknown guidance gate mode {self.mode}")
        return alpha.view(batch_size, 1, 1)


def normalize_timestep(timestep: torch.Tensor | float, batch_size: int) -> torch.Tensor:
    if not isinstance(timestep, torch.Tensor):
        timestep = torch.tensor([float(timestep)])
    t = timestep.flatten().float()
    if t.numel() == 1:
        t = t.expand(batch_size)
    if t.numel() != batch_size:
        raise ValueError(f"timestep batch mismatch: got {t.numel()}, expected {batch_size}")
    if t.max() > 1.5:
        t = t / 1000.0
    return t.clamp(0.0, 1.0)
