from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Small LoRA wrapper for optional adapter-only fine-tuning."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 1.0) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(x)) * (self.alpha / self.rank)
