from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch


@dataclass
class ControlConfig:
    warmup_steps: int = 500
    total_steps: int = 20_000
    min_lr_ratio: float = 0.05
    gradient_clip_norm: float = 2.0
    loss_ema_decay: float = 0.98
    loss_weight_min: float = 0.25
    loss_weight_max: float = 4.0


class WarmupCosineController:
    """Resumable base schedule with a persistent stability contraction.

    The multiplier is deliberately separate from the analytic schedule.  A
    rollback/plateau contraction therefore cannot be silently undone by the
    next scheduler or confidence update.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, config: ControlConfig) -> None:
        self.optimizer = optimizer
        self.config = config
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.stability_multiplier = 1.0
        self.step_number = 0

    def factor(self, step: int) -> float:
        warmup = max(1, self.config.warmup_steps)
        if step <= warmup:
            return max(step, 1) / warmup
        span = max(1, self.config.total_steps - warmup)
        progress = min(1.0, max(0.0, (step - warmup) / span))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_lr_ratio + (1.0 - self.config.min_lr_ratio) * cosine

    def step(self, step: int) -> list[float]:
        self.step_number = int(step)
        factor = self.factor(step) * self.stability_multiplier
        values = []
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            lr = base_lr * factor
            group["lr"] = lr
            values.append(lr)
        return values

    def contract(self, multiplier: float) -> None:
        if not 0.0 < multiplier <= 1.0:
            raise ValueError("LR contraction multiplier must be in (0, 1]")
        self.stability_multiplier *= float(multiplier)

    def state_dict(self) -> dict:
        return {
            "base_lrs": self.base_lrs,
            "stability_multiplier": self.stability_multiplier,
            "step_number": self.step_number,
        }

    def load_state_dict(self, state: Mapping | None) -> None:
        if not state:
            return
        saved = [float(value) for value in state["base_lrs"]]
        if len(saved) != len(self.optimizer.param_groups):
            raise ValueError("scheduler parameter-group count changed across resume")
        self.base_lrs = saved
        self.stability_multiplier = float(state["stability_multiplier"])
        self.step_number = int(state["step_number"])


class BoundedLossBalancer:
    """EMA-normalize objectives without permitting unbounded adaptive weights."""

    def __init__(self, names: Iterable[str], config: ControlConfig) -> None:
        self.names = tuple(names)
        self.config = config
        self.ema: dict[str, float] = {}

    def combine(
        self,
        losses: Mapping[str, torch.Tensor],
        authority: Mapping[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        authority = authority or {}
        weighted = []
        weights: dict[str, float] = {}
        for name in self.names:
            value = losses[name]
            raw = max(abs(float(value.detach().float().cpu())), 1e-8)
            previous = self.ema.get(name, raw)
            decay = self.config.loss_ema_decay
            scale = decay * previous + (1.0 - decay) * raw
            self.ema[name] = scale
            target = float(authority.get(name, 1.0)) / max(scale, 1e-8)
            weight = min(self.config.loss_weight_max, max(self.config.loss_weight_min, target))
            weights[name] = weight
            weighted.append(value * weight)
        return torch.stack(weighted).sum(), weights

    def state_dict(self) -> dict:
        return {"ema": dict(self.ema)}

    def load_state_dict(self, state: Mapping | None) -> None:
        if state:
            self.ema = {str(key): float(value) for key, value in state.get("ema", {}).items()}


def build_geoss_optimizer(model, *, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """Use module-specific trust scales while preserving the GeoSS model."""
    core = model.module if hasattr(model, "module") else model
    specifications = (
        ("geometry", core.anchor_queries.parameters(), 0.35, 0.0),
        ("evidence", core.ray_sampler.parameters(), 0.50, weight_decay),
        ("fusion", core.aggregator.parameters(), 1.00, weight_decay),
    )
    groups = []
    owned: set[int] = set()
    for name, parameters, multiplier, decay in specifications:
        selected = [parameter for parameter in parameters if parameter.requires_grad]
        if not selected:
            continue
        owned.update(id(parameter) for parameter in selected)
        groups.append({"name": name, "params": selected, "lr": lr * multiplier, "weight_decay": decay})
    remainder = [
        parameter for parameter in core.parameters()
        if parameter.requires_grad and id(parameter) not in owned
    ]
    if remainder:
        groups.append({"name": "other", "params": remainder, "lr": lr * 0.5, "weight_decay": weight_decay})
    if not groups:
        raise ValueError("GeoSS has no trainable parameters")
    return torch.optim.AdamW(groups, betas=(0.9, 0.99), eps=1e-8)


def snapshot_group_parameters(optimizer: torch.optim.Optimizer) -> list[list[torch.Tensor]]:
    return [[parameter.detach().float().clone() for parameter in group["params"]] for group in optimizer.param_groups]


@torch.no_grad()
def group_update_ratios(
    optimizer: torch.optim.Optimizer,
    before: list[list[torch.Tensor]],
) -> dict[str, float]:
    ratios = {}
    for index, (group, old_values) in enumerate(zip(optimizer.param_groups, before)):
        update_sq = 0.0
        parameter_sq = 0.0
        for parameter, old in zip(group["params"], old_values):
            current = parameter.detach().float()
            update_sq += float((current - old.to(current.device)).square().sum().cpu())
            parameter_sq += float(current.square().sum().cpu())
        ratios[str(group.get("name", index))] = math.sqrt(update_sq) / (math.sqrt(parameter_sq) + 1e-12)
    return ratios


def gradient_group_norms(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    norms = {}
    for index, group in enumerate(optimizer.param_groups):
        square = 0.0
        for parameter in group["params"]:
            if parameter.grad is not None:
                square += float(parameter.grad.detach().float().square().sum().cpu())
        norms[str(group.get("name", index))] = math.sqrt(square)
    return norms
