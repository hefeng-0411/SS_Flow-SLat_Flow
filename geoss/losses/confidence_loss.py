from __future__ import annotations

from typing import Dict

import torch


def confidence_calibration_loss(
    confidence: torch.Tensor,
    geo_error: torch.Tensor,
    regularizer_weight: float = 0.01,
) -> Dict[str, torch.Tensor]:
    conf = torch.nan_to_num(confidence.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(1e-4, 1.0 - 1e-4)
    clean_error = torch.nan_to_num(geo_error.float(), nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    err = clean_error.detach() if clean_error.requires_grad else clean_error
    weighted = (conf * clean_error).mean()
    entropy_reg = regularizer_weight * (-(conf * conf.log() + (1 - conf) * (1 - conf).log())).mean()
    loss = weighted + entropy_reg
    corr = _safe_corr(conf.reshape(-1), err.reshape(-1))
    return {"confidence_calibration": loss, "confidence_error_corr": corr.detach(), "loss": loss}


def _safe_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nan_to_num(a.float(), nan=0.0, posinf=1.0, neginf=0.0)
    b = torch.nan_to_num(b.float(), nan=0.0, posinf=1.0, neginf=0.0)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return torch.where(denom > 1e-8, (a * b).sum() / denom.clamp_min(1e-8), torch.zeros((), device=a.device))
