from __future__ import annotations

from typing import Dict

import torch


def confidence_calibration_loss(
    confidence: torch.Tensor,
    geo_error: torch.Tensor,
    regularizer_weight: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """Calibrate confidence to reconstruction correctness.

    ``geo_error`` is an error (zero is correct), not a confidence target.  The
    historical objective minimized ``confidence * error`` and had no term
    rewarding confidence on correct anchors, making the all-zero gate an
    optimum.  Brier calibration removes that degenerate solution.
    """
    conf = torch.nan_to_num(confidence.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(1e-4, 1.0 - 1e-4)
    clean_error = torch.nan_to_num(geo_error.float(), nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    err = clean_error.detach()
    target = 1.0 - err
    brier = (conf - target).square().mean()
    # A very small barrier prevents exact sigmoid saturation without dictating
    # whether a particular anchor should be trusted.
    barrier = -(conf.log() + (1.0 - conf).log()).mean()
    loss = brier + regularizer_weight * barrier
    corr = _safe_corr(conf.reshape(-1), err.reshape(-1))
    return {
        "confidence_calibration": loss,
        "confidence_brier": brier.detach(),
        "confidence_error_corr": corr.detach(),
        "loss": loss,
    }


def _safe_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nan_to_num(a.float(), nan=0.0, posinf=1.0, neginf=0.0)
    b = torch.nan_to_num(b.float(), nan=0.0, posinf=1.0, neginf=0.0)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return torch.where(denom > 1e-8, (a * b).sum() / denom.clamp_min(1e-8), torch.zeros((), device=a.device))
