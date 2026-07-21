from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F


def probability_binary_cross_entropy(
    probability: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Evaluate probability-space BCE safely inside a surrounding AMP region.

    PyTorch intentionally rejects ``binary_cross_entropy`` while CUDA autocast
    is enabled: a sigmoid evaluated in reduced precision can underflow and make
    the probability-space gradient unrecoverable.  Some project interfaces
    (rendered alpha, ray transmittance, externally calibrated confidence) only
    expose probabilities rather than logits.  For those interfaces, disabling
    autocast locally and promoting reduced-precision tensors to FP32 preserves
    the exact BCE objective and its gradients without disabling AMP for the
    expensive model forward.
    """
    if not torch.is_floating_point(probability):
        raise TypeError("BCE probability input must be a floating-point tensor.")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"Unsupported BCE reduction={reduction!r}.")

    compute_dtype = (
        torch.float32
        if probability.dtype in {torch.float16, torch.bfloat16}
        else probability.dtype
    )
    autocast_context = (
        torch.amp.autocast(device_type=probability.device.type, enabled=False)
        if probability.device.type in {"cpu", "cuda"}
        else contextlib.nullcontext()
    )
    with autocast_context:
        probability_compute = probability.to(dtype=compute_dtype)
        target_compute = target.to(device=probability.device, dtype=compute_dtype)
        weight_compute = (
            weight.to(device=probability.device, dtype=compute_dtype)
            if weight is not None
            else None
        )
        return F.binary_cross_entropy(
            probability_compute,
            target_compute,
            weight=weight_compute,
            reduction=reduction,
        )
