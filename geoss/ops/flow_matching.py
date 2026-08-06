from __future__ import annotations

from typing import Literal

import torch


try:  # Triton is optional on CPU/test hosts.
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - availability depends on the runtime image
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _flow_matching_pair_kernel(
        x0_ptr,
        noise_ptr,
        timestep_ptr,
        xt_ptr,
        velocity_ptr,
        elements_per_sample: tl.constexpr,
        num_elements: tl.constexpr,
        sigma_min: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < num_elements
        sample_index = offsets // elements_per_sample
        x0 = tl.load(x0_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        noise = tl.load(noise_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        timestep = tl.load(timestep_ptr + sample_index, mask=valid, other=0.0).to(tl.float32)
        noise_scale = sigma_min + (1.0 - sigma_min) * timestep
        xt = (1.0 - timestep) * x0 + noise_scale * noise
        velocity = (1.0 - sigma_min) * noise - x0
        tl.store(xt_ptr + offsets, xt, mask=valid)
        tl.store(velocity_ptr + offsets, velocity, mask=valid)


Backend = Literal["auto", "torch", "triton"]


def fused_flow_matching_available() -> bool:
    return bool(_TRITON_AVAILABLE and torch.cuda.is_available())


@torch.no_grad()
def flow_matching_pair(
    x0: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
    sigma_min: float,
    *,
    backend: Backend = "auto",
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Build ``x_t`` and its conditional-flow target in one memory pass.

    Stage-2 inputs and targets are frozen, so no autograd state is required for
    this operation.  On CUDA, the Triton path fuses the two affine expressions
    and avoids materializing four full 3-D-grid intermediates.  CPU and runtimes
    without Triton use the same equations through a portable PyTorch fallback.
    """

    if backend not in {"auto", "torch", "triton"}:
        raise ValueError(f"Unsupported flow-matching backend {backend!r}.")
    if x0.shape != noise.shape:
        raise ValueError(f"x0/noise shape mismatch: {tuple(x0.shape)} vs {tuple(noise.shape)}")
    if x0.ndim < 2:
        raise ValueError(f"x0 must have a batch dimension and payload dimensions, got {tuple(x0.shape)}")
    if timestep.ndim != 1 or timestep.shape[0] != x0.shape[0]:
        raise ValueError(f"timestep must be [B], got {tuple(timestep.shape)} for B={x0.shape[0]}")
    if not x0.is_floating_point() or x0.dtype != noise.dtype:
        raise TypeError("x0 and noise must have the same floating dtype.")
    if x0.device != noise.device or x0.device != timestep.device:
        raise ValueError("x0, noise, and timestep must share one device.")
    sigma = float(sigma_min)
    if not 0.0 <= sigma < 1.0:
        raise ValueError(f"sigma_min must be in [0,1), got {sigma}.")

    use_triton = backend == "triton" or (
        backend == "auto" and _TRITON_AVAILABLE and x0.device.type == "cuda"
    )
    if use_triton:
        if not _TRITON_AVAILABLE or x0.device.type != "cuda":
            raise RuntimeError("The Triton flow-matching backend requires Triton and a CUDA tensor.")
        x0_contiguous = x0.contiguous()
        noise_contiguous = noise.contiguous()
        timestep_contiguous = timestep.contiguous().float()
        xt = torch.empty_like(x0_contiguous)
        velocity = torch.empty_like(x0_contiguous)
        num_elements = x0_contiguous.numel()
        elements_per_sample = num_elements // x0_contiguous.shape[0]
        grid = (triton.cdiv(num_elements, 256),)
        _flow_matching_pair_kernel[grid](
            x0_contiguous,
            noise_contiguous,
            timestep_contiguous,
            xt,
            velocity,
            elements_per_sample=elements_per_sample,
            num_elements=num_elements,
            sigma_min=sigma,
            BLOCK_SIZE=256,
        )
        return xt, velocity, "triton"

    t_view = timestep.float().view(x0.shape[0], *([1] * (x0.ndim - 1)))
    x_t = (1.0 - t_view) * x0 + (sigma + (1.0 - sigma) * t_view) * noise
    target_v = (1.0 - sigma) * noise - x0
    return x_t, target_v, "torch"
