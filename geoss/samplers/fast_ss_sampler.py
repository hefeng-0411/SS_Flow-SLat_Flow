"""Five-step geometry-conditioned sampler matching TRELLIS FlowEuler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from geoss.integration.trellis_ss_hook import ss_grid_to_tokens, tokens_to_ss_grid
from geoss.models.ss_flow_adapter import SSFlowAdapter, SSFlowAdapterOutput
from geoss.models.voxel_fusion_engine import VoxelFusionOutput


@dataclass(frozen=True)
class FastSSSamplerOutput:
    samples: torch.Tensor
    active_voxels: Optional[torch.Tensor]
    diagnostics: Dict[str, Any]
    timings_ms: Dict[str, Any]


class FastGeometryConditionedSSSampler:
    """Deterministic TRELLIS-compatible Euler/Heun integration from ``t=1`` to ``0``."""

    def __init__(
        self,
        base_model: torch.nn.Module,
        adapter: SSFlowAdapter,
        *,
        decoder: Optional[torch.nn.Module] = None,
        sigma_min: float = 1e-5,
        num_steps: int = 5,
        integrator: str = "euler",
        rescale_t: float = 3.0,
        cfg_strength: float = 5.0,
        cfg_interval: tuple[float, float] = (0.5, 1.0),
    ) -> None:
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        if integrator not in {"euler", "heun"}:
            raise ValueError(f"Unsupported integrator={integrator!r}")
        self.base_model = base_model.eval().requires_grad_(False)
        self.adapter = adapter
        self.decoder = decoder.eval().requires_grad_(False) if decoder is not None else None
        self.sigma_min = float(sigma_min)
        self.num_steps = int(num_steps)
        self.integrator = integrator
        self.rescale_t = float(rescale_t)
        self.cfg_strength = float(cfg_strength)
        self.cfg_interval = cfg_interval

    @torch.inference_mode()
    def sample(
        self,
        noise: torch.Tensor,
        *,
        base_condition: torch.Tensor,
        voxel_fusion: VoxelFusionOutput,
        negative_condition: Optional[torch.Tensor] = None,
        adapter_enabled: bool = True,
        profile: bool = False,
        vggt_time_ms: Optional[float] = None,
    ) -> FastSSSamplerOutput:
        if noise.ndim != 5 or noise.shape[-3:] != (16, 16, 16):
            raise ValueError(f"TRELLIS SS noise must be [B,C,16,16,16], got {tuple(noise.shape)}")
        if base_condition is None:
            raise ValueError("Real TRELLIS base_condition is mandatory")
        cached_condition = voxel_fusion.dense_tokens
        cached_observation = voxel_fusion.observation_mask
        cached_confidence = voxel_fusion.voxel_confidence
        if cached_condition.shape[0] != noise.shape[0]:
            raise ValueError("Voxel condition batch does not match SS noise batch")

        times = _trellis_time_schedule(self.num_steps, self.rescale_t)
        state = noise
        step_timings = []
        last_output = None
        for step_index in range(self.num_steps):
            t = times[step_index]
            t_next = times[step_index + 1]
            timer = _CudaTimer(state.device, profile)
            timer.start()
            velocity, adapter_output = self._velocity(
                state,
                t,
                base_condition,
                cached_condition,
                cached_observation,
                cached_confidence,
                negative_condition,
                adapter_enabled,
            )
            dt = t_next - t
            if self.integrator == "euler" or step_index == self.num_steps - 1:
                state = state + dt * velocity
            else:
                predictor = state + dt * velocity
                velocity_next, _ = self._velocity(
                    predictor,
                    t_next,
                    base_condition,
                    cached_condition,
                    cached_observation,
                    cached_confidence,
                    negative_condition,
                    adapter_enabled,
                )
                state = state + 0.5 * dt * (velocity + velocity_next)
            step_timings.append(timer.stop())
            last_output = adapter_output

        active_voxels = None
        if self.decoder is not None:
            logits = self.decoder(state)
            active_voxels = torch.argwhere(logits > 0)[:, [0, 2, 3, 4]].to(torch.int32)
            if active_voxels.numel() == 0:
                raise RuntimeError("TRELLIS SS decoder produced empty active support")
        diagnostics = {} if last_output is None else last_output.diagnostics
        fusion_stage_times = voxel_fusion.timings_ms
        fusion_total = (
            sum(value for value in fusion_stage_times.values() if value is not None)
            if profile and fusion_stage_times
            else None
        )
        timings = {
            "per_step_ss": step_timings,
            "total_ss_sampling": sum(value for value in step_timings if value is not None) if profile else None,
            "num_steps": self.num_steps,
            "integrator": self.integrator,
            "vggt": vggt_time_ms if profile else None,
            "voxel_fusion": fusion_total,
            "voxel_fusion_stages": fusion_stage_times if profile else {},
        }
        return FastSSSamplerOutput(state, active_voxels, diagnostics, timings)

    def _velocity(
        self,
        state: torch.Tensor,
        t: float,
        condition: torch.Tensor,
        voxel_condition: torch.Tensor,
        observation: torch.Tensor,
        confidence: torch.Tensor,
        negative_condition: Optional[torch.Tensor],
        adapter_enabled: bool,
    ) -> tuple[torch.Tensor, SSFlowAdapterOutput]:
        t_model = torch.full((state.shape[0],), 1000.0 * t, device=state.device, dtype=torch.float32)
        conditional = self.base_model(state, t_model, _expand_condition(condition, state.shape[0]))
        if negative_condition is not None and self.cfg_interval[0] <= t <= self.cfg_interval[1]:
            unconditional = self.base_model(state, t_model, _expand_condition(negative_condition, state.shape[0]))
            base = (1.0 + self.cfg_strength) * conditional - self.cfg_strength * unconditional
        else:
            base = conditional
        adapter_output = self.adapter(
            ss_grid_to_tokens(state),
            voxel_condition,
            t_model,
            observation,
            confidence,
            v_base=ss_grid_to_tokens(base),
            enabled=adapter_enabled,
        )
        return tokens_to_ss_grid(adapter_output.v_final, (16, 16, 16)), adapter_output


def _trellis_time_schedule(steps: int, rescale_t: float) -> tuple[float, ...]:
    # Exact local TRELLIS convention: linspace(1,0), then rational t rescale.
    # Build this tiny schedule on CPU so the hot loop has no CUDA ``.item()``
    # synchronization between ODE steps.
    time = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    rescaled = rescale_t * time / (1.0 + (rescale_t - 1.0) * time)
    return tuple(float(value) for value in rescaled)


def _expand_condition(condition: torch.Tensor, batch_size: int) -> torch.Tensor:
    if condition.shape[0] == batch_size:
        return condition
    if condition.shape[0] == 1:
        return condition.expand(batch_size, *condition.shape[1:])
    raise ValueError(f"TRELLIS condition batch {condition.shape[0]} cannot serve SS batch {batch_size}")


class _CudaTimer:
    def __init__(self, device: torch.device, enabled: bool) -> None:
        self.enabled = enabled and device.type == "cuda"
        self.start_event = torch.cuda.Event(enable_timing=True) if self.enabled else None
        self.end_event = torch.cuda.Event(enable_timing=True) if self.enabled else None

    def start(self) -> None:
        if self.start_event is not None:
            self.start_event.record()

    def stop(self) -> Optional[float]:
        if self.end_event is None or self.start_event is None:
            return None
        self.end_event.record()
        self.end_event.synchronize()  # profiling boundary only
        return float(self.start_event.elapsed_time(self.end_event))


__all__ = ["FastGeometryConditionedSSSampler", "FastSSSamplerOutput"]
