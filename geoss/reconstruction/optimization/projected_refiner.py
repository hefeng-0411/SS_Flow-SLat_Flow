from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoss.reconstruction.evidence import FUSED_EVIDENCE_DIM
from geoss.reconstruction.fields import (
    FIELD_CHANNELS,
    PosteriorFieldState,
    SparseBrickLevel,
    query_posterior_field,
)
from geoss.reconstruction.priors import TrellisFieldPrior


class EvidenceFieldInitializer(nn.Module):
    """Amortize calibrated information-form evidence into a field posterior."""

    def __init__(self, hidden_channels: int = 96) -> None:
        super().__init__()
        self.input = nn.Conv3d(FUSED_EVIDENCE_DIM + 5, hidden_channels, 3, padding=1)
        self.blocks = nn.Sequential(
            _Residual3D(hidden_channels),
            _Residual3D(hidden_channels, dilation=2),
            _Residual3D(hidden_channels),
        )
        self.output = nn.Conv3d(hidden_channels, 38, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        evidence_volume: torch.Tensor,
        prior: TrellisFieldPrior,
        *,
        bounds: Tuple[float, float] = (-0.5, 0.5),
    ) -> PosteriorFieldState:
        if evidence_volume.ndim != 5 or evidence_volume.shape[1] != FUSED_EVIDENCE_DIM:
            raise ValueError(
                f"evidence_volume must be [B,{FUSED_EVIDENCE_DIM},D,H,W], got {tuple(evidence_volume.shape)}"
            )
        information = evidence_volume[:, 1:2].float().clamp(-20.0, 20.0).exp()
        prior_precision = F.softplus(prior.log_precision.float())
        observation_fraction = information / (information + prior_precision + 1e-4)
        sdf_mean = observation_fraction * evidence_volume[:, 0:1] + (
            1.0 - observation_fraction
        ) * prior.sdf_mean
        sdf_log_variance = -(information + prior_precision + 1e-4).log()
        state_probability = evidence_volume[:, 2:7].clamp_min(1e-6)
        state_logits = state_probability.log()
        outlier_probability = evidence_volume[:, 38:39].clamp(1e-5, 1.0 - 1e-5)
        outlier_logit = torch.logit(outlier_probability)
        observed_rgb = evidence_volume[:, 7:10].clamp(1e-5, 1.0 - 1e-5)
        surface_fraction = state_probability[:, 1:2]
        diffuse_logits = surface_fraction * torch.logit(observed_rgb) + (
            1.0 - surface_fraction
        ) * prior.diffuse_logits
        base = torch.cat(
            (
                sdf_mean,
                sdf_log_variance,
                outlier_logit,
                state_logits,
                diffuse_logits,
                prior.directional_sh,
            ),
            dim=1,
        )
        network_input = torch.cat(
            (
                evidence_volume,
                prior.sdf_mean,
                prior.log_precision,
                prior.diffuse_logits,
            ),
            dim=1,
        )
        residual = self.output(self.blocks(self.input(network_input)))
        scaled = _scale_state_delta(residual, sdf_scale=0.05)
        volume = _stabilize_state_volume(base + scaled)
        return PosteriorFieldState(volume=volume, bounds=bounds)


class PosteriorStateRefiner(nn.Module):
    """Update explicit posterior state after recomputing physical visibility."""

    def __init__(self, hidden_channels: int = 96) -> None:
        super().__init__()
        self.input = nn.Conv3d(38 + FUSED_EVIDENCE_DIM + 2, hidden_channels, 3, padding=1)
        self.blocks = nn.Sequential(
            _Residual3D(hidden_channels),
            _Residual3D(hidden_channels, dilation=2),
            _Residual3D(hidden_channels),
        )
        self.output = nn.Conv3d(hidden_channels, 38, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        field: PosteriorFieldState,
        evidence_volume: torch.Tensor,
        prior: TrellisFieldPrior,
    ) -> PosteriorFieldState:
        state_probability = field.volume[:, FIELD_CHANNELS["evidence_logits"]].softmax(dim=1)
        observation_responsibility = (state_probability[:, 0:1] + state_probability[:, 1:2]).clamp(
            0.0,
            1.0,
        )
        hidden = self.blocks(
            self.input(
                torch.cat(
                    (
                        field.volume,
                        evidence_volume,
                        prior.sdf_mean,
                        prior.log_precision,
                    ),
                    dim=1,
                )
            )
        )
        raw_delta = self.output(hidden)
        delta = _scale_state_delta(raw_delta, sdf_scale=0.03)
        # Learned SDF correction is explicitly an observation update.  The
        # distinct completion proposal is projected later.
        delta = torch.cat(
            (
                delta[:, FIELD_CHANNELS["sdf_mean"]] * observation_responsibility,
                delta[:, 1:],
            ),
            dim=1,
        )
        return field.replace_volume(_stabilize_state_volume(field.volume + delta))


class AdaptiveBrickRefiner(nn.Module):
    """Activate and predict local residual bricks around uncertain surfaces."""

    def __init__(
        self,
        *,
        max_bricks: int = 96,
        brick_resolution: int = 4,
        hidden_dim: int = 160,
        surface_band_cells: float = 3.0,
    ) -> None:
        super().__init__()
        self.max_bricks = int(max_bricks)
        self.brick_resolution = int(brick_resolution)
        self.surface_band_cells = float(surface_band_cells)
        self.predictor = nn.Sequential(
            nn.Linear(FUSED_EVIDENCE_DIM + 38, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 38),
        )
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)

    def select(
        self,
        field: PosteriorFieldState,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return brick xyz indices, node positions, valid mask, and C0 envelope."""

        b, _, d, h, w = field.volume.shape
        if not (d == h == w):
            raise ValueError("Adaptive brick refinement currently requires a cubic coarse lattice")
        bricks_per_axis = d - 1
        sdf = field.volume[:, 0, :-1, :-1, :-1]
        log_variance = field.volume[:, 1, :-1, :-1, :-1]
        surface_probability = field.volume[:, FIELD_CHANNELS["evidence_logits"]].softmax(dim=1)[
            :, 1, :-1, :-1, :-1
        ]
        cell_size = float(field.bounds[1] - field.bounds[0]) / bricks_per_axis
        score = (
            -sdf.abs() / max(cell_size, 1e-8)
            + surface_probability
            + 0.1 * (0.5 * log_variance).exp() / max(cell_size, 1e-8)
        )
        flattened = score.reshape(b, -1)
        count = min(self.max_bricks, flattened.shape[1])
        selected_score, selected = flattened.topk(count, dim=1, largest=True, sorted=False)
        iz = selected // (bricks_per_axis * bricks_per_axis)
        remainder = selected % (bricks_per_axis * bricks_per_axis)
        iy = remainder // bricks_per_axis
        ix = remainder % bricks_per_axis
        indices = torch.stack((ix, iy, iz), dim=-1)
        selected_sdf = sdf.reshape(b, -1).gather(1, selected)
        selected_surface = surface_probability.reshape(b, -1).gather(1, selected)
        valid = (
            selected_sdf.abs() <= self.surface_band_cells * cell_size
        ) | (selected_surface >= 0.15)

        local = torch.linspace(
            0.0,
            1.0,
            self.brick_resolution + 1,
            device=field.volume.device,
            dtype=field.volume.dtype,
        )
        lz, ly, lx = torch.meshgrid(local, local, local, indexing="ij")
        local_xyz = torch.stack((lx, ly, lz), dim=-1)
        origin = field.volume.new_tensor(field.bounds[0]) + indices.to(field.volume.dtype) * cell_size
        points = origin[:, :, None, None, None, :] + cell_size * local_xyz[None, None]
        envelope = (
            torch.sin(torch.pi * lx)
            * torch.sin(torch.pi * ly)
            * torch.sin(torch.pi * lz)
        )[None, None, ..., None]
        return indices, points, valid, envelope

    def decode(
        self,
        field: PosteriorFieldState,
        *,
        brick_indices: torch.Tensor,
        brick_points: torch.Tensor,
        brick_valid: torch.Tensor,
        envelope: torch.Tensor,
        fused_evidence: torch.Tensor,
    ) -> SparseBrickLevel:
        b, m, gz, gy, gx, _ = brick_points.shape
        flat_points = brick_points.reshape(b, m * gz * gy * gx, 3)
        base = query_posterior_field(field, flat_points, require_gradient=False)
        if base.raw is None:
            raise RuntimeError("Brick refinement requires raw field samples")
        if fused_evidence.shape != (b, flat_points.shape[1], FUSED_EVIDENCE_DIM):
            raise ValueError(
                f"Fine evidence must be [B,{flat_points.shape[1]},{FUSED_EVIDENCE_DIM}], "
                f"got {tuple(fused_evidence.shape)}"
            )
        raw = self.predictor(torch.cat((fused_evidence, base.raw), dim=-1))
        cell_size = float(field.bounds[1] - field.bounds[0]) / (field.volume.shape[-1] - 1)
        residual = _scale_point_state_delta(raw, sdf_scale=0.5 * cell_size)
        residual = residual.reshape(b, m, gz, gy, gx, 38) * envelope
        residual = residual * brick_valid[:, :, None, None, None, None].to(residual.dtype)
        values = residual.permute(0, 1, 5, 2, 3, 4).contiguous()
        return SparseBrickLevel(
            brick_indices=brick_indices,
            values=values,
            valid=brick_valid,
            bricks_per_axis=field.volume.shape[-1] - 1,
            brick_resolution=self.brick_resolution,
        )


class _Residual3D(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


def _scale_state_delta(delta: torch.Tensor, *, sdf_scale: float) -> torch.Tensor:
    out = delta.clone()
    out[:, FIELD_CHANNELS["sdf_mean"]] = sdf_scale * torch.tanh(
        delta[:, FIELD_CHANNELS["sdf_mean"]]
    )
    out[:, FIELD_CHANNELS["sdf_log_variance"]] = 0.5 * torch.tanh(
        delta[:, FIELD_CHANNELS["sdf_log_variance"]]
    )
    out[:, FIELD_CHANNELS["outlier_logit"]] = 0.5 * torch.tanh(
        delta[:, FIELD_CHANNELS["outlier_logit"]]
    )
    out[:, FIELD_CHANNELS["evidence_logits"]] = 0.5 * torch.tanh(
        delta[:, FIELD_CHANNELS["evidence_logits"]]
    )
    out[:, FIELD_CHANNELS["diffuse_logits"]] = 0.5 * torch.tanh(
        delta[:, FIELD_CHANNELS["diffuse_logits"]]
    )
    out[:, FIELD_CHANNELS["directional_sh"]] = 0.1 * torch.tanh(
        delta[:, FIELD_CHANNELS["directional_sh"]]
    )
    return out


def _scale_point_state_delta(delta: torch.Tensor, *, sdf_scale: float) -> torch.Tensor:
    out = delta.clone()
    out[..., 0:1] = sdf_scale * torch.tanh(delta[..., 0:1])
    out[..., 1:3] = 0.5 * torch.tanh(delta[..., 1:3])
    out[..., 3:8] = 0.5 * torch.tanh(delta[..., 3:8])
    out[..., 8:11] = 0.5 * torch.tanh(delta[..., 8:11])
    out[..., 11:38] = 0.1 * torch.tanh(delta[..., 11:38])
    return out


def _stabilize_state_volume(volume: torch.Tensor) -> torch.Tensor:
    # Functional concatenation keeps the field differentiable without
    # mutating views that clamp backward has saved.
    return torch.cat(
        (
            volume[:, FIELD_CHANNELS["sdf_mean"]],
            volume[:, FIELD_CHANNELS["sdf_log_variance"]].clamp(-14.0, 2.0),
            volume[:, FIELD_CHANNELS["outlier_logit"]].clamp(-12.0, 12.0),
            volume[:, FIELD_CHANNELS["evidence_logits"]].clamp(-20.0, 20.0),
            volume[:, FIELD_CHANNELS["diffuse_logits"]].clamp(-12.0, 12.0),
            volume[:, FIELD_CHANNELS["directional_sh"]],
        ),
        dim=1,
    )
