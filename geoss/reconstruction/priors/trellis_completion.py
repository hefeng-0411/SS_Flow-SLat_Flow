from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoss.reconstruction.fields import FIELD_CHANNELS, PosteriorFieldState, make_canonical_grid


@dataclass
class TrellisFieldPrior:
    """TRELLIS completion distribution expressed in RAPC field coordinates."""

    sdf_mean: torch.Tensor  # [B,1,D,H,W]
    log_precision: torch.Tensor  # [B,1,D,H,W]
    diffuse_logits: torch.Tensor  # [B,3,D,H,W]
    directional_sh: torch.Tensor  # [B,27,D,H,W]
    source: str
    # Appearance confidence is deliberately distinct from geometry precision:
    # SLAT can be informative about color without becoming authoritative over
    # the sparse-structure SDF.
    appearance_log_precision: Optional[torch.Tensor] = None  # [B,1,D,H,W]


@dataclass
class TrellisPriorInput:
    """Conditioning-generated latent payload decoded inside RAPC/DDP forward."""

    ss_latent_grid: torch.Tensor  # [B,C,D,H,W]
    slat_dense_grid: Optional[torch.Tensor] = None  # [B,C,64,64,64]
    slat_occupancy: Optional[torch.Tensor] = None  # [B,1,64,64,64]


@dataclass
class CompletionProposal:
    sdf_delta: torch.Tensor  # [B,1,D,H,W]
    diffuse_delta: torch.Tensor  # [B,3,D,H,W]
    directional_delta: torch.Tensor  # [B,27,D,H,W]
    completion_responsibility: torch.Tensor  # [B,1,D,H,W]
    prior_precision: torch.Tensor  # [B,1,D,H,W]


class TrellisPriorAdapter(nn.Module):
    """Distill TRELLIS structure latents into an explicit field prior.

    The adapter is a prior decoder only.  Observations never enter TRELLIS
    latent space, and the resulting SDF proposal must still pass through the
    observation-authority projector.
    """

    def __init__(
        self,
        *,
        ss_latent_channels: int = 8,
        slat_latent_channels: int = 8,
        hidden_channels: int = 64,
        output_resolution: int = 32,
        bounds: Tuple[float, float] = (-0.5, 0.5),
    ) -> None:
        super().__init__()
        self.output_resolution = int(output_resolution)
        self.bounds = bounds
        self.encoder = nn.Sequential(
            nn.Conv3d(ss_latent_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            _Residual3D(hidden_channels),
            _Residual3D(hidden_channels),
        )
        self.slat_encoder = nn.Sequential(
            nn.Conv3d(slat_latent_channels + 1, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            _Residual3D(hidden_channels),
        )
        self.sdf_head = nn.Conv3d(hidden_channels, 1, 3, padding=1)
        self.precision_head = nn.Conv3d(hidden_channels, 1, 3, padding=1)
        self.diffuse_head = nn.Conv3d(hidden_channels, 3, 1)
        self.directional_head = nn.Conv3d(hidden_channels, 27, 1)
        self.appearance_precision_head = nn.Conv3d(hidden_channels, 1, 1)
        nn.init.zeros_(self.sdf_head.weight)
        nn.init.zeros_(self.sdf_head.bias)
        nn.init.zeros_(self.diffuse_head.weight)
        nn.init.zeros_(self.diffuse_head.bias)
        nn.init.zeros_(self.directional_head.weight)
        nn.init.zeros_(self.directional_head.bias)
        nn.init.zeros_(self.appearance_precision_head.weight)
        nn.init.constant_(self.appearance_precision_head.bias, -2.0)

    def forward(
        self,
        *,
        ss_latent_grid: Optional[torch.Tensor] = None,
        slat_dense_grid: Optional[torch.Tensor] = None,
        slat_occupancy: Optional[torch.Tensor] = None,
        decoded_prior_sdf: Optional[torch.Tensor] = None,
        decoded_prior_log_precision: Optional[torch.Tensor] = None,
        decoded_prior_diffuse_logits: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> TrellisFieldPrior:
        resolution = self.output_resolution
        if decoded_prior_sdf is not None:
            if decoded_prior_sdf.ndim != 5 or decoded_prior_sdf.shape[1] != 1:
                raise ValueError("decoded_prior_sdf must be [B,1,D,H,W]")
            sdf = F.interpolate(
                decoded_prior_sdf.float(),
                size=(resolution, resolution, resolution),
                mode="trilinear",
                align_corners=True,
            )
            b = sdf.shape[0]
            log_precision = (
                F.interpolate(
                    decoded_prior_log_precision.float(),
                    size=sdf.shape[-3:],
                    mode="trilinear",
                    align_corners=True,
                )
                if decoded_prior_log_precision is not None
                else torch.full_like(sdf, -2.0)
            )
            diffuse = (
                F.interpolate(
                    decoded_prior_diffuse_logits.float(),
                    size=sdf.shape[-3:],
                    mode="trilinear",
                    align_corners=True,
                )
                if decoded_prior_diffuse_logits is not None
                else torch.zeros(b, 3, *sdf.shape[-3:], device=sdf.device, dtype=sdf.dtype)
            )
            directional = torch.zeros(b, 27, *sdf.shape[-3:], device=sdf.device, dtype=sdf.dtype)
            return TrellisFieldPrior(
                sdf_mean=sdf,
                log_precision=log_precision.clamp(-10.0, 8.0),
                diffuse_logits=diffuse,
                directional_sh=directional,
                source="trellis_decoded_spatial_prior",
                appearance_log_precision=log_precision.clamp(-10.0, 8.0),
            )
        if ss_latent_grid is not None:
            if ss_latent_grid.ndim == 4:
                ss_latent_grid = ss_latent_grid.unsqueeze(0)
            if ss_latent_grid.ndim != 5:
                raise ValueError("ss_latent_grid must be [B,C,D,H,W]")
            structure_features = self.encoder(ss_latent_grid.float())
            structure_features = F.interpolate(
                structure_features,
                size=(resolution, resolution, resolution),
                mode="trilinear",
                align_corners=True,
            )
            appearance_features = structure_features
            appearance_source = "structure_only"
            if slat_dense_grid is not None:
                if slat_dense_grid.ndim == 4:
                    slat_dense_grid = slat_dense_grid.unsqueeze(0)
                if slat_dense_grid.ndim != 5 or slat_dense_grid.shape[0] != structure_features.shape[0]:
                    raise ValueError("slat_dense_grid must be [B,C,D,H,W] with the SS batch size")
                if slat_occupancy is None:
                    slat_occupancy = (slat_dense_grid.abs().sum(dim=1, keepdim=True) > 0).to(
                        slat_dense_grid.dtype
                    )
                if slat_occupancy.ndim == 4:
                    slat_occupancy = slat_occupancy.unsqueeze(1)
                if slat_occupancy.shape[:2] != (slat_dense_grid.shape[0], 1):
                    raise ValueError("slat_occupancy must be [B,1,D,H,W]")
                slat_input = torch.cat(
                    (slat_dense_grid.float(), slat_occupancy.float()),
                    dim=1,
                )
                slat_features = self.slat_encoder(slat_input)
                slat_features = F.interpolate(
                    slat_features,
                    size=(resolution, resolution, resolution),
                    mode="trilinear",
                    align_corners=True,
                )
                appearance_features = structure_features + slat_features
                appearance_source = "generated_slat"
            sphere = _sphere_sdf(
                structure_features.shape[0],
                resolution,
                device=structure_features.device,
                dtype=structure_features.dtype,
                bounds=self.bounds,
            )
            # Geometry depends only on sparse structure.  Generated SLAT is
            # admitted exclusively through appearance heads, preventing color
            # evidence from silently moving the surface.
            sdf = sphere + 0.25 * torch.tanh(self.sdf_head(structure_features))
            return TrellisFieldPrior(
                sdf_mean=sdf,
                log_precision=self.precision_head(structure_features).clamp(-10.0, 8.0),
                diffuse_logits=self.diffuse_head(appearance_features),
                directional_sh=self.directional_head(appearance_features),
                source=f"trellis_sparse_structure_distillation+{appearance_source}",
                appearance_log_precision=self.appearance_precision_head(
                    appearance_features
                ).clamp(-10.0, 8.0),
            )
        if batch_size is None or device is None:
            raise ValueError("batch_size and device are required when no TRELLIS prior input is available")
        sphere = _sphere_sdf(batch_size, resolution, device=device, dtype=dtype, bounds=self.bounds)
        return TrellisFieldPrior(
            sdf_mean=sphere,
            log_precision=torch.full_like(sphere, -8.0),
            diffuse_logits=torch.zeros(batch_size, 3, resolution, resolution, resolution, device=device, dtype=dtype),
            directional_sh=torch.zeros(
                batch_size,
                27,
                resolution,
                resolution,
                resolution,
                device=device,
                dtype=dtype,
            ),
            source="weak_sphere_fallback_no_trellis_authority",
            appearance_log_precision=torch.full_like(sphere, -8.0),
        )


class ObservationConditionedCompletion(nn.Module):
    """Predict a completion proposal in explicit field space."""

    def __init__(
        self,
        *,
        hidden_channels: int = 96,
        semantic_dim: int = 0,
    ) -> None:
        super().__init__()
        # Current state: sdf, log variance, 5 responsibilities, diffuse (10).
        # Prior: sdf, log precision, diffuse (5).
        input_channels = 15
        self.semantic_dim = int(semantic_dim)
        self.semantic_projection = (
            nn.Linear(self.semantic_dim, hidden_channels) if self.semantic_dim > 0 else None
        )
        self.input = nn.Conv3d(input_channels, hidden_channels, 3, padding=1)
        self.blocks = nn.Sequential(
            _Residual3D(hidden_channels, dilation=1),
            _Residual3D(hidden_channels, dilation=2),
            _Residual3D(hidden_channels, dilation=1),
        )
        self.sdf_head = nn.Conv3d(hidden_channels, 1, 3, padding=1)
        self.diffuse_head = nn.Conv3d(hidden_channels, 3, 1)
        self.directional_head = nn.Conv3d(hidden_channels, 27, 1)
        nn.init.zeros_(self.sdf_head.weight)
        nn.init.zeros_(self.sdf_head.bias)
        nn.init.zeros_(self.diffuse_head.weight)
        nn.init.zeros_(self.diffuse_head.bias)
        nn.init.zeros_(self.directional_head.weight)
        nn.init.zeros_(self.directional_head.bias)

    def forward(
        self,
        field: PosteriorFieldState,
        prior: TrellisFieldPrior,
        *,
        semantic_condition: Optional[torch.Tensor] = None,
    ) -> CompletionProposal:
        volume = field.volume
        if prior.sdf_mean.shape[-3:] != volume.shape[-3:]:
            raise ValueError("TRELLIS prior and posterior field resolutions differ")
        state_probability = volume[:, FIELD_CHANNELS["evidence_logits"]].softmax(dim=1)
        current = torch.cat(
            (
                volume[:, FIELD_CHANNELS["sdf_mean"]],
                volume[:, FIELD_CHANNELS["sdf_log_variance"]],
                state_probability,
                volume[:, FIELD_CHANNELS["diffuse_logits"]],
                prior.sdf_mean,
                prior.log_precision,
                prior.diffuse_logits,
            ),
            dim=1,
        )
        hidden = self.input(current)
        if self.semantic_projection is not None:
            if semantic_condition is None or semantic_condition.shape[-1] != self.semantic_dim:
                raise ValueError(f"semantic_condition must be [B,{self.semantic_dim}]")
            hidden = hidden + self.semantic_projection(semantic_condition).to(hidden.dtype)[..., None, None, None]
        hidden = self.blocks(hidden)
        # These are posterior state responsibilities, not a learned confidence
        # gate.  Contradictory regions retain only half a prior step so they
        # remain uncertain rather than being silently overwritten.
        completion_responsibility = (
            state_probability[:, 2:3]
            + state_probability[:, 3:4]
            + 0.5 * state_probability[:, 4:5]
        ).clamp(0.0, 1.0)
        prior_precision = F.softplus(prior.log_precision.float()).to(volume.dtype)
        appearance_precision = F.softplus(
            (
                prior.appearance_log_precision
                if prior.appearance_log_precision is not None
                else prior.log_precision
            ).float()
        ).to(volume.dtype)
        current_sdf = volume[:, FIELD_CHANNELS["sdf_mean"]]
        information_fraction = prior_precision / (1.0 + prior_precision)
        appearance_information_fraction = appearance_precision / (
            1.0 + appearance_precision
        )
        explicit_prior_delta = information_fraction * (prior.sdf_mean - current_sdf)
        learned_delta = 0.1 * torch.tanh(self.sdf_head(hidden))
        sdf_delta = completion_responsibility * (explicit_prior_delta + learned_delta)
        diffuse_delta = completion_responsibility * (
            appearance_information_fraction
            * (prior.diffuse_logits - volume[:, FIELD_CHANNELS["diffuse_logits"]])
            + 0.25 * torch.tanh(self.diffuse_head(hidden))
        )
        directional_delta = completion_responsibility * (
            appearance_information_fraction * (
                prior.directional_sh - volume[:, FIELD_CHANNELS["directional_sh"]]
            )
            + 0.1 * torch.tanh(self.directional_head(hidden))
        )
        return CompletionProposal(
            sdf_delta=sdf_delta,
            diffuse_delta=diffuse_delta,
            directional_delta=directional_delta,
            completion_responsibility=completion_responsibility,
            prior_precision=prior_precision,
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


def _sphere_sdf(batch_size, resolution, *, device, dtype, bounds):
    points = make_canonical_grid(
        resolution,
        batch_size=batch_size,
        bounds=bounds,
        device=device,
        dtype=dtype,
    )
    return (points.norm(dim=-1) - 0.35).reshape(
        batch_size,
        resolution,
        resolution,
        resolution,
    )[:, None]
