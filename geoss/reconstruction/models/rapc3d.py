from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from geoss.geometry.alignment import GeometryAlignment
from geoss.reconstruction.decoders import MeshAsset, SurfaceGaussianAsset, UnifiedSurfaceDecoder
from geoss.reconstruction.evidence import FUSED_EVIDENCE_DIM, RayEvidence, RayEvidenceBuilder, VGGTDepthCalibrator
from geoss.reconstruction.fields import FIELD_CHANNELS, PosteriorFieldState, make_canonical_grid
from geoss.reconstruction.optimization import (
    AdaptiveBrickRefiner,
    EvidenceFieldInitializer,
    PosteriorStateRefiner,
)
from geoss.reconstruction.posterior import (
    MatrixFreeAuthorityProjector,
    ObservationConstraintBundle,
    RayConstraintBuilder,
    flatten_hierarchical_sdf,
    make_hierarchical_constraints,
    replace_hierarchical_sdf,
)
from geoss.reconstruction.priors import (
    ObservationConditionedCompletion,
    TrellisFieldPrior,
    TrellisPriorInput,
    TrellisPriorAdapter,
)
from geoss.reconstruction.rays import camera_rays_from_pixels
from geoss.reconstruction.visibility import (
    SDFRayRenderer,
    field_transmittance_to_points,
)


@dataclass(frozen=True)
class RAPC3DConfig:
    field_resolution: int = 32
    field_bounds: Tuple[float, float] = (-0.5, 0.5)
    posterior_iterations: int = 4
    hidden_channels: int = 96
    visibility_samples: int = 24
    visibility_point_chunk: int = 2048
    constraint_pixel_stride: int = 8
    surface_tolerance: float = 0.01
    # A one-standard-deviation calibrated credible event.  The threshold is
    # applied to predicted error probability, never raw VGGT confidence.
    constraint_reliability_probability: float = 0.68
    free_margin: float = 0.002
    max_bricks: int = 96
    brick_resolution: int = 4
    use_adaptive_bricks: bool = True
    max_gaussians: int = 24_576
    render_samples: int = 96
    shell_beta: float = 0.008
    shell_eta: float = 12.0
    trellis_ss_latent_channels: int = 8
    trellis_slat_latent_channels: int = 8
    semantic_condition_dim: int = 0


@dataclass
class RAPC3DOutput:
    field: PosteriorFieldState
    evidence: RayEvidence
    prior: TrellisFieldPrior
    constraints: ObservationConstraintBundle
    gaussians: Optional[SurfaceGaussianAsset]
    render: Optional[Dict[str, torch.Tensor]]
    diagnostics: Dict[str, Any] = dataclass_field(default_factory=dict)
    meshes: Optional[List[MeshAsset]] = None


class RAPC3D(nn.Module):
    """Ray-Authoritative Posterior Completion with surface-tethered Gaussians."""

    def __init__(self, config: Optional[RAPC3DConfig] = None) -> None:
        super().__init__()
        self.config = config or RAPC3DConfig()
        calibrator = VGGTDepthCalibrator()
        self.geometry_alignment = GeometryAlignment(canonical_extent=0.5)
        self.evidence_builder = RayEvidenceBuilder(depth_calibrator=calibrator)
        self.constraint_builder = RayConstraintBuilder(
            depth_calibrator=calibrator,
            pixel_stride=self.config.constraint_pixel_stride,
            surface_tolerance=self.config.surface_tolerance,
            reliability_probability=self.config.constraint_reliability_probability,
        )
        self.prior_adapter = TrellisPriorAdapter(
            ss_latent_channels=self.config.trellis_ss_latent_channels,
            slat_latent_channels=self.config.trellis_slat_latent_channels,
            hidden_channels=64,
            output_resolution=self.config.field_resolution,
            bounds=self.config.field_bounds,
        )
        self.initializer = EvidenceFieldInitializer(hidden_channels=self.config.hidden_channels)
        self.state_refiner = PosteriorStateRefiner(hidden_channels=self.config.hidden_channels)
        self.completion = ObservationConditionedCompletion(
            hidden_channels=self.config.hidden_channels,
            semantic_dim=self.config.semantic_condition_dim,
        )
        self.authority = MatrixFreeAuthorityProjector()
        self.brick_refiner = AdaptiveBrickRefiner(
            max_bricks=self.config.max_bricks,
            brick_resolution=self.config.brick_resolution,
        )
        self.surface_decoder = UnifiedSurfaceDecoder(max_gaussians=self.config.max_gaussians)
        self.renderer = SDFRayRenderer(
            num_samples=self.config.render_samples,
            beta=self.config.shell_beta,
            eta=self.config.shell_eta,
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        trellis_prior: Optional[TrellisFieldPrior | TrellisPriorInput] = None,
        render_pixels: Optional[torch.Tensor] = None,
        decode_gaussians: bool = True,
        extract_mesh: bool = False,
    ) -> RAPC3DOutput:
        inputs = self._prepare_observations(batch)
        images = inputs["images"]
        b = images.shape[0]
        resolution = self.config.field_resolution
        grid_points = make_canonical_grid(
            resolution,
            batch_size=b,
            bounds=self.config.field_bounds,
            device=images.device,
            dtype=torch.float32,
        )
        if isinstance(trellis_prior, TrellisPriorInput):
            prior = self.prior_adapter(
                ss_latent_grid=trellis_prior.ss_latent_grid,
                slat_dense_grid=trellis_prior.slat_dense_grid,
                slat_occupancy=trellis_prior.slat_occupancy,
            )
        elif isinstance(trellis_prior, TrellisFieldPrior):
            prior = trellis_prior
        elif trellis_prior is None:
            prior = self._make_prior(batch, b, images.device)
        else:
            raise TypeError(
                "trellis_prior must be TrellisPriorInput, TrellisFieldPrior, or None"
            )
        evidence = self._build_evidence(inputs, grid_points, physical_visibility=None)
        evidence_volume = _points_to_volume(evidence.fused, resolution)
        field = self.initializer(
            evidence_volume,
            prior,
            bounds=self.config.field_bounds,
        )
        constraints = self.constraint_builder(
            aligned_depth=inputs["aligned_depth"],
            masks=inputs["masks"],
            K=inputs["K"],
            c2w=inputs["c2w"],
            resolution=resolution,
            raw_vggt_confidence=inputs.get("raw_vggt_confidence"),
            camera_reliability=inputs.get("camera_reliability"),
            bounds=self.config.field_bounds,
        )
        iteration_diagnostics: List[Dict[str, torch.Tensor]] = []
        camera_centers = inputs["c2w"][..., :3, 3]

        for _ in range(self.config.posterior_iterations):
            visibility = field_transmittance_to_points(
                field,
                grid_points,
                camera_centers,
                num_samples=self.config.visibility_samples,
                beta=self.config.shell_beta,
                eta=self.config.shell_eta,
                point_chunk_size=self.config.visibility_point_chunk,
            )
            evidence = self._build_evidence(
                inputs,
                grid_points,
                physical_visibility=visibility,
            )
            evidence_volume = _points_to_volume(evidence.fused, resolution)
            field = self.state_refiner(field, evidence_volume, prior)
            completion = self.completion(
                field,
                prior,
                semantic_condition=batch.get("semantic_condition"),
            )
            field, update_diagnostics = self._apply_authoritative_completion(
                field,
                completion.sdf_delta,
                completion.diffuse_delta,
                completion.directional_delta,
                completion.prior_precision,
                constraints,
            )
            iteration_diagnostics.append(update_diagnostics)

        if self.config.use_adaptive_bricks:
            field, brick_diagnostics = self._refine_bricks(
                field,
                inputs,
                camera_centers,
                constraints,
                prior,
            )
        else:
            brick_diagnostics = {"enabled": torch.tensor(False, device=images.device)}

        final_visibility = field_transmittance_to_points(
            field,
            grid_points,
            camera_centers,
            num_samples=self.config.visibility_samples,
            beta=self.config.shell_beta,
            eta=self.config.shell_eta,
            point_chunk_size=self.config.visibility_point_chunk,
        )
        evidence = self._build_evidence(
            inputs,
            grid_points,
            physical_visibility=final_visibility,
        )
        gaussians = self.surface_decoder(field) if decode_gaussians else None
        rendered = None
        if render_pixels is not None:
            if render_pixels.ndim != 4 or render_pixels.shape[:2] != images.shape[:2]:
                raise ValueError("render_pixels must be [B,V,P,2] matching conditioning cameras")
            rays = camera_rays_from_pixels(render_pixels, inputs["K"], inputs["c2w"])
            rendered = self.renderer(
                field,
                rays.origins.reshape(b, -1, 3),
                rays.directions.reshape(b, -1, 3),
            )
            rendered["pixel_shape"] = torch.tensor(
                render_pixels.shape[1:3],
                device=images.device,
                dtype=torch.long,
            )
        meshes = self.surface_decoder.extract_meshes(field) if extract_mesh else None
        return RAPC3DOutput(
            field=field,
            evidence=evidence,
            prior=prior,
            constraints=constraints,
            gaussians=gaussians,
            render=rendered,
            meshes=meshes,
            diagnostics={
                "authority_iterations": iteration_diagnostics,
                "bricks": brick_diagnostics,
                "prior_source": prior.source,
                "surface_constraint_count": constraints.surface_reliable.sum(dim=-1),
                "free_constraint_count": constraints.free_reliable.sum(dim=-1),
                "test_time_ground_truth_latents_used": False,
            },
        )

    def _prepare_observations(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        required = ("images", "masks", "K", "c2w", "w2c")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"RAPC-3D observation batch is missing {missing}")
        out = {key: batch[key] for key in required}
        if batch.get("aligned_depth") is not None:
            out["aligned_depth"] = batch["aligned_depth"]
            out["aligned_pointmap"] = batch.get("aligned_pointmap")
            out["camera_reliability"] = batch.get("alignment_confidence")
        elif batch.get("vggt_pointmap") is not None:
            alignment = self.geometry_alignment(
                vggt_depth=batch.get("vggt_depth"),
                vggt_pointmap=batch["vggt_pointmap"],
                K=batch["K"],
                c2w=batch["c2w"],
                w2c=batch["w2c"],
                masks=batch["masks"],
                vggt_confidence=batch.get("vggt_confidence"),
                dataset_depth=None,
                vggt_camera=batch.get("vggt_camera"),
            )
            out["aligned_depth"] = alignment["aligned_depth"]
            out["aligned_pointmap"] = alignment["aligned_pointmap"]
            out["camera_reliability"] = alignment["alignment_confidence"]
            out["alignment_debug"] = alignment
        else:
            raise KeyError(
                "RAPC-3D requires aligned_depth or VGGT point maps for a calibrated observation likelihood. "
                "Raw unaligned VGGT depth is not accepted."
            )
        out["raw_vggt_confidence"] = batch.get(
            "vggt_confidence_raw",
            batch.get("vggt_confidence"),
        )
        out["vggt_features"] = batch.get("vggt_features")
        return out

    def _make_prior(
        self,
        batch: Dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> TrellisFieldPrior:
        # `ss_latent_grid` from the MeshFleet dataset is a ground-truth asset
        # latent and is deliberately ignored.  Only a latent generated by the
        # conditioning-only TRELLIS pipeline is accepted at test time.
        generated_ss = batch.get("trellis_generated_ss_latent_grid")
        return self.prior_adapter(
            ss_latent_grid=generated_ss,
            decoded_prior_sdf=batch.get("trellis_generated_prior_sdf"),
            decoded_prior_log_precision=batch.get("trellis_generated_prior_log_precision"),
            decoded_prior_diffuse_logits=batch.get("trellis_generated_prior_diffuse_logits"),
            batch_size=batch_size,
            device=device,
            dtype=torch.float32,
        )

    def _build_evidence(
        self,
        inputs: Dict[str, torch.Tensor],
        points: torch.Tensor,
        *,
        physical_visibility: Optional[torch.Tensor],
    ) -> RayEvidence:
        return self.evidence_builder(
            points=points,
            images=inputs["images"],
            masks=inputs["masks"],
            K=inputs["K"],
            c2w=inputs["c2w"],
            w2c=inputs["w2c"],
            aligned_depth=inputs["aligned_depth"],
            raw_vggt_confidence=inputs.get("raw_vggt_confidence"),
            aligned_pointmap=inputs.get("aligned_pointmap"),
            vggt_features=inputs.get("vggt_features"),
            camera_reliability=inputs.get("camera_reliability"),
            physical_visibility=physical_visibility,
        )

    def _apply_authoritative_completion(
        self,
        field: PosteriorFieldState,
        sdf_delta: torch.Tensor,
        diffuse_delta: torch.Tensor,
        directional_delta: torch.Tensor,
        prior_precision: torch.Tensor,
        constraints: ObservationConstraintBundle,
    ) -> Tuple[PosteriorFieldState, Dict[str, torch.Tensor]]:
        b = field.volume.shape[0]
        current = field.volume[:, 0].reshape(b, -1)
        proposal = sdf_delta.reshape(b, -1)
        observation_precision = (-field.volume[:, 1]).float().exp()
        metric = (1.0 + observation_precision + prior_precision.float()).reshape(b, -1)
        update = self.authority(
            current,
            proposal,
            metric_diagonal=metric,
            surface_constraints=constraints.surface,
            free_constraints=constraints.free,
            free_margin=self.config.free_margin,
        )
        # Assemble the state functionally.  Assigning an updated slice back
        # into the same cloned tensor after reading that slice creates a
        # version-counter conflict in autograd (the appearance branch retains
        # the pre-assignment view for its backward pass).
        volume = torch.cat(
            (
                update.value.reshape(b, 1, *field.volume.shape[-3:]),
                field.volume[:, 1:8],
                (
                    field.volume[:, FIELD_CHANNELS["diffuse_logits"]]
                    + diffuse_delta
                ).clamp(-12.0, 12.0),
                (
                    field.volume[:, FIELD_CHANNELS["directional_sh"]]
                    + directional_delta
                ).clamp(-8.0, 8.0),
            ),
            dim=1,
        )
        return field.replace_volume(volume), update.diagnostics

    def _refine_bricks(
        self,
        field: PosteriorFieldState,
        inputs: Dict[str, torch.Tensor],
        camera_centers: torch.Tensor,
        constraints: ObservationConstraintBundle,
        prior: TrellisFieldPrior,
    ) -> Tuple[PosteriorFieldState, Dict[str, torch.Tensor]]:
        indices, brick_points, brick_valid, envelope = self.brick_refiner.select(field)
        b = field.volume.shape[0]
        flat_points = brick_points.reshape(b, -1, 3)
        visibility = field_transmittance_to_points(
            field,
            flat_points,
            camera_centers,
            num_samples=self.config.visibility_samples,
            beta=self.config.shell_beta,
            eta=self.config.shell_eta,
            point_chunk_size=self.config.visibility_point_chunk,
        )
        fine_evidence = self._build_evidence(
            inputs,
            flat_points,
            physical_visibility=visibility,
        )
        level = self.brick_refiner.decode(
            field,
            brick_indices=indices,
            brick_points=brick_points,
            brick_valid=brick_valid,
            envelope=envelope,
            fused_evidence=fine_evidence.fused,
        )
        refined = field.with_residual_levels((level,))
        surface = make_hierarchical_constraints(
            refined,
            constraints.surface_points,
            valid=constraints.surface_reliable,
        )
        free = make_hierarchical_constraints(
            refined,
            constraints.free_points,
            valid=constraints.free_reliable,
        )
        current = flatten_hierarchical_sdf(refined)
        coarse_precision = (1.0 + (-field.volume[:, 1]).float().exp() + F_softplus(prior.log_precision)).reshape(
            b,
            -1,
        )
        detail_count = current.shape[1] - coarse_precision.shape[1]
        detail_metric = coarse_precision.mean(dim=-1, keepdim=True).expand(b, detail_count)
        metric = torch.cat((coarse_precision, detail_metric), dim=-1)
        authority = self.authority(
            current,
            torch.zeros_like(current),
            metric_diagonal=metric,
            surface_constraints=surface,
            free_constraints=free,
            free_margin=self.config.free_margin,
        )
        refined = replace_hierarchical_sdf(refined, authority.value)
        diagnostics = {
            "enabled": torch.tensor(True, device=field.volume.device),
            "active_brick_count": brick_valid.sum(dim=-1),
            **authority.diagnostics,
        }
        return refined, diagnostics


def _points_to_volume(points: torch.Tensor, resolution: int) -> torch.Tensor:
    if points.ndim != 3 or points.shape[-1] != FUSED_EVIDENCE_DIM:
        raise ValueError(f"points must be [B,N,{FUSED_EVIDENCE_DIM}]")
    b = points.shape[0]
    if points.shape[1] != resolution**3:
        raise ValueError(f"Expected {resolution**3} field nodes, got {points.shape[1]}")
    return points.transpose(1, 2).reshape(
        b,
        FUSED_EVIDENCE_DIM,
        resolution,
        resolution,
        resolution,
    )


def F_softplus(value: torch.Tensor) -> torch.Tensor:
    # Local alias keeps the principal geometry reductions explicit and FP32.
    return torch.nn.functional.softplus(value.float())
