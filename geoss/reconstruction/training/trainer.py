from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import torch
import torch.nn as nn

from geoss.geometry.alignment import align_vggt_batch
from geoss.reconstruction.models import RAPC3D
from geoss.reconstruction.objectives import RAPC3DObjective
from geoss.reconstruction.priors import TrellisFieldPrior, TrellisPriorInput
from geoss.reconstruction.rays import (
    camera_depth_to_ray_distance,
    deterministic_pixel_grid,
    sample_image_at_pixels,
)


@dataclass(frozen=True)
class RAPCTrainerConfig:
    learning_rate: float = 2e-4
    calibration_learning_rate: float = 5e-5
    prior_learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    render_pixel_stride: int = 8
    gradient_clip_norm: float = 5.0
    use_bfloat16_features: bool = True
    audit_interval: int = 500


class RAPCTrainer:
    """Training-state owner for the reconstruction algorithm.

    Distributed process creation, logging, and checkpoint persistence remain
    outside this class; the complete differentiable scientific path lives
    here.
    """

    def __init__(
        self,
        model: nn.Module,
        vggt: nn.Module,
        objective: RAPC3DObjective,
        *,
        config: Optional[RAPCTrainerConfig] = None,
        prior_provider: Optional[
            Callable[
                [Dict[str, torch.Tensor]],
                Optional[TrellisFieldPrior | TrellisPriorInput],
            ]
        ] = None,
    ) -> None:
        self.model = model
        self.vggt = vggt
        self.objective = objective
        self.config = config or RAPCTrainerConfig()
        self.prior_provider = prior_provider
        core_model = model.module if hasattr(model, "module") else model
        if not isinstance(core_model, RAPC3D):
            raise TypeError(f"RAPCTrainer requires RAPC3D or DDP[RAPC3D], got {type(model)!r}")
        calibration_parameters = list(core_model.evidence_builder.depth_calibrator.parameters())
        prior_parameters = list(core_model.prior_adapter.parameters()) + list(core_model.completion.parameters())
        excluded = {id(parameter) for parameter in calibration_parameters + prior_parameters}
        core_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in excluded
        ]
        self.optimizer = torch.optim.AdamW(
            (
                {"params": core_parameters, "lr": self.config.learning_rate},
                {"params": calibration_parameters, "lr": self.config.calibration_learning_rate},
                {"params": prior_parameters, "lr": self.config.prior_learning_rate},
            ),
            weight_decay=self.config.weight_decay,
        )
        self.step_index = 0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.model.train()
        self.vggt.eval()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            device_type = batch["images"].device.type
            use_amp = self.config.use_bfloat16_features and device_type == "cuda"
            feature_dtype = (
                torch.bfloat16
                if use_amp and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            with torch.autocast(device_type=device_type, dtype=feature_dtype, enabled=use_amp):
                vggt_output = self.vggt(batch["images"], use_cache=False)
            batch = {**batch, **vggt_output}
            if batch.get("aligned_depth") is None and batch.get("vggt_pointmap") is not None:
                core_model = self.model.module if hasattr(self.model, "module") else self.model
                batch = align_vggt_batch(batch, core_model.geometry_alignment)
        prior = self.prior_provider(batch) if self.prior_provider is not None else None
        pixels = self._render_pixels(batch)
        output = self.model(
            batch,
            trellis_prior=prior,
            render_pixels=pixels,
            decode_gaussians=True,
            extract_mesh=False,
        )
        targets = {
            "rgb": sample_image_at_pixels(batch["images"], pixels),
            "mask": sample_image_at_pixels(batch["masks"], pixels),
            "image_shape": self._render_grid_shape(batch),
        }
        if batch.get("aligned_depth") is not None:
            camera_depth = sample_image_at_pixels(batch["aligned_depth"], pixels)
            targets["depth"] = camera_depth_to_ray_distance(
                pixels,
                camera_depth,
                batch["K"],
            )
        losses = self.objective(output, batch, render_targets=targets)
        losses["total"].backward()
        gradient_audit = _gradient_audit(self.model)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
        parameter_before = None
        run_update_audit = self.step_index % max(self.config.audit_interval, 1) == 0
        if run_update_audit:
            parameter_before = {
                name: parameter.detach().float().clone()
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            }
        self.optimizer.step()
        if parameter_before is not None:
            update_norm = _parameter_update_norm(self.model, parameter_before)
        else:
            update_norm = losses["total"].new_tensor(float("nan"))
        self.step_index += 1
        return {
            **losses,
            **gradient_audit,
            "parameter_update_norm": update_norm,
            "surface_constraint_count": output.diagnostics["surface_constraint_count"].float().mean(),
            "free_constraint_count": output.diagnostics["free_constraint_count"].float().mean(),
        }

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.model.eval()
        self.vggt.eval()
        vggt_output = self.vggt(batch["images"], use_cache=False)
        batch = {**batch, **vggt_output}
        if batch.get("aligned_depth") is None and batch.get("vggt_pointmap") is not None:
            core_model = self.model.module if hasattr(self.model, "module") else self.model
            batch = align_vggt_batch(batch, core_model.geometry_alignment)
        prior = self.prior_provider(batch) if self.prior_provider is not None else None
        pixels = self._render_pixels(batch)
        output = self.model(
            batch,
            trellis_prior=prior,
            render_pixels=pixels,
            decode_gaussians=True,
            extract_mesh=False,
        )
        targets = {
            "rgb": sample_image_at_pixels(batch["images"], pixels),
            "mask": sample_image_at_pixels(batch["masks"], pixels),
            "image_shape": self._render_grid_shape(batch),
        }
        if batch.get("aligned_depth") is not None:
            camera_depth = sample_image_at_pixels(batch["aligned_depth"], pixels)
            targets["depth"] = camera_depth_to_ray_distance(
                pixels,
                camera_depth,
                batch["K"],
            )
        return self.objective(output, batch, render_targets=targets)

    def _render_pixels(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        b, v, _, h, w = batch["images"].shape
        return deterministic_pixel_grid(
            h,
            w,
            stride=self.config.render_pixel_stride,
            batch_size=b,
            num_views=v,
            device=batch["images"].device,
            dtype=batch["images"].dtype,
        )

    def _render_grid_shape(self, batch: Dict[str, torch.Tensor]) -> tuple[int, int]:
        height, width = batch["images"].shape[-2:]
        offset = self.config.render_pixel_stride // 2
        rows = len(range(offset, height, self.config.render_pixel_stride))
        columns = len(range(offset, width, self.config.render_pixel_stride))
        return max(rows, 1), max(columns, 1)


def collate_rapc_samples(samples: Iterable[Dict]) -> Dict:
    """Collate incomplete MeshFleet samples without UID substitution.

    Missing geometry supervision is represented by explicit validity masks;
    RGB/camera observations remain trainable and no sample is silently mapped
    to another UID.
    """

    samples = list(samples)
    if not samples:
        raise ValueError("Cannot collate an empty RAPC batch")
    required_tensor_keys = ("images", "masks", "K", "c2w", "w2c")
    batch: Dict = {
        key: torch.stack([sample[key] for sample in samples], dim=0)
        for key in required_tensor_keys
    }
    batch["uid"] = [sample["uid"] for sample in samples]
    batch["metadata"] = [sample.get("metadata", {}) for sample in samples]
    has_geometry = torch.tensor(
        [
            _sample_has_geometry(sample)
            and "gt_occ" in sample
            for sample in samples
        ],
        dtype=torch.bool,
    )
    batch["geometry_supervision_valid"] = has_geometry
    reference_occ = next((sample["gt_occ"] for sample in samples if "gt_occ" in sample), None)
    if reference_occ is not None:
        batch["gt_occ"] = torch.stack(
            [
                sample["gt_occ"] if "gt_occ" in sample else torch.zeros_like(reference_occ)
                for sample in samples
            ],
            dim=0,
        )
    point_counts = [sample.get("gt_sparse_xyz", torch.empty(0, 3)).shape[0] for sample in samples]
    maximum_points = max(point_counts)
    if maximum_points > 0:
        reference = next(sample["gt_sparse_xyz"] for sample in samples if "gt_sparse_xyz" in sample)
        points = reference.new_zeros((len(samples), maximum_points, 3))
        valid = torch.zeros(len(samples), maximum_points, dtype=torch.bool)
        for index, sample in enumerate(samples):
            value = sample.get("gt_sparse_xyz")
            if value is None:
                continue
            points[index, : value.shape[0]] = value
            valid[index, : value.shape[0]] = True
        batch["gt_sparse_xyz"] = points
        batch["gt_sparse_valid"] = valid
        # MeshFleet loader exposes voxel centers in the historical GeoSS
        # [-1,1] convention; RAPC's calibrated camera world is [-0.5,0.5].
        batch["gt_sparse_to_field_scale"] = torch.full((len(samples),), 0.5)
    for optional in (
        "trellis_cond_image",
        "vggt_depth",
        "vggt_pointmap",
        "vggt_confidence",
        "vggt_confidence_raw",
    ):
        values = [sample.get(optional) for sample in samples]
        if all(isinstance(value, torch.Tensor) and value.shape == values[0].shape for value in values):
            batch[optional] = torch.stack(values, dim=0)
    return batch


def _gradient_audit(model: nn.Module) -> Dict[str, torch.Tensor]:
    squared_norm = None
    parameter_count = 0
    nonzero_count = 0
    finite = True
    reference = next(model.parameters())
    module_norms: Dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        value = gradient.detach().float()
        finite = finite and bool(torch.isfinite(value).all().item())
        norm_square = value.square().sum()
        squared_norm = norm_square if squared_norm is None else squared_norm + norm_square
        parameter_count += value.numel()
        nonzero_count += int((value != 0).sum().item())
        module = name.split(".", 1)[0]
        module_norms[module] = module_norms.get(module, value.new_zeros(())) + norm_square
    total_norm = reference.new_zeros(()) if squared_norm is None else squared_norm.sqrt().to(reference.dtype)
    out = {
        "gradient_norm": total_norm,
        "nonzero_gradient_fraction": reference.new_tensor(
            nonzero_count / max(parameter_count, 1),
            dtype=torch.float32,
        ),
        "finite_gradients": reference.new_tensor(float(finite)),
    }
    out.update({f"gradient_norm/{name}": value.sqrt().to(reference.dtype) for name, value in module_norms.items()})
    return out


def _parameter_update_norm(model: nn.Module, before: Dict[str, torch.Tensor]) -> torch.Tensor:
    squared = None
    reference = next(model.parameters())
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        difference = parameter.detach().float() - before[name].to(parameter.device)
        value = difference.square().sum()
        squared = value if squared is None else squared + value
    return reference.new_zeros(()) if squared is None else squared.sqrt().to(reference.dtype)


def _sample_has_geometry(sample: Dict) -> bool:
    value = sample.get("has_gt", False)
    if isinstance(value, torch.Tensor):
        return bool(value.detach().float().item())
    return bool(value)
