from __future__ import annotations

import contextlib
import sys
from types import MethodType
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

from geoss.integration.trellis_ss_hook import GeoSSTrellisSSWrapper, ss_grid_to_tokens, tokens_to_ss_grid
from geoss.io.asset_io import write_internal_mesh
from geoss.metrics.gaussian_metrics import gaussian_statistics
from geoss.slat.integration.trellis_slat_hook import GeoVisTrellisSLATWrapper


class RealTrellisGeoPipeline:
    """Explicit real TRELLIS image->SS->SLAT->decoder wrapper.

    It avoids the TRELLIS `run()` convenience method so adapter contexts are
    passed into the samplers instead of silently ignored.
    """

    def __init__(self, trellis_root: Optional[str], pipeline_path: str, device: str = "cuda") -> None:
        if trellis_root:
            sys.path.insert(0, str(Path(trellis_root)))
        try:
            from trellis.pipelines import TrellisImageTo3DPipeline
        except ModuleNotFoundError as exc:
            # TRELLIS imports rembg/pymatting at module import time.  Report an
            # environment contract failure explicitly so it cannot be confused
            # with a corrupt Stage-3/4 adapter checkpoint.
            missing = exc.name or "<unknown>"
            raise RuntimeError(
                "TRELLIS runtime dependency is missing from the active Python "
                f"environment: {missing!r} (python={sys.executable}). Install "
                "the project requirements with this same interpreter before "
                "running real inference."
            ) from exc

        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(pipeline_path)
        self.pipeline.to(device)
        self.device = torch.device(device)
        self._require_models(
            "sparse_structure_flow_model",
            "sparse_structure_decoder",
            "slat_flow_model",
            "slat_decoder_gs",
        )

    def install_ss_adapter(self, velocity_adapter) -> None:
        self.pipeline.models["sparse_structure_flow_model"] = GeoSSTrellisSSWrapper(
            self.pipeline.models["sparse_structure_flow_model"],
            velocity_adapter,
            use_geoss_adapter=True,
        ).to(self.device)

    def install_slat_adapter(self, velocity_adapter) -> None:
        self.pipeline.models["slat_flow_model"] = GeoVisTrellisSLATWrapper(
            self.pipeline.models["slat_flow_model"],
            velocity_adapter,
            use_geovis_slat=True,
        ).to(self.device)

    def ss_velocity_hook(self, x_t: torch.Tensor, t: torch.Tensor, cond, base_velocity: torch.Tensor, context: Dict[str, torch.Tensor]):
        flow = self.pipeline.models["sparse_structure_flow_model"]
        if not isinstance(flow, GeoSSTrellisSSWrapper):
            raise RuntimeError("ss_velocity_hook requires install_ss_adapter() before sampling.")
        out = flow.velocity_adapter(
            ss_latent_tokens=ss_grid_to_tokens(x_t),
            geo_tokens=context["geo_tokens"],
            geo_confidence=context["geo_confidence"],
            timestep=t,
            v_base=ss_grid_to_tokens(base_velocity),
            voxel_xyz=context.get("ss_voxel_xyz"),
            anchor_xyz=context.get("anchor_xyz"),
            anchor_metadata=context.get("anchor_metadata"),
        )
        return tokens_to_ss_grid(out["v_geo"], tuple(x_t.shape[-3:])), out

    def slat_velocity_hook(self, x_t, t: torch.Tensor, cond, ss_context, base_velocity, context: Dict[str, torch.Tensor]):
        flow = self.pipeline.models["slat_flow_model"]
        if not isinstance(flow, GeoVisTrellisSLATWrapper):
            raise RuntimeError("slat_velocity_hook requires install_slat_adapter() before sampling.")
        return flow(x_t, t, cond, geovis_slat_context=context), dict(flow.last_debug)

    @torch.no_grad()
    def run(
        self,
        images: List,
        *,
        geoss_context: Optional[Dict[str, torch.Tensor]] = None,
        geovis_slat_context: Optional[Dict[str, torch.Tensor]] = None,
        coords_override: Optional[torch.Tensor] = None,
        formats: Iterable[str] = ("gaussian", "mesh"),
        seed: int = 42,
        ss_sampler_params: Optional[dict] = None,
        slat_sampler_params: Optional[dict] = None,
        multi_image_mode: str = "multidiffusion",
        preprocess_images: bool = True,
    ) -> Dict[str, object]:
        images = self._prepare_conditioning_images(images, preprocess=preprocess_images)
        cond = self.pipeline.get_cond(images)
        num_images = int(cond["cond"].shape[0])
        if num_images > 1:
            if multi_image_mode not in {"multidiffusion", "stochastic"}:
                raise ValueError(f"Unsupported multi_image_mode={multi_image_mode!r}")
            cond["neg_cond"] = cond["neg_cond"][:1]
        torch.manual_seed(seed)
        ss_params = ss_sampler_params or {}
        slat_params = slat_sampler_params or {}
        if coords_override is not None:
            coords = self._validate_coords(coords_override)
        else:
            ss_context = _adapter_aware_sampler(
                self.pipeline.sparse_structure_sampler,
                num_images=num_images,
                mode=multi_image_mode,
                context_key="geoss_context",
            ) if num_images > 1 or geoss_context is not None else contextlib.nullcontext()
            with ss_context:
                coords = self.sample_sparse_structure(cond, geoss_context=geoss_context, sampler_params=ss_params)
        slat_context = _adapter_aware_sampler(
            self.pipeline.slat_sampler,
            num_images=num_images,
            mode=multi_image_mode,
            context_key="geovis_slat_context",
        ) if num_images > 1 or geovis_slat_context is not None else contextlib.nullcontext()
        with slat_context:
            slat = self.sample_slat(cond, coords, geovis_slat_context=geovis_slat_context, sampler_params=slat_params)
        decoded = self.pipeline.decode_slat(slat, list(formats))
        decoded["coords"] = coords
        decoded["slat"] = slat
        decoded["conditioning_metadata"] = {
            "num_images": num_images,
            "multi_image_mode": multi_image_mode if num_images > 1 else "single_image",
        }
        return decoded

    def _prepare_conditioning_images(self, images, *, preprocess: bool):
        if isinstance(images, torch.Tensor):
            if images.ndim == 5:
                if images.shape[0] != 1:
                    raise ValueError(f"TRELLIS inference currently expects one object, got tensor {tuple(images.shape)}")
                images = images[0]
            if images.ndim != 4 or images.shape[1] != 3:
                raise ValueError(f"TRELLIS conditioning tensor must be [N,3,H,W], got {tuple(images.shape)}")
            images = images.to(device=self.device, dtype=torch.float32)
            if images.shape[-2:] != (518, 518):
                images = torch.nn.functional.interpolate(
                    images, size=(518, 518), mode="bicubic", align_corners=False, antialias=True
                ).clamp(0.0, 1.0)
            return images
        if not isinstance(images, list) or not images:
            raise TypeError("TRELLIS conditioning images must be a non-empty PIL list or tensor batch.")
        return [self.pipeline.preprocess_image(image) for image in images] if preprocess else images

    def _validate_coords(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] != 4 or coords.numel() == 0:
            raise ValueError(f"Stage-2 sparse coordinates must be non-empty [N,4], got {tuple(coords.shape)}")
        return coords.to(device=self.device, dtype=torch.int32).contiguous()

    def sample_sparse_structure(self, cond: dict, *, geoss_context: Optional[Dict[str, torch.Tensor]], sampler_params: dict) -> torch.Tensor:
        flow_model = self.pipeline.models["sparse_structure_flow_model"]
        reso = flow_model.resolution
        noise = torch.randn(1, flow_model.in_channels, reso, reso, reso, device=self.device)
        params = {**self.pipeline.sparse_structure_sampler_params, **sampler_params}
        sample_kwargs = {**cond, **params, "verbose": True}
        if geoss_context is not None:
            sample_kwargs["geoss_context"] = _to_device(geoss_context, self.device)
        z_s = self.pipeline.sparse_structure_sampler.sample(flow_model, noise, **sample_kwargs).samples
        decoder = self.pipeline.models["sparse_structure_decoder"]
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()
        if coords.numel() == 0:
            raise RuntimeError("TRELLIS sparse structure decoder produced zero active voxels.")
        return coords

    def sample_slat(self, cond: dict, coords: torch.Tensor, *, geovis_slat_context: Optional[Dict[str, torch.Tensor]], sampler_params: dict):
        from trellis.modules import sparse as sp

        flow_model = self.pipeline.models["slat_flow_model"]
        in_channels = _flow_in_channels(flow_model, name="slat_flow_model")
        if isinstance(flow_model, GeoVisTrellisSLATWrapper):
            adapter_channels = int(flow_model.velocity_adapter.slat_dim)
            if adapter_channels != in_channels:
                raise RuntimeError(
                    "Stage-3 adapter/TRELLIS SLat interface mismatch: "
                    f"adapter slat_dim={adapter_channels}, TRELLIS in_channels={in_channels}. "
                    "Use a checkpoint trained for this exact TRELLIS SLat model."
                )
        # The sampler requires only the native TRELLIS latent width here.  The
        # Stage-2 coordinates select the structure; they must not be mistaken
        # for SLat features or silently projected to a different channel count.
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], in_channels, device=self.device),
            coords=coords,
        )
        params = {**self.pipeline.slat_sampler_params, **sampler_params}
        sample_kwargs = {**cond, **params, "verbose": True}
        if geovis_slat_context is not None:
            sample_kwargs["geovis_slat_context"] = _to_device(geovis_slat_context, self.device)
        slat = self.pipeline.slat_sampler.sample(flow_model, noise, **sample_kwargs).samples
        slat_feats = slat.feats if hasattr(slat, "feats") else slat
        if not isinstance(slat_feats, torch.Tensor) or slat_feats.ndim != 2 or slat_feats.shape[-1] != in_channels:
            shape = list(slat_feats.shape) if isinstance(slat_feats, torch.Tensor) else type(slat_feats).__name__
            raise RuntimeError(
                "TRELLIS SLat sampler returned an invalid latent contract: "
                f"expected [N,{in_channels}], got {shape}."
            )
        std = torch.tensor(self.pipeline.slat_normalization["std"], device=slat.device)[None]
        mean = torch.tensor(self.pipeline.slat_normalization["mean"], device=slat.device)[None]
        if std.shape[-1] != in_channels or mean.shape[-1] != in_channels:
            raise RuntimeError(
                "TRELLIS SLat normalization contract does not match its flow model: "
                f"in_channels={in_channels}, std={tuple(std.shape)}, mean={tuple(mean.shape)}."
            )
        return slat * std + mean

    def save_outputs(
        self,
        outputs: Dict[str, object],
        output_dir: Path,
        *,
        export_textured_glb: bool = False,
    ) -> Dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: Dict[str, object] = {}
        if isinstance(outputs.get("conditioning_metadata"), dict):
            saved["conditioning_metadata"] = dict(outputs["conditioning_metadata"])
        gaussian = outputs.get("gaussian")
        if isinstance(gaussian, list) and gaussian:
            path = output_dir / "asset_gaussian.ply"
            gaussian[0].save_ply(str(path))
            saved["gaussian_ply"] = str(path)
            saved["gaussian_statistics"] = gaussian_statistics(gaussian[0])
        mesh = outputs.get("mesh")
        if isinstance(mesh, list) and mesh:
            # MeshExtractResult is not an exportable trimesh object.  Preserve
            # the decoder's internal canonical frame for CD/F-score; TRELLIS'
            # public GLB conversion rotates vertices into y-up exchange space.
            internal_path = output_dir / "asset_mesh_internal.ply"
            write_internal_mesh(mesh[0], internal_path, real_mode=True)
            saved["mesh_internal_ply"] = str(internal_path)
            path = output_dir / "asset_mesh.glb"
            if hasattr(mesh[0], "export"):
                mesh[0].export(str(path))
                saved["mesh_glb"] = str(path)
            elif export_textured_glb and isinstance(gaussian, list) and gaussian:
                from trellis.utils.postprocessing_utils import to_glb

                textured = to_glb(gaussian[0], mesh[0])
                textured.export(str(path))
                saved["mesh_glb"] = str(path)
        # Persist plain tensors, not TRELLIS SparseTensor Python objects. This
        # keeps Stage-2→3 artifacts portable and compatible with weights-only
        # loading during isolated evaluation workers.
        slat = outputs.get("slat")
        slat_feats = slat.feats if hasattr(slat, "feats") else slat
        coords = outputs.get("coords")
        if not isinstance(coords, torch.Tensor) or not isinstance(slat_feats, torch.Tensor):
            raise TypeError("TRELLIS sampler must return tensor coordinates and SLAT features for Stage-2 handoff.")
        torch.save(
            {"coords": coords.detach().cpu().contiguous(), "slat": slat_feats.detach().cpu().contiguous()},
            output_dir / "trellis_latents.pt",
        )
        saved["latents"] = str(output_dir / "trellis_latents.pt")
        return saved

    def _require_models(self, *names: str) -> None:
        missing = [name for name in names if name not in self.pipeline.models or self.pipeline.models[name] is None]
        if missing:
            raise RuntimeError(f"TRELLIS pipeline is missing required real decoder/flow models: {missing}")


def _to_device(context: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in context.items()}


def _flow_in_channels(flow_model: object, *, name: str) -> int:
    """Read and validate TRELLIS sampler metadata with a diagnostic failure."""
    value = getattr(flow_model, "in_channels", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(
            f"TRELLIS {name} must expose a positive integer in_channels for sampling; "
            f"got {value!r} from {type(flow_model).__name__}."
        )
    return value


@contextlib.contextmanager
def _adapter_aware_sampler(sampler, *, num_images: int, mode: str, context_key: str):
    """Apply multi-view conditioning while keeping adapters out of CFG's negative branch.

    TRELLIS' native multi-image patch directly invokes ``FlowEulerSampler`` and
    cannot distinguish adapter context between conditional and unconditional
    predictions.  This unified patch averages (or cycles) image-conditioned
    velocities, injects geometry only into those positive predictions, and
    preserves the sampler's configured CFG interval.
    """
    if mode not in {"multidiffusion", "stochastic"}:
        raise ValueError(f"Unsupported multi-image mode {mode!r}")
    old_inference_model = sampler._inference_model
    cursor = {"step": 0}

    def _patched(
        sampler_self,
        model,
        x_t,
        t,
        cond=None,
        neg_cond=None,
        cfg_strength=None,
        cfg_interval=None,
        **kwargs,
    ):
        adapter_context = kwargs.pop(context_key, None)
        if cond is None:
            return _call_trellis_flow_model(model, x_t, t, cond, context_key, adapter_context, kwargs)
        cfg_active = neg_cond is not None and cfg_strength is not None
        if cfg_active and cfg_interval is not None:
            cfg_active = bool(cfg_interval[0] <= t <= cfg_interval[1])
        positive_kwargs = dict(kwargs)
        if cfg_active and context_key == "geovis_slat_context":
            amplification = 1.0 + float(cfg_strength)
            if amplification <= 0:
                raise ValueError(f"SLAT CFG residual amplification must be positive, got {amplification}.")
            positive_kwargs["geovis_residual_scale"] = 1.0 / amplification
        count = min(num_images, int(cond.shape[0]))
        if mode == "stochastic" and count > 1:
            index = cursor["step"] % count
            cursor["step"] += 1
            indices = [index]
        else:
            indices = list(range(count))
        predictions = [
            _call_trellis_flow_model(
                model, x_t, t, cond[index : index + 1], context_key, adapter_context, positive_kwargs
            )
            for index in indices
        ]
        pred = predictions[0]
        for prediction in predictions[1:]:
            pred = pred + prediction
        if len(predictions) > 1:
            pred = pred / len(predictions)
        if not cfg_active:
            return pred
        neg_pred = _call_trellis_flow_model(model, x_t, t, neg_cond, context_key, None, kwargs)
        return (1.0 + cfg_strength) * pred - cfg_strength * neg_pred

    sampler._inference_model = MethodType(_patched, sampler)
    try:
        yield sampler
    finally:
        sampler._inference_model = old_inference_model


def _call_trellis_flow_model(model, x_t, t, cond, context_key: str, context, kwargs):
    batch_size = int(x_t.shape[0])
    t_tensor = torch.tensor([1000.0 * float(t)] * batch_size, device=x_t.device, dtype=torch.float32)
    if cond is not None and cond.shape[0] == 1 and batch_size > 1:
        cond = cond.repeat(batch_size, *([1] * (cond.ndim - 1)))
    model_kwargs = dict(kwargs)
    if context is not None:
        model_kwargs[context_key] = context
    return model(x_t, t_tensor, cond, **model_kwargs)
