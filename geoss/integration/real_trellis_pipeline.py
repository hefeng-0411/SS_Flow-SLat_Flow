from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

from geoss.integration.trellis_ss_hook import GeoSSTrellisSSWrapper, ss_grid_to_tokens, tokens_to_ss_grid
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
        from trellis.pipelines import TrellisImageTo3DPipeline

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
        formats: Iterable[str] = ("gaussian", "mesh"),
        seed: int = 42,
        ss_sampler_params: Optional[dict] = None,
        slat_sampler_params: Optional[dict] = None,
    ) -> Dict[str, object]:
        cond = self.pipeline.get_cond(images)
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, geoss_context=geoss_context, sampler_params=ss_sampler_params or {})
        slat = self.sample_slat(cond, coords, geovis_slat_context=geovis_slat_context, sampler_params=slat_sampler_params or {})
        decoded = self.pipeline.decode_slat(slat, list(formats))
        decoded["coords"] = coords
        decoded["slat"] = slat
        return decoded

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
        noise = sp.SparseTensor(feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device), coords=coords)
        params = {**self.pipeline.slat_sampler_params, **sampler_params}
        sample_kwargs = {**cond, **params, "verbose": True}
        if geovis_slat_context is not None:
            sample_kwargs["geovis_slat_context"] = _to_device(geovis_slat_context, self.device)
        slat = self.pipeline.slat_sampler.sample(flow_model, noise, **sample_kwargs).samples
        std = torch.tensor(self.pipeline.slat_normalization["std"], device=slat.device)[None]
        mean = torch.tensor(self.pipeline.slat_normalization["mean"], device=slat.device)[None]
        return slat * std + mean

    def save_outputs(self, outputs: Dict[str, object], output_dir: Path) -> Dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: Dict[str, str] = {}
        gaussian = outputs.get("gaussian")
        if isinstance(gaussian, list) and gaussian:
            path = output_dir / "asset_gaussian.ply"
            gaussian[0].save_ply(str(path))
            saved["gaussian_ply"] = str(path)
            saved["gaussian_statistics"] = gaussian_statistics(gaussian[0])
        mesh = outputs.get("mesh")
        if isinstance(mesh, list) and mesh:
            path = output_dir / "asset_mesh.glb"
            if hasattr(mesh[0], "export"):
                mesh[0].export(str(path))
                saved["mesh_glb"] = str(path)
        torch.save({"coords": outputs.get("coords"), "slat": outputs.get("slat")}, output_dir / "trellis_latents.pt")
        saved["latents"] = str(output_dir / "trellis_latents.pt")
        return saved

    def _require_models(self, *names: str) -> None:
        missing = [name for name in names if name not in self.pipeline.models or self.pipeline.models[name] is None]
        if missing:
            raise RuntimeError(f"TRELLIS pipeline is missing required real decoder/flow models: {missing}")


def _to_device(context: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in context.items()}
