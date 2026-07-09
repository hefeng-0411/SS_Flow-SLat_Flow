from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .optional_deps import require_dependency


def mode_summary(
    *,
    dry_run: bool,
    use_real_vggt: bool,
    use_real_trellis: bool,
    use_decoder: bool,
    render_eval: bool,
    real_data: bool,
) -> Dict[str, Any]:
    return {
        "vggt_mode": "real" if use_real_vggt else "mock",
        "trellis_mode": "real" if use_real_trellis else "mock",
        "data_mode": "real" if real_data else "synthetic",
        "decoder_enabled": bool(use_decoder),
        "render_eval_enabled": bool(render_eval),
        "dry_run": bool(dry_run),
        "official_metrics": bool((not dry_run) and use_real_vggt and use_real_trellis and real_data),
    }


def validate_real_mode(
    *,
    cfg: Mapping[str, Any] | None,
    args: Any,
    mode: str,
    required: Iterable[str],
) -> Dict[str, Any]:
    """Fail fast for real train/eval/infer instead of silently falling back.

    `required` names are logical resources: vggt, trellis, dataset, decoder,
    prediction, render, gt. The function accepts either CLI attributes or config
    keys so existing scripts can adopt it without a large argument rewrite.
    """
    cfg = cfg or {}
    validate_config_switches(cfg)
    dry_run = bool(getattr(args, "dry_run", False) or cfg.get("dry_run", False))
    allow_mock = bool(cfg.get("allow_mock", dry_run))
    allow_synthetic = bool(cfg.get("allow_synthetic", dry_run))
    requested_real = bool(getattr(args, mode, False) or cfg.get(mode, False) or not dry_run)
    if dry_run:
        return mode_summary(
            dry_run=True,
            use_real_vggt=False,
            use_real_trellis=False,
            use_decoder=bool(cfg.get("use_decoder", False)),
            render_eval=bool(cfg.get("render_eval", False)),
            real_data=False,
        )
    validate_dependency_switches(cfg, mode=mode)
    if requested_real and (allow_mock or allow_synthetic):
        raise ValueError(
            f"{mode}=true requires allow_mock=false and allow_synthetic=false. "
            "Use --dry_run true for mock/synthetic smoke tests."
        )

    missing: list[str] = []
    for item in required:
        if item == "vggt" and not _has_any(args, cfg, ("vggt_checkpoint", "vggt_pretrained")):
            missing.append("VGGT checkpoint or pretrained name")
        elif item == "trellis" and not _has_any(args, cfg, ("trellis_model_path", "trellis_checkpoint", "trellis_pipeline")):
            missing.append("TRELLIS checkpoint/pipeline")
        elif item == "dataset" and not _has_dataset(args, cfg):
            missing.append("real dataset/input path")
        elif item == "decoder" and not _has_any(args, cfg, ("trellis_model_path", "trellis_pipeline", "decoder_path")):
            missing.append("TRELLIS decoder/pipeline")
        elif item == "prediction" and not _path_exists(_get(args, cfg, "prediction")):
            missing.append("real prediction file")
        elif item == "render" and not _path_exists(_get(args, cfg, "render_dir")) and not _path_exists(_get(args, cfg, "input_dir")):
            missing.append("real rendered result directory")
        elif item == "gt" and not _path_exists(_get(args, cfg, "gt_occ")) and not _has_dataset(args, cfg):
            missing.append("real GT or real input images")
    if missing:
        raise FileNotFoundError(f"{mode}=true is missing: {', '.join(missing)}")

    return mode_summary(
        dry_run=False,
        use_real_vggt=_has_any(args, cfg, ("vggt_checkpoint", "vggt_pretrained")),
        use_real_trellis=_has_any(args, cfg, ("trellis_model_path", "trellis_checkpoint", "trellis_pipeline")),
        use_decoder=_has_any(args, cfg, ("trellis_model_path", "trellis_pipeline", "decoder_path")),
        render_eval=bool(cfg.get("render_eval", getattr(args, "render_eval", False))),
        real_data=True,
    )


def assert_no_official_mock(metrics: Mapping[str, Any]) -> None:
    if metrics.get("official_metrics") and (
        metrics.get("vggt_mode") != "real" or metrics.get("trellis_mode") != "real" or metrics.get("data_mode") != "real"
    ):
        raise ValueError("Mock/synthetic outputs cannot be marked as official metrics.")


def validate_config_switches(cfg: Mapping[str, Any]) -> None:
    if cfg.get("use_real_vggt") and cfg.get("allow_mock"):
        raise ValueError("use_real_vggt=true is incompatible with allow_mock=true.")
    if cfg.get("use_real_trellis") and cfg.get("allow_mock"):
        raise ValueError("use_real_trellis=true is incompatible with allow_mock=true.")
    if cfg.get("render_eval") and not cfg.get("use_decoder", False):
        raise ValueError("render_eval=true requires use_decoder=true.")
    if cfg.get("use_render_loss") and not cfg.get("use_decoder", False):
        raise ValueError("use_render_loss=true requires use_decoder=true.")


def validate_dependency_switches(cfg: Mapping[str, Any], *, mode: str) -> None:
    deps = _section(cfg, "dependencies")
    rendering = _section(cfg, "rendering")
    geometry = _section(cfg, "geometry")
    evaluation = _section(cfg, "evaluation")
    real = bool(cfg.get(mode, False) or cfg.get("real_train", False) or cfg.get("real_eval", False) or cfg.get("real_infer", False))
    if deps.get("require_gsplat_for_real_eval") and (mode in {"real_eval", "real_infer"} or evaluation.get("use_gsplat_render_metrics")):
        require_dependency("gsplat", real_mode=real, feature="real 3DGS render metrics")
    if deps.get("require_kornia_for_projection") or geometry.get("use_kornia_projection"):
        require_dependency("kornia", real_mode=real, feature="differentiable camera projection")
    if deps.get("require_pycolmap_for_pose_check") or geometry.get("use_pycolmap_pose_check"):
        require_dependency("pycolmap", real_mode=real, feature="pose/SfM consistency check")
    if geometry.get("use_open3d_icp_refine") or _section(cfg, "anchors").get("dynamic_anchor_backend") == "open3d":
        require_dependency("open3d", real_mode=real, feature="ICP/dynamic anchor point cloud ops")
    if geometry.get("use_point_cloud_utils_metrics") or evaluation.get("use_geometry_metrics"):
        require_dependency("point_cloud_utils", real_mode=real and bool(evaluation.get("use_geometry_metrics")), feature="geometry metrics")
    renderer = rendering.get("renderer")
    renderer_active = bool(rendering.get("enable_render_loss") or evaluation.get("use_gsplat_render_metrics") or cfg.get("render_eval"))
    if renderer == "gsplat" and renderer_active:
        require_dependency("gsplat", real_mode=real, feature="configured renderer=gsplat")
    if (renderer == "pytorch3d" and renderer_active) or deps.get("require_pytorch3d_for_mesh_render"):
        require_dependency("pytorch3d", real_mode=real, feature="configured mesh renderer")
    if (renderer == "nvdiffrast" and renderer_active) or deps.get("require_nvdiffrast_for_advanced_render"):
        require_dependency("nvdiffrast", real_mode=real, feature="configured advanced renderer")
    if deps.get("require_kaolin_for_voxel_ops"):
        require_dependency("kaolin", real_mode=real, feature="voxel/SDF ops")


def _section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = cfg.get(name, {})
    return value if isinstance(value, Mapping) else {}


def _get(args: Any, cfg: Mapping[str, Any], name: str) -> Any:
    return getattr(args, name, None) or cfg.get(name) or _get_nested_alias(cfg, name)


def _has_any(args: Any, cfg: Mapping[str, Any], names: Iterable[str]) -> bool:
    for name in names:
        value = _get(args, cfg, name)
        if isinstance(value, bool):
            if value:
                return True
        elif value:
            return True
    return False


def _has_dataset(args: Any, cfg: Mapping[str, Any]) -> bool:
    return _has_any(
        args,
        cfg,
        (
            "meshfleet_root",
            "srn_root",
            "objaverse_rendered_root",
            "input",
            "input_dir",
            "dataset_root",
            "render_dir",
        ),
    )


def _get_nested_alias(cfg: Mapping[str, Any], name: str) -> Any:
    dataset = _section(cfg, "dataset")
    aliases = {
        "meshfleet_root": (dataset, "root"),
        "dataset_root": (dataset, "root"),
        "input_dir": (dataset, "root"),
        "trellis_model_path": (_section(cfg, "trellis"), "model_path"),
        "trellis_checkpoint": (_section(cfg, "trellis"), "checkpoint"),
        "trellis_pipeline": (_section(cfg, "trellis"), "pipeline"),
        "vggt_checkpoint": (_section(cfg, "vggt"), "checkpoint"),
        "vggt_pretrained": (_section(cfg, "vggt"), "pretrained"),
    }
    target = aliases.get(name)
    if target is None:
        return None
    section, key = target
    return section.get(key)


def _path_exists(value: Any) -> bool:
    if not value:
        return False
    return Path(str(value)).exists()
