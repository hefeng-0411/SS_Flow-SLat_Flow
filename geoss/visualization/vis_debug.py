from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch

from geoss.utils.visualization import write_point_cloud_ply


def export_debug_bundle(
    output_dir: str | Path,
    *,
    aligned_pointmap: Optional[torch.Tensor] = None,
    dynamic_anchors: Optional[torch.Tensor] = None,
    anchor_confidence: Optional[torch.Tensor] = None,
    ss_active_voxels: Optional[torch.Tensor] = None,
    slat_active_voxels: Optional[torch.Tensor] = None,
    cameras: Optional[Dict[str, torch.Tensor]] = None,
    metrics: Optional[Dict] = None,
) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    if aligned_pointmap is not None:
        pts = aligned_pointmap[0].permute(0, 2, 3, 1).reshape(-1, 3)
        write_point_cloud_ply(out / "aligned_vggt_pointmap.ply", pts)
        written["aligned_vggt_pointmap"] = str(out / "aligned_vggt_pointmap.ply")
    if dynamic_anchors is not None:
        write_point_cloud_ply(out / "dynamic_anchors.ply", dynamic_anchors[0], anchor_confidence[0] if anchor_confidence is not None else None)
        written["dynamic_anchors"] = str(out / "dynamic_anchors.ply")
    if ss_active_voxels is not None:
        write_point_cloud_ply(out / "ss_active_voxels.ply", ss_active_voxels[0])
        written["ss_active_voxels"] = str(out / "ss_active_voxels.ply")
    if slat_active_voxels is not None:
        write_point_cloud_ply(out / "slat_active_voxels.ply", slat_active_voxels[0])
        written["slat_active_voxels"] = str(out / "slat_active_voxels.ply")
    if cameras is not None:
        cam_json = {k: v.detach().cpu().tolist() for k, v in cameras.items() if isinstance(v, torch.Tensor)}
        (out / "camera_frustums.json").write_text(json.dumps(cam_json, indent=2), encoding="utf-8")
        written["camera_frustums"] = str(out / "camera_frustums.json")
    if metrics is not None:
        (out / "metrics_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        written["metrics_summary"] = str(out / "metrics_summary.json")
    return written
