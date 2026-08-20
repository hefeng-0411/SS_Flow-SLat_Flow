from __future__ import annotations

import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoss.slat.models.visibility_evidence_sampler import VisibilityEvidenceSampler


def test_visibility_evidence_sampler_synthetic_depth_rules():
    B, L, N, H, W = 1, 3, 1, 32, 32
    images = torch.rand(B, N, 3, H, W)
    masks = torch.ones(B, N, 1, H, W)
    masks[:, :, :, :, :8] = 0.0
    depths = torch.ones(B, N, 1, H, W) * 2.0
    uv = torch.tensor([[[[16.0, 16.0]], [[16.0, 16.0]], [[4.0, 16.0]]]])
    z_active = torch.tensor([[[[2.0]], [[2.4]], [[2.0]]]])
    in_bounds = torch.ones(B, L, N, 1)
    out = VisibilityEvidenceSampler(evidence_dim=32, feature_dim=4)(torch.zeros(B, L, 3), uv, z_active, images, masks, in_bounds, depths=depths)
    surface_vis = out["visibility"][0, 0, 0, 0]
    occluded_vis = out["visibility"][0, 1, 0, 0]
    mask_out_vis = out["visibility"][0, 2, 0, 0]
    assert surface_vis > occluded_vis
    assert surface_vis > mask_out_vis
    assert out["occlusion_score"][0, 1, 0, 0] > 0.5
    assert out["view_slat_tokens"].shape == (B, L, N, 32)


if __name__ == "__main__":
    test_visibility_evidence_sampler_synthetic_depth_rules()
