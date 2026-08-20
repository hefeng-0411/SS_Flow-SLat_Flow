from __future__ import annotations

import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoss.slat.models.geovis_slat_adapter import GeoVisSLATAdapter
from geoss.utils.coordinates import c2w_to_w2c


def test_geovis_slat_adapter_forward_shapes():
    B, N, L, C, H, W = 1, 3, 64, 8, 64, 64
    K = torch.eye(3).view(1, 1, 3, 3).repeat(B, N, 1, 1)
    K[..., 0, 0] = W
    K[..., 1, 1] = H
    K[..., 0, 2] = W / 2
    K[..., 1, 2] = H / 2
    c2w = torch.eye(4).view(1, 1, 4, 4).repeat(B, N, 1, 1)
    c2w[..., 2, 3] = -2.0
    batch = {
        "images": torch.rand(B, N, 3, H, W),
        "masks": torch.ones(B, N, 1, H, W),
        "depths": torch.ones(B, N, 1, H, W) * 2.0,
        "K": K,
        "c2w": c2w,
        "w2c": c2w_to_w2c(c2w),
        "ss_active_indices": torch.randint(24, 40, (B, L, 3)),
        "ss_confidence": torch.rand(B, L, 1),
        "slat_latent_tokens": torch.randn(B, L, C),
        "v_slat_base": torch.randn(B, L, C) * 0.01,
        "timestep": torch.tensor([0.5]),
        "vggt_features": torch.rand(B, N, 16, H // 4, W // 4),
    }
    out = GeoVisSLATAdapter(slat_dim=C, evidence_dim=32, hidden_dim=32, feature_dim=16, num_heads=4)(batch)
    assert out["slat_cond_tokens"].shape == (B, L, C)
    assert out["v_slat_geo"].shape == (B, L, C)
    assert out["view_weights"].shape == (B, L, N, 1)
    assert out["slat_confidence"].std(unbiased=False) >= 0


if __name__ == "__main__":
    test_geovis_slat_adapter_forward_shapes()
