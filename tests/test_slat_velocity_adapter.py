from __future__ import annotations

import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoss.slat.models.slat_velocity_adapter import SLATVelocityAdapter


def test_slat_velocity_adapter_identity_and_clipping():
    B, L, C = 2, 32, 8
    x = torch.randn(B, L, C)
    cond = torch.randn(B, L, C)
    slat_conf = torch.ones(B, L, 1)
    ss_conf = torch.ones(B, L, 1)
    v_base = torch.randn(B, L, C)
    adapter = SLATVelocityAdapter(slat_dim=C, cond_dim=C, hidden_dim=32, trust_region=0.05, enabled=True)
    disabled = adapter(x, cond, slat_conf, ss_conf, torch.tensor([0.5, 0.5]), v_base, use_geovis_slat=False)
    assert torch.equal(disabled["v_slat_geo"], v_base)
    out = adapter(x, cond, slat_conf, ss_conf, torch.tensor([1000.0, 1000.0]), v_base)
    assert out["delta_v_slat"].abs().max() <= 0.050001
    assert out["v_slat_geo"].shape == v_base.shape


if __name__ == "__main__":
    test_slat_velocity_adapter_identity_and_clipping()
