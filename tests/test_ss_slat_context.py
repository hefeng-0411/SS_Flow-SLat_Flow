from __future__ import annotations

import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoss.slat.integration.ss_slat_context import build_ss_slat_context


def test_ss_slat_context_nearest_anchor_mapping():
    B, L, M, C = 1, 12, 24, 8
    indices = torch.randint(20, 44, (B, L, 3))
    anchor_xyz = torch.rand(B, M, 3) * 2 - 1
    geoss_output = {
        "anchor_xyz": anchor_xyz,
        "geo_tokens": torch.randn(B, M, C),
        "geo_confidence": torch.rand(B, M, 1),
    }
    out = build_ss_slat_context(ss_active_indices=indices, geoss_output=geoss_output, resolution=64, target_dim=C)
    assert out["ss_active_indices"].shape == (B, L, 3)
    assert out["ss_active_xyz"].shape == (B, L, 3)
    assert out["ss_geo_tokens"].shape == (B, L, C)
    assert out["ss_confidence"].shape == (B, L, 1)
    assert out["ss_to_slat_token_map"].shape == (B, L)


if __name__ == "__main__":
    test_ss_slat_context_nearest_anchor_mapping()
