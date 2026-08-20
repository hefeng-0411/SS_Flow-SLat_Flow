from __future__ import annotations

import pytest
import torch

from geoss.slat.integration.ss_slat_context import build_ss_slat_context
from geoss.slat.utils.slat_token_mapping import align_token_dim


def test_align_token_dim_refuses_truncation():
    with pytest.raises(ValueError):
        align_token_dim(torch.randn(1, 4, 256), 8)


def test_soft_ss_to_slat_context_keeps_high_dim_tokens():
    B, L, M = 1, 5, 16
    ctx = build_ss_slat_context(
        ss_active_xyz=torch.rand(B, L, 3),
        geoss_output={
            "anchor_xyz": torch.rand(B, M, 3),
            "geo_tokens": torch.rand(B, M, 64),
            "geo_confidence": torch.ones(B, M, 1),
        },
        target_dim=8,
    )
    assert ctx["ss_geo_tokens"].shape == (B, L, 64)
    assert ctx["ss_confidence"].shape == (B, L, 1)
