from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoss.slat.integration.trellis_slat_hook import GeoVisTrellisSLATWrapper, make_cfg_geovis_context
from geoss.slat.models.slat_velocity_adapter import SLATVelocityAdapter


class DenseFlow(nn.Module):
    def forward(self, x, t, cond=None, **kwargs):
        return x * 0.25


class FakeSparseTensor:
    def __init__(self, feats, coords, layout):
        self.feats = feats
        self.coords = coords
        self.layout = layout
        self.shape = (len(layout),)

    def replace(self, feats):
        return FakeSparseTensor(feats, self.coords, self.layout)


class SparseFlow(nn.Module):
    def forward(self, x, t, cond=None, **kwargs):
        return x.replace(x.feats * 0.25)


def test_trellis_slat_hook_dense_identity_and_debug():
    B, L, C = 1, 16, 8
    x = torch.randn(B, L, C)
    context = {
        "slat_cond_tokens": torch.randn(B, L, C),
        "slat_confidence": torch.ones(B, L, 1),
        "ss_confidence": torch.ones(B, L, 1),
    }
    wrapper = GeoVisTrellisSLATWrapper(DenseFlow(), SLATVelocityAdapter(slat_dim=C, cond_dim=C, hidden_dim=32), use_geovis_slat=True)
    identity = wrapper(x, torch.tensor([500.0]), None, use_geovis_slat=False)
    assert torch.equal(identity, x * 0.25)
    controlled = wrapper(x, torch.tensor([500.0]), None, geovis_slat_context=context)
    assert controlled.shape == x.shape
    assert wrapper.last_debug["enabled"] is True
    assert make_cfg_geovis_context(context, branch="uncond", apply_to_uncond=False) is None
    assert make_cfg_geovis_context(context, branch="cond", apply_to_uncond=False) is context


def test_trellis_slat_hook_sparse_mask_keeps_canonical_rank():
    token_count, channels = 7, 8
    feats = torch.randn(token_count, channels)
    coords = torch.cat(
        [torch.zeros(token_count, 1, dtype=torch.long), torch.randint(0, 64, (token_count, 3))],
        dim=-1,
    )
    sparse = FakeSparseTensor(feats, coords, [slice(0, token_count)])
    context = {
        "slat_cond_tokens": torch.randn(1, token_count, channels),
        "slat_confidence": torch.ones(1, token_count, 1),
        "ss_confidence": torch.ones(1, token_count, 1),
        "ss_active_indices": coords[:, 1:].unsqueeze(0),
    }
    wrapper = GeoVisTrellisSLATWrapper(
        SparseFlow(),
        SLATVelocityAdapter(slat_dim=channels, cond_dim=channels, hidden_dim=32),
        use_geovis_slat=True,
    )
    controlled = wrapper(
        sparse,
        torch.tensor([500.0]),
        None,
        geovis_slat_context=context,
    )
    assert isinstance(controlled, FakeSparseTensor)
    assert controlled.feats.shape == feats.shape
    assert wrapper.last_debug["valid_token_ratio"] == 1


if __name__ == "__main__":
    test_trellis_slat_hook_dense_identity_and_debug()
