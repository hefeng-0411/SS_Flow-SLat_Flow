import torch
import torch.nn as nn

from geoss.integration.trellis_ss_hook import GeoSSTrellisSSWrapper
from geoss.models.ss_velocity_adapter import SSVelocityAdapter


class TinyFlow(nn.Module):
    resolution = 4
    in_channels = 8
    out_channels = 8

    def forward(self, x, t, cond, **kwargs):
        return x * 0.125


def test_trellis_ss_hook_debug_enabled():
    base = TinyFlow()
    adapter = SSVelocityAdapter(latent_dim=8, geo_dim=16, hidden_dim=32)
    wrapper = GeoSSTrellisSSWrapper(base, adapter)
    x = torch.randn(1, 8, 4, 4, 4)
    cond = torch.randn(1, 4, 16)
    ctx = {"geo_tokens": torch.randn(1, 32, 16), "geo_confidence": torch.rand(1, 32, 1)}
    y = wrapper(x, torch.tensor([100.0]), cond, geoss_context=ctx)
    assert y.shape == x.shape
    assert wrapper.last_debug["enabled"]
    assert wrapper.last_debug["delta_v_geo"].numel() > 0
    assert wrapper.last_debug["alpha_t"].numel() > 0
    assert wrapper.last_debug["token_confidence"].numel() > 0
    assert wrapper.last_debug["clipping_ratio"].numel() > 0
    assert wrapper.last_debug["velocity_norm"].numel() > 0
    assert len(wrapper.debug_trajectory) == 1
