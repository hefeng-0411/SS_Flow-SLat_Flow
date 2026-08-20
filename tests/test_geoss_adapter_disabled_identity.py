import torch
import torch.nn as nn

from geoss.integration.trellis_ss_hook import GeoSSTrellisSSWrapper


class TinyIdentityFlow(nn.Module):
    resolution = 4
    in_channels = 8
    out_channels = 8

    def forward(self, x, t, cond, **kwargs):
        return torch.sin(x)


def test_disabled_identity_error_under_threshold():
    base = TinyIdentityFlow()
    wrapper = GeoSSTrellisSSWrapper(base)
    x = torch.randn(1, 8, 4, 4, 4)
    cond = torch.randn(1, 2, 8)
    expected = base(x, torch.tensor([10.0]), cond)
    actual = wrapper(x, torch.tensor([10.0]), cond, geoss_context=None, use_geoss_adapter=False)
    assert (actual - expected).abs().max().item() < 1e-6
