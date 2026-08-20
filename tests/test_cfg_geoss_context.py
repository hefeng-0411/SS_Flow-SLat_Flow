import torch
import torch.nn as nn

from geoss.integration.trellis_ss_hook import GeoSSSamplerWrapper, GeoSSTrellisSSWrapper, split_geoss_context_for_cfg


class TinySampler:
    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        return model(x_t, torch.tensor([1000 * t]), cond, **kwargs)


class BranchAwareFlow(nn.Module):
    resolution = 2
    in_channels = 1
    out_channels = 1

    def forward(self, x, t, cond, geoss_context=None, **kwargs):
        return x + (0.5 if geoss_context is not None else 0.0)


def test_cfg_context_default_conditional_only():
    ctx = {"geo_tokens": torch.ones(1, 4, 1), "geo_confidence": torch.ones(1, 4, 1)}
    cond_ctx, uncond_ctx = split_geoss_context_for_cfg(ctx, geoss_apply_to_uncond=False)
    assert cond_ctx is ctx
    assert uncond_ctx is None

    sampler = TinySampler()
    model = BranchAwareFlow()
    x = torch.zeros(1, 1, 2, 2, 2)
    with GeoSSSamplerWrapper(sampler, geoss_apply_to_uncond=False) as patched:
        y = patched._inference_model(model, x, 0.5, cond=torch.ones(1, 1), neg_cond=torch.zeros(1, 1), cfg_strength=1.0, geoss_context=ctx)
    assert torch.allclose(y, torch.ones_like(x))
    assert patched.last_geoss_cfg_debug["cond_geoss"] is True
    assert patched.last_geoss_cfg_debug["uncond_geoss"] is False
