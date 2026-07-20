from __future__ import annotations

import torch

from geoss.integration.real_trellis_pipeline import _adapter_aware_sampler


class DummySampler:
    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        return model(x_t, torch.tensor([1000.0 * t]), cond, **kwargs)


class ContextModel:
    def __init__(self):
        self.calls = []

    def __call__(self, x, timestep, cond, geoss_context=None):
        self.calls.append({"cond": float(cond.mean()), "has_context": geoss_context is not None})
        value = cond.mean() + (10.0 if geoss_context is not None else 0.0)
        return torch.ones_like(x) * value


def test_multidiffusion_averages_views_and_excludes_context_from_cfg_negative():
    sampler = DummySampler()
    model = ContextModel()
    x = torch.zeros(1, 2)
    cond = torch.tensor([[1.0], [3.0]])
    neg_cond = torch.tensor([[0.0]])
    with _adapter_aware_sampler(
        sampler,
        num_images=2,
        mode="multidiffusion",
        context_key="geoss_context",
    ):
        pred = sampler._inference_model(
            model,
            x,
            0.5,
            cond=cond,
            neg_cond=neg_cond,
            cfg_strength=2.0,
            cfg_interval=(0.0, 1.0),
            geoss_context={"geometry": torch.ones(1)},
        )
    # Positive mean=(11+13)/2=12; negative=0; TRELLIS CFG=(1+2)*12-2*0.
    assert torch.allclose(pred, torch.full_like(x, 36.0))
    assert [call["has_context"] for call in model.calls] == [True, True, False]
