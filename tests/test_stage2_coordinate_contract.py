from __future__ import annotations

import torch

from geoss.integration.trellis_ss_hook import _ss_grid_xyz as inference_ss_grid_xyz
from scripts.train_sparse_ray_ss_velocity import (
    _compute_geoss_context,
    _ss_grid_xyz as training_ss_grid_xyz,
)


def test_stage2_training_and_inference_use_identical_voxel_centers() -> None:
    grid = torch.zeros(2, 8, 16, 16, 16)
    training = training_ss_grid_xyz(grid, torch.float32)
    inference = inference_ss_grid_xyz(grid, torch.float32)
    assert torch.equal(training, inference)
    assert training.shape == (2, 4096, 3)
    assert training[0, 0].tolist() == [-0.9375, -0.9375, -0.9375]
    assert training[0, -1].tolist() == [0.9375, 0.9375, 0.9375]


def test_stage2_context_delegates_to_exact_stage1_context_operator() -> None:
    class DummyVGGT(torch.nn.Module):
        def forward(self, images, use_cache=False):
            assert use_cache is False
            return {"vggt_pointmap": torch.zeros(1, 1, 3, 1, 1)}

    class DummyGeoSS(torch.nn.Module):
        def forward(self, batch):
            assert "ss_latent_tokens" not in batch
            assert "vggt_pointmap" in batch
            return {
                "geo_tokens": torch.ones(1, 2, 4),
                "geo_confidence": torch.full((1, 2, 1), 0.75),
                "anchor_xyz": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]]]),
                "anchor_metadata": torch.zeros(1, 2, 7),
            }

    context = _compute_geoss_context(
        {"images": torch.zeros(1, 1, 3, 2, 2), "ss_latent_tokens": torch.zeros(1, 4, 8)},
        DummyGeoSS(),
        DummyVGGT(),
    )
    assert context["anchor_xyz"].tolist() == [[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]]]
    assert context["geo_confidence"].mean().item() == 0.75
