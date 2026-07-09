from __future__ import annotations

import torch

from geoss.models.sparse_anchor_queries import SparseAnchorQueries
from geoss.models.ss_velocity_adapter import SSVelocityAdapter


def test_dynamic_anchor_metadata_and_sources():
    B, N, H, W = 1, 2, 8, 8
    pointmap = torch.rand(B, N, 3, H, W) * 2 - 1
    masks = torch.ones(B, N, 1, H, W)
    anchors = SparseAnchorQueries(num_anchors=2048, feature_dim=32).forward_dynamic(B, aligned_pointmap=pointmap, masks=masks)
    assert anchors["anchor_xyz"].shape == (B, 2048, 3)
    assert anchors["anchor_metadata"].shape == (B, 2048, 7)
    assert (anchors["source_type"] > 0).any()


def test_ss_local_attention_uses_anchor_geometry():
    B, L, M, C, G = 1, 16, 64, 8, 32
    adapter = SSVelocityAdapter(latent_dim=C, geo_dim=G, hidden_dim=32, num_heads=4, local_attention=True, knn_k=8)
    out = adapter(
        torch.randn(B, L, C),
        torch.randn(B, M, G),
        torch.ones(B, M, 1),
        torch.rand(B),
        torch.zeros(B, L, C),
        voxel_xyz=torch.rand(B, L, 3) * 2 - 1,
        anchor_xyz=torch.rand(B, M, 3) * 2 - 1,
        anchor_metadata=torch.rand(B, M, 7),
    )
    assert out["delta_v_geo"].shape == (B, L, C)
    assert bool(out["debug"]["local_attention"])
