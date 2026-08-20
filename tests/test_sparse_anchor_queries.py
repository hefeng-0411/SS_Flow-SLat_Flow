import torch

from geoss.models.sparse_anchor_queries import SparseAnchorQueries


def test_sparse_anchor_queries_shapes_and_range():
    for num_anchors in (2048, 4096, 8192):
        module = SparseAnchorQueries(num_anchors=num_anchors, feature_dim=32)
        xyz, feat = module(batch_size=2, device="cpu")
        assert xyz.shape == (2, num_anchors, 3)
        assert feat.shape == (2, num_anchors, 32)
        assert xyz.min() >= -1.0
        assert xyz.max() <= 1.0
        assert torch.isfinite(xyz).all()
        assert torch.isfinite(feat).all()
