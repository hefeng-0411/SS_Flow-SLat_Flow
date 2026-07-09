import torch

from geoss.utils.projection import projection_roundtrip_check, project_points, unproject_depth


def test_projection_roundtrip_sparse_points():
    points = torch.tensor([[[0.0, 0.0, 2.0], [0.1, -0.2, 3.0]]])
    K = torch.tensor([[[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]]])
    c2w = torch.eye(4).view(1, 4, 4)
    result = projection_roundtrip_check(points, K, c2w, tolerance=1e-5)
    assert result["ok"]


def test_unproject_depth_shape():
    depth = torch.ones(1, 1, 4, 4)
    K = torch.tensor([[[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]])
    c2w = torch.eye(4).view(1, 4, 4)
    pts = unproject_depth(depth, K, c2w)
    assert pts.shape == (1, 4, 4, 3)
