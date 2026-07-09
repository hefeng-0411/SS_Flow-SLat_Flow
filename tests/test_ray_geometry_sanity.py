import torch

from geoss.models.ray_evidence_sampler import RayEvidenceSampler


def test_ray_geometry_occ_free_invalid_rules():
    B, N, H, W = 1, 1, 32, 32
    K = torch.tensor([[[[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]]]])
    c2w = torch.eye(4).view(1, 1, 4, 4)
    w2c = torch.eye(4).view(1, 1, 4, 4)
    masks = torch.zeros(B, N, 1, H, W)
    masks[:, :, :, 12:20, 12:20] = 1
    depths = torch.ones(B, N, 1, H, W) * 2.0
    anchors = torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [0.6, 0.0, 2.0], [10.0, 0.0, 2.0]]])
    sampler = RayEvidenceSampler(evidence_dim=8, depth_near_threshold=0.05, depth_free_margin=0.05)
    out = sampler(anchors, K, c2w, w2c, masks, depths=depths)
    assert out["occ_score"][0, 0, 0, 0] > out["free_score"][0, 0, 0, 0]
    assert out["free_score"][0, 1, 0, 0] > out["occ_score"][0, 1, 0, 0]
    assert out["free_score"][0, 2, 0, 0] > out["occ_score"][0, 2, 0, 0]
    assert out["ray_valid"][0, 3, 0, 0] == 0
