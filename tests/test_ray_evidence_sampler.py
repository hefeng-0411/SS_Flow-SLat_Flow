import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoss.datasets.vehicle_multiview_dataset import make_dry_run_batch
from geoss.models.ray_evidence_sampler import RayEvidenceSampler


def test_ray_evidence_sampler_shapes():
    batch = make_dry_run_batch(batch_size=1, num_views=2, image_size=32, latent_tokens=8)
    anchor_xyz = torch.rand(1, 64, 3) * 0.5
    sampler = RayEvidenceSampler(evidence_dim=16)
    out = sampler(anchor_xyz, batch["K"], batch["c2w"], batch["w2c"], batch["masks"], depths=batch["depths"])
    assert out["view_tokens"].shape == (1, 64, 2, 16)
    assert out["occ_score"].shape == (1, 64, 2, 1)
    assert out["ray_valid"].shape == (1, 64, 2, 1)


def test_ray_evidence_sampler_depth_missing_does_not_crash():
    B, N, H, W = 1, 1, 32, 32
    K = torch.tensor([[[[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]]]])
    c2w = torch.eye(4).view(1, 1, 4, 4)
    w2c = torch.eye(4).view(1, 1, 4, 4)
    masks = torch.zeros(B, N, 1, H, W)
    anchors = torch.tensor([[[0.0, 0.0, 1.0]]])
    sampler = RayEvidenceSampler(evidence_dim=8)
    out = sampler(anchors, K, c2w, w2c, masks, depths=None)
    assert torch.isfinite(out["view_tokens"]).all()
    assert out["free_score"][0, 0, 0, 0] > out["occ_score"][0, 0, 0, 0]


if __name__ == "__main__":
    test_ray_evidence_sampler_shapes()
    test_ray_evidence_sampler_depth_missing_does_not_crash()
