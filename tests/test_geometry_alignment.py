from __future__ import annotations

import torch

from geoss.geometry.alignment import GeometryAlignment, estimate_sim3_umeyama


def test_umeyama_recovers_simple_sim3():
    src = torch.randn(1, 64, 3)
    dst = src * 2.0 + torch.tensor([[[0.5, -0.25, 1.0]]])
    sim = estimate_sim3_umeyama(src, dst)
    aligned = sim.scale[:, None, None] * torch.matmul(src, sim.rotation.transpose(1, 2)) + sim.translation[:, None]
    assert torch.allclose(aligned, dst, atol=1e-4)


def test_geometry_alignment_outputs_canonical_tensors():
    B, N, H, W = 1, 2, 8, 8
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    pointmap = torch.stack([xx, yy, torch.ones_like(xx)], dim=0).view(1, 1, 3, H, W).expand(B, N, 3, H, W)
    depth = torch.ones(B, N, 1, H, W)
    masks = torch.ones(B, N, 1, H, W)
    K = torch.eye(3).view(1, 1, 3, 3).expand(B, N, 3, 3)
    c2w = torch.eye(4).view(1, 1, 4, 4).expand(B, N, 4, 4)
    out = GeometryAlignment()(vggt_depth=depth, vggt_pointmap=pointmap, K=K, c2w=c2w, w2c=c2w, masks=masks)
    assert out["aligned_pointmap"].shape == pointmap.shape
    assert out["aligned_depth"].shape == depth.shape
    assert out["alignment_confidence"].min() >= 0


def test_geometry_alignment_uses_camera_center_sim3():
    B, N, H, W = 1, 4, 3, 3
    target_centers = torch.tensor(
        [[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [0.0, -2.0, 0.0]]]
    )
    source_centers = (target_centers - torch.tensor([[[0.5, -0.25, 1.0]]])) / 2.0
    target_c2w = torch.eye(4).view(1, 1, 4, 4).repeat(B, N, 1, 1)
    source_c2w = target_c2w.clone()
    target_c2w[..., :3, 3] = target_centers
    source_c2w[..., :3, 3] = source_centers
    source_points = torch.randn(B, N, 3, H, W) * 0.1
    K = torch.eye(3).view(1, 1, 3, 3).expand(B, N, 3, 3)
    out = GeometryAlignment()(
        vggt_depth=torch.ones(B, N, 1, H, W),
        vggt_pointmap=source_points,
        K=K,
        c2w=target_c2w,
        w2c=torch.linalg.inv(target_c2w),
        masks=torch.ones(B, N, 1, H, W),
        vggt_camera={"c2w": source_c2w, "w2c": torch.linalg.inv(source_c2w), "K": K},
    )
    assert bool(out["camera_alignment_valid"].item())
    assert torch.allclose(out["sim3_scale"], torch.tensor([2.0]), atol=1e-4)
    assert out["alignment_residual"].item() < 1e-4
