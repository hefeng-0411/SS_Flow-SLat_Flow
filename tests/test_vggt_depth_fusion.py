import torch

from geoss.reconstruction.vggt_depth_fusion import (
    VGGTDepthFusionConfig,
    aligned_pointmap_depth_evidence,
    carve_visual_hull_with_depth,
)


def _camera_pair(image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    K = torch.tensor(
        [
            [40.0, 0.0, (image_size - 1) / 2],
            [0.0, 40.0, (image_size - 1) / 2],
            [0.0, 0.0, 1.0],
        ]
    ).repeat(2, 1, 1)
    w2c = torch.eye(4).repeat(2, 1, 1)
    w2c[:, 2, 3] = 2.0
    return K, w2c


def test_reprojection_filter_accepts_points_on_known_camera_rays():
    size = 16
    K, w2c = _camera_pair(size)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    camera_depth = torch.full((size, size), 2.0)
    camera_x = (xx - K[0, 0, 2]) * camera_depth / K[0, 0, 0]
    camera_y = (yy - K[0, 1, 2]) * camera_depth / K[0, 1, 1]
    # w2c translates canonical world z by +2, so camera z=2 is world z=0.
    pointmap = torch.stack(
        [camera_x, camera_y, camera_depth - 2.0],
        dim=0,
    ).repeat(2, 1, 1, 1)
    masks = torch.ones(2, 1, size, size)
    depth, confidence, report = aligned_pointmap_depth_evidence(
        pointmap,
        K,
        w2c,
        masks,
        torch.ones(2, size, size),
        reprojection_sigma_pixels=2.0,
        max_reprojection_error_pixels=4.0,
    )
    assert torch.allclose(depth, torch.full_like(depth, 2.0), atol=1e-5)
    assert float(confidence.min()) > 0.99
    assert max(report["per_view_median_reprojection_error_pixels"]) < 1e-4


def test_depth_fusion_carves_observed_front_free_space_only():
    size = 16
    K, w2c = _camera_pair(size)
    hull = torch.ones(24, 24, 24, dtype=torch.bool)
    depth = torch.full((2, 1, size, size), 2.0)
    confidence = torch.ones_like(depth)
    masks = torch.ones_like(depth)
    fused, report = carve_visual_hull_with_depth(
        hull,
        depth,
        confidence,
        masks,
        K,
        w2c,
        VGGTDepthFusionConfig(
            confidence_threshold=0.1,
            free_space_margin=0.0,
            min_depth_views=2,
            min_not_free_fraction=1.0,
            chunk_size=4096,
        ),
    )
    # Restrict the assertion to voxels inside the cameras' finite field of
    # view; unsupported hull corners must be preserved by design.
    assert not bool(fused[8:16, 8:16, 2].any())
    assert bool(fused[8:16, 8:16, -2].all())
    assert report["carved_voxels"] > 0
    assert 0.1 < report["retained_fraction"] < 0.9


def test_depth_fusion_preserves_hull_without_qualified_evidence():
    size = 16
    K, w2c = _camera_pair(size)
    hull = torch.zeros(20, 20, 20, dtype=torch.bool)
    hull[4:16, 4:16, 4:16] = True
    fused, report = carve_visual_hull_with_depth(
        hull,
        torch.full((2, 1, size, size), 2.0),
        torch.zeros(2, 1, size, size),
        torch.ones(2, 1, size, size),
        K,
        w2c,
        VGGTDepthFusionConfig(confidence_threshold=0.1, chunk_size=2048),
    )
    assert torch.equal(fused, hull)
    assert report["carved_voxels"] == 0
