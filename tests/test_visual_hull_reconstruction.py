from __future__ import annotations

import pytest
import torch

from geoss.reconstruction.visual_hull import VisualHullConfig, carve_visual_hull


def _centered_cameras(views: int, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    K = torch.eye(3).repeat(views, 1, 1)
    K[:, 0, 0] = 50.0
    K[:, 1, 1] = 50.0
    K[:, 0, 2] = (image_size - 1) / 2
    K[:, 1, 2] = (image_size - 1) / 2
    w2c = torch.eye(4).repeat(views, 1, 1)
    w2c[:, 2, 3] = 2.0
    return K, w2c


def test_visual_hull_uses_silhouette_intersection_in_canonical_space():
    image_size = 64
    masks = torch.zeros(2, 1, image_size, image_size)
    masks[:, :, 20:44, 20:44] = 1.0
    K, w2c = _centered_cameras(2, image_size)
    occupancy, report = carve_visual_hull(
        masks,
        K,
        w2c,
        VisualHullConfig(
            resolution=32,
            min_valid_views=2,
            min_view_fraction=1.0,
            mask_dilation_pixels=0,
            closing_iterations=0,
            chunk_size=4096,
        ),
    )
    assert occupancy.shape == (32, 32, 32)
    assert occupancy[16, 16, 16]
    assert not occupancy[0, 0, 16]
    assert 0.0 < report["occupancy_fraction"] < 0.9
    assert report["uses_gt_3d"] is False
    assert report["uses_heldout_views"] is False


def test_visual_hull_rejects_uninformative_full_frame_masks():
    image_size = 32
    masks = torch.ones(2, 1, image_size, image_size)
    K, w2c = _centered_cameras(2, image_size)
    with pytest.raises(RuntimeError, match="implausibly large"):
        carve_visual_hull(
            masks,
            K,
            w2c,
            VisualHullConfig(
                resolution=16,
                min_valid_views=2,
                mask_dilation_pixels=0,
                closing_iterations=0,
            ),
        )
