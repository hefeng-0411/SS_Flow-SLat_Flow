import torch

from geoss.utils.coordinates import occ_index_to_anchor_center
from geoss.utils.voxelization import sample_occupancy_at_points


def test_anchor_occ_sampling_matches_index_center():
    gt = torch.zeros(1, 16, 16, 16)
    gt[0, 3, 4, 5] = 1
    anchor = occ_index_to_anchor_center(torch.tensor([[3, 4, 5]]), 16).view(1, 1, 3)
    sample = sample_occupancy_at_points(gt, anchor)
    assert sample.item() == 1.0
