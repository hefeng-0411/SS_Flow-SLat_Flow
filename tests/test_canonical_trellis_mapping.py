import torch

from geoss.utils.coordinates import anchor_to_occ_index, canonical_to_trellis_grid, occ_index_to_anchor_center, trellis_grid_to_canonical


def test_canonical_trellis_roundtrip_centers():
    idx = torch.tensor([[0, 0, 0], [15, 15, 15], [7, 8, 9]])
    centers = occ_index_to_anchor_center(idx, 16)
    idx2 = anchor_to_occ_index(centers, 16)
    assert torch.equal(idx, idx2)
    grid = canonical_to_trellis_grid(centers, 16)
    recon = trellis_grid_to_canonical(grid, 16)
    assert torch.allclose(centers, recon)
