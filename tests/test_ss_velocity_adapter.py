import torch

from geoss.integration.trellis_ss_hook import ss_grid_to_tokens, tokens_to_ss_grid
from geoss.models.ss_velocity_adapter import SSVelocityAdapter


def test_ss_velocity_adapter_shapes():
    B, L, C, M, G = 2, 32, 8, 64, 16
    adapter = SSVelocityAdapter(latent_dim=C, geo_dim=G, hidden_dim=32, num_heads=4)
    ss = torch.randn(B, L, C)
    geo = torch.randn(B, M, G)
    conf = torch.rand(B, M, 1)
    v_base = torch.randn(B, L, C)
    out = adapter(ss, geo, conf, torch.tensor([100.0, 500.0]), v_base)
    assert out["v_geo"].shape == (B, L, C)
    assert out["delta_v_geo"].shape == (B, L, C)
    assert out["token_confidence"].shape == (B, L, 1)


def test_ss_grid_token_roundtrip():
    grid = torch.randn(1, 8, 4, 4, 4)
    tokens = ss_grid_to_tokens(grid)
    recon = tokens_to_ss_grid(tokens, (4, 4, 4))
    assert torch.allclose(grid, recon)
