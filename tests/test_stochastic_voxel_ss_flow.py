from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from geoss.datasets.dataset_stochastic_meshfleet import (
    StochasticMeshFleetDataset,
    stochastic_meshfleet_collate,
)
from geoss.integration.vggt_geometry_wrapper import VGGTGeometryBatch, VGGTGeometryWrapper
from geoss.models.ss_flow_adapter import SSFlowAdapter
from geoss.models.voxel_fusion_engine import (
    ConfidenceSparseVoxelFusion,
    VoxelFusionOutput,
    align_vggt_reference_to_dataset,
    unproject_depth_batched,
)
from geoss.ops.flow_matching import construct_flow_training_pair
from geoss.samplers.fast_ss_sampler import FastGeometryConditionedSSSampler
from geoss.utils.projection import project_points


def _make_meshfleet_object(root: Path, views: int = 10, missing: tuple[int, ...] = ()) -> None:
    uid = "unit_object"
    render = root / "train" / "renders" / uid
    latent = root / "train" / "ss_latents"
    render.mkdir(parents=True)
    latent.mkdir(parents=True)
    frames = []
    for index in range(views):
        if index not in missing:
            rgba = np.zeros((16, 16, 4), dtype=np.uint8)
            rgba[..., :3] = index * 10
            rgba[..., 3] = 255
            Image.fromarray(rgba, "RGBA").save(render / f"{index:03d}.png")
        transform = np.eye(4)
        transform[0, 3] = index / 100.0
        frames.append(
            {
                "file_path": f"{index:03d}.png",
                "camera_angle_x": 0.7,
                "transform_matrix": transform.tolist(),
            }
        )
    (render / "transforms.json").write_text(
        json.dumps({"aabb": [[-0.5] * 3, [0.5] * 3], "scale": 1.0, "offset": [0, 0, 0], "frames": frames})
    )
    np.savez(latent / f"{uid}.npz", mean=np.zeros((8, 16, 16, 16), dtype=np.float32))


def test_dataset_stochastic_views_are_bounded_distinct_and_reproducible(tmp_path: Path):
    _make_meshfleet_object(tmp_path)
    first = StochasticMeshFleetDataset(str(tmp_path), min_views=1, max_views=8, seed=9, rank=2, image_size=16, load_gt_occupancy=False)
    second = StochasticMeshFleetDataset(str(tmp_path), min_views=1, max_views=8, seed=9, rank=2, image_size=16, load_gt_occupancy=False)
    first.set_epoch(3)
    second.set_epoch(3)
    a, b = first[0], second[0]
    assert 1 <= a["num_views"] <= 8
    assert torch.equal(a["view_ids"], b["view_ids"])
    assert a["view_ids"].unique().numel() == a["num_views"]
    epoch_selections = []
    for epoch in range(6):
        first.set_epoch(epoch)
        epoch_selections.append(tuple(first[0]["view_ids"].tolist()))
    assert len(set(epoch_selections)) > 1


def test_dataset_gapped_render_ids_keep_image_camera_and_metadata_aligned(tmp_path: Path):
    _make_meshfleet_object(tmp_path, views=10, missing=(1, 4))
    dataset = StochasticMeshFleetDataset(
        str(tmp_path),
        min_views=8,
        max_views=8,
        seed=5,
        use_stochastic_views=False,
        image_size=16,
        load_gt_occupancy=False,
    )
    sample = dataset[0]

    assert sample["num_views"] == 8
    assert set(sample["view_ids"].tolist()) == {0, 2, 3, 5, 6, 7, 8, 9}
    assert torch.equal(sample["view_ids"], sample["view_metadata_indices"])
    assert sample["metadata"]["missing_frame_ids"] == ["001", "004"]
    assert sample["metadata"]["num_frames_total"] == 10
    assert sample["metadata"]["num_frames_available"] == 8
    for position, view_id in enumerate(sample["view_ids"].tolist()):
        expected_rgb = view_id * 10 / 255.0
        assert torch.allclose(
            sample["images"][position].mean(), torch.tensor(expected_rgb), atol=1e-6
        )
        assert sample["c2w_dataset"][position, 0, 3].item() == pytest.approx(
            view_id / 100.0
        )
        assert Path(sample["metadata"]["selected_frame_paths"][position]).stem == f"{view_id:03d}"
    batch = stochastic_meshfleet_collate([sample])
    assert torch.equal(batch["view_ids"][0], sample["view_ids"])
    assert torch.equal(batch["view_metadata_indices"][0], sample["view_metadata_indices"])


def test_alignment_reports_rays_that_no_scale_can_place_in_canonical_box():
    points = torch.tensor(
        [[[[[0.0, 2.0]], [[0.0, 0.0]], [[2.0, 0.0]]]]], dtype=torch.float32
    )
    vggt_w2c = torch.eye(4).view(1, 1, 4, 4)
    dataset_c2w = torch.eye(4).view(1, 1, 4, 4)
    dataset_c2w[..., 2, 3] = -2.0
    aligned, scale, alignment_inlier = align_vggt_reference_to_dataset(
        points,
        vggt_w2c,
        dataset_c2w,
        torch.zeros(1, 3),
        torch.full((1, 3), 0.5),
        torch.ones(1, 1, 1, 2, dtype=torch.bool),
        torch.ones(1, 1, 1, 2),
    )

    assert torch.isfinite(aligned).all()
    assert torch.isfinite(scale).all()
    assert alignment_inlier.flatten().tolist() == [True, False]


def test_project_unproject_roundtrip():
    depth = torch.full((1, 1, 1, 3, 4), 2.0)
    K = torch.tensor([[[[4.0, 0.0, 1.5], [0.0, 4.0, 1.0], [0.0, 0.0, 1.0]]]])
    c2w = torch.eye(4).view(1, 1, 4, 4)
    points = unproject_depth_batched(depth, K, c2w)[0, 0].permute(1, 2, 0).reshape(-1, 3)
    projection = project_points(points, K[0, 0], torch.eye(4))
    y, x = torch.meshgrid(torch.arange(3), torch.arange(4), indexing="ij")
    expected = torch.stack([x, y], dim=-1).reshape(-1, 2).float()
    assert torch.allclose(projection["uv"], expected, atol=1e-5)
    assert torch.allclose(projection["depth"], torch.full((12, 1), 2.0), atol=1e-5)


def test_weighted_duplicate_voxel_reduction_and_observation_mask():
    fusion = ConfidenceSparseVoxelFusion(
        input_feature_dim=2,
        projected_feature_dim=2,
        positional_frequencies=1,
        require_spconv=False,
        use_spconv_refinement=False,
    )
    fusion.feature_projection = torch.nn.Identity()
    fusion.condition_dim = 2 + 3 + 6
    points = torch.tensor([[[[[-0.9, -0.9]], [[-0.9, -0.9]], [[-0.9, -0.9]]]]])
    # Reshape as [B,V,3,H,W], two view observations in one voxel.
    points = torch.tensor([[[[[-0.9]], [[-0.9]], [[-0.9]]], [[[ -0.9]], [[-0.9]], [[-0.9]]]]])
    features = torch.tensor([[[[[2.0]], [[4.0]]], [[[10.0]], [[14.0]]]]])
    confidence = torch.tensor([[[0.25], [0.75]]])
    weights = confidence.clone()
    valid = torch.ones(1, 2, 1, 1, dtype=torch.bool)
    output = fusion._reduce_to_voxels(points, features, confidence, weights, valid, torch.ones(1, 2, dtype=torch.bool), torch.ones(1))
    expected = torch.tensor([(0.25 * 2 + 0.75 * 10), (0.25 * 4 + 0.75 * 14)])
    assert torch.allclose(output.dense_tokens[0, 0, :2], expected, atol=1e-6)
    assert bool(output.observation_mask[0, 0, 0])
    assert not bool(output.observation_mask[0, 1, 0])
    assert output.observation_count[0, 0, 0] == 2


def test_zero_init_and_spatial_gate_invariants():
    adapter = SSFlowAdapter(latent_dim=3, condition_dim=5, hidden_dim=16, num_heads=4, num_blocks=1, trust_region=10.0)
    x = torch.randn(1, 4, 3)
    cond = torch.randn(1, 4, 5)
    base = torch.randn_like(x)
    observed = torch.ones(1, 4, 1, dtype=torch.bool)
    confidence = torch.ones(1, 4, 1)
    initialized = adapter(x, cond, torch.zeros(1), observed, confidence, v_base=base)
    assert torch.equal(initialized.v_final, base)
    torch.nn.init.zeros_(adapter.residual_head.weight)
    torch.nn.init.constant_(adapter.residual_head.bias, 0.2)
    unobserved = adapter(x, cond, torch.zeros(1), torch.zeros_like(observed), confidence, v_base=base, enabled=False)
    assert torch.equal(unobserved.v_final, base)
    full = adapter(x, cond, torch.zeros(1), observed, confidence, v_base=base)
    assert torch.allclose(full.v_final, base + full.delta_v_geo, atol=1e-6)


def test_trellis_flow_pair_matches_sigma_min_equation():
    x0 = torch.randn(2, 8, 2, 2, 2)
    noise = torch.randn_like(x0)
    t = torch.tensor([0.2, 0.8])
    xt, target, backend = construct_flow_training_pair(x0, noise, t, 1e-5, backend="torch")
    view = t.view(2, 1, 1, 1, 1)
    assert torch.allclose(xt, (1 - view) * x0 + (1e-5 + (1 - 1e-5) * view) * noise)
    assert torch.allclose(target, (1 - 1e-5) * noise - x0)
    assert backend == "torch"


class _ConstantBase(torch.nn.Module):
    def forward(self, x, t, cond):
        return torch.full_like(x, 0.125)


def test_five_step_sampler_is_deterministic():
    adapter = SSFlowAdapter(latent_dim=8, condition_dim=5, hidden_dim=16, num_heads=4, num_blocks=1)
    sampler = FastGeometryConditionedSSSampler(_ConstantBase(), adapter, num_steps=5, rescale_t=3.0)
    noise = torch.randn(1, 8, 16, 16, 16, generator=torch.Generator().manual_seed(4))
    fusion = VoxelFusionOutput(
        sparse_tensor=None,
        dense_tokens=torch.zeros(1, 4096, 5),
        voxel_indices=torch.empty(0, 4, dtype=torch.int32),
        voxel_xyz=torch.empty(0, 3),
        voxel_confidence=torch.ones(1, 4096, 1),
        observation_mask=torch.ones(1, 4096, 1, dtype=torch.bool),
        accumulated_weight=torch.ones(1, 4096, 1),
        observation_count=torch.ones(1, 4096, 1),
        valid_mask=torch.ones(1, 4096, dtype=torch.bool),
        alignment_scale=torch.ones(1),
        condition_dim=5,
    )
    condition = torch.ones(1, 3, 1024)
    first = sampler.sample(noise.clone(), base_condition=condition, voxel_fusion=fusion, adapter_enabled=False)
    second = sampler.sample(noise.clone(), base_condition=condition, voxel_fusion=fusion, adapter_enabled=False)
    assert torch.equal(first.samples, second.samples)


def test_production_vggt_rejects_mock():
    with pytest.raises(RuntimeError, match="Production VGGT"):
        VGGTGeometryWrapper(mock=True, require_real=True)


def test_production_trainer_has_no_mock_or_random_condition_tokens():
    source = (Path(__file__).parents[1] / "scripts" / "train_stochastic_voxel_ss_flow.py").read_text()
    assert "MockSSFlow" not in source
    assert "random condition" not in source.lower()
    assert "pipeline.encode_image(first_view)" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP smoke requires a GPU")
def test_bf16_adapter_forward_backward_is_finite():
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is unavailable")
    device = torch.device("cuda")
    adapter = SSFlowAdapter(latent_dim=8, condition_dim=9, hidden_dim=32, num_heads=4, num_blocks=1).to(device)
    torch.nn.init.normal_(adapter.residual_head.weight, std=1e-3)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = adapter(
            torch.randn(1, 16, 8, device=device),
            torch.randn(1, 16, 9, device=device),
            torch.tensor([500.0], device=device),
            torch.ones(1, 16, 1, device=device, dtype=torch.bool),
            torch.ones(1, 16, 1, device=device),
            v_base=torch.randn(1, 16, 8, device=device),
        )
        loss = output.v_final.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in adapter.parameters())
