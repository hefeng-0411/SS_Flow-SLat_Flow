from pathlib import Path
import sys
import json
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from geoss.datasets.meshfleet_trellis_dataset import MeshFleetTrellisDataset
from geoss.datasets.vehicle_multiview_dataset import VehicleMultiViewDataset


ROOT = Path(r"D:\VsCode\MVG\Base\MeshFleet_TRELLIS")


def test_meshfleet_sample_shapes_if_available():
    if not (ROOT / "test" / "sdvas").exists():
        return
    dataset = MeshFleetTrellisDataset(ROOT, split="test", category="sdvas", num_views=4, image_size=64, occ_resolution=16)
    assert len(dataset) > 0
    sample = dataset[0]
    assert sample["images"].shape == (4, 3, 64, 64)
    assert sample["masks"].shape == (4, 1, 64, 64)
    assert sample["K"].shape == (4, 3, 3)
    assert sample["c2w"].shape == (4, 4, 4)
    assert sample["w2c"].shape == (4, 4, 4)
    assert sample["ss_latent_grid"].shape == (8, 16, 16, 16)
    assert sample["ss_latent_tokens"].shape == (4096, 8)
    assert sample["gt_occ"].shape == (16, 16, 16)
    assert sample["gt_occ"].any()
    assert sample["gt_sparse_xyz"].min() >= -1.0
    assert sample["gt_sparse_xyz"].max() <= 1.0
    assert sample["metadata"]["voxel_coordinate"]["source"] in {"trellis_centered_half_cube", "canonical_or_clamped"}


def test_meshfleet_collate_keeps_variable_fields_if_available():
    if not (ROOT / "test" / "sdvas").exists():
        return
    dataset = MeshFleetTrellisDataset(ROOT, split="test", category="sdvas", num_views=2, image_size=32, occ_resolution=8)
    batch = VehicleMultiViewDataset.collate_fn([dataset[0]])
    assert batch["images"].shape == (1, 2, 3, 32, 32)
    assert batch["ss_latent_grid"].shape == (1, 8, 16, 16, 16)
    assert torch.is_tensor(batch["gt_occ"])


def test_meshfleet_flat_split_layout():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_flat_sample(root / "train")
        dataset = MeshFleetTrellisDataset(root, split="train", num_views=1, image_size=16, occ_resolution=8)
        assert len(dataset) == 1
        sample = dataset[0]
        assert sample["metadata"]["layout"] == "flat_split"
        assert sample["images"].shape == (1, 3, 16, 16)
        assert sample["trellis_slat_feats"].shape == (4, 8)
        assert sample["trellis_slat_indices"].shape == (4, 3)

        direct = MeshFleetTrellisDataset(root / "train", split="train", num_views=1, image_size=16, occ_resolution=8)
        assert len(direct) == 1
        assert direct[0]["metadata"]["layout"] == "flat_split"


def test_meshfleet_missing_view_is_skipped_without_shape_drift():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_flat_sample(root / "train")
        dataset = MeshFleetTrellisDataset(root, split="train", num_views=2, image_size=16, occ_resolution=8)
        sample = dataset[0]
        assert sample["images"].shape == (2, 3, 16, 16)
        assert sample["K"].shape == (2, 3, 3)
        assert sample["c2w"].shape == (2, 4, 4)
        assert sample["metadata"]["num_frames_total"] == 2
        assert sample["metadata"]["num_frames_available"] == 1
        assert sample["metadata"]["missing_frames_skipped"] == 1


def _write_flat_sample(split_root: Path) -> None:
    uid = "flat_uid"
    render_dir = split_root / "renders" / uid
    render_dir.mkdir(parents=True)
    image = Image.new("RGBA", (16, 16), (128, 96, 64, 255))
    image.save(render_dir / "000.png")
    transforms = {
        "camera_angle_x": 0.8,
        "w": 16,
        "h": 16,
        "frames": [
            {
                "file_path": "000.png",
                "transform_matrix": np.eye(4, dtype=np.float32).tolist(),
            },
            {
                "file_path": "064.png",
                "transform_matrix": np.eye(4, dtype=np.float32).tolist(),
            }
        ],
    }
    (render_dir / "transforms.json").write_text(json.dumps(transforms), encoding="utf-8")
    voxel_dir = split_root / "voxels"
    voxel_dir.mkdir(parents=True)
    (voxel_dir / f"{uid}.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n",
        encoding="utf-8",
    )
    ss_dir = split_root / "ss_latents" / "ss_enc_conv3d_16l8_fp16"
    ss_dir.mkdir(parents=True)
    np.savez_compressed(ss_dir / f"{uid}.npz", mean=np.zeros((8, 16, 16, 16), dtype=np.float32))
    slat_dir = split_root / "latents" / "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"
    slat_dir.mkdir(parents=True)
    np.savez_compressed(
        slat_dir / f"{uid}.npz",
        feats=np.zeros((4, 8), dtype=np.float32),
        coords=np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=np.uint8),
    )
    feat_dir = split_root / "features" / "dinov2_vitl14_reg"
    feat_dir.mkdir(parents=True)
    np.savez_compressed(
        feat_dir / f"{uid}.npz",
        patchtokens=np.zeros((4, 16), dtype=np.float32),
        indices=np.zeros((4, 3), dtype=np.uint8),
    )


if __name__ == "__main__":
    test_meshfleet_sample_shapes_if_available()
    test_meshfleet_collate_keeps_variable_fields_if_available()
    test_meshfleet_flat_split_layout()
    test_meshfleet_missing_view_is_skipped_without_shape_drift()
