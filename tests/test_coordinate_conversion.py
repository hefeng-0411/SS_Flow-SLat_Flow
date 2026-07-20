from pathlib import Path

import torch

from geoss.utils.coordinates import c2w_to_w2c, opengl_to_opencv, parse_objaverse_camera, parse_srn_intrinsics, parse_srn_pose, w2c_to_c2w


def test_srn_parse_and_inverse(tmp_path: Path):
    intr = tmp_path / "intrinsics.txt"
    pose = tmp_path / "000000.txt"
    intr.write_text("100 32 32 0")
    pose.write_text(" ".join(str(x) for x in torch.eye(4).reshape(-1).tolist()))
    K = parse_srn_intrinsics(intr)
    c2w = parse_srn_pose(pose)
    w2c = c2w_to_w2c(c2w)
    assert K.shape == (3, 3)
    assert torch.allclose(w2c_to_c2w(w2c), c2w)


def test_opengl_to_opencv_axes():
    c2w = torch.eye(4)
    out = opengl_to_opencv(c2w)
    assert torch.allclose(out[1, 1], torch.tensor(-1.0))
    assert torch.allclose(out[2, 2], torch.tensor(-1.0))


def test_explicit_intrinsics_are_rescaled_with_image_resize():
    camera = {
        "transform_matrix": torch.eye(4).tolist(),
        "w": 1024,
        "h": 512,
        "fl_x": 800.0,
        "fl_y": 600.0,
        "cx": 512.0,
        "cy": 256.0,
    }
    _, K = parse_objaverse_camera(camera, image_size=(256, 256), assume_opengl=False)
    assert torch.allclose(K, torch.tensor([[200.0, 0.0, 128.0], [0.0, 300.0, 128.0], [0.0, 0.0, 1.0]]))


def test_intrinsic_matrix_is_rescaled_with_image_resize():
    camera = {
        "transform_matrix": torch.eye(4).tolist(),
        "w": 800,
        "h": 400,
        "K": [[600.0, 0.0, 400.0], [0.0, 500.0, 200.0], [0.0, 0.0, 1.0]],
    }
    _, K = parse_objaverse_camera(camera, image_size=(200, 200), assume_opengl=False)
    assert torch.allclose(K, torch.tensor([[150.0, 0.0, 100.0], [0.0, 250.0, 100.0], [0.0, 0.0, 1.0]]))
