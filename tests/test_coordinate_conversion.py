from pathlib import Path

import torch

from geoss.utils.coordinates import c2w_to_w2c, opengl_to_opencv, parse_srn_intrinsics, parse_srn_pose, w2c_to_c2w


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
