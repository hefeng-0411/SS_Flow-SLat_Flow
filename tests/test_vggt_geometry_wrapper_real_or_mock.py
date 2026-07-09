import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from geoss.integration.vggt_geometry_wrapper import VGGTGeometryWrapper


def test_vggt_geometry_wrapper_mock_and_optional_real():
    images_4d = torch.rand(2, 3, 32, 32)
    wrapper = VGGTGeometryWrapper(mock=True)
    out = wrapper(images_4d)
    assert out["vggt_depth"].shape == (1, 2, 1, 32, 32)
    assert out["vggt_pointmap"].shape == (1, 2, 3, 32, 32)
    assert "feature_shape_info" in out
    assert out["vggt_depth"].requires_grad is False
    assert out["vggt_pointmap"].requires_grad is False

    log_dir = Path("outputs/test_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log = {"mock_depth": list(out["vggt_depth"].shape), "real": "skipped"}

    ckpt = os.environ.get("VGGT_CHECKPOINT")
    root = os.environ.get("VGGT_ROOT", r"D:\VsCode\MVG\Base\vggt")
    if ckpt and Path(ckpt).exists():
        real = VGGTGeometryWrapper(vggt_root=root, checkpoint=ckpt, mock=False)
        assert real.model.training is False
        assert all(not p.requires_grad for p in real.model.parameters())
        real_out = real(torch.rand(1, 1, 3, 518, 518))
        log["real"] = {
            "depth": None if real_out["vggt_depth"] is None else list(real_out["vggt_depth"].shape),
            "pointmap": None if real_out["vggt_pointmap"] is None else list(real_out["vggt_pointmap"].shape),
            "features": None if real_out["vggt_features"] is None else list(real_out["vggt_features"].shape),
        }
    (log_dir / "test_vggt_geometry_wrapper_real_or_mock.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    test_vggt_geometry_wrapper_mock_and_optional_real()
