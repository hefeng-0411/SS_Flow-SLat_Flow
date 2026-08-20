from __future__ import annotations

from argparse import Namespace

import pytest

from geoss.utils.optional_deps import availability_report
from geoss.utils.run_mode import validate_config_switches, validate_real_mode


def test_real_train_missing_resources_fails_fast():
    args = Namespace(dry_run=False, real_train=True)
    with pytest.raises(FileNotFoundError):
        validate_real_mode(cfg={"allow_mock": False, "allow_synthetic": False}, args=args, mode="real_train", required=("vggt", "trellis", "dataset"))


def test_illegal_real_mock_config_rejected():
    with pytest.raises(ValueError):
        validate_config_switches({"use_real_vggt": True, "allow_mock": True})
    with pytest.raises(ValueError):
        validate_config_switches({"render_eval": True, "use_decoder": False})


def test_real_infer_missing_decoder_fails_fast():
    args = Namespace(
        dry_run=False,
        real_infer=True,
        vggt_checkpoint="vggt.pt",
        trellis_model_path=None,
        input="dataset",
    )
    with pytest.raises(FileNotFoundError):
        validate_real_mode(cfg={"allow_mock": False, "allow_synthetic": False}, args=args, mode="real_infer", required=("vggt", "trellis", "dataset", "decoder"))


def test_real_eval_missing_prediction_fails_fast():
    args = Namespace(dry_run=False, real_eval=True, prediction=None, gt_occ="gt.pt")
    with pytest.raises(FileNotFoundError):
        validate_real_mode(cfg={"allow_mock": False, "allow_synthetic": False}, args=args, mode="real_eval", required=("prediction",))


def test_real_eval_missing_gsplat_when_required_fails_fast():
    if availability_report()["gsplat"]:
        return
    args = Namespace(dry_run=False, real_eval=True, prediction="pred.npz", gt_occ="gt.pt")
    cfg = {
        "allow_mock": False,
        "allow_synthetic": False,
        "real_eval": True,
        "dependencies": {"require_gsplat_for_real_eval": True},
        "evaluation": {"use_gsplat_render_metrics": True},
    }
    with pytest.raises(ImportError):
        validate_real_mode(cfg=cfg, args=args, mode="real_eval", required=())
