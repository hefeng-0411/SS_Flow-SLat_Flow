from __future__ import annotations

from argparse import Namespace

import pytest

from geoss.utils.env_check import collect_env
from geoss.utils.optional_deps import availability_report, require_dependency
from geoss.utils.run_mode import validate_real_mode


def test_optional_dependency_report_is_non_importing():
    report = availability_report()
    assert "gsplat" in report
    assert "kornia" in report
    assert isinstance(report["pycolmap"], bool)


def test_require_dependency_dry_mode_returns_false_for_missing_package():
    assert require_dependency("__definitely_missing_package__", real_mode=False) is False
    with pytest.raises(ImportError):
        require_dependency("__definitely_missing_package__", real_mode=True)


def test_real_eval_requires_configured_renderer_dependency():
    args = Namespace(dry_run=False, real_eval=True, prediction="x", gt_occ="x")
    cfg = {
        "allow_mock": False,
        "allow_synthetic": False,
        "real_eval": True,
        "dependencies": {"require_gsplat_for_real_eval": True},
        "rendering": {"renderer": "gsplat"},
    }
    if not availability_report()["gsplat"]:
        with pytest.raises(ImportError):
            validate_real_mode(cfg=cfg, args=args, mode="real_eval", required=())


def test_env_check_shape():
    info = collect_env()
    assert "python" in info
    assert "optional_dependencies" in info
