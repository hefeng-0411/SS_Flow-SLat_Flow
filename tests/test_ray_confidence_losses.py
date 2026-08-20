from __future__ import annotations

import torch

from geoss.losses.confidence_calibration import confidence_calibration_loss, target_confidence_from_errors


def test_confidence_calibration_uses_error_target():
    target = target_confidence_from_errors(depth_error=torch.zeros(4, 1), mask_error=torch.zeros(4, 1), sigma=0.5)
    terms = confidence_calibration_loss(torch.full((4, 1), 0.9), target)
    assert terms["loss"].ndim == 0
    assert terms["ece"] >= 0
