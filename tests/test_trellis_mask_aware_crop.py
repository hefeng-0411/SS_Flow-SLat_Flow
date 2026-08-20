import torch

from geoss.integration.real_trellis_pipeline import mask_aware_trellis_crop


def test_mask_aware_crop_matches_trellis_square_foreground_framing():
    image = torch.zeros(1, 3, 100, 100)
    mask = torch.zeros(1, 1, 100, 100)
    image[:, 0, 30:70, 40:60] = 1.0
    mask[:, :, 30:70, 40:60] = 1.0
    cropped = mask_aware_trellis_crop(
        image,
        mask,
        output_size=100,
        padding=1.2,
    )
    foreground = cropped[0, 0] > 0.5
    coords = torch.nonzero(foreground, as_tuple=False)
    height = int(coords[:, 0].max() - coords[:, 0].min() + 1)
    width = int(coords[:, 1].max() - coords[:, 1].min() + 1)
    assert cropped.shape == (1, 3, 100, 100)
    assert 78 <= height <= 86
    assert 38 <= width <= 46
    assert float(cropped[:, 1:].abs().max()) == 0.0


def test_mask_aware_crop_rejects_empty_conditioning_mask():
    image = torch.zeros(1, 3, 32, 32)
    mask = torch.zeros(1, 1, 32, 32)
    try:
        mask_aware_trellis_crop(image, mask, output_size=64)
    except RuntimeError as exc:
        assert "no foreground" in str(exc)
    else:
        raise AssertionError("Empty conditioning mask must fail explicitly.")
