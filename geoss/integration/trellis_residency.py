from __future__ import annotations

import gc
from typing import Iterable

import torch


def configure_trellis_training_residency(
    pipeline,
    *,
    required_models: Iterable[str],
    device: torch.device,
) -> dict[str, object]:
    """Keep only the frozen TRELLIS modules used by one training stage.

    TRELLIS' generic ``Pipeline.to`` moves every flow model and decoder to the
    GPU.  Residual training needs a strict subset.  Retaining the identical
    pretrained modules while dropping unused references changes neither the
    forward function nor its numerical precision, but avoids co-resident
    models consuming VRAM needed by VGGT and the trainable adapter.
    """
    requested = tuple(dict.fromkeys(str(name) for name in required_models))
    models = getattr(pipeline, "models", None)
    if not isinstance(models, dict):
        raise TypeError("TRELLIS pipeline must expose a models dictionary.")
    missing = [name for name in requested if name not in models or models[name] is None]
    if missing:
        raise RuntimeError(f"TRELLIS pipeline is missing required training models: {missing}")

    original_names = tuple(models)
    retained = {name: models[name] for name in requested}
    removed = tuple(name for name in original_names if name not in retained)
    pipeline.models = retained
    for model in retained.values():
        model.to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    # All removed modules are still on CPU at this point.  Collect them before
    # constructing VGGT so rank-local host memory also has a bounded peak.
    del models
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "policy": "exact_stage_required_models_v1",
        "required_models": list(requested),
        "removed_models": list(removed),
        "device": str(device),
        "numerical_contract": "identical retained pretrained modules; no quantization or precision change",
    }
