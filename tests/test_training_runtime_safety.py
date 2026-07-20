from __future__ import annotations

import torch

from geoss.integration.trellis_residency import configure_trellis_training_residency
from scripts.launch_meshfleet_multigpu_sequence import _recover_allocator_incompatibility


class _DummyPipeline:
    def __init__(self) -> None:
        self.models = {
            "sparse_structure_flow_model": torch.nn.Linear(4, 4),
            "slat_flow_model": torch.nn.Linear(4, 4),
            "image_cond_model": torch.nn.Linear(4, 4),
            "slat_decoder_gs": torch.nn.Linear(4, 4),
        }


def test_allocator_assert_removes_only_unsupported_option() -> None:
    env = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.9"
    }
    recovered = _recover_allocator_incompatibility(
        'RuntimeError: !block->expandable_segment_ INTERNAL ASSERT FAILED', env, "stage2", 4
    )
    assert recovered is True
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:512,garbage_collection_threshold:0.9"


def test_trellis_residency_keeps_exact_required_modules_and_precision() -> None:
    pipeline = _DummyPipeline()
    flow = pipeline.models["slat_flow_model"]
    image_encoder = pipeline.models["image_cond_model"]
    flow_dtype = next(flow.parameters()).dtype
    report = configure_trellis_training_residency(
        pipeline,
        required_models=("slat_flow_model", "image_cond_model"),
        device=torch.device("cpu"),
    )
    assert list(pipeline.models) == ["slat_flow_model", "image_cond_model"]
    assert pipeline.models["slat_flow_model"] is flow
    assert pipeline.models["image_cond_model"] is image_encoder
    assert next(flow.parameters()).dtype == flow_dtype
    assert all(not parameter.requires_grad for model in pipeline.models.values() for parameter in model.parameters())
    assert set(report["removed_models"]) == {"sparse_structure_flow_model", "slat_decoder_gs"}
