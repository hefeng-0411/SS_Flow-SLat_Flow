from __future__ import annotations

import torch
import pytest

from geoss.integration.trellis_residency import configure_trellis_training_residency
from scripts.launch_meshfleet_multigpu_sequence import (
    _infer_stage_manifest,
    _recover_allocator_incompatibility,
)
from scripts.train_geovis_slat import _validate_checkpoint_model_config


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


def test_stage3_manifest_is_inferred_from_dataset_audit_siblings(tmp_path) -> None:
    stage2 = tmp_path / "stage2_train_uids.json"
    stage3 = tmp_path / "stage3_train_uids.json"
    stage2.write_text('{"uids": ["stage2"]}', encoding="utf-8")
    stage3.write_text('{"uids": ["stage3"]}', encoding="utf-8")
    selected = _infer_stage_manifest(
        explicit=None,
        fallback=None,
        infer_from=(str(stage2),),
        filename="stage3_train_uids.json",
    )
    assert selected == str(stage3.resolve())


def test_stage4_weights_only_init_allows_only_nonstructural_control_curriculum() -> None:
    checkpoint_model = {
        "slat_dim": 8,
        "hidden_dim": 128,
        "fusion_mode": "aligned",
        "trust_region": 0.15,
        "confidence_floor": 0.05,
    }
    current_model = {
        **checkpoint_model,
        "trust_region": 0.20,
        "confidence_floor": 0.0,
    }
    state = {"config": {"model": checkpoint_model}}
    overrides = _validate_checkpoint_model_config(
        state,
        current_model,
        context="Initialization",
        allow_runtime_control_overrides=True,
    )
    assert set(overrides) == {"trust_region", "confidence_floor"}
    with pytest.raises(RuntimeError, match="trust_region"):
        _validate_checkpoint_model_config(state, current_model, context="Resume")

    structurally_incompatible = {**current_model, "hidden_dim": 256}
    with pytest.raises(RuntimeError, match="hidden_dim"):
        _validate_checkpoint_model_config(
            state,
            structurally_incompatible,
            context="Initialization",
            allow_runtime_control_overrides=True,
        )
