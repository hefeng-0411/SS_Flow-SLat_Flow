# Phase II current-state file inventory

This inventory is based only on the current filesystem, generated artifacts,
and changes made during the Phase-II task. Repository history and version-control
commands are forbidden by the workspace instructions, so “new” and “modified”
below describe this task's observed work rather than a historical Git diff.

## Phase-II architecture and objective files

Created for Phase II:

- `geoss/slat/utils/normalization.py`: strict TRELLIS raw-VAE/normalized-flow
  conversion and tensor-contract version.
- `geoss/slat/losses/factorized_control_loss.py`: separate correction-demand
  calibration and heteroscedastic residual uncertainty.
- `geoss/slat/losses/decoded_asset_loss.py`: differentiable clean-state
  inversion, frozen TRELLIS Gaussian decoding, RGB/foreground/SSIM/LPIPS/mask,
  optional depth, and decoded geometry loss.
- `configs/phase2_ablation_normalized_single_view.yaml`.
- `configs/phase2_ablation_multiview_teacher.yaml`.
- `configs/phase2_ablation_factorized_control.yaml`.
- `configs/phase2_ablation_decoded_appearance.yaml`.
- `configs/phase2_decoded_asset.yaml`.
- `scripts/profile_meshfleet_distribution.py`.
- `docs/PHASE2_MATHEMATICAL_AUDIT.md`.
- `docs/PHASE2_REMOTE_RUNBOOK.md`.
- `docs/PHASE2_FILE_INVENTORY.md`.

Modified for Phase-II coupling/training:

- `geoss/slat/models/geovis_slat_aggregator.py`.
- `geoss/slat/models/geovis_slat_adapter.py`.
- `geoss/slat/models/slat_velocity_adapter.py`.
- `geoss/slat/integration/trellis_slat_hook.py`.
- `geoss/slat/losses/__init__.py`.
- `geoss/losses/render_losses.py`.
- `geoss/renderers/gsplat_renderer.py`.
- `geoss/integration/real_trellis_pipeline.py`.
- `geoss/io/asset_io.py`.
- `scripts/train_geovis_slat.py`.
- `scripts/train_geovis_slat_joint.py`.
- `scripts/infer_geovis_slat.py`.
- `scripts/launch_meshfleet_multigpu_sequence.py`.
- `scripts/refine_gaussian_conditioning.py`.
- `configs/geovis_slat.yaml`.
- `configs/geovis_slat_joint.yaml`.

Modified or revalidated for correctness/provenance:

- `geoss/losses/occupancy_loss.py`.
- `geoss/utils/coordinates.py`.
- `geoss/integration/vggt_geometry_wrapper.py`.
- `scripts/train_sparse_ray_geoss.py`.
- `scripts/train_sparse_ray_ss_velocity.py`.
- `scripts/infer_sparse_ray_geoss_ss.py`.
- `scripts/eval_geovis_slat.py`.
- `scripts/evaluate_meshfleet_sequence.py`.
- `scripts/inspect_meshfleet_dataset.py`.
- `scripts/analyze_evaluation_suite.py`.
- `docs/TECHNICAL_REPORT_PROTOCOL_V2.md`.

Test files changed or extended:

- `tests/test_coordinate_conversion.py`.
- `tests/test_evaluation_protocol_v2.py`.
- `tests/test_slat_velocity_adapter.py`.
- `tests/test_multiview_trellis_sampler.py`.
- `tests/test_geometry_alignment.py`.

## Revalidated Phase-I paths

The audit also inspected the current dataset, metric, coordinate, VGGT, sparse
structure, and active-voxel paths, notably:

- `geoss/datasets/meshfleet_trellis_dataset.py` and
  `geoss/datasets/vehicle_multiview_dataset.py`;
- `geoss/eval/render_metrics.py` and `geoss/metrics/geometry_metrics.py`;
- `geoss/geometry/alignment.py`;
- `geoss/integration/trellis_ss_hook.py`;
- `geoss/models/ray_evidence_sampler.py` and
  `geoss/models/sparse_ray_geoss_adapter.py`;
- `geoss/slat/utils/active_voxel_utils.py`;
- `geoss/slat/losses/appearance_feature_loss.py`,
  `view_consistency_loss.py`, `visibility_confidence_loss.py`, and
  `slat_velocity_loss.py`.

## Deleted, disabled, incomplete, and unused paths

- No source path was deleted during Phase II.
- `geoss/slat/losses/render_proxy_loss.py` remains as a legacy disabled API and
  is not part of the production objective; decoded supervision replaces it.
- Dry-run mock/synthetic code remains deliberately isolated behind explicit
  `--dry_run`; real train/eval/infer paths fail closed.
- Phase-I SLAT checkpoints are incompatible with the strict
  `phase2_factorized_control_v2` contract and must not be partially loaded.
- Existing legacy evaluation JSON remains diagnostic/non-official.
- Complete remote training, CUDA gradient execution, ablation, validation, and
  final test are pending because `/mnt/sda2` and a PyTorch/CUDA runtime are not
  accessible from this workspace.

## Generated local artifacts

- `outputs/dataset_profile_local`: one-object structural distribution profile.
- `outputs/dataset_audit_local`: deterministic local manifests and integrity
  summary.
- `outputs/evaluation_suite_analysis`: legacy-output audit and failure analysis.

Python bytecode caches were created by compilation checks and were not removed,
in accordance with the explicit no-cache-cleaning rule.
