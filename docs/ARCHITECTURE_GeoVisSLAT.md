# GeoVis-SLAT Architecture

GeoVis-SLAT is an SS-conditioned adapter for TRELLIS SLAT Flow. It does not modify Sparse-Ray GeoSS, VGGT, TRELLIS source, SLAT decoders, 3DGS, or mesh refinement.

## SLAT Audit

| Item | Local TRELLIS Finding | Status |
| --- | --- | --- |
| SLAT Flow model | `D:\VsCode\MVG\Base\TRELLIS\trellis\models\structured_latent_flow.py::SLatFlowModel` | real audited |
| DiT block | `trellis\modules\sparse\transformer\modulated.py::ModulatedSparseTransformerCrossBlock` | real audited |
| Latent representation | `trellis.modules.sparse.SparseTensor`; `feats=[sum_active,C]`, `coords=[sum_active,4]` with batch id plus xyz indices | real audited |
| Token-active voxel mapping | One SLAT feature row per active sparse coord. Dataset stores `coords=[L,3]`, collate prepends batch id | real audited |
| Velocity output | `SLatFlowModel.forward(...)->SparseTensor` after `out_layer`; output coords/layout match input | real audited |
| Flow loss | `trellis\trainers\flow_matching\sparse_flow_matching.py::SparseFlowMatchingTrainer.training_losses`; MSE between `pred.feats` and target velocity feats | real audited |
| Sampling | `trellis\pipelines\samplers\flow_euler.py`; Euler update uses predicted velocity | real audited |
| Condition injection | Cross-attention context `cond` passed to each `ModulatedSparseTransformerCrossBlock` | real audited |
| Checkpoint loading | TRELLIS pipeline loads model config/checkpoint externally; GeoVis wrapper wraps a loaded model and does not alter state dict | wrapper-compatible |
| SS to SLAT interface | `TrellisImageTo3DPipeline.sample_slat(cond, coords, ...)` creates SLAT noise on SS `coords` | real audited |
| Decoder interface | `decode_slat` consumes final `SparseTensor`; GeoVis-SLAT does not call or modify decoder | untouched |
| CFG | `ClassifierFreeGuidanceSamplerMixin` calls model for conditional and negative branches | real audited |

## Core Formula

`SLATVelocityAdapter` implements:

```text
v_slat_geo = v_slat_base + beta(t) * slat_confidence * ss_confidence * clip_tau(t)(Delta_v_slat)
```

Location: `geoss/slat/models/slat_velocity_adapter.py`.

Disabled path: `use_geovis_slat=False` returns `v_slat_base` directly.

## Data Flow

1. `ActiveVoxelProjector` maps TRELLIS active indices to canonical centers with `occ_index_to_anchor_center`, then projects with existing OpenCV `project_points`.
2. `VisibilityEvidenceSampler` grid-samples RGB, masks, optional rendered/VGGT depth, and optional VGGT features per active voxel and view.
3. `GeoVisSLATAggregator` aggregates only the `N` view tokens belonging to the same active voxel. It does not attend over full image tokens.
4. `SLATVelocityAdapter` cross-attends noisy SLAT tokens to GeoVis condition tokens and gates the velocity residual.
5. `GeoVisTrellisSLATWrapper` converts real TRELLIS `SparseTensor` feats to padded `[B,L,C]`, applies the adapter, and restores `SparseTensor.feats`.

## Visibility Rules

- In bounds, inside mask, near depth surface gives high visibility.
- Behind the visible surface by `occlusion_margin` gives high occlusion and low visibility.
- Mask outside gives low appearance reliability.
- Missing rendered depth falls back to VGGT depth.
- If no depth exists, visibility uses mask and feature evidence only.
- Cross-view feature/RGB variance lowers appearance consistency and SLAT confidence.

## Confidence

`slat_confidence` combines learned confidence with:

- visible view support;
- low occlusion;
- low depth residual;
- low appearance conflict;
- square-rooted SS geometry confidence.

Final velocity uses `joint_confidence = slat_confidence * ss_confidence`.

## CFG Policy

`GeoVisTrellisSLATWrapper` defaults to applying GeoVis context only on the conditional branch. For unconditional CFG calls, pass `geovis_branch="uncond"`; the wrapper returns base velocity unless `geovis_slat_apply_to_uncond=True`.

## Real vs Fallback

Real:
- TRELLIS SLAT target audit and native MeshFleet `latents/<model>/<uid>.npz` loading.
- Active voxel projection, visibility evidence, view aggregation, confidence-gated velocity residual.
- Dense and SparseTensor-compatible hook boundary.

Fallback:
- Real TRELLIS checkpoint execution is not launched by dry-run scripts.
- Decoder/render metrics are optional and disabled by default.
- If SRN/Objaverse samples lack native SLAT latents, scripts use auxiliary view/visibility supervision and mark the source.
