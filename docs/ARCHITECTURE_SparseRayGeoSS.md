# Architecture

## Audited VGGT Interfaces

- Input: `VGGT.forward(images)` accepts `[S,3,H,W]` or `[B,S,3,H,W]`, range `[0,1]`.
- Camera: `pose_enc [B,S,9]`, decoded by `pose_encoding_to_extri_intri` to OpenCV `w2c [B,S,3,4]` and `K [B,S,3,3]`.
- Depth: DPT output is normalized by `VGGTGeometryWrapper` to `[B,N,1,H,W]`.
- Point map: `world_points` is normalized by the wrapper to `[B,N,3,H,W]`.
- Features: public predictions do not expose features. The wrapper uses `model.aggregator(images)` and returns final patch tokens.

## Audited TRELLIS SS Interfaces

- Model: `trellis/models/sparse_structure_flow.py::SparseStructureFlowModel`.
- DiT block: `trellis/modules/transformer/modulated.py::ModulatedTransformerCrossBlock`.
- Velocity: final `SparseStructureFlowModel.forward()` output `[B,out_channels,16,16,16]`.
- Flow loss: `trellis/trainers/flow_matching/flow_matching.py`, target velocity `(1-sigma_min)*noise - x_0`.
- Sampler: `trellis/pipelines/samplers/flow_euler.py`, Euler update with predicted velocity.
- Hook: `GeoSSTrellisSSWrapper` wraps the SS flow model and injects after `v_base` is produced.

## New Modules

- `SparseAnchorQueries`: `[B,M,3]` anchor xyz and `[B,M,C]` anchor features.
- `RayEvidenceSampler`: projects anchors into every view and emits `[B,M,N,C_e]` evidence tokens.
- `CrossViewEvidenceAggregator`: anchor query attends to per-view evidence tokens.
- `SSVelocityAdapter`: latent tokens attend to geo tokens and output clipped velocity residuals.
- `GeoSSTrellisSSWrapper`: converts TRELLIS SS grids to tokens and back.

The velocity control equation is:

```text
v_geo = v_base + alpha(t) * token_confidence * clip_tau(t)(Delta_v_geo)
```

Module I/O:

```text
SparseAnchorQueries(B) -> anchor_xyz [B,M,3], anchor_feat [B,M,C_a]
RayEvidenceSampler(anchor_xyz,K,c2w,w2c,masks,depths) -> view_tokens [B,M,N,C_e]
CrossViewEvidenceAggregator(view_tokens,anchor_feat,ray_valid) -> geo_tokens [B,M,C_g], geo_confidence [B,M,1]
SSVelocityAdapter(ss_latent_tokens,geo_tokens,geo_confidence,t,v_base) -> v_geo [B,L,C_ss]
GeoSSTrellisSSWrapper(x [B,C,D,H,W]) -> v_geo_grid [B,C,D,H,W]
```

## Ray Evidence Rules

For anchor `i` and view `j`:

1. Project with OpenCV `uv_ij = project(anchor_i, K_j, w2c_j)`.
2. Mark invalid if the point is behind the camera or outside image bounds.
3. Sample mask and depth at `uv_ij`. Depth priority is rendered/GT `depths`, then `vggt_depth`; if both are absent, depth-based occupied/free evidence is skipped while mask-outside free evidence remains valid.
4. `occupied_geometry = in_bounds & mask & abs(z_anchor - z_surface) < surface_threshold`.
5. `free_geometry = in_bounds & (mask == 0 or z_anchor < z_surface - free_margin)`.
6. `conflict_score` is the fraction of valid views where both occupied and free evidence exist for the same anchor across views.

Outputs include `view_tokens`, `occ_score`, `free_score`, `visibility`, absolute `depth_residual`, `signed_depth_residual`, `ray_valid`, `conflict_score`, and `evidence_debug`.

## Confidence Loop

`CrossViewEvidenceAggregator` returns:

```text
alpha_occ = softplus(occ_evidence) + 1
alpha_free = softplus(free_evidence) + 1
p_occ = alpha_occ / (alpha_occ + alpha_free)
uncertainty = 2 / (alpha_occ + alpha_free)
geo_confidence = exp(-uncertainty) * exp(-conflict_score)
```

`geo_confidence` is used by `SSVelocityAdapter` through attention-weighted `token_confidence`, by prior preservation loss, by confidence calibration loss, by eval correlation hooks, and by PLY visualization.

## CFG Handling

TRELLIS CFG calls the SS flow once for the conditional branch and once for the unconditional branch. `GeoSSSamplerWrapper` patches a sampler instance without editing TRELLIS source:

- default: `geoss_apply_to_uncond=false`; GeoSS context is applied only to the conditional branch.
- optional: `geoss_apply_to_uncond=true`; both branches receive the same geometry context.

The disabled adapter path still returns the base `v_base` with error under `1e-6` in tests.
