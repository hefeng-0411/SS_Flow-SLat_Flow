# Phase II mathematical audit and architecture decision

## Status and audit boundary

This document describes the executable repository state on 2026-07-20. The
workspace instruction forbids version-control operations and history inspection,
so a historical diff cannot be reconstructed mechanically. The inventory below
is based on the current source tree, the Phase-I report, generated outputs, and
the Phase-II changes explicitly made in this task. No source path was deleted in
Phase II.

Local execution cannot validate CUDA/TRELLIS behavior: the Windows interpreter
has no usable PyTorch environment, WSL has no PyTorch, model weights are absent,
the local training split is empty, and `/mnt/sda2` is not mounted. Therefore all
post-redesign metric claims remain pending the mandatory remote experiments.

## Results status

The only numerical outputs currently present are legacy diagnostics. They are
retained for failure analysis but are not protocol-v2 results: they use
conditioning/unspecified views, omit geometry metrics, contain proxy fields, and
cover inconsistent object counts. Phase II does not relabel them as official.

| Method | Split/protocol | N | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CD ↓ | F-score ↑ | Runtime | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original TRELLIS | legacy diagnostic | 8 | 15.2603 | 0.1933 | 0.2357 | absent | absent | absent | absent |
| Current SS Flow | legacy diagnostic | 8 | 15.2614 | 0.1945 | 0.2353 | absent | absent | absent | absent |
| Current SLAT Flow | legacy diagnostic | 8 | 15.2063 | 0.1919 | 0.2349 | absent | absent | absent | absent |
| Legacy “joint” | legacy diagnostic | 6 | 14.5669 | 0.2179 | 0.2374 | absent | absent | absent | absent |
| Phase-II normalized single-view | validation/test v2 | pending remote | pending | pending | pending | pending | pending | emitted | emitted |
| Phase-II multi-view CFG teacher | validation/test v2 | pending remote | pending | pending | pending | pending | pending | emitted | emitted |
| Phase-II factorized control | validation/test v2 | pending remote | pending | pending | pending | pending | pending | emitted | emitted |
| Phase-II decoded appearance | validation/test v2 | pending remote | pending | pending | pending | pending | pending | emitted | emitted |
| Phase-II decoded asset | validation/test v2 | pending remote | pending | pending | pending | pending | pending | emitted | emitted |

The supplied comparison target `(22.632, 0.911, 0.090, 0.0895, 0.953)` is not
claimed exceeded—or even directly comparable—until its background, held-out
camera policy, LPIPS backbone, Chamfer convention, and F-score threshold are
matched.

## Current-state inventory

Phase I changed or introduced the following functional areas:

- dataset contracts: `geoss/datasets/meshfleet_trellis_dataset.py`,
  `geoss/datasets/vehicle_multiview_dataset.py`;
- evaluation: `geoss/eval/render_metrics.py`,
  `geoss/metrics/geometry_metrics.py`, `scripts/eval_geovis_slat.py`,
  `scripts/evaluate_meshfleet_sequence.py`,
  `scripts/analyze_evaluation_suite.py`;
- coordinate/export contracts: `geoss/io/asset_io.py`,
  `geoss/utils/coordinates.py`, `geoss/slat/utils/active_voxel_utils.py`,
  `geoss/models/ray_evidence_sampler.py`;
- VGGT alignment/confidence: `geoss/integration/vggt_geometry_wrapper.py`,
  `geoss/geometry/alignment.py`;
- TRELLIS coupling: `geoss/integration/real_trellis_pipeline.py`,
  `geoss/integration/trellis_ss_hook.py`,
  `geoss/slat/integration/trellis_slat_hook.py`;
- SS/SLAT adapters and masked losses under `geoss/models`, `geoss/slat/models`,
  and `geoss/slat/losses`;
- training/inference launchers under `scripts/train_*`, `scripts/infer_*`, and
  `scripts/launch_meshfleet_multigpu_sequence.py`;
- dataset/evaluation audit tools and protocol-v2 tests.

Phase II adds or changes:

- exact raw-VAE ↔ normalized-flow SLAT conversion in
  `geoss/slat/utils/normalization.py` and both training/inference paths;
- factorized reliability, correction demand, and residual variance in
  `geoss/slat/models/geovis_slat_aggregator.py`,
  `geoss/slat/models/geovis_slat_adapter.py`, and
  `geoss/slat/models/slat_velocity_adapter.py`;
- calibrated factorized-control supervision in
  `geoss/slat/losses/factorized_control_loss.py`;
- differentiable decoded Gaussian supervision in
  `geoss/slat/losses/decoded_asset_loss.py`;
- physically correct TRELLIS Gaussian activation and degree-zero SH color
  conversion in `geoss/renderers/gsplat_renderer.py`;
- differentiable SSIM and removal of invalid cross-camera pixel-variance loss in
  `geoss/losses/render_losses.py`;
- GT-calibrated GeoSS correctness in `scripts/train_sparse_ray_geoss.py` and
  `geoss/losses/occupancy_loss.py`;
- a real weights-only Stage-3→Stage-4 handoff and decoded-supervision config in
  `scripts/train_geovis_slat.py`, `scripts/train_geovis_slat_joint.py`,
  `scripts/launch_meshfleet_multigpu_sequence.py`, and
  `configs/phase2_decoded_asset.yaml`;
- full-dataset distribution profiling in
  `scripts/profile_meshfleet_distribution.py`;
- expanded mathematical/gradient/normalization tests.

The finalized Phase-II implementation also removes padded fake voxels before
calling TRELLIS' sparse teacher, repads teacher predictions by exact coordinate
identity, records sparse indices in saved GeoVis contexts, and rejects a context
whose coordinates differ from the live sampler state. The aggregator's training
reference is now the frozen multi-view teacher's clean-state estimate rather
than the ground-truth SLAT. Periodic checkpoints are retained for validation on
final decoded metrics.

Checkpoint implication: Phase-II SLAT checkpoints have additional demand and
variance heads and declare tensor contract `phase2_factorized_control_v2`.
Phase-I checkpoints do not strictly load into this architecture. This is an
intentional fail-fast incompatibility; partial evaluation with randomly
initialized missing heads would not be scientifically valid. Phase-I generated
assets remain evaluable as a frozen comparison.

## Foundational audit of the actual pipeline

### 1. Images and cameras

The loader returns sRGB `I ∈ [0,1]^{B×N×3×H×W}`, alpha-derived masks
`M ∈ [0,1]^{B×N×1×H×W}`, OpenCV intrinsics `K`, camera-to-world `C`, and
world-to-camera `W=C⁻¹`. RGBA is composited onto a declared background.
Conditioning and evaluation camera sets are disjoint. Pixel coordinates use
`u=fx X/Z+cx`, `v=fy Y/Z+cy`. Camera projection is differentiable with respect
to 3D points, not camera metadata. Cost is `O(BNHW)` I/O.

### 2. VGGT extraction

Frozen VGGT maps all views jointly to patch features, depth, world pointmaps,
camera encodings, and raw `1+exp(r)` confidence. The bounded reliability mapping
is `q=1-1/conf`. The predictions are learned geometric proxies in VGGT's
sequence frame, not canonical MeshFleet geometry. VGGT has global multi-view
attention and approximately quadratic token attention cost. It is frozen; no
final-asset gradient reaches VGGT.

### 3. Coordinate registration

When known cameras are available, a weighted camera-center Sim(3)
`x_canonical=s R x_vggt+t` is estimated and applied to world points. Depth is
recomputed after transformation in the known OpenCV camera. This removes the
old point/ray frame mismatch. A similarity transform cannot correct non-rigid
VGGT depth distortion; pose-degenerate cases are explicitly low trust.

### 4. Sparse anchors

Static grid anchors and dynamic VGGT point/boundary/uncertainty samples form
`A∈[-1,1]^{B×M×3}` with metadata and learned queries. Dynamic selection is
non-differentiable with respect to top-k indices. It preserves observed surfaces
but can underrepresent mirrors, transparent windows, and unseen undersides.

### 5–7. Projection, ray evidence, and aggregation

Every anchor is projected into every view. Sampled mask, RGB/features, predicted
depth residual, signed free-space evidence, validity, and conflict are encoded as
`E∈R^{B×M×N×Ce}`. Attention aggregates along the view axis only, producing
occupancy/free evidence and geometry tokens. Cost is `O(BMNCe)` plus attention
`O(BMN H)`. Geometry is equivariant only to coordinate transforms that are
applied consistently to cameras and anchors.

### 8. Geometry reliability

Positive evidential parameters yield occupancy probability and epistemic proxy
uncertainty. Phase I incorrectly calibrated confidence against the prediction's
distance from its own threshold. Phase II calibrates correctness against sampled
GT occupancy during training. Reliability remains distinct from the downstream
velocity residual magnitude.

### 9. Sparse-structure flow correction

Frozen TRELLIS predicts base velocity `v_ss`. Local k-NN anchor attention predicts
`δv_ss`; the controlled velocity is

`v'_ss = v_ss + α(t) q_ss clip(δv_ss, -τ(t), τ(t))`.

The residual target is the frozen-prior error `v*−v_ss`. Raw and effective
residual losses make the correction direction trainable even with gating.
k-NN distance cost is chunked `O(B L M)` time and `O(B chunk M)` memory. This
stage is supervised in latent space and occupancy; it does not receive decoded
mesh gradients.

### 10–11. SS-to-SLAT and active-voxel projection

TRELLIS active coordinates `i∈{0,…,63}³` map to decoder world coordinates
`x=(i+0.5)/64−0.5`. GeoSS coordinates use `[-1,1]`, so the explicit boundary is
`x_world=x_geoss/2`. Active voxels project through every physical camera. Rows
remain aligned by sparse coordinate; Phase II never truncates active voxels by
default.

### 12. Visibility and appearance evidence

For each active voxel and view, the system samples RGB, VGGT features, alpha,
depth, signed residual, in-bounds state, and occlusion. Visibility is a physical
support proxy. Reflective surfaces violate Lambertian correspondence; their
view disagreement must represent appearance uncertainty rather than geometric
free space.

### 13. Factorized SLAT control

View attention produces condition token `c_l` and three distinct variables:

- evidence reliability `r_l∈[0,1]` from visibility, occlusion, depth support, and
  appearance agreement;
- correction demand `d_l∈[0,1]`, supervised by
  `1−exp(−||v*−v_base||/κ)`;
- residual variance `σ_l²>0`, supervised by heteroscedastic residual NLL.

The adapter predicts direction/magnitude `δv_l`; its effective rule is

`v'_l = v_base,l + β(t) r_l d_l / sqrt(1+σ_l²) clip(δv_l,−τ(t),τ(t))`.

Thus reliable evidence does not itself imply a large correction, and uncertainty
does not specify a correction direction. Frozen TRELLIS remains responsible for
unseen-region prior completion.

Aligned fusion costs `O(BLC)` instead of global `O(BL²)`. Multi-view attention at
each voxel costs `O(BLNH)`.

### 14. TRELLIS flow and latent normalization

MeshFleet `latents/*.npz` contain raw VAE features. TRELLIS flow operates on
`z=(z_raw−μ)/σ` and only denormalizes after sampling. Phase I omitted this
normalization during training, so its frozen teacher was queried off-distribution.
Phase II enforces this boundary and fails on missing/mismatched statistics.

For the TRELLIS path

`x_t=(1−t)x_0+[σ_min+(1−σ_min)t]ε`,
`v*=(1−σ_min)ε−x_0`,

the differentiable clean estimate is

`x̂_0=(1−σ_min)x_t−[σ_min+(1−σ_min)t]v'`.

Training now averages frozen conditional velocities from every conditioning view,
and applies the configured TRELLIS negative branch, CFG strength, and guidance
interval. This matches inference-time MultiDiffusion instead of training the
adapter against a single-image, non-CFG teacher. Variable-length objects are represented
with true sparse rows only; padded batch rows never enter sparse attention. The
reference supplied to the evidence aggregator is

`x̂_0,base=(1−σ_min)x_t−[σ_min+(1−σ_min)t]v_base`,

not the clean target. This matches the Stage-2 generated prior available during
inference and prevents clean-latent leakage into the conditioning branch.

The adapter is injected into the positive CFG branch only. Without correction,
CFG would multiply its residual by `1+g`. The sampler therefore applies the
positive-branch scale `1/(1+g)` whenever CFG is active, giving

`(1+g)(v_pos + δv/(1+g)) - g v_neg = v_CFG + δv`.

The learned residual consequently has the same meaning in training and inference
and remains invariant when the frozen prior's CFG strength changes.

### 15. Decoding and final-asset gradients

The frozen Gaussian decoder maps denormalized sparse `x̂_0` to positions, degree-0
SH color, activated scale/rotation/opacity. Decoder weights remain frozen, but
autograd through the decoder to the adapter is enabled. TRELLIS physical
attributes are used exactly; private raw scale/opacity tensors are forbidden.
Degree-zero SH is converted by `rgb=clamp(0.5+C0 f_dc,0,1)`,
`C0=0.28209479177387814`.

The decoded objective is

`L_asset = λ_rgb ||R(G)−I||_1 + λ_fg ||M⊙(R(G)−I)||_1/||M||_1
          + λ_ssim(1−SSIM) + λ_lpips LPIPS(R(G),I) + λ_mask BCE(A,M)
          + λ_depth L_depth + λ_geo CD(G_xyz,V_gt)`.

The LPIPS network and TRELLIS decoder are frozen and in evaluation mode, but
neither is executed under `torch.inference_mode`, so their Jacobians with respect
to the rendered image/SLAT remain available to the adapter. Exported PLY scale and opacity parameterizations are
tagged explicitly as log-scale/logit; numerical-range guessing is not used for
project-produced assets.

It uses conditioning/training views only. Held-out validation/test renders never
enter training or refinement. Decoder/rasterizer memory dominates; supervision is
scheduled on a small view subset and can be increased when server VRAM permits.

### 16. Conditioning-only refinement

Optional per-object refinement changes decoded Gaussian DC color, opacity, and
optionally scale using conditioning images only. It cannot alter learned test
geometry or use held-out views. It is reported as a separate method because its
runtime and optimization protocol differ from feed-forward inference.

## Architecture decision gate

**Selected: Option B — retain VGGT and TRELLIS, redesign their coupling.**

Evidence for retaining them:

- VGGT provides joint multi-view point/depth/camera features that are expensive
  to reproduce from the available vehicle subset.
- TRELLIS provides a pretrained sparse generative prior and differentiable
  Gaussian/mesh decoders with exportable assets.
- The local sample has 7,996 active SLAT sites with 1,024-D extracted features;
  discarding those priors would require substantially larger full retraining.

Evidence against retaining the previous coupling:

- Stage-3 queried the frozen flow in the wrong latent distribution.
- “Stage-4 joint” called the same adapter-only Stage-3 loop and started from
  scratch; it was neither joint nor decoder-supervised.
- the render proxy was explicitly disabled;
- configured SSIM had no gradient and pixel variance across unrelated cameras
  encouraged texture averaging;
- Gaussian SH coefficients and private unactivated attributes were rendered as
  if they were physical RGB/scale/opacity;
- one scalar mixed reliability and correction gating;
- training used a single-image frozen teacher while inference averaged views;
- the nominal `best` training checkpoint optimized latent loss, not final
  metrics. Phase II therefore archives periodic checkpoints and selects among
  them only with the complete validation protocol.

These are coupling/objective defects, not evidence that the pretrained priors
themselves are unusable. A complete Option-C replacement is therefore not yet
justified. It becomes the next gate only if decoded-supervision Option B fails to
improve the complete validation manifest.

## Local dataset evidence

`outputs/dataset_profile_local` contains the representative profile. The only
local test object has 150 standard renders, 12+12 held-out evaluation renders,
7,996 SLAT/voxel sites, 8 raw latent channels, and 1,024-D precomputed features.
Its raw SLAT mean/std are approximately `0.403/2.321`, reinforcing that raw VAE
features cannot be assumed to be standard-normal flow states. This single object
is structural evidence only, never a population-level conclusion.

## Mandatory remote decision experiment

The first remote ablation isolates the discovered contracts:

1. frozen Phase-I assets re-evaluated with corrected SH/physical renderer;
2. normalized flow training only;
3. normalized flow + multi-view teacher;
4. factorized control without decoded loss;
5. decoded RGB/mask loss;
6. decoded RGB/mask + geometry loss;
7. conditioning-only refinement on the best feed-forward model.

Every row must use the same UID manifest, seed, camera/view protocol, render
resolution, geometry sampling, and completeness accounting. Only validation may
select the winning combination; test is run once after selection.
