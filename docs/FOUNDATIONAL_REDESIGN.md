# Foundational redesign: evidence, decision, and validation gates

## Empirical diagnosis

The paired 94-object v2 result is decisive:

| Method | PSNR | SSIM | LPIPS | CD | F-score |
|---|---:|---:|---:|---:|---:|
| Original TRELLIS | 21.35969 | 0.91794 | 0.07591 | 0.15265 | 0.07785 |
| Stage 2 | 21.35785 | 0.91793 | 0.07598 | 0.15277 | 0.07760 |
| Stage 3 | 21.35833 | 0.91795 | 0.07602 | 0.15274 | 0.07788 |
| Stage 4 | 21.35913 | 0.91796 | 0.07602 | 0.15274 | 0.07789 |
| Final refinement | 21.07476 | 0.91366 | 0.09526 | 0.15274 | 0.07789 |

Stages 2–4 do not have a practically measurable effect on the final assets.
The final refinement changes only Gaussian DC color, opacity, and optionally
scale; it copies the mesh, cannot improve CD/F-score, and materially damages all
three appearance metrics. These stages are retained only as reproducible
baselines. They are not the primary redesign.

The decoded SLAT geometry objective does not supervise the decoded mesh. It
matches selected high-opacity Gaussian centers to occupancy targets while Stage
3/4 reuse fixed sparse coordinates. The placeholder render-proxy loss is zero.
Consequently, a nonzero latent residual or adapter norm is not evidence of
surface change.

## Dataset and preprocessing findings

MeshFleet/TRELLIS normalizes every asset into the canonical cube
`[-0.5,0.5]^3`. Blender writes OpenGL camera-to-world matrices; the project
loader explicitly converts them to OpenCV cameras with positive camera z. The
same canonical scale and offset are stored with conditioning and evaluation
renders.

TRELLIS voxelization rasterizes the normalized render mesh at `64^3`. This
surface discretization loses geometry below roughly `1/64` of the canonical
extent and is a poor ceiling for wheels, mirrors, arches, trim, and thin
components.

Precomputed DINO features are sampled at ground-truth voxel positions during
dataset preprocessing. Voxels, SS latents, SLAT latents, and these
voxel-indexed features are therefore supervised training products, not legal
test-time observations. Inference datasets now use
`load_3d_modalities=False`, so these products and mesh paths are not even
materialized in an inference sample.

Incomplete objects remain in the corpus. Frozen stage-specific manifests select
UIDs with the modalities required by each stage. UID lookup is exact: failed
objects are never replaced by a later dataset index. Official evaluation uses
one predeclared manifest, counts every failure, and requires
`official_complete=true`.

## Architecture decision

The residual SS/SLAT chain is rejected as the primary architecture. The current
controlled redesign has four independent candidates:

1. **Original TRELLIS** is the immutable learned-prior baseline.
2. **Mask-cropped native multi-image TRELLIS** restores the alpha-bbox square
   crop and 1.2 padding used by TRELLIS preprocessing. The previous tensor path
   silently bypassed it.
3. **Direct visual hull** reconstructs explicit geometry only from conditioning
   masks and calibrated cameras at `160^3`; it copies Original TRELLIS
   appearance byte-for-byte.
4. **VGGT depth-fused hull** conservatively carves the silhouette hull using
   confidence-qualified VGGT world points aligned to the known camera rig. It
   also copies Original TRELLIS appearance byte-for-byte.

This separates appearance, silhouette geometry, and learned depth evidence.
Only components with a paired effect on final held-out metrics will survive.
Original and mask-cropped TRELLIS now run in a native worker that does not load
VGGT or GeoSS. The former baseline loaded and executed both models before
discarding their outputs. Resetting the TRELLIS sampling seed made that work
mathematically irrelevant but costly in latency and VRAM. The native worker
also obeys the evaluator's selected conditioning render set.

Foundation-model conditioning is loaded at 518×518, the native spatial input
of both TRELLIS' DINO image tower and VGGT, while held-out assets are still
rendered and scored at the frozen 256×256 evaluation resolution. The former
path downsampled conditioning images to 256 and then enlarged them to 518,
irreversibly discarding texture and silhouette detail before either foundation
model received the image.

Native multi-view TRELLIS uses MultiDiffusion at both the sparse-structure and
SLAT flows, with the repository-recommended 12 Euler steps and CFG strengths
7.5/3.0 recorded explicitly in provenance. Stochastic view cycling remains
available as a separately named validation ablation; sampler settings are no
longer implicit model-package state.

The native worker also supports deterministic multi-seed generation. It renders
each candidate back into the known conditioning cameras and selects by a
foreground-weighted RGB/SSIM/alpha score without opening evaluation views or
3D targets. The checked run defaults to seed 42 for the first causal ablation;
afterward, `TRELLIS_CANDIDATE_SEEDS=42,3407,20260720` can be validated as an
explicit compute-for-quality ablation on the same frozen UID population.

## Mathematical specification

Let canonical voxel centers be

```text
x ∈ Ω = [-0.5,0.5]^3
```

and let calibrated conditioning camera `v` have projection

```text
p_v(x) = π(K_v [R_v | t_v] x).
```

For alpha mask `M_v`, the visual hull is

```text
H(x) = 1[
  valid_views(x) ≥ V_min
  and
  mean_v 1[M_v(p_v(x)) ≥ τ_mask] ≥ ρ_hull
].
```

No depth, held-out render, mesh, voxel, or latent target enters this
calculation.

VGGT produces per-pixel world points, confidence, and cameras. Its predicted
camera centers are aligned to the known MeshFleet camera centers by a weighted
Sim(3). A point-map pixel is admitted only if the aligned 3D point reprojects
through the known camera to within `ε_reproj` pixels of the same source pixel.
Its weight is

```text
w_v(p) = c_v(p) exp(-e_v(p)^2 / (2 σ_reproj^2)).
```

For a voxel with projected camera depth `z_v(x)` and admitted VGGT surface depth
`d_v(p_v(x))`, observed free space is

```text
F_v(x) = 1[z_v(x) < d_v(p_v(x)) - δ_free].
```

The fused volume retains unsupported hull voxels and removes only repeatedly
observed free space:

```text
G(x) = H(x) and (
  evidence_count(x) < D_min
  or
  mean_evidence_v (1 - F_v(x)) ≥ ρ_depth
).
```

Marching cubes extracts an exportable closed mesh in the same canonical
coordinates used by the evaluator. This is conservative by design: uncertainty
falls back to the silhouette prior instead of inventing empty space.

As a local coordinate-only sanity check, the representative UID
`17a53839...e43a17` produced an 8-view, `80^3` hull occupying 8.15% of the
canonical cube. Its bounds were approximately
`[-0.214,-0.484,-0.198]` to `[0.214,0.5,0.151]`, close to the available
surface-voxel bounds. A diagnostic boundary-voxel comparison gave CD `0.0245`
and F-score `0.2803` at threshold `0.01`. This is not an official result: it
uses voxel centers rather than deterministic mesh-surface samples, one object,
and lower resolution. It is recorded only as evidence that camera conversion,
scale, and axis order are plausible enough for the remote experiment.

The appearance-control candidates copy the Original TRELLIS Gaussian and verify
source/output SHA-256 equality. Their held-out PSNR, SSIM, and LPIPS must
therefore be invariant; a difference is an evaluator determinism or provenance
failure.

## Controlled remote experiment

Run:

```bash
MAX_SAMPLES=2 OVERWRITE=true \
  bash scripts/run_foundational_geometry_ablation.sh

MAX_SAMPLES=0 OVERWRITE=false \
  bash scripts/run_foundational_geometry_ablation.sh
```

The first command is a runtime gate. The second uses the complete frozen
validation manifest. Results are selected from:

```text
summary.json
  by_ablation.<method>.official_complete
  by_ablation.<method>.official_metrics
  paired_vs_original_trellis.<method>
```

Paired deltas use identical UID intersections and improvement-positive signs.
They report mean, median, CI95, p10, worst regression, win rate, and an exact
two-sided sign test. Unmatched successful-subset means are not admissible.

The corrected evaluator is `meshfleet_heldout_v3`. It requires explicit
negative provenance for test-time GT latents, mesh, voxels, and evaluation
views, and it adds foreground-crop metrics. Cached v2 asset evaluations are
invalidated automatically.

The old conditioning refiner is also replaced by a conservative v2 path. It
learns bounded Gaussian residuals from an optimization subset of conditioning
views and selects checkpoints on disjoint, angularly interleaved conditioning
views. Opacity stays frozen by default. A candidate must improve the selection
objective without foreground-L1, SSIM, or mask regression; otherwise the source
Gaussian is restored byte-for-byte. This prevents the previously observed
automatic appearance regression, but remains an ablation rather than a claimed
gain until complete paired validation is available.

## Current status and non-claims

The local Windows machine has no usable PyTorch/CUDA runtime. Source compilation
and shell syntax checks pass, but tensor tests, asset generation, and full
held-out metrics must run on the remote TRELLIS environment. No metric
improvement is claimed until the smoke test and complete paired validation
finish. The final test manifest must remain untouched until the architecture
and hyperparameters are frozen on validation.
