# SS_Flow reconstruction audit, protocol v2, and remote runbook

## Status and scientific scope

This repository now contains a leakage-checked evaluation path and corrected
coordinate/data contracts. Final trained validation/test numbers are deliberately
left pending: the local machine has no usable PyTorch environment, no TRELLIS or
VGGT weights, an empty local training split, and no mount of `/mnt/sda2`. Legacy
numbers below are retained only as diagnostic evidence and are not comparable to
the supplied target.

The primary remote paths are:

- project: `/mnt/sda2/hef/Base/SS_Flow`
- TRELLIS source: `/mnt/sda2/hef/Base/TRELLIS`
- VGGT source: `/mnt/sda2/hef/Base/vggt`
- complete MeshFleet/TRELLIS dataset: `/mnt/sda2/hef/Base/dataset`

## Dataset provenance and modality roles

The vehicle assets originate from Objaverse-XL and are selected by the SHA256
identifiers in `meshfleet_with_vehicle_categories_df.csv`. The MeshFleet pipeline
downloads source assets, normalizes heterogeneous formats through `oxl_processing`
(`export_glb.py`, `fbx_processing.py`), renders them with Blender
(`objaverse_xl_batched_renderer.py`, `blender_script.py`), filters poor assets with
the quality-classifier/embedding pipeline, and packages TRELLIS features, sparse
structure latents, structured latents, voxels, and optional WebDataset shards.

Protocol-v2 assigns the available folders these roles:

| Folder | Role |
|---|---|
| `mesh_normalized` | Canonical ground-truth geometry for CD/F-score only |
| `renders` | Multi-view training/conditioning images |
| `renders_cond` | Explicit image-to-3D conditioning view(s) |
| `renders_eval_70`, `renders_eval_90` | Held-out appearance evaluation only |
| `voxels` | Training-only occupancy/geometry supervision |
| `features` | Precomputed conditioning features; not a held-out target |
| `latents` | Training-only TRELLIS structured-latent targets |
| `ss_latents` | Training-only TRELLIS sparse-structure targets |

The local representative test object has all nine modalities. It has 150 ordinary
renders, one available conditional image (24 cameras declared), 12 eval-70 views,
and 12 eval-90 views. Its conditioning and evaluation camera sets do not overlap.
The local `train` directory is empty; full-dataset training must therefore run on
the server.

## Findings in the supplied evaluation suite

Thirty legacy JSON files cover eight objects for the first three methods and only
six objects for Stage 4. All 30 files claim official status, but none contains CD
or F-score. They also predate the held-out protocol and contain mislabeled proxy
metrics. The complete machine-readable audit is written by
`scripts/analyze_evaluation_suite.py`.

| Legacy diagnostic method | N | PSNR | SSIM | LPIPS | CD | F-score |
|---|---:|---:|---:|---:|---:|---:|
| Original TRELLIS | 8 | 15.2603 | 0.1933 | 0.2357 | absent | absent |
| Stage 2 GeoSS | 8 | 15.2614 | 0.1945 | 0.2353 | absent | absent |
| Stage 3 GeoVis-SLAT | 8 | 15.2063 | 0.1919 | 0.2349 | absent | absent |
| Stage 4 joint | 6 | 14.5669 | 0.2179 | 0.2374 | absent | absent |

Paired against original TRELLIS, Stage 2 changes mean PSNR by only +0.0010 dB;
Stage 3 changes it by -0.0541 dB; and Stage 4 changes it by -0.0556 dB on its six
available pairs. These are statistically and practically negligible changes.
`test_000001` is the worst object by PSNR. The worst legacy original views include
view 6 of that object (9.6268 dB) and view 3 (9.9462 dB). Mean performance declines
strongly with later view indices, indicating view-dependent failures rather than
uniform decoder noise.

## Root causes corrected in code

1. The legacy SSIM implementation used whole-image statistics rather than the
   standard local Gaussian window. It is now an 11x11, sigma-1.5 local SSIM.
2. `masked_LPIPS` was pixel L1 under an LPIPS name. Both full and masked LPIPS now
   use the learned VGG LPIPS network on `[-1,1]`; foreground L1 has its own name.
3. RGB-vector cosine and `1-|pred-gt|` proxies are no longer reported as DINO or
   multi-view consistency.
4. The evaluator used the same ordinary renders for conditioning and scoring.
   It now requires `renders_eval_70` or `renders_eval_90`, checks camera overlap,
   composites RGBA onto an explicit background, and records frame IDs.
5. TRELLIS `Gaussian.save_ply` rotates its internal z-up asset by +90 degrees about
   x for public y-up export. The old evaluator rendered that PLY with unrotated
   MeshFleet cameras. Protocol-v2 exactly inverts both positions and wxyz
   orientations before rendering.
6. TRELLIS mesh decoding returns `MeshExtractResult`, not an object with `export`.
   Inference now always saves `asset_mesh_internal.ply` without public-axis rotation;
   this is the geometry metric input. Textured GLB export is separately optional.
7. TRELLIS SLAT decoder coordinates live in world `[-0.5,0.5]`, while GeoSS
   occupancy uses `[-1,1]`. Old ray/SLAT projection doubled object extent. The code
   now converts explicitly at the projection boundary and uses exact voxel centers.
8. VGGT confidence is `1+exp(raw)`, not a probability. Clamping it to `[0,1]`
   collapsed every confidence to one. It is now mapped to `1-1/conf`, preserving
   order in `[0,1]`, with raw confidence retained for provenance.
9. The old VGGT alignment fit a point cloud to a target constructed from its own
   bounding box, then returned unchanged dataset cameras. Points and rays were in
   different frames. Alignment now estimates a weighted Sim(3) between VGGT and
   known conditioning camera centers, transforms world points, and recomputes depth
   in the known camera frame. Pose-less fallback is explicitly capped at low trust.
10. Stage-3 inference read test-object ground-truth SLAT coordinates/features and
    paired them by row order with predicted coordinates. That leakage path has been
    removed. Inference now requires Stage-2 predicted coordinates/SLAT and records
    `test_time_ground_truth_latents_used=false`.
11. TRELLIS itself received one image while VGGT received the multi-view sequence.
    Both SS and SLAT sampling now use all conditioning images with adapter-aware
    multidiffusion. Geometry adapters are applied to positive CFG predictions only.
12. The previous SLAT global attention mixed already aligned voxel rows and required
    quadratic memory. The production adapter uses per-voxel aligned fusion and
    propagates explicit padding masks through control and auxiliary losses.
13. Dataset failures previously substituted a later UID, and evaluation indices
    were clamped. Both behaviors now fail loudly. Full evaluation defaults to the
    complete split, reports missing/failed objects, and accepts deterministic UID
    manifests.

## Protocol-v2 metric definitions

- PSNR: per-view sRGB `[0,1]` RGB MSE, followed by per-object and dataset means.
- SSIM: local 11x11 Gaussian window, sigma 1.5, data range 1.
- LPIPS: `lpips.LPIPS(net="vgg")` on sRGB tensors mapped to `[-1,1]`.
- CD: one half the sum of mean nearest-neighbor Euclidean distances in each
  direction, using 100,000 deterministically sampled surface points per mesh.
- Chamfer-L2: the analogous symmetric mean squared distance, reported separately.
- F-score: harmonic mean of precision/recall at Euclidean distance `<0.01` in the
  shared canonical TRELLIS/MeshFleet frame.
- No per-object ICP, recentering, rescaling, or best-view selection is allowed.
- Means, medians, standard deviations, p10/worst cases, per-view results, failures,
  runtime, and sampled peak VRAM are persisted.

The supplied target can only be compared after confirming that its CD convention
and F-score threshold are identical. Protocol-v2 will not rename or tune metric
definitions to manufacture agreement.

## Remote commands

Install the project dependencies in the same Python environment as TRELLIS/VGGT:

```bash
cd /mnt/sda2/hef/Base/SS_Flow
python -m pip install -r requirements/all.txt
```

Audit the complete dataset and create deterministic train/validation/test manifests:

```bash
python scripts/inspect_meshfleet_dataset.py \
  --data_root /mnt/sda2/hef/Base/dataset \
  --output_dir outputs/dataset_audit_full \
  --splits train,test \
  --validation_percent 10 \
  --split_seed 20260720 \
  --strict
```

Train all adapter stages on every training-manifest object and every active SLAT
voxel (`--active_tokens 0`). Adjust GPU count to the server:

```bash
python scripts/launch_meshfleet_multigpu_sequence.py \
  --data_root /mnt/sda2/hef/Base/dataset \
  --output_root outputs/meshfleet_protocol_v2 \
  --train_manifest outputs/dataset_audit_full/train_uids.json \
  --meshfleet_split train \
  --vggt_root /mnt/sda2/hef/Base/vggt \
  --trellis_root /mnt/sda2/hef/Base/TRELLIS \
  --trellis_model_path microsoft/TRELLIS-image-large \
  --num_views 8 --image_size 256 --active_tokens 0 \
  --gpus 0,1,2,3 --nproc_per_node 4
```

Run leakage-free validation first. This command evaluates all validation UIDs and
includes the conditioning-only Gaussian refinement as a separately named method:

```bash
python scripts/evaluate_meshfleet_sequence.py \
  --data_root /mnt/sda2/hef/Base/dataset \
  --run_root outputs/meshfleet_protocol_v2 \
  --output_dir outputs/meshfleet_protocol_v2/validation_protocol_v2 \
  --split train \
  --uid_manifest outputs/dataset_audit_full/validation_uids.json \
  --max_samples 0 \
  --num_views 8 --eval_num_views 12 --image_size 256 \
  --conditioning_view_set renders --eval_view_set renders_eval_70 \
  --geometry_samples 100000 --fscore_threshold 0.01 \
  --run_refined_final true --refinement_steps 150 \
  --vggt_root /mnt/sda2/hef/Base/vggt \
  --trellis_root /mnt/sda2/hef/Base/TRELLIS \
  --gpus 0,1,2,3
```

After selecting checkpoints/hyperparameters on validation only, run the complete
test manifest once:

```bash
python scripts/evaluate_meshfleet_sequence.py \
  --data_root /mnt/sda2/hef/Base/dataset \
  --run_root outputs/meshfleet_protocol_v2 \
  --output_dir outputs/meshfleet_protocol_v2/test_protocol_v2 \
  --split test \
  --uid_manifest outputs/dataset_audit_full/test_uids.json \
  --max_samples 0 \
  --num_views 8 --eval_num_views 12 --image_size 256 \
  --conditioning_view_set renders --eval_view_set renders_eval_70 \
  --geometry_samples 100000 --fscore_threshold 0.01 \
  --run_refined_final true --refinement_steps 150 \
  --save_visuals true \
  --vggt_root /mnt/sda2/hef/Base/vggt \
  --trellis_root /mnt/sda2/hef/Base/TRELLIS \
  --gpus 0,1,2,3
```

Generate the legacy-output audit at any time with:

```bash
python scripts/analyze_evaluation_suite.py \
  --input_dir outputs/evaluation_suite_metrics \
  --output_dir outputs/evaluation_suite_analysis
```

## Required final result table

The remote evaluator writes scalar CSV/JSON summaries and per-object/per-view
records. Populate this table only from `validation_protocol_v2/summary.json` and
`test_protocol_v2/summary.json`; never copy the legacy diagnostics into it.

| Method | Split | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CD ↓ | F-score ↑ | Runtime | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Original multi-view TRELLIS | validation/test | pending remote run | pending | pending | pending | pending | emitted | emitted |
| Corrected GeoSS Stage 2 | validation/test | pending remote run | pending | pending | pending | pending | emitted | emitted |
| Corrected aligned GeoVis-SLAT | validation/test | pending remote run | pending | pending | pending | pending | emitted | emitted |
| Final conditioning-refined asset | validation/test | pending remote run | pending | pending | pending | pending | emitted | emitted |

## Remaining limitations

- No trustworthy post-correction metric can be produced on the local machine.
- The target paper/protocol's exact CD aggregation and F-score threshold have not
  been supplied; comparison must remain conditional until those are confirmed.
- Conditioning-only Gaussian refinement updates view-independent DC color and
  opacity (optionally scale). It improves observed texture/silhouette without
  evaluation leakage, but cannot invent unseen high-frequency texture.
- Textured GLB baking is implemented as an opt-in export because TRELLIS's bake is
  substantially slower than raw canonical mesh/PLY evaluation.
- Any architecture/hyperparameter change must be retained only after the complete
  validation manifest improves jointly and worst-case regressions are acceptable.
