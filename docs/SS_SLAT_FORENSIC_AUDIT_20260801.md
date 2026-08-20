# SS–SLAT forensic audit and training-topology decision

Date: 2026-08-01 (Asia/Shanghai)

Status: the two-step failure, stage separation, gradient paths, dataset population,
runtime contracts, and one-object branch learnability were verified locally. A
full held-out decoded-asset improvement is **not** established. The current
adapter chain remains a diagnostic baseline, not the recommended primary asset
architecture.

## 1. Executive verdict

1. The SLAT command performed two updates because
   `scripts/train_geovis_slat.py` defined `--steps` with the smoke-test default
   `2`, while `configs/real_train_slat_only.yaml` supplied no `steps` value. The
   config-default merge therefore retained `2`. The outer loop is directly
   bounded by `end_step`; one `scaler.step(optimizer)` is attempted per outer
   iteration. The two events were optimizer-update attempts, not epochs,
   validation events, checkpoints, flow time steps, or exhaustion of the
   dataloader.
2. Git history shows that the `default=2` and step-less SLAT config arrived in
   the initial commit `68067b3` on 2026-07-08. There is no design rationale or
   data-dependent derivation. It was a retained smoke-test default.
3. SS and SLAT are legitimately separated at native TRELLIS' hard support
   boundary:

   ```python
   coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()
   ```

   The comparison, `argwhere`, and integer cast produce a discrete sparse index
   set. Native TRELLIS uses this at
   `/mnt/sda/hf/MVG/Base/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:191`;
   this repository reproduces it at
   `geoss/integration/real_trellis_pipeline.py:354`.
4. The present trainers are more disconnected than that mathematical boundary
   requires. SLAT training reads cached teacher `feats` and `coords` from NPZ,
   never loads an SS adapter/checkpoint, and constructs its “SS” context from
   those cached coordinates. Therefore

   ```text
   ∂ L_SLAT / ∂ θ_SS = 0
   ```

   exactly: `θ_SS` is not in the SLAT graph. The current SLAT checkpoint also
   trains on teacher support while inference uses SS-predicted support.
5. The strongest deployable native topology is **strict sequential training at
   an explicit discrete boundary**, with hash-addressed checkpoint pairing and
   decoded held-out selection. Alternating predicted-support refresh is a
   research ablation, not a production recommendation yet: the repository has
   no valid target SLAT values for predicted-only coordinates. “Joint”
   end-to-end gradients are false without changing the discrete model; a soft
   occupancy/STE relaxation would be a new, unvalidated architecture.
6. Real one-object checks prove that both adapter graphs are connected and can
   improve their same-sample latent residual relative to frozen TRELLIS. The
   gains are small. They do not prove decoded mesh or 3DGS improvement.
7. Existing paired 94-object v2 evidence already shows no practically measurable
   effect from Stages 2–4. This audit therefore does not recommend scaling the
   residual chain until decoded validation falsifies that historical result.

The actual operators are therefore

```text
S_thetaS(I,C,G_V) = z_SS                    # dense 16^3 SS latent
X_SS             = (z_SS, A(z_SS))
A(z_SS)          = int(argwhere(D_SS(z_SS) > 0))
A_thetaL(X_SS,I,C,G_V) = z_SLAT[A(z_SS)]    # values on a fixed sparse domain
D_T(X_SS,X_SLAT) = (M, G_3DGS).
```

For the implemented trainer, replace `A(z_SS)` by cached `A_teacher`; hence
`A_thetaL` has no `thetaS` argument and
`d L_final / d thetaS` through the current SLAT-training graph is exactly zero.
The decoder can transmit continuous gradients to SLAT values on fixed support,
but cannot restore a derivative through creation or deletion of integer support.

## 2. Exact two-step root cause

### 2.1 Historical evidence

At initial commit `68067b3`:

```text
scripts/train_geovis_slat.py:331
    parser.add_argument("--steps", type=int, default=2)

configs/real_train_slat_only.yaml
    # no steps entry
```

The current pre-fix config merge read `cfg.get("steps") == None`. Because the
CLI value still equalled its parser default, nothing replaced it.

### 2.2 Control flow

The trainer computes:

```text
end_step = steps                         if steps_are_total
           start_step + steps            otherwise

while step < end_step:
    step += 1
    ... forward / backward ...
    scaler.step(optimizer)                unless gradients are non-finite
```

`next_from_loader` restarts the iterator after `StopIteration`, so dataloader
exhaustion cannot terminate training. `drop_last=False`. There is no epoch cap,
`max_steps`, Lightning trainer, validation limit, or scheduler-controlled stop.
The early-stop controller was not the cause; before this audit its YAML boolean
was not even mapped into the SLAT CLI namespace.

### 2.3 Why SS did not show the same symptom

The SS parser also historically defaulted to two, but
`configs/real_train_ss.yaml` explicitly set `steps: 1000`, and the SS config
mapping applied it. Hence the shown two-rank SS command targeted 1,000 update
attempts while the four-rank SLAT command retained two.

### 2.4 Runtime-log finding

No prior `train_geovis_slat.jsonl`, `train_sparse_ray_ss_velocity.jsonl`,
`geovis_slat_adapter_last.pt`, or `ss_velocity_adapter_last.pt` existed under
`/mnt/sda/hf` for the reported commands. The source/config/history evidence is
decisive, but claims about those missing historical logs cannot be made.

## 3. Dataset and split audit

The full payload-validating audit is in
`outputs/forensics_20260801/dataset_audit/summary.json`.

### 3.1 Raw population

| Population | Train | Test |
|---|---:|---:|
| Render directories / discovered UIDs | 1,365 | 251 |
| Voxels + SS latents + SLAT latents + features | 1,357 | 249 |
| Strictly valid under requested payload/render contract | 1,030 | 203 |
| Train/test UID overlap | 0 | 0 |

Eight train and two test renders lack the four training asset modalities. The
raw train intersection used by the original unmanifested commands is 1,357.

### 3.2 Frozen leakage-free manifests

With seed `20260801`, deterministic 10% validation selection, at least eight
usable primary views, and payload validation:

| Manifest | UIDs |
|---|---:|
| Stage-1 train | 1,205 |
| Stage-2 SS train | 1,205 |
| Stage-3 SLAT train | 1,205 |
| Stage-4 decoded train | 1,205 |
| Validation evaluation | 152 |
| Test evaluation | 251 |

The production configs now use the Stage-2 and Stage-3 manifests. This prevents
validation leakage and avoids silently repeating views from weak primary render
payloads.

### 3.3 Payload contracts observed

Representative real archives:

| Payload | Shape / dtype | Semantics |
|---|---|---|
| SS `mean` | `[8,16,16,16]`, FP32 after load | dense TRELLIS SS clean latent; training-only teacher |
| SLAT `feats` | `[N_A,8]`, FP32 after load | sparse clean structured latent; training-only teacher |
| SLAT `coords` | `[N_A,3]`, integer | active 64³ support; training-only teacher |
| DINO patchtokens | `[N_A,1024]`, source FP16 | voxel-indexed preprocessing product; not legal inference input |
| RGB views | `[V,3,H,W]`, FP32 `[0,1]` | legal conditioning observation |
| masks | `[V,1,H,W]`, FP32 | legal conditioning observation |
| `K`, `c2w`, `w2c` | `[V,3,3]`, `[V,4,4]`, `[V,4,4]` | known calibrated camera rig |

Variable SLAT lengths are kept as lists by collation and padded to the largest
`N_A` in each batch. The audited causal UID has 10,639 active tokens.

## 4. Exact call graphs

### 4.1 SS training

```text
CLI / YAML
  -> validate real mode and initialize rank/device
  -> build manifest-filtered MeshFleet loader
  -> compute and enforce update/exposure budget
  -> load TRELLIS SS flow + DINO image conditioner (frozen)
  -> load VGGT and GeoSS context adapter (frozen)
  -> construct SSVelocityAdapter (trainable)
  -> for each outer update:
       cached x0_SS [B,8,16,16,16]
       epsilon ~ N(0,I), t ~ U(0,1)
       x_t = (1-t)x0 + [sigma_min + (1-sigma_min)t]epsilon
       v_target = (1-sigma_min)epsilon - x0
       frozen TRELLIS v_base(x_t,1000t,image_cond)
       frozen VGGT -> depth/pointmap/features/confidence/camera
       frozen GeoSS -> 4096 geometry anchors/tokens/confidence
       SSVelocityAdapter -> raw/clipped/gated delta and v_geo
       FP32 loss reductions -> backward -> gradient checks -> AdamW
       JSONL + atomic async checkpoint
```

There is no SS decoder, mesh decoder, Gaussian decoder, renderer, held-out
validation pass, or asset metric in this loop.

### 4.2 SLAT training

```text
CLI / YAML
  -> validate real mode and initialize rank/device
  -> build manifest-filtered MeshFleet loader
  -> compute and enforce update/exposure budget
  -> load TRELLIS SLAT flow + DINO image conditioner (frozen)
  -> load VGGT (frozen)
  -> construct GeoVisSLATAdapter (trainable)
  -> for each outer update and microstep:
       cached teacher feats [N_A,8] and coords [N_A,3]
       pad -> x0_raw [B,L,8], indices [B,L,3], valid [B,L,1]
       normalize with TRELLIS SLAT mean/std
       build_ss_slat_context(indices)  # no SS model/checkpoint
       epsilon ~ N(0,I), t ~ U(0,1)
       x_t and v_target by conditional flow matching
       frozen VGGT refresh on the exact training views
       frozen multi-view CFG-matched TRELLIS v_base on fixed coords
       GeoVis projection -> RGB/mask/depth/pointmap/feature evidence
       evidence aggregation -> gated SLAT residual and v_geo
       latent + view + appearance + visibility + control losses
       optional decoded Gaussian loss (disabled in real_train_slat_only)
       backward -> gradient checks/clipping -> AdamW
       JSONL + atomic async checkpoint
```

`scripts/train_geovis_slat_joint.py` calls this same SLAT loop; it does not add
an SS graph. `scripts/train_sparse_ray_joint.py` runs stages sequentially. These
names must not be interpreted as end-to-end joint optimization.

### 4.3 Native inference

```text
conditioning images
  -> DINO conditioning tokens
  -> SS flow sampling: z_SS
  -> SS decoder occupancy logits
  -> (logits > 0) -> argwhere -> int coords       [gradient boundary]
  -> SLAT flow sampling on those fixed coords
  -> GS / radiance-field / mesh decoder
  -> 3D asset
```

The native pipeline's public `run` is also decorated with `@torch.no_grad()`;
that is appropriate for inference and independent of the structural discrete
boundary.

### 4.4 Source/function trace by transition

| Transition | SS implementation | SLAT implementation |
|---|---|---|
| CLI/config | `scripts/train_sparse_ray_ss_velocity.py:main/_apply_config_defaults` | `scripts/train_geovis_slat.py:main/_apply_config_defaults` |
| dataset/eligibility | `MeshFleetTrellisDataset`, `_build_meshfleet_loader` | `MeshFleetTrellisDataset`, `build_real_loader` |
| distributed sampler/collation | `geoss/utils/distributed.py:build_dataloader`, `meshfleet_collate_fn` | same loader utility and collation |
| VGGT forward | `VGGTGeometryWrapper.forward` under frozen/no-grad execution | `VGGTGeometryWrapper.forward` under frozen/no-grad execution |
| TRELLIS teacher target | cached SS NPZ `mean` | cached SLAT NPZ `feats/coords` |
| frozen base velocity | `trellis_ss_base_velocity` | `trellis_slat_base_velocity` with multi-view CFG |
| learned forward | `GeoSSTrellisSSWrapper -> SSVelocityAdapter` | `GeoVisSLATAdapter.forward` |
| loss | inline CFM residual/velocity/prior loss | `compute_losses` plus optional `DecodedAssetSupervisor` |
| AMP/backward/reduction | autocast + scaler/backward; DDP reducer when `P>1` | autocast + scaler/backward; accumulation and DDP reducer |
| optimizer | AdamW, one attempted step per outer iteration | AdamW, one attempted step after `A` microsteps |
| scheduler | none; evidence controller may contract LR | none; evidence controller may contract LR |
| validation | no held-out pass in stage loop | no held-out pass in stage loop |
| checkpoint/stop | `_save_velocity_checkpoint`, `EarlyStopper` | `_save_slat_checkpoint`, `EarlyStopper` |

Both dataloaders restart after exhaustion. SS uses no gradient accumulation in
this entry point; SLAT uses the configured `grad_accum_steps`. Cached latent
targets and projected preprocessing features are training-only. Images,
calibrated cameras, and online VGGT predictions are legal at inference.

## 5. Major tensor/device/gradient contracts

| Tensor | Typical shape | Dtype/device in training | Requires grad | Inference legality |
|---|---|---|---|---|
| images | `[B,8,3,128,128]` | FP32 CUDA | no | legal |
| masks | `[B,8,1,128,128]` | FP32 CUDA | no | legal |
| cameras | `[B,8,3,3]`, `[B,8,4,4]` | FP32 CUDA | no | legal |
| VGGT depth | `[B,8,1,128,128]` | FP32 CUDA | detached | legal predicted evidence |
| VGGT pointmap | `[B,8,3,128,128]` | FP32 CUDA | detached | legal predicted evidence |
| VGGT dense/tokens | `[B,8,C,h,w]` or tokens | FP32 CUDA | detached | legal predicted evidence |
| SS clean latent | `[B,8,16,16,16]` | FP32 CUDA | no | **training target only** |
| SS noisy latent | same | BF16 activation / FP32 contract | input edge | sampled online |
| SS tokens | `[B,4096,8]` | BF16/FP32 CUDA | adapter edge | internal |
| SS base velocity | `[B,4096,8]` | FP32 CUDA | detached | frozen prior |
| geometry anchors | `[B,4096,3]` plus `[B,4096,256]` | FP32 CUDA | detached | predicted |
| SS adapter residual | `[B,4096,8]` | BF16/FP32 CUDA | yes | learned correction |
| SLAT clean values | `[B,L,8]` | FP32 CUDA | no | **training target only** |
| SLAT indices | `[B,L,3]` | INT64 CUDA | never | teacher during current training; predicted at inference |
| sparse TRELLIS input | `[sum L_b,8]`, coords `[sum L_b,4]` | FP32 + INT32 CUDA | features only | internal |
| SLAT base velocity | `[B,L,8]` | FP32 CUDA | detached | frozen prior |
| GeoVis evidence | `[B,L,8,C]`-family tensors | BF16/FP32 CUDA | yes through adapter sampling/fusion | predicted |
| SLAT adapter residual | `[B,L,8]` | BF16/FP32 CUDA | yes | learned correction |
| decoded Gaussians, optional | decoder-dependent `G` rows | FP32 CUDA | decoder frozen; grad to SLAT features | legal only when decoder branch enabled |

TRELLIS active lattice centers are mapped as

```text
x = (index + 0.5) / 64 - 0.5
```

in `geoss/slat/utils/active_voxel_utils.py:7-24`. Stored sparse coordinate order
must remain TRELLIS-native; batch coordinates are prepended only when creating a
`SparseTensor`.

## 6. Gradient paths and mathematical reason for separation

### 6.1 Within-stage paths

SS:

```text
L_SS -> v_geo -> delta_clipped -> delta_head
     -> local attention -> latent_proj / geo_proj -> θ_SS_adapter
```

SLAT:

```text
L_SLAT -> v_slat_geo -> gated residual -> velocity adapter
       -> evidence aggregator -> view sampler/projector -> θ_SLAT_adapter
```

VGGT, the TRELLIS image conditioner, both base flow models, and the frozen GeoSS
context network are detached/frozen. Optional decoded supervision freezes the
GS decoder weights but retains autograd from rendered losses to SLAT features.

### 6.2 Across-stage path

Let the SS adapter produce `z_S(θ_S)` and let

```text
A(θ_S) = int(argwhere(D_SS(z_S(θ_S)) > 0)).
```

SLAT features exist only on the support `A`. The hard comparison and index
creation do not have a useful Jacobian. Even before that issue, the current
trainer uses cached `A_teacher` and `x0_SLAT_teacher`; it never evaluates
`z_S(θ_S)`. Thus the implemented graph is

```text
L_SLAT = L_SLAT(θ_L ; A_teacher, x0_SLAT_teacher, images, cameras)
```

and not a function of `θ_S`.

### 6.3 What can and cannot be differentiated

- Dense SS flow, SS decoder logits before threshold, SLAT feature flow on fixed
  support, the frozen GS decoder's continuous outputs, and Gaussian rendering
  can be differentiated conditionally.
- Hard occupancy thresholding, `argwhere`, integer sparse indices, sparse
  support insertion/removal, top-k Gaussian selection, and mesh topology/faces
  are discrete.
- FlexiCubes vertices/SDF/deformation can carry gradients conditional on a
  selected topology; topology changes are not smoothly differentiable.

## 7. Loss system actually implemented

SS uses

```text
L_SS = MSE(delta_effective, delta_target)
     + lambda_raw MSE(delta_raw, delta_target)
     + lambda_v L_velocity
     + lambda_p L_prior.
```

SLAT uses

```text
L_SLAT = L_flow_raw + L_flow_effective
       + 0.25 L_view + 0.25 L_appearance
       + 0.20 L_visibility_confidence
       + lambda_control L_factorized_control
       + lambda_v L_velocity + lambda_p L_prior
       + L_decoded_asset.
```

In `real_train_slat_only.yaml`, `L_decoded_asset = 0`: the Gaussian decoder and
renderer are not loaded. There is no mesh supervision. The decoded Stage-4
configuration can supervise Gaussian RGB/foreground/SSIM/LPIPS/mask/depth and a
Gaussian-center/occupancy geometry proxy, but it still does not supervise the
decoded mesh.

This audit added the missing same-sample controls:

```text
gain_SS   = MSE(0, delta_target) - MSE(delta_effective, delta_target)
gain_SLAT = MSE(0, delta_target) - MSE(delta_effective, delta_target).
```

Positive gain proves a correction helps the exact sampled target relative to the
frozen base; it is not an asset-quality metric.

No conventional learning-rate scheduler exists. The evidence-rate controller
may issue an LR multiplier intervention, but checkpoint selection currently
uses training telemetry, not held-out decoded assets.

## 8. Exact update arithmetic

The loops are update-bounded and cycle forever over the loader. For diagnostic
epoch arithmetic with `drop_last=False`:

```text
samples_per_rank = ceil(N / P)
batches_per_rank = ceil(samples_per_rank / B)
updates_per_sampler_epoch = ceil(batches_per_rank / A)
effective_global_batch = P * B * A
nominal_presentations = remaining_updates * effective_global_batch
nominal_passes = nominal_presentations / N
```

Here `N` is the eligible object count, `P` ranks, `B` microbatch/rank, and `A`
gradient-accumulation steps. The correct epoch-capped remaining-update formula
would be `max(0, min(E*U_epoch, S_max) - R)`, not the malformed expression in
the directive. These scripts have no `E` cap.

### 8.1 Original commands, no manifest (`N=1,357`)

| Stage | P | B | A | Hard updates | U/epoch | Presentations | Nominal passes |
|---|---:|---:|---:|---:|---:|---:|---:|
| SS | 2 | 1 | 1 | 1,000 | 679 | 2,000 | 1.474 |
| SLAT | 4 | 1 | 1 | **2** | 340 | **8** | **0.005895** |

SLAT touched at most 0.5895% of eligible objects. Adaptive growth requires eight
successful low-utilization steps, so a two-step run cannot even change batch
size.

### 8.2 Corrected manifest topology (`N=1,205`)

| Stage | P | B | A | Hard updates | U/epoch | Padding/epoch | Nominal passes |
|---|---:|---:|---:|---:|---:|---:|---:|
| SS | 2 | 1 | 1 | 1,000 | 603 | 1 object | 1.660 |
| SLAT | 4 | 1 | 1 | 1,000 | 302 | 3 objects | 3.320 |

`steps_are_total: true` now makes resume semantics unambiguous. Every log and
checkpoint carries configured steps, resume start, target, remaining update
attempts, dataset size, effective batch, padding, and exposure arithmetic.
Statistical early stopping is deferred until the minimum exposure is reached;
non-finite/zero-gradient failures and explicit time limits remain immediately
terminal.

## 9. Real causal experiments completed

UID:
`3f887ef6642a1ba1ffa095e0f8cb8312c047384846e8b94cc02a1969ad42fefa`.

Both runs used real cached MeshFleet payloads, real frozen TRELLIS, real frozen
VGGT, eight views, BF16 autocast, and no mock/synthetic fallback.

### 9.1 SS

| Result | Value |
|---|---:|
| Total optimizer updates | 25 |
| Measured same-sample causal updates | 5 |
| Positive / negative / tie gains | 5 / 0 / 0 |
| Mean causal residual gain | `5.8323e-6` |
| Median causal residual gain | `5.6475e-6` |
| Critical gradients finite and >0 | 25/25 updates |
| Skipped optimizer updates | 0 |
| Peak allocated / reserved VRAM | 9.646 / 13.191 GiB |
| Checkpoint | 2,463,730 bytes |
| SHA-256 | `04a7688c488a2be1b4ccf89f4d6c2c1b866b6c2b43a3e4e205a13109362fab76` |

This passes the adapter connectivity/causal latent-residual gate. The effect is
tiny and does not establish changed occupancy or a better decoded asset.

### 9.2 SLAT

| Result | Value |
|---|---:|
| Total optimizer updates | 100 |
| Positive / negative / tie gains | 68 / 29 / 3 |
| Mean causal residual gain | `3.1400e-4` |
| Median causal residual gain | `1.7993e-6` |
| Last-20 positive / negative | 17 / 3 |
| Last-20 median gain | `1.5199e-6` |
| First-5 / last-5 total loss means | 0.65309 / 0.38961 |
| Critical gradients finite and >0 | 100/100 updates |
| Skipped optimizer updates | 0 |
| Checkpoint | 17,488,330 bytes |
| SHA-256 | `87bc608598a7e072ad9a00e4abc88f93b80b2f8e9e13229d36798ded2a95c481` |

This passes the fixed-support latent-flow connectivity/learnability gate after a
longer run. Its checkpoint explicitly states teacher-only support, no upstream SS
checkpoint, zero cross-stage gradient, and train/inference support mismatch.

### 9.3 Produced diagnostics

- raw JSONL logs for every update;
- atomic SS and SLAT checkpoints;
- SLAT confidence and active-support PLY files;
- visibility and velocity NPZ diagnostics;
- per-run preflight budget/provenance JSON;
- `outputs/forensics_20260801/one_object_checkpoint_pair.json` with checkpoint
  hashes and topology status.

The pair manifest correctly reports
`diagnostic_pair_only_slat_was_teacher_support_trained`; it does not pretend the
SLAT checkpoint consumed the SS checkpoint.

## 10. Asset evidence and non-claims

The current environment lacks `gsplat`, `lpips`, `kornia`, and PyTorch3D in the
TRELLIS environment. The production latent trainers do not need them after
removing a stale Kornia requirement, but decoded Gaussian supervision requires
at least `gsplat` and its renderer. No decoded one-object asset comparison was
therefore run, and no new PSNR/SSIM/LPIPS/Chamfer/F-score claim is made.

Historical paired 94-object v2 results from `docs/FOUNDATIONAL_REDESIGN.md` are:

| Method | PSNR | SSIM | LPIPS | CD | F-score |
|---|---:|---:|---:|---:|---:|
| Original TRELLIS | 21.35969 | 0.91794 | 0.07591 | 0.15265 | 0.07785 |
| Stage 2 | 21.35785 | 0.91793 | 0.07598 | 0.15277 | 0.07760 |
| Stage 3 | 21.35833 | 0.91795 | 0.07602 | 0.15274 | 0.07788 |
| Stage 4 | 21.35913 | 0.91796 | 0.07602 | 0.15274 | 0.07789 |
| Final refinement | 21.07476 | 0.91366 | 0.09526 | 0.15274 | 0.07789 |

Stages 2–4 had no practical asset effect, and final refinement hurt appearance.
The current v3 audit also says all 30 cached records are non-official, so they
cannot replace a fresh complete paired validation.

The required final table can only be populated with that historical paired
evidence; unmeasured fields are deliberately not fabricated:

| Method | Completion | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CD ↓ | F-score ↑ | Runtime | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original TRELLIS (historical v2) | 94 paired | 21.35969 | 0.91794 | 0.07591 | 0.15265 | 0.07785 | unreported | unreported |
| Current SS-only | latent probe only | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unreported | 9.646 GiB allocated |
| Current SS→SLAT | disconnected latent probe | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unreported | unmeasured |
| Alternating/joint candidate | not validly constructible yet | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |
| Final optimized system | not trained | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |

## 11. Topology decision

### Selected: strict sequential native topology with explicit boundary

1. Train SS adapter against the frozen SS prior.
2. Select SS by held-out decoded occupancy/surface metrics, not training loss.
3. Freeze and hash the SS checkpoint.
4. Generate a versioned support cache from that checkpoint for diagnostic
   support-agreement evaluation.
5. Train SLAT on a mathematically valid fixed support. Teacher-support training
   must remain explicitly labelled teacher-only unless valid targets for
   predicted support are constructed.
6. Pair SS and SLAT checkpoints by hash and validate the complete decoded chain
   on identical held-out UIDs/views/seeds.

### Rejected as current production choices

- **Alternating block optimization:** potentially useful, but predicted-only
  support cells have no ground-truth SLAT feature targets. Implementing it now
  would require an explicit union/intersection/free-space target rule and could
  teach arbitrary zero latents.
- **Multi-rate joint optimization:** no SS-to-SLAT derivative exists through
  native integer support, so this is merely scheduling two independent blocks.
- **Differentiable relaxation:** a soft occupancy field, STE, or probabilistic
  sparse support changes the model distribution and decoder interface. It is a
  research project, not a safe correction.

For final asset quality, the empirical primary recommendation remains the
foundational redesign: native multi-image TRELLIS plus explicit calibrated
silhouette/VGGT geometry candidates, with paired held-out asset selection. The
SS/SLAT adapters should survive only if they produce a measurable paired decoded
gain.

## 12. Freezing and fine-tuning decision

- Keep VGGT fully frozen. It supplies geometric evidence; end-to-end tuning a
  1B foundation model on 1,205 vehicles is high-risk and unnecessary to test the
  adapter hypothesis.
- Keep TRELLIS image conditioning and both flow backbones frozen during the
  causal adapter phase.
- Keep decoder weights frozen while allowing decoder autograd to SLAT features
  in decoded supervision.
- Do not unfreeze a full TRELLIS stage until the adapter shows nonzero decoded
  asset Jacobian/effect and stable held-out improvement. If justified, unfreeze
  only the late SLAT flow block or decoder output head with a much smaller LR and
  a separate ablation.

## 13. Hardware and performance findings

Host at audit time:

- four NVIDIA RTX A6000 GPUs, 49,140 MiB each;
- no GPU index 4–7, so the shown SLAT command is invalid on this host;
- all GPU-to-GPU links report `SYS` (PCIe/host path), not NVLink;
- GPUs 0–2 were occupied by unrelated workloads during the audit; GPU 3 alone
  was used for bounded causal checks;
- Python 3.10.20, PyTorch 2.4.0+cu118, CUDA runtime 11.8, cuDNN 9.1,
  xFormers 0.0.27, spconv 2.3.8.

Existing good practices: BF16 activations, FP32 reductions, frozen-base
inference mode, activation checkpointing, local KNN chunking, pinned workers,
atomic async checkpoint publication, and adaptive batch/OOM rollback.

Remaining performance issues:

1. SLAT batches pad variable sparse token counts to batch maximum; packed
   offsets would reduce wasted attention/projection work.
2. Standard `DistributedSampler(drop_last=False)` pads one SS or three SLAT UIDs
   per corrected sampler epoch. The padding is now logged; removing it safely
   requires uneven-input DDP join semantics or a deterministic global stream.
3. The four A6000s cross NUMA/host links. Rank/process/data-worker CPU affinity
   should follow GPU NUMA locality.
4. No conventional schedule is configured; decoded validation, not a more
   elaborate scheduler, is the first missing quality control.

## 14. Reference-project reuse decision

The project
`/mnt/sda/hf/MVG/Base/MeshFleet/second_jet_contact_measure` demonstrates:

- deterministic UID streams without padded duplicate IDs;
- packed sparse coordinates/values with offsets and validity masks;
- pinned host batches and asynchronous transfer;
- exact data cursors and RNG state on resume;
- atomic `fsync` + rename checkpoint commits and hashes.

Recommended reuse: deterministic global UID cursor, packed sparse offsets,
explicit validity/provenance, and exact resume/checkpoint manifests.

Do not copy: second-jet physics, halo exchange, solver states, spatial process
grids, and unrelated metric-tensor machinery.

| Dimension | Current SS_Flow | `second_jet_contact_measure` | Decision |
|---|---|---|---|
| manifests | stage-specific JSON UID sets after this audit | deterministic global UID stream | retain manifests; adopt exact cursor semantics later |
| sparse batching | SLAT padding to batch maximum | packed values/coords plus offsets | packed offsets are relevant |
| VGGT/TRELLIS | core reconstruction evidence/backbones | absent/not analogous | no mechanism to copy |
| stages/losses | SS and SLAT CFM residual stages | physics solver stages/objectives | not transferable |
| checkpoint handoff | atomic checkpoints plus new hashes/provenance | atomic fsync/rename with exact cursor/RNG | adopt stronger cursor/RNG manifest |
| precision/distribution | BF16 learned blocks, FP32 geometry/reductions, DDP | mixed sparse/distributed mechanics | preserve current numerical contract |
| error/memory handling | OOM rollback and adaptive batch | explicit validity, pinned batches, recovery | validity/pinning are relevant |

## 15. Corrections implemented

1. Removed implicit two-step defaults from real SS/SLAT CLIs; real training now
   requires an explicit CLI/config budget.
2. Added initial-topology budget arithmetic and a minimum-dataset-pass fail-fast
   guard, with an explicit smoke-test override.
3. Added minimum-exposure gating for nonfatal statistical early stopping.
4. Made `steps_are_total`, manifests, VGGT settings, early-stop settings, and
   runtime settings actually map from configs.
5. Fixed CLI precedence so explicit boolean values equal to parser defaults are
   not overwritten by YAML.
6. Corrected stale `/mnt/sda2/...` paths and removed the invalid automatic
   `CUDA_VISIBLE_DEVICES=4,5,6,7` handoff.
7. Wired leakage-free Stage-2/Stage-3 manifests.
8. Added local DINO torch-hub resolution so a populated cache does not query
   GitHub for the default branch.
9. Canonicalized single-process CUDA devices to indexed `cuda:N`, fixing false
   device-contract failures.
10. Fixed nested VGGT config mapping in SLAT.
11. Fixed the SS dry run to supply spatial anchors required by local attention.
12. Added same-sample frozen-base causal gains, update/support provenance, and
    hash-addressed pair manifests.
13. Corrected decoder telemetry: an available pipeline no longer claims a
    decoder was loaded/executed.

## 16. Validation completed

- 26 focused budget/runtime/control/early-stop tests passed in the final run.
- SS synthetic dry run passed on CPU and exercised local attention.
- SLAT synthetic dry run passed on CPU with 64 active tokens.
- Full payload audit completed over train and test.
- Real SS: 25/25 updates, no skips, all critical gradients positive.
- Real SLAT: 100/100 updates, no skips, all critical gradients positive.
- Checkpoints and diagnostic artifacts were written and hashed.

The complete repository suite result is **130 passed, 7 failed**. All newly
affected tests pass.
The seven failures are pre-existing repository inconsistencies outside this
patch: two static-string policies contradicted existing dry-run/comment text;
one checkpoint test assumes a particular mismatch-key order; and four older
ray-coordinate, SS–SLAT map-shape, and local-attention fallback contracts disagree
with their current implementations. They are not represented as passing and
were not silently rewritten as part of this topology audit.

The one-object SLAT summary was generated before the decoder-telemetry repair
and therefore contains the stale field `decoder_enabled: true`; the config,
residency record, and loss fields show that no decoder was loaded or executed.
The repaired behavior is covered by a focused regression test.

## 17. Remaining production gates

The mission is not complete until all of the following pass:

1. install/verify the decoded renderer dependencies in the TRELLIS environment;
2. run decoded one-object SS and SLAT causal checks;
3. train on the frozen 1,205-UID manifests on free GPUs;
4. select checkpoints on the 152-UID validation-evaluation manifest;
5. generate both mesh and 3DGS assets with fixed views/seeds;
6. compare against original TRELLIS with paired PSNR, SSIM, LPIPS, silhouette,
   Chamfer, F-score, worst-case regression, and sign-test summaries;
7. freeze architecture/hyperparameters before opening the 251-UID test set.

Until then, the accurate conclusion is: the two-step bug is fixed, both adapter
branches are learnable in their own latent objectives, native separation is
mathematically justified, current cross-stage training is disconnected, and no
new final-asset improvement has been demonstrated.
