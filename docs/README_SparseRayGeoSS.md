# Sparse-Ray GeoSS Adapter

Sparse-Ray GeoSS is an SS-only adapter for TRELLIS sparse structure flow. It uses VGGT only as a multi-view geometry evidence source, then queries sparse 3D anchors in canonical space, aggregates ray-level occupied/free-space evidence, and injects a confidence-gated velocity residual into TRELLIS SS Flow.

Core formula:

```text
v_geo = v_base + alpha(t) * token_confidence * clip_tau(t)(Delta_v_geo)
```

This repository extension intentionally does not modify TRELLIS SLAT, decoders, 3DGS, mesh refinement, or VGGT source code.

Pipeline:

```text
Multi-view RGB
-> VGGT camera/depth/pointmap/features
-> Sparse 3D anchors
-> Ray-level occupied/free-space evidence
-> Cross-view evidence aggregation
-> geo_tokens + confidence
-> TRELLIS SS Flow velocity residual
-> geometry-controlled Sparse Structure
```

Core shapes:

```text
images [B,N,3,H,W]
anchor_xyz [B,M,3]
view_tokens [B,M,N,C_e]
geo_tokens [B,M,C]
ss_latent_tokens [B,L,C_ss]
v_base/delta_v_geo/v_geo [B,L,C_ss]
token_confidence [B,L,1]
```

Dry-run commands:

```bash
python scripts/train_sparse_ray_geoss.py --config configs/sparse_ray_geoss.yaml --dry_run true
python scripts/train_sparse_ray_ss_velocity.py --config configs/sparse_ray_ss_velocity.yaml --dry_run true
python scripts/train_sparse_ray_joint.py --config configs/sparse_ray_joint.yaml --dry_run true
python scripts/infer_sparse_ray_geoss_ss.py --input examples/car_views --dry_run true
```

Two-step training smoke checks:

```bash
python scripts/train_sparse_ray_geoss.py --config configs/sparse_ray_geoss.yaml --steps 2 --device cpu --output_dir outputs/stage1_train_2step
python scripts/train_sparse_ray_ss_velocity.py --config configs/sparse_ray_ss_velocity.yaml --steps 2 --device cpu --output_dir outputs/stage2_train_2step
python scripts/train_sparse_ray_joint.py --config configs/sparse_ray_joint.yaml --steps 2 --device cpu --output_dir outputs/joint_train_2step
```

MeshFleet_TRELLIS local smoke checks:

```bash
python scripts/train_sparse_ray_geoss.py ^
  --config configs/sparse_ray_geoss.yaml ^
  --meshfleet_root D:/VsCode/MVG/Base/MeshFleet_TRELLIS ^
  --meshfleet_split test ^
  --meshfleet_category sdvas ^
  --num_views 4 ^
  --image_size 64 ^
  --meshfleet_occ_resolution 16 ^
  --steps 2 ^
  --device cpu ^
  --output_dir outputs/meshfleet_stage1_2step

python scripts/train_sparse_ray_ss_velocity.py ^
  --config configs/sparse_ray_ss_velocity.yaml ^
  --meshfleet_root D:/VsCode/MVG/Base/MeshFleet_TRELLIS ^
  --meshfleet_split test ^
  --meshfleet_category sdvas ^
  --num_views 4 ^
  --image_size 64 ^
  --meshfleet_occ_resolution 16 ^
  --steps 2 ^
  --device cpu ^
  --output_dir outputs/meshfleet_stage2_2step

python scripts/infer_sparse_ray_geoss_ss.py ^
  --config configs/sparse_ray_geoss.yaml ^
  --meshfleet_root D:/VsCode/MVG/Base/MeshFleet_TRELLIS ^
  --meshfleet_split test ^
  --meshfleet_category sdvas ^
  --num_views 4 ^
  --image_size 64 ^
  --meshfleet_occ_resolution 16 ^
  --geoss_checkpoint outputs/meshfleet_stage1_2step/geoss_adapter_last.pt ^
  --ss_adapter_checkpoint outputs/meshfleet_stage2_2step/ss_velocity_adapter_last.pt ^
  --device cpu ^
  --output_dir outputs/meshfleet_infer_smoke
```

The current local `D:/VsCode/MVG/Base/MeshFleet_TRELLIS/train` directory is
empty. Use `test/sdvas` for local verification, or reconstruct the train shards
before starting full training.

## VGGT Checkpoint Configuration

`VGGTGeometryWrapper` supports three modes:

- `mock=True`: deterministic local fallback for dry-runs and tests.
- local checkpoint: pass `vggt_root` and `checkpoint`; checkpoint dictionaries with `state_dict`, `model`, or `model_state_dict` are accepted.
- pretrained interface: pass `pretrained_name`, e.g. `facebook/VGGT-1B`, when the local environment already has access to the model. The code does not require network access for normal dry-run or synthetic training.

CLI example:

```bash
python scripts/train_sparse_ray_geoss.py \
  --config configs/sparse_ray_geoss.yaml \
  --srn_root /path/to/SRN_ROOT \
  --vggt_root D:/VsCode/MVG/Base/vggt \
  --vggt_checkpoint /path/to/vggt_checkpoint.pt \
  --steps 1000
```

## Current Real vs Fallback Status

Real implemented: anchor queries, projection, ray occupied/free rules, cross-view aggregation, evidential confidence, TRELLIS SS velocity wrapper, disabled identity, synthetic two-step training, synthetic eval metrics.

Real local data path implemented: MeshFleet_TRELLIS reconstructed folders with
renders, alpha masks, transforms, voxel PLY, TRELLIS SS latents, DINO/TRELLIS
features, and normalized mesh paths.

Fallback when assets are absent: VGGT mock mode, mock TRELLIS base flow,
synthetic dataset batches, unsupported ablation baselines. Fallbacks are logged
in command outputs and are not reported as complete baselines.

## TRELLIS Checkpoint Configuration

`train_sparse_ray_ss_velocity.py` can load a real TRELLIS SS Flow checkpoint through:

```bash
python scripts/train_sparse_ray_ss_velocity.py \
  --config configs/sparse_ray_ss_velocity.yaml \
  --trellis_root D:/VsCode/MVG/Base/TRELLIS \
  --trellis_model_path /path/to/trellis_sparse_structure_flow
```

Without `--trellis_model_path`, the script uses an explicit mock SS Flow and reports
`mock_trellis` in logs. The GeoSS hook itself remains external and does not change
TRELLIS checkpoint keys.
