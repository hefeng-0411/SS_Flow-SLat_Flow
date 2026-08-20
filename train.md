Below is the current command guide for the project as it stands.

Assumed server layout from your configs:

```bash
# Dataset root
/mnt/sda2/hef/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97

# Repos
/mnt/sda2/hef/Base/SS_Flow-SLat_Flow
/mnt/sda2/hef/Base/TRELLIS
/mnt/sda2/hef/Base/vggt
```

**1. Enter Project**
```bash
cd /mnt/sda2/hef/Base/SS_Flow-SLat_Flow
```

**2. Recommended Full Multi-GPU Training**
Uses GPUs `4,5,6,7`, 4 processes, adaptive batch from config, early stop, and Stage 1 SS velocity training automatically launches Stage 2 SLAT training on convergence.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
  scripts/train_sparse_ray_ss_velocity.py \
  --config configs/real_train_ss.yaml
# Stage 1: SS velocity residual training
# Uses configs/real_train_ss.yaml
# Adaptive batch + early stopping enabled in config
# On plateau convergence, config workflow launches Stage 2:
# scripts/train_geovis_slat.py --config configs/real_train_slat_only.yaml
```

**3. Stage 1 Only: SS Training**
Multi-GPU:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
  scripts/train_sparse_ray_ss_velocity.py \
  --config configs/real_train_ss.yaml
# Output checkpoint:
# outputs/real_train_ss/ss_velocity_adapter_last.pt
# Best checkpoint:
# outputs/real_train_ss/ss_velocity_adapter_best.pt
```

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=4 python \
  scripts/train_sparse_ray_ss_velocity.py \
  --config configs/real_train_ss.yaml
# Runs on one A800 only
```

Resume Stage 1:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
  scripts/train_sparse_ray_ss_velocity.py \
  --config configs/real_train_ss.yaml \
  --resume outputs/real_train_ss/ss_velocity_adapter_last.pt
# Resume optimizer + adapter state from previous checkpoint
```

**4. Stage 2 Only: SLAT Training**
Run this manually if you disable auto-stage2 or want to retrain SLAT separately.

Multi-GPU:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
  scripts/train_geovis_slat.py \
  --config configs/real_train_slat_only.yaml
# Output checkpoint:
# outputs/real_train_slat/geovis_slat_adapter_last.pt
# Best checkpoint:
# outputs/real_train_slat/geovis_slat_adapter_best.pt
```

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=4 python \
  scripts/train_geovis_slat.py \
  --config configs/real_train_slat_only.yaml
```

Resume Stage 2:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
  scripts/train_geovis_slat.py \
  --config configs/real_train_slat_only.yaml \
  --resume outputs/real_train_slat/geovis_slat_adapter_last.pt
```

**5. Optional High-Level Multi-Stage Launcher**
This launcher runs staged training with probing/restart logic. It includes `stage1`, `stage2`, `slat`, and `slat_joint`.

```bash
python scripts/launch_meshfleet_multigpu_sequence.py \
  --data_root /mnt/sda2/hef/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97 \
  --gpus 4,5,6,7 \
  --nproc_per_node 4 \
  --vggt_root /mnt/sda2/hef/Base/vggt \
  --vggt_pretrained facebook/VGGT-1B \
  --trellis_root /mnt/sda2/hef/Base/TRELLIS \
  --trellis_model_path microsoft/TRELLIS-image-large \
  --output_root outputs/meshfleet_full_4gpu_sequence
# Runs sequential multi-stage training with per-stage batch probing
```

Start from Stage 2 only:

```bash
python scripts/launch_meshfleet_multigpu_sequence.py \
  --data_root /mnt/sda2/hef/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97 \
  --gpus 4,5,6,7 \
  --nproc_per_node 4 \
  --vggt_root /mnt/sda2/hef/Base/vggt \
  --trellis_root /mnt/sda2/hef/Base/TRELLIS \
  --trellis_model_path microsoft/TRELLIS-image-large \
  --output_root outputs/meshfleet_full_4gpu_sequence \
  --start_at stage2
```

**6. Full Joint Training**
⚠️ `configs/real_train_full_joint.yaml` exists, but the direct joint script currently exposed in `scripts/train_sparse_ray_joint.py` does not appear to consume that config as a complete real pipeline in the same way as the SS/SLAT dedicated scripts. Prefer the high-level launcher for joint-style staged training.

If you still want to run the available joint entry:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
  scripts/train_sparse_ray_joint.py \
  --config configs/sparse_ray_joint.yaml \
  --device cuda \
  --meshfleet_root /mnt/sda2/hef/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97 \
  --meshfleet_split train \
  --vggt_root /mnt/sda2/hef/Base/vggt \
  --vggt_pretrained facebook/VGGT-1B \
  --trellis_root /mnt/sda2/hef/Base/TRELLIS \
  --trellis_model_path microsoft/TRELLIS-image-large
# ⚠️ Verify outputs before treating this as the main recommended path
```

**7. Inference / Generation**
SS + TRELLIS decode inference using Stage 1 checkpoint:

```bash
CUDA_VISIBLE_DEVICES=4 python \
  scripts/infer_sparse_ray_geoss_ss.py \
  --config configs/real_infer.yaml \
  --ss_adapter_checkpoint outputs/real_train_ss/ss_velocity_adapter_last.pt \
  --output_dir outputs/real_infer_ss
# Generates TRELLIS decoder outputs with GeoSS SS adapter
# Uses dataset/test split from configs/real_infer.yaml unless overridden
```

Infer a specific MeshFleet index:

```bash
CUDA_VISIBLE_DEVICES=4 python \
  scripts/infer_sparse_ray_geoss_ss.py \
  --config configs/real_infer.yaml \
  --ss_adapter_checkpoint outputs/real_train_ss/ss_velocity_adapter_last.pt \
  --meshfleet_index 0 \
  --output_dir outputs/real_infer_ss_idx0
# meshfleet_index selects object from test split
```

Override dataset/input path:

```bash
CUDA_VISIBLE_DEVICES=4 python \
  scripts/infer_sparse_ray_geoss_ss.py \
  --config configs/real_infer.yaml \
  --ss_adapter_checkpoint <PATH_TO_STAGE1_CKPT> \
  --meshfleet_root /mnt/sda2/hef/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97 \
  --meshfleet_split test \
  --output_dir outputs/real_infer_custom
```

SLAT inference entry point:

```bash
CUDA_VISIBLE_DEVICES=4 python \
  scripts/infer_geovis_slat.py \
  --config configs/real_infer.yaml \
  --slat_adapter_checkpoint outputs/real_train_slat/geovis_slat_adapter_last.pt \
  --geovis_context <PATH_TO_GEOVIS_CONTEXT_NPZ> \
  --input <PATH_TO_INPUT_IMAGE_OR_IMAGE_DIR> \
  --output_dir outputs/real_infer_slat
# ⚠️ Requires --geovis_context produced by an SS/GeoVis context stage
```

**8. Evaluation**
Evaluate SS occupancy prediction:

```bash
python scripts/eval_sparse_ray_geoss.py \
  --real_eval \
  --prediction <PATH_TO_PREDICTION_NPZ> \
  --gt_occ <PATH_TO_GT_OCC_PT> \
  --output_dir outputs/eval_sparse_ray_geoss
# prediction npz should contain occupancy, commonly key "occ"
# gt_occ can be a tensor or dict with "gt_occ"
```

Evaluate generated SLAT / render / Gaussian outputs:

```bash
python scripts/eval_geovis_slat.py \
  --real_eval \
  --input_dir outputs/real_infer_slat \
  --prediction <PATH_TO_PRED_RENDER_NPZ> \
  --gt_render <PATH_TO_GT_RENDER_NPZ> \
  --gaussian_ply outputs/real_infer_ss/asset_gaussian.ply \
  --output_dir outputs/eval_geovis_slat
# prediction and gt_render npz should contain "rgb"; optional "mask"
# gaussian_ply enables Gaussian statistics
```

Evaluate geometry point metrics if you have point npz files:

```bash
python scripts/eval_geovis_slat.py \
  --real_eval \
  --input_dir outputs/real_infer_slat \
  --prediction <PATH_TO_PRED_RENDER_NPZ> \
  --gt_render <PATH_TO_GT_RENDER_NPZ> \
  --pred_points <PATH_TO_PRED_POINTS_NPZ> \
  --gt_points <PATH_TO_GT_POINTS_NPZ> \
  --output_dir outputs/eval_geovis_slat_geometry
# point npz files should contain key "points"
```

Ablation summary over multiple run folders:

```bash
python scripts/eval_ablation_table.py \
  --runs_dir outputs/ablation_runs \
  --output_json outputs/ablation_summary.json \
  --output_csv outputs/ablation_summary.csv
# Scans run directories and writes unified JSON/CSV table
```

**9. Qualitative / Visualization**
Visualize SS/GeoSS demo artifacts:

```bash
python scripts/visualize_sparse_ray_geoss.py \
  --output_dir outputs/vis_sparse_ray_geoss
# Writes visualize_sparse_ray_geoss_demo.ply
```

Visualize SLAT inference outputs:

```bash
python scripts/visualize_geovis_slat.py \
  --input_dir outputs/real_infer_slat \
  --output_dir outputs/vis_geovis_slat
# Consumes slat_visibility_debug.npz and original_slat.npz if present
```

Plot training curves:

```bash
python scripts/plot_training_metrics.py \
  --output_root outputs/real_train_ss \
  --report_dir outputs/reports/ss_training \
  --smooth 0.9 \
  --formats png,pdf
# Generates metric plots from JSONL training logs
```

For the staged launcher outputs:

```bash
python scripts/plot_training_metrics.py \
  --output_root outputs/meshfleet_full_4gpu_sequence \
  --report_dir outputs/reports/full_sequence \
  --smooth 0.9 \
  --formats png,pdf
```

**10. Checkpoint Placeholders**
Use these common checkpoint paths:

```bash
# Stage 1 SS adapter
outputs/real_train_ss/ss_velocity_adapter_last.pt
outputs/real_train_ss/ss_velocity_adapter_best.pt

# Stage 2 SLAT adapter
outputs/real_train_slat/geovis_slat_adapter_last.pt
outputs/real_train_slat/geovis_slat_adapter_best.pt

# Launcher sequence outputs
outputs/meshfleet_full_4gpu_sequence/stage1_geoss/geoss_adapter_last.pt
outputs/meshfleet_full_4gpu_sequence/stage2_ss_velocity/ss_velocity_adapter_last.pt
outputs/meshfleet_full_4gpu_sequence/stage3_geovis_slat/geovis_slat_adapter_last.pt
outputs/meshfleet_full_4gpu_sequence/stage4_geovis_slat_joint/geovis_slat_adapter_last.pt
```

**11. Fast Iteration Commands**
Short SS smoke training on 4 GPUs:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
  scripts/train_sparse_ray_ss_velocity.py \
  --config configs/real_train_ss.yaml \
  --steps 20 \
  --save_every 10 \
  --output_dir outputs/debug_ss_20step
# Fast train-loop check with real data/model
```

Short SLAT smoke training:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
  scripts/train_geovis_slat.py \
  --config configs/real_train_slat_only.yaml \
  --steps 20 \
  --save_every 10 \
  --output_dir outputs/debug_slat_20step
```

Dry-run only, no real metrics:

```bash
python scripts/train_sparse_ray_ss_velocity.py \
  --config configs/dry_run_debug.yaml \
  --dry_run true \
  --output_dir outputs/dry_run_ss
# Mock/synthetic only; not valid for paper metrics
```