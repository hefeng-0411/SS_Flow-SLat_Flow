# Phase II remote training and evaluation runbook

All commands run from `/mnt/sda2/hef/Base/SS_Flow`. They assume four GPUs; change
`--nproc_per_node` and batch limits to match the server. Test evaluation is the
last command and must not be run until validation selects a frozen method.

## 1. Environment and dataset gate

```bash
cd /mnt/sda2/hef/Base/SS_Flow
python -m pip install -r requirements/all.txt
python - <<'PY'
import torch
print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.device_count())
PY

python scripts/inspect_meshfleet_dataset.py \
  --data_root /mnt/sda2/hef/Base/dataset \
  --output_dir outputs/phase2/dataset_audit \
  --splits train,test --validation_percent 10 --split_seed 20260720 \
  --min_train_views 8 --min_eval_views 12 \
  --hash_files --validate_payloads --strict --strict_scope manifests

python scripts/profile_meshfleet_distribution.py \
  --data_root /mnt/sda2/hef/Base/dataset \
  --output_dir outputs/phase2/dataset_profile \
  --splits train,test --images_per_set 8 --hash_meshes
```

`all_discovered_valid=false` is expected when the raw corpus contains incomplete
objects. Stop only if `manifests_valid=false`, train/test overlap is nonzero, a
required manifest is empty, or the selected manifest contains camera overlap.
Do not reuse v2 manifests: they were generated from all discovered UIDs before
validity filtering. The evaluator now requires v3 auditor provenance by default.
Keep incomplete files in place: the auditor assigns UIDs to stage-specific
manifests instead of deleting objects. `train_uids.json`,
`validation_uids.json`, and `test_uids.json` are the conservative, fully
complete manifests. `test_evaluation_uids.json` is the broader protocol-valid
population whose inference inputs and evaluation GT exist even if an unused
training cache is absent; choose one population before experiments and never
change it after seeing model results.

## 2. Reproduce the corrected SS foundation

```bash
python scripts/launch_meshfleet_multigpu_sequence.py \
  --data_root /mnt/sda2/hef/Base/dataset \
  --output_root outputs/phase2/foundation \
  --stage1_train_manifest outputs/phase2/dataset_audit/stage1_train_uids.json \
  --stage2_train_manifest outputs/phase2/dataset_audit/stage2_train_uids.json \
  --meshfleet_split train \
  --vggt_root /mnt/sda2/hef/Base/vggt \
  --trellis_root /mnt/sda2/hef/Base/TRELLIS \
  --trellis_model_path microsoft/TRELLIS-image-large \
  --num_views 8 --image_size 256 --active_tokens 0 \
  --gpus 0,1,2,3 --nproc_per_node 4 \
  --stop_after stage2
```

## 3. Controlled SLAT ablations

Shared arguments:

```bash
COMMON_ARGS="--device cuda \
  --meshfleet_root /mnt/sda2/hef/Base/dataset \
  --meshfleet_split train \
  --train_manifest outputs/phase2/dataset_audit/stage3_train_uids.json \
  --num_views 8 --image_size 256 --active_tokens 0 \
  --vggt_root /mnt/sda2/hef/Base/vggt \
  --vggt_pretrained facebook/VGGT-1B \
  --trellis_root /mnt/sda2/hef/Base/TRELLIS \
  --trellis_model_path microsoft/TRELLIS-image-large \
  --real_train --amp true --amp_dtype bf16 \
  --gradient_checkpointing true --steps 100000 \
  --batch_size 1 --grad_accum_steps 4 \
  --save_every 2000 --visualize_every 1000"
```

Run equal-budget, from-scratch latent ablations:

```bash
torchrun --standalone --nproc_per_node=4 scripts/train_geovis_slat.py \
  --config configs/phase2_ablation_normalized_single_view.yaml \
  --output_dir outputs/phase2/ablation_normalized_single_view $COMMON_ARGS

torchrun --standalone --nproc_per_node=4 scripts/train_geovis_slat.py \
  --config configs/phase2_ablation_multiview_teacher.yaml \
  --output_dir outputs/phase2/ablation_multiview_teacher $COMMON_ARGS

torchrun --standalone --nproc_per_node=4 scripts/train_geovis_slat.py \
  --config configs/phase2_ablation_factorized_control.yaml \
  --output_dir outputs/phase2/ablation_factorized_control $COMMON_ARGS
```

Each run retains `geovis_slat_adapter_step_XXXXXXXX.pt` snapshots. Evaluate
those snapshots on validation (first a fixed deterministic probe subset, then
the complete validation manifest for the top candidates) using the Section 4
command with `--run_stage3 true --run_stage4 false`, the matching
`--config_slat`, and `--slat_checkpoint`. Choose the factorized checkpoint by
complete decoded validation metrics, not by the training-loss `*_best.pt`
alias. Record its exact path as `FACTOR_CKPT` below.

Fine-tune the same selected factorized checkpoint for equal decoded-supervision
budgets. Both branches start from the same checkpoint so the geometry term is the
only intended difference:

```bash
FACTOR_CKPT=outputs/phase2/ablation_factorized_control/geovis_slat_adapter_step_XXXXXXXX.pt
DECODE_ARGS="--device cuda \
  --meshfleet_root /mnt/sda2/hef/Base/dataset \
  --meshfleet_split train \
  --train_manifest outputs/phase2/dataset_audit/stage4_train_uids.json \
  --num_views 8 --image_size 256 --active_tokens 0 \
  --vggt_root /mnt/sda2/hef/Base/vggt \
  --vggt_pretrained facebook/VGGT-1B \
  --trellis_root /mnt/sda2/hef/Base/TRELLIS \
  --trellis_model_path microsoft/TRELLIS-image-large \
  --real_train --amp true --amp_dtype bf16 \
  --gradient_checkpointing true --steps 30000 \
  --batch_size 1 --grad_accum_steps 4 \
  --init_checkpoint $FACTOR_CKPT \
  --save_every 1000 --visualize_every 500"

torchrun --standalone --nproc_per_node=4 scripts/train_geovis_slat_joint.py \
  --config configs/phase2_ablation_decoded_appearance.yaml \
  --output_dir outputs/phase2/ablation_decoded_appearance $DECODE_ARGS

torchrun --standalone --nproc_per_node=4 scripts/train_geovis_slat_joint.py \
  --config configs/phase2_decoded_asset.yaml \
  --output_dir outputs/phase2/final_decoded_asset $DECODE_ARGS
```

If the decoder/rasterizer exceeds VRAM, reduce only per-GPU batch size first;
preserve all active tokens and use gradient accumulation. Record every OOM and
the resulting batch schedule.

The launcher intentionally omits `expandable_segments` from
`PYTORCH_CUDA_ALLOC_CONF`. Some PyTorch/CUDA builds print that the feature is
unsupported and then fail with `!block->expandable_segment_ INTERNAL ASSERT`
when VGGT and TRELLIS coexist. This allocator setting does not affect model
mathematics. Stage 2 retains only TRELLIS' sparse-structure flow and DINO image
encoder on GPU; Stage 3 retains SLAT flow plus DINO; Stage 4 additionally keeps
the Gaussian decoder required for decoded supervision. These are the exact
pretrained modules used by each objective—no token, view, precision, or loss is
removed.

Stage 3/4 run the demand-calibration and heteroscedastic residual objectives in
FP32 inside the BF16 training region. The demand head exposes its pre-sigmoid
logits to `binary_cross_entropy_with_logits`; rendered-alpha and ray losses,
which genuinely have only probability inputs, use a narrowly scoped FP32 BCE.
This keeps the mathematical objectives and all gradients while avoiding
PyTorch's intentional `binary_cross_entropy is unsafe to autocast` failure.
The sequential launcher also finds sibling `stage3_train_uids.json` and
`stage4_train_uids.json` files when earlier stage manifests from the same audit
directory are supplied. Explicit `--stage3_train_manifest` and
`--stage4_train_manifest` arguments remain preferred for recorded production
runs.

## 4. Validation protocol

For every checkpoint, evaluate the complete validation manifest with the same
command template. The example below evaluates the final decoded model; substitute
the appropriate config/checkpoint/output identifier for each ablation.

```bash
python scripts/evaluate_meshfleet_sequence.py \
  --data_root /mnt/sda2/hef/Base/dataset \
  --run_root outputs/phase2/foundation \
  --output_dir outputs/phase2/validation_final_decoded_asset \
  --split train \
  --uid_manifest outputs/phase2/dataset_audit/validation_uids.json \
  --max_samples 0 --num_views 8 --eval_num_views 12 --image_size 256 \
  --conditioning_view_set renders --eval_view_set renders_eval_70 \
  --geometry_samples 100000 --geometry_seed 20260720 \
  --fscore_threshold 0.01 --save_visuals true \
  --run_original_trellis true --run_stage1 true --run_stage2 true \
  --run_stage3 false --run_stage4 true --run_refined_final true \
  --config_slat_joint configs/phase2_decoded_asset.yaml \
  --slat_joint_checkpoint outputs/phase2/foundation/stage4_geovis_slat_joint/geovis_slat_adapter_best.pt \
  --geoss_checkpoint outputs/phase2/foundation/stage1_geoss/geoss_adapter_best.pt \
  --ss_checkpoint outputs/phase2/foundation/stage2_ss_velocity/ss_velocity_adapter_best.pt \
  --vggt_root /mnt/sda2/hef/Base/vggt \
  --trellis_root /mnt/sda2/hef/Base/TRELLIS \
  --gpus 0,1,2,3 --parallel true --overwrite true
```

Selection uses `summary.json -> by_ablation -> official_metrics`, requires
`official_complete=true`, and considers mean, CI95, median, p10/worst cases,
failure count, runtime, VRAM, and saved visuals. Do not select on latent loss.
The sampler keeps the learned SLAT residual invariant to TRELLIS CFG strength;
training reads the same CFG strength/interval from the loaded pipeline.

## 5. Final test, once

Freeze the winning validation config/checkpoints and run:

```bash
python scripts/evaluate_meshfleet_sequence.py \
  --data_root /mnt/sda2/hef/Base/dataset \
  --run_root outputs/phase2/foundation \
  --output_dir outputs/phase2/final_test \
  --split test \
  --uid_manifest outputs/phase2/dataset_audit/test_uids.json \
  --max_samples 0 --num_views 8 --eval_num_views 12 --image_size 256 \
  --conditioning_view_set renders --eval_view_set renders_eval_70 \
  --geometry_samples 100000 --geometry_seed 20260720 \
  --fscore_threshold 0.01 --save_visuals true \
  --run_original_trellis true --run_stage1 true --run_stage2 true \
  --run_stage3 false --run_stage4 true --run_refined_final true \
  --config_slat_joint configs/phase2_decoded_asset.yaml \
  --slat_joint_checkpoint outputs/phase2/foundation/stage4_geovis_slat_joint/geovis_slat_adapter_best.pt \
  --geoss_checkpoint outputs/phase2/foundation/stage1_geoss/geoss_adapter_best.pt \
  --ss_checkpoint outputs/phase2/foundation/stage2_ss_velocity/ss_velocity_adapter_best.pt \
  --vggt_root /mnt/sda2/hef/Base/vggt \
  --trellis_root /mnt/sda2/hef/Base/TRELLIS \
  --gpus 0,1,2,3 --parallel true --overwrite true
```

Do not compare against the supplied target until conditioning views, LPIPS
backbone, background, resolution, CD convention, normalization, and F-score
threshold have been confirmed identical.
