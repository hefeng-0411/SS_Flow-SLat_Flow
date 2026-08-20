#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/mnt/sda2/hef/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97}"
RUN_ROOT="${RUN_ROOT:-outputs/phase2/foundation}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/phase2/validation_final_decoded_asset}"
UID_MANIFEST="${UID_MANIFEST:-outputs/phase2/dataset_audit/validation_evaluation_uids.json}"
PHYSICAL_GPUS="${PHYSICAL_GPUS:-0,1}"
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-6}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
REFINEMENT_SOURCE_STAGE="${REFINEMENT_SOURCE_STAGE:-original_trellis}"
TRELLIS_CANDIDATE_SEEDS="${TRELLIS_CANDIDATE_SEEDS:-42}"
OVERWRITE="${OVERWRITE:-false}"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

exec python scripts/evaluate_meshfleet_sequence.py \
  --data_root "${DATA_ROOT}" \
  --run_root "${RUN_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --split train \
  --uid_manifest "${UID_MANIFEST}" \
  --max_samples "${MAX_SAMPLES}" \
  --num_views 8 \
  --eval_num_views 12 \
  --image_size 256 \
  --foundation_conditioning_image_size 518 \
  --trellis_multi_image_mode multidiffusion \
  --trellis_ss_steps 12 --trellis_ss_cfg_strength 7.5 \
  --trellis_slat_steps 12 --trellis_slat_cfg_strength 3.0 \
  --trellis_candidate_seeds "${TRELLIS_CANDIDATE_SEEDS}" \
  --conditioning_view_set renders \
  --eval_view_set renders_eval_70 \
  --geometry_samples 100000 \
  --geometry_seed 20260720 \
  --fscore_threshold 0.01 \
  --save_visuals true \
  --run_original_trellis true \
  --run_trellis_mask_cropped true \
  --run_direct_visual_hull true \
  --run_vggt_depth_fused true \
  --run_stage1 true \
  --run_stage2 true \
  --run_stage3 true \
  --run_stage4 true \
  --run_refined_final true \
  --refinement_source_stage "${REFINEMENT_SOURCE_STAGE}" \
  --refinement_validation_views 2 \
  --refinement_validation_every 10 \
  --refinement_validation_patience 5 \
  --refinement_min_relative_improvement 0.002 \
  --refinement_max_color_delta 0.08 \
  --refinement_optimize_opacity false \
  --config_slat_joint configs/phase2_decoded_asset.yaml \
  --geoss_checkpoint "${RUN_ROOT}/stage1_geoss/geoss_adapter_best.pt" \
  --ss_checkpoint "${RUN_ROOT}/stage2_ss_velocity/ss_velocity_adapter_best.pt" \
  --slat_checkpoint "${RUN_ROOT}/stage3_geovis_slat/geovis_slat_adapter_best.pt" \
  --slat_joint_checkpoint "${RUN_ROOT}/stage4_geovis_slat_joint/geovis_slat_adapter_best.pt" \
  --vggt_root /mnt/sda2/hef/Base/vggt \
  --trellis_root /mnt/sda2/hef/Base/TRELLIS \
  --gpus "${PHYSICAL_GPUS}" \
  --parallel true \
  --scheduler_mode stage_major \
  --auto_workers_per_gpu true \
  --workers_per_gpu 1 \
  --max_workers_per_gpu "${MAX_WORKERS_PER_GPU}" \
  --min_free_vram_gb 8 \
  --stage_vram_gb "original_trellis=14,trellis_mask_cropped=14,direct_visual_hull=4,vggt_depth_fused=18,stage1_geoss_context=13,stage2_geoss_ss=16,stage3_geovis_slat=16,stage4_geovis_slat_joint=16,final_conditioning_refined=8,asset_evaluation=4" \
  --worker_admission_warmup_seconds 30 \
  --worker_timeout_seconds 3600 \
  --worker_stall_timeout_seconds 300 \
  --worker_terminate_grace_seconds 15 \
  --worker_monitor_interval_seconds 2 \
  --worker_cpu_threads 4 \
  --timeout_retry_limit 1 \
  --oom_retry_limit 2 \
  --overwrite "${OVERWRITE}" \
  "$@"
