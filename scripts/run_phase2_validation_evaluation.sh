#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/mnt/sda2/hef/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97}"
RUN_ROOT="${RUN_ROOT:-outputs/phase2/foundation}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/phase2/validation_final_decoded_asset}"
UID_MANIFEST="${UID_MANIFEST:-outputs/phase2/dataset_audit/validation_uids.json}"
PHYSICAL_GPUS="${PHYSICAL_GPUS:-0,1}"
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-6}"
OVERWRITE="${OVERWRITE:-false}"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

exec python scripts/evaluate_meshfleet_sequence.py \
  --data_root "${DATA_ROOT}" \
  --run_root "${RUN_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --split train \
  --uid_manifest "${UID_MANIFEST}" \
  --max_samples 0 \
  --num_views 8 \
  --eval_num_views 12 \
  --image_size 256 \
  --conditioning_view_set renders \
  --eval_view_set renders_eval_70 \
  --geometry_samples 100000 \
  --geometry_seed 20260720 \
  --fscore_threshold 0.01 \
  --save_visuals true \
  --run_original_trellis true \
  --run_stage1 true \
  --run_stage2 true \
  --run_stage3 true \
  --run_stage4 true \
  --run_refined_final true \
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
  --stage_vram_gb "original_trellis=16,stage1_geoss_context=13,stage2_geoss_ss=16,stage3_geovis_slat=16,stage4_geovis_slat_joint=16,final_conditioning_refined=8,asset_evaluation=4" \
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
