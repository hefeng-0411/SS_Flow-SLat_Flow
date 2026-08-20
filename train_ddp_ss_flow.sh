#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

PROJECT_ROOT="/mnt/sda/hf/MVG/Base/SS_Flow_SLat_Flow"
cd "$PROJECT_ROOT"

/mnt/sda/hf/miniconda3/envs/trellis/bin/torchrun \
  --standalone \
  --nproc_per_node=8 \
  scripts/train_stochastic_voxel_ss_flow.py \
  --config configs/stochastic_voxel_ss_flow.yaml \
  --trellis-root /mnt/sda/hf/MVG/Base/TRELLIS \
  --trellis-model microsoft/TRELLIS-image-large \
  --vggt-root /mnt/sda/hf/MVG/Base/vggt \
  --vggt-pretrained facebook/VGGT-1B \
  --hf-cache-root /mnt/sda/hf/.cache/huggingface/hub \
  --meshfleet-root /mnt/sda/hf/MVG/Base/MeshFleet \
  --dataset-root /mnt/sda/hf/MVG/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97 \
  --extensions-root /mnt/sda/hf/MVG/Base/extensions \
  --min-views 1 \
  --max-views 8 \
  --grid-resolution 16 \
  --precision bf16 \
  --gradient-checkpointing \
  --num-workers 8 \
  --persistent-workers \
  --pin-memory "$@"
