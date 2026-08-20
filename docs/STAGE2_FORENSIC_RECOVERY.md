# Stage 2 forensic recovery and training-server runbook

This document describes the repaired Stage 2 control plane. It does not claim
that GPU training was run in the debug checkout. Runtime acceptance evidence
must be collected on the four-A100 training server that owns `outputs/`.

## Execution DAG

```text
launch_meshfleet_multigpu_sequence.main
  -> parse launcher CLI
  -> sanitize inherited torchrun variables
  -> configure CUDA allocator/NCCL environment
  -> resolve visible GPU count and world size
  -> _make_stages
       -> select stage-specific UID manifests
       -> select stage1_geoss/geoss_adapter_best.pt for Stage 2
       -> serialize Stage 2 model/data/optimizer arguments
  -> _slice_stages
  -> _validate_stage_manifests
  -> _assert_selected_input_contracts
       -> use Stage 1 last-checkpoint early-stop metadata to certify best.pt
  -> _probe_batch
       -> _probe_batch_binary or _probe_batch_halve
       -> _resolved_grad_accum_steps
       -> _torchrun_command(execution_mode="probe")
            -> minimum_dataset_passes=0
            -> auto_expand_training_budget=false
            -> adaptive_batch=false
            -> early_stop=false
            -> save_best=false
            -> fault_tolerant_save_every=0
       -> isolated _probe_bs<N>/ output
       -> typed failure classification
       -> reduce batch only for verified CUDA OOM or measured VRAM headroom
  -> selected microbatch/accumulation/effective-batch metadata
  -> stop here when --probe_only is present
  -> _run_full_stage
       -> _torchrun_command(execution_mode="train")
            -> minimum_dataset_passes>=1
            -> auto_expand_training_budget=true
            -> adaptive_max_batch_size=largest proven microbatch
       -> append exact command to launcher_stage.log
       -> resume only from Stage 2 last/best candidates, never probe dirs
       -> lower microbatch only for FailureKind.CUDA_OOM

torch.distributed.run (one Stage 2 process per GPU)
  -> train_sparse_ray_ss_velocity.main
       -> parse CLI and apply config only where CLI is absent
       -> install SIGTERM -> KeyboardInterrupt translation
       -> run_training
            -> validate real VGGT/TRELLIS/dataset mode
            -> init_distributed
            -> enforce execution-mode contract
            -> MeshFleetTrellisDataset(post-filter manifest)
            -> DistributedSampler + DataLoader(drop_last=false)
            -> compute_training_budget(N,W,B_rank,A,U)
            -> optionally expand U to the runtime minimum
            -> enforce_minimum_dataset_passes
            -> load TRELLIS pipeline and retain SS flow + image encoder
            -> freeze TRELLIS SS flow
            -> construct trainable SSVelocityAdapter
            -> strict resume of Stage 2 adapter/optimizer/scheduler if requested
            -> wrap only Stage 2 adapter in DDP
            -> create AdamW + warmup/cosine scheduler
            -> strict-load Stage 1 GeoSS best checkpoint
            -> SHA-256 and parameter-summary handoff report
            -> freeze Stage 1 GeoSS and VGGT
            -> optimizer-update loop
                 -> fetch A rank-local microbatches
                 -> VGGT at native 518 input, return evidence at data resolution
                 -> exact frozen Stage 1 context-only forward
                 -> frozen TRELLIS base velocity
                 -> Stage 2 confidence-gated residual velocity
                 -> FP32 loss reductions
                 -> backward with DDP no_sync for microsteps 1..A-1
                 -> finite/nonzero gradient assertions
                 -> optimizer step + scheduler step
                 -> metrics, VRAM, timing, throughput, unique UID tracking
                 -> real mode only: best/candidate/last checkpoint policy
            -> rank-wise unique-coverage and hardware telemetry gather
       -> write probe_summary.json or training_summary.json
       -> finally: destroy process group without masking the primary exception
```

## Budget contract

The CLI batch size is a rank-local DataLoader microbatch. One optimizer update
contains `A` microbatches, so

```text
B_eff = B_rank * W * A
P_nominal = U * B_eff / N
U_min = ceil(P_min * N / B_eff)
```

For `N=1226`, `W=4`, `A=1`, the observed microbatches 8, 4, and 2 require
39, 77, and 154 optimizer updates respectively. `DistributedSampler` pads two
objects for this cardinality/world size. That padding contributes sample
presentations but is not counted as an additional unique object.

When the proven microbatch falls below the intended maximum, the launcher raises
accumulation by `ceil(B_rank,target * A_target / B_rank,selected)`. AdamW's
learning rate is not rescaled because losses are averaged over accumulation and
the effective batch is preserved. The chosen derivation is logged.

## Training-server commands

Set these shell variables to the remote paths before executing the commands:

```bash
REPO=/mnt/sda/hf/MVG/Base/SS_Flow
DATA=/mnt/sda/hf/MVG/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97
OUT=$REPO/outputs/phase2/foundation
PY=/mnt/sda/hf/miniconda3/envs/trellis/bin/python
```

Single-rank graph/capacity probe (GPU 4 shown; change to the training server's
first assigned device if necessary):

```bash
cd "$REPO"
"$PY" scripts/launch_meshfleet_multigpu_sequence.py \
  --data_root "$DATA" --output_root "$OUT" \
  --stage1_train_manifest "$OUT/../dataset_audit/stage1_train_uids.json" \
  --stage2_train_manifest "$OUT/../dataset_audit/stage2_train_uids.json" \
  --meshfleet_split train --vggt_root /mnt/sda/hf/MVG/Base/vggt \
  --trellis_root /mnt/sda/hf/MVG/Base/TRELLIS \
  --trellis_model_path microsoft/TRELLIS-image-large \
  --num_views 8 --image_size 256 --active_tokens 0 \
  --gpus 4 --nproc_per_node 1 --start_at stage2 --stop_after stage2 \
  --stage2_max_batch_size 1 --probe_only
```

Four-rank feasibility search:

```bash
cd "$REPO"
"$PY" scripts/launch_meshfleet_multigpu_sequence.py \
  --data_root "$DATA" --output_root "$OUT" \
  --stage1_train_manifest "$OUT/../dataset_audit/stage1_train_uids.json" \
  --stage2_train_manifest "$OUT/../dataset_audit/stage2_train_uids.json" \
  --meshfleet_split train --vggt_root /mnt/sda/hf/MVG/Base/vggt \
  --trellis_root /mnt/sda/hf/MVG/Base/TRELLIS \
  --trellis_model_path microsoft/TRELLIS-image-large \
  --num_views 8 --image_size 256 --active_tokens 0 \
  --gpus 4,5,6,7 --nproc_per_node 4 \
  --start_at stage2 --stop_after stage2 --stage2_max_batch_size 8 --probe_only
```

Real Stage 2 launch after reviewing the selected configuration. The example
assumes the probe selected microbatch 4, so accumulation 2 preserves effective
global batch 32. Replace both values with the probe result:

```bash
cd "$REPO"
"$PY" scripts/launch_meshfleet_multigpu_sequence.py \
  --data_root "$DATA" --output_root "$OUT" \
  --stage1_train_manifest "$OUT/../dataset_audit/stage1_train_uids.json" \
  --stage2_train_manifest "$OUT/../dataset_audit/stage2_train_uids.json" \
  --meshfleet_split train --vggt_root /mnt/sda/hf/MVG/Base/vggt \
  --trellis_root /mnt/sda/hf/MVG/Base/TRELLIS \
  --trellis_model_path microsoft/TRELLIS-image-large \
  --num_views 8 --image_size 256 --active_tokens 0 \
  --gpus 4,5,6,7 --nproc_per_node 4 \
  --start_at stage2 --stop_after stage2 --no_auto_batch \
  --stage2_max_batch_size 4 --stage2_grad_accum_steps 2 \
  --stage2_minimum_dataset_passes 1 --stage2_steps 100000
```

`active_tokens=0` is a Stage 3/4 sentinel meaning “keep every active SLAT
voxel.” Stage 2 does not consume this argument.

## Runtime acceptance checklist

- Every `_probe_bs*/probe_metadata.json` has `is_probe=true`, return code zero,
  `promotable_checkpoint=false`, and no best/candidate/last files.
- `probe_summary.json` proves fetch, forward, backward, nonzero finite gradients,
  optimizer step, per-rank VRAM, latency, throughput, and clean rank exit.
- `launcher_stage.log` contains the exact probe and real commands.
- `training_preflight.json` records the runtime post-filter `N` and a nominal
  pass count at least one.
- `training_summary.json` and the final JSONL coverage event report actual unique
  UID exposure separately from nominal presentations and sampler padding.
- Stage 1 handoff reports strict zero missing/unexpected keys and a SHA-256.
- No NCCL destroy warning, surviving torchrun worker, or stale CUDA process.
- Promotion beyond Stage 2 must use the repository's held-out decoded-asset
  evaluation protocol. Stage 2 loss alone is not evidence of texture quality.
