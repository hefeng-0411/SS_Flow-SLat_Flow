# Stage 1/2 training re-engineering v2

## Evidence behind the change

The supplied report is not a numerical-divergence trace. Stage 1 reaches its
best raw region early and then its adaptive controller applies 13 learning-rate
contractions by update 509. The persistent scheduler multiplier reaches
`0.000716`, so the largest parameter group is running near `7.16e-8`. The plotted
Stage-1 `loss` is also an EMA-reweighted optimization scalar whose scale changes
over time; it is not a stationary convergence metric.

Stage 2 is finite but causally inactive. At update 3612 the effective correction
norm is about `0.00316`, the correction/base ratio is about `0.00116`, and the
effective residual gain over frozen TRELLIS is only about `2.49e-5`. The legacy
auxiliary asks the raw residual to match the effective target even though
inference multiplies it by the timestep/confidence gate. Those objectives have
different optima whenever the gate is below one.

## New numerical contract

- Stage 1 optimizes the bounded balanced objective but selects checkpoints from
  `loss_stationary`, whose physical weights never change.
- Neither convergence fitting nor LR intervention observes scheduler warmup.
- The persistent LR control multiplier is bounded to `[0.1, 2.0]`.
- Stage 2 samples the same logit-normal timestep law as TRELLIS' original
  `FlowMatchingTrainer`.
- Stage-2 CFM supervision is normalized by frozen-base MSE. A preconditioned
  auxiliary bypasses only the trust projection while retaining the exact
  timestep/confidence gate; it therefore shares the effective loss optimum.
- The trust region uses `tau*tanh(raw/tau)`, avoiding the zero derivative of a
  hard clamp.
- Checkpoints carry an optimization-contract version. Legacy model weights are
  warm-started, but their AdamW, scheduler, loss-balancer, and early-stop state
  are reset. Stage 2 and inference reject an unrepaired Stage-1 handoff.

## Memory and execution contract

- Frozen VGGT and TRELLIS forwards run under inference mode and retain no
  autograd graph.
- Trainable Stage-1 and Stage-2 attention uses non-reentrant analytic
  recomputation.
- Grid-to-token routing is a shared-storage strided view; there is no explicit
  grid-sized transpose copy.
- CUDA runs use a fused Triton kernel to construct `x_t` and the CFM velocity
  target in one pass. `auto` falls back to the exact PyTorch equations when
  Triton/CUDA is unavailable.
- When sequential frozen giant-model forwards leave at least 8 GiB inactive and
  reserve at least 90% of VRAM, only the inactive CUDA allocator cache is
  released before adapter backpropagation. Live tensors are never moved or
  invalidated.
- TRELLIS Flow-Euler inference retains only the current ODE state. It executes
  the upstream sampler's exact `sample_once` transition and time grid without
  accumulating `pred_x_t`/`pred_x_0` trajectories.

## Training-server acceptance gates

Use a new output root for the cleanest audit. Reusing an existing root is also
supported: the first resumed update performs the versioned warm-start reset.
Do not copy only the old best checkpoint into Stage 2; the launcher deliberately
rejects that handoff.

Before promotion, require all of the following from the training server:

1. Stage-1 `loss_stationary` improves on a held-out split and no LR action occurs
   during warmup.
2. Stage-2 `normalized_effective_residual < 1` and its confidence interval is
   below one on held-out objects.
3. `causal_residual_gain > 0`, a non-null update/weight ratio, finite gradients,
   and non-collapsed effective correction norms hold across timestep bins.
4. Probe VRAM stays below the configured headroom and allocator reclamation does
   not dominate step latency.
5. Decoded held-out geometry improves over frozen TRELLIS. Training MSE alone is
   not sufficient for checkpoint promotion.

No implementation can guarantee convergence independent of data, initialization,
and hardware. These gates turn that uncertainty into a falsifiable promotion
contract instead of silently declaring a low training scalar successful.
