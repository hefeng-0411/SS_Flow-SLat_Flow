from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainingBudget:
    """Auditable update/exposure arithmetic for the initial batch topology.

    The production loops are update-bounded and cycle their dataloaders.  An
    "epoch" is therefore diagnostic rather than a stopping condition.  Keeping
    the arithmetic in one place prevents a smoke-test default from silently
    masquerading as a real training run.
    """

    dataset_objects: int
    world_size: int
    initial_microbatch_per_rank: int
    grad_accum_steps: int
    drop_last: bool
    samples_per_rank_per_sampler_epoch: int
    padded_samples_per_sampler_epoch: int
    dataloader_batches_per_rank: int
    optimizer_updates_per_sampler_epoch: int
    planned_optimizer_updates: int
    initial_global_batch_size: int
    initial_effective_global_batch_size: int
    nominal_sample_presentations: int
    nominal_dataset_passes: float
    unique_object_fraction_upper_bound: float

    def as_dict(self) -> dict:
        return asdict(self)


def compute_training_budget(
    *,
    dataset_objects: int,
    world_size: int,
    microbatch_per_rank: int,
    grad_accum_steps: int,
    planned_optimizer_updates: int,
    drop_last: bool = False,
) -> TrainingBudget:
    values = {
        "dataset_objects": dataset_objects,
        "world_size": world_size,
        "microbatch_per_rank": microbatch_per_rank,
        "grad_accum_steps": grad_accum_steps,
        "planned_optimizer_updates": planned_optimizer_updates,
    }
    for name, value in values.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive, got {value!r}")

    dataset_objects = int(dataset_objects)
    world_size = int(world_size)
    microbatch_per_rank = int(microbatch_per_rank)
    grad_accum_steps = int(grad_accum_steps)
    planned_optimizer_updates = int(planned_optimizer_updates)

    if drop_last:
        samples_per_rank = dataset_objects // world_size
        batches_per_rank = samples_per_rank // microbatch_per_rank
    else:
        samples_per_rank = math.ceil(dataset_objects / world_size)
        batches_per_rank = math.ceil(samples_per_rank / microbatch_per_rank)
    if batches_per_rank <= 0:
        raise ValueError(
            "The selected distributed/batch topology produces no dataloader batches: "
            f"N={dataset_objects}, world_size={world_size}, microbatch={microbatch_per_rank}, "
            f"drop_last={drop_last}."
        )

    effective_global_batch = microbatch_per_rank * world_size * grad_accum_steps
    presentations = planned_optimizer_updates * effective_global_batch
    return TrainingBudget(
        dataset_objects=dataset_objects,
        world_size=world_size,
        initial_microbatch_per_rank=microbatch_per_rank,
        grad_accum_steps=grad_accum_steps,
        drop_last=bool(drop_last),
        samples_per_rank_per_sampler_epoch=samples_per_rank,
        padded_samples_per_sampler_epoch=max(0, samples_per_rank * world_size - dataset_objects),
        dataloader_batches_per_rank=batches_per_rank,
        optimizer_updates_per_sampler_epoch=math.ceil(batches_per_rank / grad_accum_steps),
        planned_optimizer_updates=planned_optimizer_updates,
        initial_global_batch_size=microbatch_per_rank * world_size,
        initial_effective_global_batch_size=effective_global_batch,
        nominal_sample_presentations=presentations,
        nominal_dataset_passes=presentations / dataset_objects,
        unique_object_fraction_upper_bound=min(1.0, presentations / dataset_objects),
    )


def enforce_minimum_dataset_passes(
    budget: TrainingBudget,
    *,
    minimum_dataset_passes: float,
    stage: str,
) -> None:
    minimum_dataset_passes = float(minimum_dataset_passes)
    if minimum_dataset_passes < 0:
        raise ValueError(
            f"minimum_dataset_passes must be non-negative, got {minimum_dataset_passes}."
        )
    if budget.nominal_dataset_passes + 1.0e-12 >= minimum_dataset_passes:
        return
    required_updates = math.ceil(
        minimum_dataset_passes
        * budget.dataset_objects
        / budget.initial_effective_global_batch_size
    )
    raise RuntimeError(
        f"{stage} training budget is not a real dataset pass: configured "
        f"{budget.planned_optimizer_updates} optimizer updates expose at most "
        f"{budget.nominal_dataset_passes:.6f} nominal passes "
        f"({budget.nominal_sample_presentations}/{budget.dataset_objects} sample presentations) "
        f"at the initial effective global batch {budget.initial_effective_global_batch_size}. "
        f"At least {required_updates} updates are required for minimum_dataset_passes="
        f"{minimum_dataset_passes:g}. For an intentional smoke test, explicitly pass "
        "--minimum_dataset_passes 0; smoke-test outputs are not evidence of training."
    )


def minimum_updates_for_dataset_passes(
    budget: TrainingBudget,
    minimum_dataset_passes: float,
) -> int:
    minimum_dataset_passes = max(0.0, float(minimum_dataset_passes))
    return math.ceil(
        minimum_dataset_passes
        * budget.dataset_objects
        / budget.initial_effective_global_batch_size
    )


def defer_nonfatal_early_stop_until_minimum_exposure(
    status,
    *,
    budget: TrainingBudget,
    step: int,
    minimum_dataset_passes: float,
) -> dict:
    """Prevent convergence heuristics from turning a real run into a smoke test.

    Structural/numerical failures and explicit time limits remain terminal.
    Only a statistical convergence/degeneration decision is deferred.
    """

    required_updates = minimum_updates_for_dataset_passes(
        budget, minimum_dataset_passes
    )
    fatal_reasons = {
        "nonfinite_primary_metric",
        "structural_zero_gradient",
        "nonfinite_gradient",
        "externally_budgeted_time_limit",
    }
    deferred = bool(
        getattr(status, "should_stop", False)
        and int(step) < required_updates
        and getattr(status, "reason", "") not in fatal_reasons
    )
    original_reason = getattr(status, "reason", None)
    if deferred:
        status.should_stop = False
        status.handoff_ready = False
        status.reason = "minimum_dataset_exposure_gate"
        diagnostics = getattr(status, "diagnostics", None)
        if isinstance(diagnostics, dict):
            diagnostics["deferred_early_stop_reason"] = original_reason
            diagnostics["minimum_exposure_required_updates"] = required_updates
    return {
        "deferred": deferred,
        "original_reason": original_reason if deferred else None,
        "required_updates": required_updates,
        "current_step": int(step),
    }
