from __future__ import annotations

import copy
import math
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping, Optional


_PROTOCOL = "evidence_rate_termination_v1"
_FLOAT32_EPSILON = 2.0**-23


@dataclass
class EarlyStopStatus:
    enabled: bool
    should_stop: bool
    is_best: bool
    metric: str
    raw_value: float | None
    score: float | None
    best_score: float | None
    bad_steps: int
    reason: str
    regime: str = "unknown"
    maturity: float = 0.0
    trend: float | None = None
    trend_uncertainty: float | None = None
    opportunity_ratio: float | None = None
    is_candidate: bool = False
    lr_multiplier: float | None = None
    selected_checkpoint: str = "last"
    handoff_ready: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EarlyStopStatus":
        known = {item.name for item in fields(cls)}
        return cls(**{key: item for key, item in value.items() if key in known})


@dataclass
class _RegressionView:
    regime: str
    identifiable: bool
    estimate: float
    estimate_variance: float
    slope: float
    slope_uncertainty: float
    noise_variance: float
    bic_constant: float
    bic_linear: float
    maturity: float
    opportunity_ratio: float


@dataclass
class _OnlineLinear:
    """O(1) sufficient statistics for y = intercept + slope * x."""

    n: int = 0
    mean_x: float = 0.0
    mean_y: float = 0.0
    sxx: float = 0.0
    sxy: float = 0.0
    syy: float = 0.0

    def update(self, x: float, y: float) -> None:
        self.n += 1
        dx = x - self.mean_x
        dy = y - self.mean_y
        self.mean_x += dx / self.n
        self.mean_y += dy / self.n
        self.sxx += dx * (x - self.mean_x)
        self.sxy += dx * (y - self.mean_y)
        self.syy += dy * (y - self.mean_y)

    def view(self, x: float) -> _RegressionView:
        # A Gaussian line has three fitted quantities (offset, slope, noise).
        # Before n > 3, its trend and residual variance are not identifiable.
        parameter_count = 3
        floor = _positive_floor(self.mean_y, self.syy)
        if self.n <= parameter_count or self.sxx <= floor:
            return _RegressionView(
                regime="warmup",
                identifiable=False,
                estimate=self.mean_y,
                estimate_variance=max(self.syy, floor),
                slope=0.0,
                slope_uncertainty=float("inf"),
                noise_variance=max(self.syy, floor),
                bic_constant=float("inf"),
                bic_linear=float("inf"),
                maturity=0.0,
                opportunity_ratio=float("inf"),
            )

        slope = self.sxy / max(self.sxx, floor)
        intercept = self.mean_y - slope * self.mean_x
        estimate = intercept + slope * x
        sse_constant = max(self.syy, floor)
        sse_linear = max(self.syy - self.sxy * self.sxy / max(self.sxx, floor), floor)
        degrees_of_freedom = self.n - 2
        noise = max(sse_linear / degrees_of_freedom, floor)
        slope_variance = noise / max(self.sxx, floor)
        estimate_variance = noise * (
            1.0 / self.n + (x - self.mean_x) ** 2 / max(self.sxx, floor)
        )

        # BIC is an MDL criterion: the linear model is retained only when its
        # coding gain pays for the additional slope parameter. No delta,
        # window, significance level, or patience is supplied by the user.
        log_n = math.log(self.n)
        bic_constant = self.n * math.log(sse_constant / self.n) + 2.0 * log_n
        bic_linear = self.n * math.log(sse_linear / self.n) + 3.0 * log_n
        if bic_linear < bic_constant:
            regime = "rapid_descent" if slope > 0.0 else "degeneration"
        else:
            regime = "fine_convergence"

        # One future compute doubling has length log(2) in x. Its optimistic
        # gain is compared with irreducible one-summary observation noise.
        # The MDL radius grows with the number of candidate model comparisons.
        # A selected one-off slope claim costs log(n) nats. Equating that code
        # length with Gaussian evidence z²/2 gives the universal radius
        # sqrt(2 log n), rather than a configured confidence multiplier.
        mdl_radius = math.sqrt(max(2.0 * math.log(self.n), 0.0))
        optimistic_slope = max(0.0, slope + mdl_radius * math.sqrt(slope_variance))
        recoverable_gain = optimistic_slope * math.log(2.0)
        observation_resolution = math.sqrt(noise)
        opportunity = recoverable_gain / max(observation_resolution, math.sqrt(floor))
        separation = abs(bic_constant - bic_linear) / max(log_n, floor)
        maturity = 1.0 - math.exp(-separation)
        return _RegressionView(
            regime=regime,
            identifiable=True,
            estimate=estimate,
            estimate_variance=estimate_variance,
            slope=slope,
            slope_uncertainty=math.sqrt(slope_variance),
            noise_variance=noise,
            bic_constant=bic_constant,
            bic_linear=bic_linear,
            maturity=maturity,
            opportunity_ratio=opportunity,
        )

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any] | None) -> "_OnlineLinear":
        if not state:
            return cls()
        return cls(
            n=_as_int(state.get("n"), 0),
            mean_x=_as_float(state.get("mean_x")) or 0.0,
            mean_y=_as_float(state.get("mean_y")) or 0.0,
            sxx=_as_float(state.get("sxx")) or 0.0,
            sxy=_as_float(state.get("sxy")) or 0.0,
            syy=_as_float(state.get("syy")) or 0.0,
        )


@dataclass
class _Signal:
    """Scale-invariant telemetry signal with one dyadic-compute regression."""

    mode: str
    scale: float | None = None
    era: int | None = None
    regression: _OnlineLinear = field(default_factory=_OnlineLinear)
    raw_abs_max: float = 0.0

    def encode(self, raw: float) -> float:
        if self.scale is None:
            # Scale comes from the signal itself. The tiny fallback is the
            # representable Python-float floor, not a metric-specific delta.
            self.scale = max(abs(raw), math.sqrt(sys.float_info.min))
        direction = 1.0 if self.mode == "max" else -1.0
        return direction * math.asinh(raw / self.scale)

    def decode(self, utility: float) -> float:
        if self.scale is None:
            return utility
        direction = 1.0 if self.mode == "max" else -1.0
        return self.scale * math.sinh(direction * utility)

    def update(self, raw: float, x: float, era: int, *, force_reset: bool = False) -> float:
        utility = self.encode(raw)
        self.raw_abs_max = max(self.raw_abs_max, abs(raw))
        if force_reset or self.era != era:
            self.era = era
            self.regression = _OnlineLinear()
        self.regression.update(x, utility)
        return utility

    def view(self, x: float) -> _RegressionView:
        return self.regression.view(x)

    def active(self) -> bool:
        if self.scale is None:
            return False
        numerical = math.sqrt(_FLOAT32_EPSILON) * max(self.raw_abs_max, self.scale)
        return self.raw_abs_max > numerical

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "scale": self.scale,
            "era": self.era,
            "regression": self.regression.state_dict(),
            "raw_abs_max": self.raw_abs_max,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "_Signal":
        return cls(
            mode=str(state.get("mode", "min")),
            scale=_as_float(state.get("scale")),
            era=_as_int(state.get("era"), 0) if state.get("era") is not None else None,
            regression=_OnlineLinear.from_state_dict(state.get("regression")),
            raw_abs_max=_as_float(state.get("raw_abs_max")) or 0.0,
        )


@dataclass
class _PendingObservation:
    x: float
    era: int
    utility: float
    raw: float
    components: dict[str, float]
    innovation_sign: int


@dataclass
class _Health:
    hard_failure: str | None
    causal_available: bool
    causal_significant: bool
    gradient_available: bool
    gradient_nonzero: bool
    warnings: list[str]
    diagnostics: dict[str, Any]


class EarlyStopper:
    """Evidence-rate adaptive termination for heterogeneous training stages.

    The controller stores only sufficient statistics. It does not use a loss
    window, patience counter, or configured improvement threshold. Trainers may
    supply a control-start update to exclude a known nonstationary LR warmup.
    Training progress after that boundary is measured in processed examples
    and partitioned into powers-of-two compute regimes. Within each regime,
    BIC selects between a constant and linear utility model.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        metric: str,
        mode: str = "min",
        max_train_hours: float = 0.0,
        min_control_updates: int = 0,
        **legacy_ignored: Any,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("early stop mode must be 'min' or 'max'")
        self.enabled = bool(enabled)
        self.metric = str(metric)
        self.mode = mode
        self.max_train_seconds = max(0.0, float(max_train_hours)) * 3600.0
        self.min_control_updates = max(0, int(min_control_updates))
        self.start_time = time.time()
        self.primary = _Signal(mode)
        self.components: dict[str, _Signal] = {}
        self.total_updates = 0
        self.last_step: int | None = None
        self.cumulative_examples = 0.0
        self.compute_unit: float | None = None
        self.current_x = 0.0
        self.current_era = 0
        self.initial_utility: float | None = None
        self.initial_variance: float | None = None
        self.champion_utility: float | None = None
        self.champion_variance: float | None = None
        self.champion_step: int | None = None
        self.pending: _PendingObservation | None = None
        self.force_regime_reset = False
        self.probe_active = False
        self.probe_kind: str | None = None
        self.probe_baseline_utility: float | None = None
        self.probe_baseline_variance: float | None = None
        self.probe_generation = 0
        self.last_raw: float | None = None
        self.last_status: EarlyStopStatus | None = None
        self.legacy_best_score: float | None = None
        self.distributed_schema: tuple[str, ...] | None = None

    @classmethod
    def from_args(
        cls,
        args: Any,
        default_metric: str = "loss",
        *,
        min_control_updates: int | None = None,
    ) -> "EarlyStopper":
        # Legacy patience/window/delta flags remain parseable. A minimum control
        # step is now a safety boundary because warmup data is nonstationary.
        return cls(
            enabled=bool(getattr(args, "early_stop", False)),
            metric=getattr(args, "early_stop_metric", None) or default_metric,
            mode=getattr(args, "early_stop_mode", "min"),
            max_train_hours=getattr(args, "max_train_hours", 0.0),
            min_control_updates=(
                max(0, int(min_control_updates))
                if min_control_updates is not None
                else max(0, int(getattr(args, "early_stop_min_steps", 0) or 0))
            ),
        )

    def update(self, record: Mapping[str, Any]) -> EarlyStopStatus:
        if not self.enabled:
            return self._status(reason="disabled", enabled=False)

        raw_object = _get_nested(record, self.metric)
        raw = _as_float(raw_object)
        if raw is None:
            if raw_object is not None:
                status = self._status(
                    reason="nonfinite_primary_metric",
                    should_stop=True,
                    selected_checkpoint="best",
                )
                status.diagnostics["invalid_value"] = repr(raw_object)
                self.last_status = status
                return status
            return self._status(reason="metric_missing")

        step = _as_int(record.get("step"), self.last_step or 0)
        if self.last_step is not None and step <= self.last_step:
            status = self._status(reason="duplicate_or_out_of_order_summary", raw=raw)
            self.last_status = status
            return status

        # Scheduler warmup is an intentional nonstationary experiment.  Fitting
        # convergence/degeneration models to it caused repeated LR contractions
        # before the base learning rate was ever reached.  Structural failures
        # still terminate immediately, but trend evidence begins only after the
        # trainer-defined control horizon.
        if step <= self.min_control_updates:
            health = _telemetry_health(record)
            if health.hard_failure is not None:
                status = self._status(
                    reason=health.hard_failure,
                    raw=raw,
                    should_stop=True,
                    selected_checkpoint="best",
                )
            else:
                status = self._status(reason="control_warmup", raw=raw, regime="warmup")
            status.diagnostics.update(
                {
                    "control_updates_deferred": True,
                    "control_start_update": self.min_control_updates + 1,
                    "updates_remaining": self.min_control_updates - step + 1,
                }
            )
            self.last_step = step
            self.last_raw = raw
            self.last_status = status
            return status

        self.total_updates += 1
        x, era = self._advance_compute(record, step)
        component_raw = _loss_components(record, primary=self.metric)
        encoded = self.primary.encode(raw)

        # Quarantine an MDL-improbable singleton. A second same-direction
        # observation is the minimum information needed to identify a shifted
        # regime; otherwise the point is discarded as a transient spike.
        outlier, innovation_sign = self._is_surprising(encoded, x, era)
        if self.pending is not None:
            if outlier and innovation_sign == self.pending.innovation_sign:
                pending = self.pending
                self.pending = None
                self._reset_signals()
                self._ingest(
                    pending.raw,
                    pending.components,
                    pending.x,
                    pending.era,
                    force_reset=True,
                )
                utility = self._ingest(raw, component_raw, x, era)
                transient_reason = "confirmed_regime_change"
            else:
                self.pending = None
                utility = self._ingest(raw, component_raw, x, era)
                transient_reason = "transient_spike_rejected"
        elif outlier:
            self.pending = _PendingObservation(
                x=x,
                era=era,
                utility=encoded,
                raw=raw,
                components=component_raw,
                innovation_sign=innovation_sign,
            )
            status = self._status(
                reason="transient_candidate",
                raw=raw,
                regime="transient",
                is_candidate=innovation_sign > 0,
            )
            status.diagnostics.update(
                {
                    "innovation_sign": innovation_sign,
                    "quarantined_until_independently_confirmed": True,
                }
            )
            self.last_step = step
            self.last_raw = raw
            self.last_status = status
            return status
        else:
            utility = self._ingest(
                raw,
                component_raw,
                x,
                era,
                force_reset=self.force_regime_reset,
            )
            self.force_regime_reset = False
            transient_reason = None

        self.last_step = step
        self.last_raw = raw
        primary_view = self.primary.view(x)
        health = _telemetry_health(record)
        if health.hard_failure is not None:
            status = self._build_status(
                raw,
                primary_view,
                health,
                reason=health.hard_failure,
                should_stop=True,
                selected_checkpoint="best",
            )
            self.last_status = status
            return status

        components_unsettled, components_degrading, component_regimes = (
            self._component_state(x)
        )
        checkpoint_eligible = (
            not components_degrading
            and health.hard_failure is None
            and (not health.causal_available or health.causal_significant)
            and (not health.gradient_available or health.gradient_nonzero)
        )
        # The first checkpoint is always retained as a recoverable reference.
        # Later promotion requires a statistically credible primary gain and
        # no evidence that a constituent objective or causal adapter path has
        # collapsed. This is deliberately stricter than minimizing raw loss.
        is_best = self._update_champion(
            utility,
            primary_view,
            step,
            eligible=checkpoint_eligible or self.champion_utility is None,
        )
        learned = self._primary_learning_proven(primary_view)
        structural_quality_ready = (
            learned
            and health.hard_failure is None
            and (not health.causal_available or health.causal_significant)
            and (not health.gradient_available or health.gradient_nonzero)
        )
        quality_ready = (
            structural_quality_ready and primary_view.regime != "degeneration"
        )
        handoff_ready = quality_ready and primary_view.identifiable
        should_stop = False
        reason = transient_reason or primary_view.regime
        lr_multiplier: float | None = None
        selected_checkpoint = "best" if self.champion_step is not None else "last"

        if primary_view.regime == "degeneration" and primary_view.identifiable:
            # A smooth adverse trend can still be a short stochastic phase.
            # Create an independent post-intervention segment before calling
            # it persistent degeneration; two identifiable models are the
            # minimum evidence for a change and its continuation.
            if self.probe_active and self.probe_kind == "convergence":
                deterioration = (
                    (
                        self.probe_baseline_utility
                        if self.probe_baseline_utility is not None
                        else primary_view.estimate
                    )
                    - primary_view.estimate
                )
                deterioration_variance = primary_view.estimate_variance + (
                    self.probe_baseline_variance or 0.0
                )
                if _mdl_positive_evidence(
                    deterioration,
                    deterioration_variance,
                    self.total_updates,
                ) and structural_quality_ready:
                    # The only intervention that could reveal hidden progress
                    # made the posterior utility worse. Stop at the pre-probe
                    # champion rather than chasing an induced adverse regime.
                    should_stop = True
                    handoff_ready = True
                    reason = "lr_probe_rejected_convergence"
                else:
                    reason = "convergence_probe_under_observation"
            elif self.probe_active and self.probe_kind == "degeneration":
                deterioration = (
                    (
                        self.probe_baseline_utility
                        if self.probe_baseline_utility is not None
                        else primary_view.estimate
                    )
                    - primary_view.estimate
                )
                deterioration_variance = primary_view.estimate_variance + (
                    self.probe_baseline_variance or 0.0
                )
                if _mdl_positive_evidence(
                    deterioration,
                    deterioration_variance,
                    self.total_updates,
                ):
                    should_stop = True
                    handoff_ready = False
                    reason = "confirmed_degeneration"
                else:
                    reason = "degeneration_under_recovery_probe"
            else:
                lr_multiplier = self._derive_lr_probe(
                    primary_view,
                    health,
                    direction=-1.0,
                )
                self.probe_active = True
                self.probe_kind = "degeneration"
                self.probe_baseline_utility = primary_view.estimate
                self.probe_baseline_variance = primary_view.estimate_variance
                self.probe_generation += 1
                self.force_regime_reset = True
                handoff_ready = False
                reason = "self_calibrating_recovery_probe"
        elif (
            primary_view.regime == "rapid_descent"
            and self.probe_active
            and self.probe_baseline_utility is not None
        ):
            probe_gain = primary_view.estimate - self.probe_baseline_utility
            probe_variance = primary_view.estimate_variance + (
                self.probe_baseline_variance or 0.0
            )
            if _mdl_positive_evidence(
                probe_gain,
                probe_variance,
                self.total_updates,
            ):
                self.probe_active = False
                self.probe_kind = None
                self.probe_baseline_utility = None
                self.probe_baseline_variance = None
                reason = "lr_probe_found_progress"
        elif (
            primary_view.regime == "fine_convergence"
            and primary_view.identifiable
            and self.probe_active
            and self.probe_kind == "degeneration"
        ):
            # The recovery intervention eliminated the adverse slope. Clear it
            # as soon as a stationary model is identifiable; component noise
            # cannot keep a stale degeneration hypothesis alive indefinitely.
            self.probe_active = False
            self.probe_kind = None
            self.probe_baseline_utility = None
            self.probe_baseline_variance = None
            reason = "recovery_probe_stabilized"
        elif (
            primary_view.regime == "fine_convergence"
            and primary_view.identifiable
            and not components_unsettled
            and primary_view.opportunity_ratio <= 1.0
        ):
            if not self.probe_active:
                lr_multiplier = self._derive_lr_probe(primary_view, health)
                self.probe_active = True
                self.probe_kind = "convergence"
                self.probe_baseline_utility = primary_view.estimate
                self.probe_baseline_variance = primary_view.estimate_variance
                self.probe_generation += 1
                self.force_regime_reset = True
                reason = "self_calibrating_lr_probe"
            else:
                probe_gain = primary_view.estimate - (
                    self.probe_baseline_utility
                    if self.probe_baseline_utility is not None
                    else primary_view.estimate
                )
                probe_variance = primary_view.estimate_variance + (
                    self.probe_baseline_variance or 0.0
                )
                if _mdl_positive_evidence(
                    probe_gain,
                    probe_variance,
                    self.total_updates,
                ):
                    self.probe_active = False
                    self.probe_kind = None
                    self.probe_baseline_utility = None
                    self.probe_baseline_variance = None
                    reason = "lr_probe_found_progress"
                elif quality_ready:
                    should_stop = True
                    reason = "evidence_limited_convergence"
                else:
                    reason = "plateau_but_quality_unproven"

        if self.max_train_seconds > 0.0 and self.elapsed_seconds >= self.max_train_seconds:
            should_stop = True
            reason = "externally_budgeted_time_limit"
            handoff_ready = quality_ready

        status = self._build_status(
            raw,
            primary_view,
            health,
            reason=reason,
            should_stop=should_stop,
            is_best=is_best,
            lr_multiplier=lr_multiplier,
            selected_checkpoint=selected_checkpoint,
            handoff_ready=handoff_ready,
        )
        status.diagnostics.update(
            {
                "protocol": _PROTOCOL,
                "compute_examples": self.cumulative_examples,
                "compute_era": self.current_era,
                "updates_seen": self.total_updates,
                "primary_learning_proven": learned,
                "structural_quality_ready": structural_quality_ready,
                "quality_ready": quality_ready,
                "components_unsettled": components_unsettled,
                "components_degrading": components_degrading,
                "component_regimes": component_regimes,
                "probe_active": self.probe_active,
                "probe_kind": self.probe_kind,
                "probe_generation": self.probe_generation,
                "causal_available": health.causal_available,
                "causal_significant": health.causal_significant,
                "health_warnings": health.warnings,
                **health.diagnostics,
            }
        )
        self.last_status = status
        return status

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.time() - self.start_time)

    def _advance_compute(self, record: Mapping[str, Any], step: int) -> tuple[float, int]:
        batch = _effective_batch(record)
        delta_step = step if self.last_step is None else step - self.last_step
        increment = max(float(delta_step), 0.0) * batch
        if self.compute_unit is None:
            self.compute_unit = max(increment, batch, math.sqrt(sys.float_info.min))
        self.cumulative_examples += increment
        ratio = max(self.cumulative_examples / self.compute_unit, 1.0)
        self.current_x = math.log(ratio)
        self.current_era = int(math.floor(math.log2(ratio)))
        return self.current_x, self.current_era

    def _ingest(
        self,
        raw: float,
        components: Mapping[str, float],
        x: float,
        era: int,
        *,
        force_reset: bool = False,
    ) -> float:
        utility = self.primary.update(raw, x, era, force_reset=force_reset)
        for name, value in components.items():
            signal = self.components.setdefault(name, _Signal("min"))
            signal.update(value, x, era, force_reset=force_reset)
        if self.initial_utility is None:
            self.initial_utility = utility
        # The initial observation's uncertainty is not identifiable from one
        # summary. Once the local Gaussian line is identifiable, its learned
        # residual variance is used as the uncertainty of that reference.
        view = self.primary.view(x)
        if self.initial_variance is None and view.identifiable:
            self.initial_variance = view.noise_variance
        return utility

    def _reset_signals(self) -> None:
        self.primary.regression = _OnlineLinear()
        self.primary.era = None
        for signal in self.components.values():
            signal.regression = _OnlineLinear()
            signal.era = None

    def _is_surprising(self, utility: float, x: float, era: int) -> tuple[bool, int]:
        if self.primary.era != era:
            return False, 0
        view = self.primary.view(x)
        if not view.identifiable:
            return False, 0
        residual = utility - view.estimate
        predictive_variance = (
            view.noise_variance + view.estimate_variance + _positive_floor(utility)
        )
        standardized_square = residual * residual / predictive_variance
        # Universal coding: a dedicated exception costs log(n) nats; use it
        # only when the Gaussian residual saves more than that code length.
        is_outlier = 0.5 * standardized_square > math.log(max(self.total_updates, 2))
        return is_outlier, 1 if residual > 0.0 else -1

    def _update_champion(
        self,
        utility: float,
        view: _RegressionView,
        step: int,
        *,
        eligible: bool,
    ) -> bool:
        estimate = view.estimate if view.identifiable else utility
        variance = view.estimate_variance
        if self.champion_utility is None:
            self.champion_utility = estimate
            self.champion_variance = variance
            self.champion_step = step
            return True
        if not eligible:
            return False
        gain = estimate - self.champion_utility
        combined_variance = variance + max(
            self.champion_variance or 0.0,
            view.noise_variance if view.identifiable else 0.0,
        )
        if _mdl_positive_evidence(gain, combined_variance, self.total_updates):
            self.champion_utility = estimate
            self.champion_variance = variance
            self.champion_step = step
            return True
        return False

    def _primary_learning_proven(self, view: _RegressionView) -> bool:
        if (
            self.initial_utility is None
            or self.initial_variance is None
            or self.champion_utility is None
        ):
            return False
        gain = self.champion_utility - self.initial_utility
        variance = (self.initial_variance or 0.0) + (
            self.champion_variance or view.estimate_variance
        )
        return _mdl_positive_evidence(gain, variance, self.total_updates)

    def _component_state(self, x: float) -> tuple[bool, bool, dict[str, str]]:
        states: dict[str, str] = {}
        unsettled = False
        degrading = False
        for name, signal in self.components.items():
            if not signal.active():
                continue
            view = signal.view(x)
            states[name] = view.regime
            if view.identifiable and view.regime in {"rapid_descent", "degeneration"}:
                unsettled = True
            if view.identifiable and view.regime == "degeneration":
                degrading = True
        return unsettled, degrading, states

    @staticmethod
    def _derive_lr_probe(
        view: _RegressionView,
        health: _Health,
        *,
        direction: float | None = None,
    ) -> float:
        # A dimensionless signal-to-noise ratio determines the perturbation
        # magnitude. exp(±1/sqrt(1+SNR²)) is finite, positive, and approaches
        # one automatically when the local response is already well resolved.
        snr = abs(view.slope) / max(
            view.slope_uncertainty,
            math.sqrt(_positive_floor(view.slope)),
        )
        magnitude = 1.0 / math.sqrt(1.0 + snr * snr)
        signed_direction = (
            direction
            if direction is not None
            else (
                1.0
                if health.causal_available and not health.causal_significant
                else -1.0
            )
        )
        return math.exp(signed_direction * magnitude)

    def _build_status(
        self,
        raw: float,
        view: _RegressionView,
        health: _Health,
        *,
        reason: str,
        should_stop: bool = False,
        is_best: bool = False,
        lr_multiplier: float | None = None,
        selected_checkpoint: str = "last",
        handoff_ready: bool = False,
    ) -> EarlyStopStatus:
        score = self.primary.decode(view.estimate)
        best = (
            self.primary.decode(self.champion_utility)
            if self.champion_utility is not None
            else self.legacy_best_score
        )
        return EarlyStopStatus(
            enabled=True,
            should_stop=should_stop,
            is_best=is_best,
            metric=self.metric,
            raw_value=raw,
            score=score,
            best_score=best,
            bad_steps=0,
            reason=reason,
            regime=view.regime,
            maturity=view.maturity,
            trend=view.slope,
            trend_uncertainty=view.slope_uncertainty,
            opportunity_ratio=view.opportunity_ratio,
            lr_multiplier=lr_multiplier,
            selected_checkpoint=selected_checkpoint,
            handoff_ready=handoff_ready,
        )

    def _status(
        self,
        *,
        reason: str,
        enabled: bool = True,
        raw: float | None = None,
        should_stop: bool = False,
        regime: str = "unknown",
        is_candidate: bool = False,
        selected_checkpoint: str = "last",
    ) -> EarlyStopStatus:
        best = (
            self.primary.decode(self.champion_utility)
            if self.champion_utility is not None
            else self.legacy_best_score
        )
        return EarlyStopStatus(
            enabled=enabled,
            should_stop=should_stop,
            is_best=False,
            metric=self.metric,
            raw_value=raw,
            score=self.last_status.score if self.last_status is not None else raw,
            best_score=best,
            bad_steps=0,
            reason=reason,
            regime=regime,
            is_candidate=is_candidate,
            selected_checkpoint=selected_checkpoint,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "protocol": _PROTOCOL,
            "metric": self.metric,
            "mode": self.mode,
            "primary": self.primary.state_dict(),
            "components": {
                name: signal.state_dict() for name, signal in self.components.items()
            },
            "total_updates": self.total_updates,
            "last_step": self.last_step,
            "cumulative_examples": self.cumulative_examples,
            "compute_unit": self.compute_unit,
            "current_x": self.current_x,
            "current_era": self.current_era,
            "initial_utility": self.initial_utility,
            "initial_variance": self.initial_variance,
            "champion_utility": self.champion_utility,
            "champion_variance": self.champion_variance,
            "champion_step": self.champion_step,
            "pending": asdict(self.pending) if self.pending is not None else None,
            "force_regime_reset": self.force_regime_reset,
            "probe_active": self.probe_active,
            "probe_kind": self.probe_kind,
            "probe_baseline_utility": self.probe_baseline_utility,
            "probe_baseline_variance": self.probe_baseline_variance,
            "probe_generation": self.probe_generation,
            "last_raw": self.last_raw,
            "min_control_updates": self.min_control_updates,
            "elapsed_seconds": self.elapsed_seconds,
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        if state.get("metric") and state.get("metric") != self.metric:
            return
        if state.get("mode") and state.get("mode") != self.mode:
            return
        if state.get("protocol") != _PROTOCOL:
            # Existing patience checkpoints remain resumable. Their best value
            # is retained for audit reporting, while adaptive trend evidence
            # starts afresh because the old window is not sufficient for the
            # new likelihood model. The caller quarantines the legacy best
            # file before the first evidence-rate champion replaces it.
            self.legacy_best_score = _as_float(state.get("best_score"))
            self.last_step = _as_int(state.get("seen_steps"), 0) or None
            return
        self.primary = _Signal.from_state_dict(state.get("primary", {}))
        self.components = {
            str(name): _Signal.from_state_dict(value)
            for name, value in (state.get("components") or {}).items()
            if isinstance(value, Mapping)
        }
        self.total_updates = _as_int(state.get("total_updates"), 0)
        self.last_step = (
            _as_int(state.get("last_step"), 0)
            if state.get("last_step") is not None
            else None
        )
        self.cumulative_examples = _as_float(state.get("cumulative_examples")) or 0.0
        self.compute_unit = _as_float(state.get("compute_unit"))
        self.current_x = _as_float(state.get("current_x")) or 0.0
        self.current_era = _as_int(state.get("current_era"), 0)
        self.initial_utility = _as_float(state.get("initial_utility"))
        self.initial_variance = _as_float(state.get("initial_variance"))
        self.champion_utility = _as_float(state.get("champion_utility"))
        self.champion_variance = _as_float(state.get("champion_variance"))
        self.champion_step = (
            _as_int(state.get("champion_step"), 0)
            if state.get("champion_step") is not None
            else None
        )
        pending = state.get("pending")
        if isinstance(pending, Mapping):
            self.pending = _PendingObservation(
                x=float(pending["x"]),
                era=int(pending["era"]),
                utility=float(pending["utility"]),
                raw=float(pending["raw"]),
                components={
                    str(key): float(value)
                    for key, value in (pending.get("components") or {}).items()
                },
                innovation_sign=int(pending["innovation_sign"]),
            )
        self.force_regime_reset = bool(state.get("force_regime_reset", False))
        self.probe_active = bool(state.get("probe_active", False))
        self.probe_kind = (
            str(state.get("probe_kind"))
            if state.get("probe_kind") is not None
            else ("convergence" if self.probe_active else None)
        )
        self.probe_baseline_utility = _as_float(state.get("probe_baseline_utility"))
        self.probe_baseline_variance = _as_float(state.get("probe_baseline_variance"))
        self.probe_generation = _as_int(state.get("probe_generation"), 0)
        self.last_raw = _as_float(state.get("last_raw"))
        self.min_control_updates = max(
            self.min_control_updates,
            _as_int(state.get("min_control_updates"), 0),
        )
        elapsed = _as_float(state.get("elapsed_seconds")) or 0.0
        self.start_time = time.time() - elapsed


def apply_early_stop_action(optimizer: Any, status: EarlyStopStatus) -> dict[str, Any]:
    """Apply the controller's one-shot, data-derived LR falsification probe."""

    multiplier = status.lr_multiplier
    if multiplier is None:
        return {"applied": False, "reason": "no_action"}
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        return {
            "applied": False,
            "reason": "invalid_multiplier",
            "requested_multiplier": multiplier,
        }
    before = [float(group["lr"]) for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = float(group["lr"]) * multiplier
    after = [float(group["lr"]) for group in optimizer.param_groups]
    return {
        "applied": True,
        "reason": status.reason,
        "multiplier": multiplier,
        "before": before,
        "after": after,
    }


def distributed_early_stop_update(
    stopper: EarlyStopper,
    record: Mapping[str, Any],
    *,
    rank: int = 0,
    source_rank: int = 0,
) -> EarlyStopStatus:
    """Make one telemetry decision authoritative across every DDP rank.

    Loss summaries are rank-local even when gradients are synchronized. Letting
    each rank fit its own regime model can therefore desynchronize LR probes.
    Rank ``source_rank`` owns the sufficient statistics and broadcasts only the
    compact decision; no loss history or model tensor is communicated.
    """

    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        status = stopper.update(record)
        status.diagnostics.setdefault("telemetry_scope", "single_process")
        status.diagnostics.setdefault("telemetry_world_size", 1)
        return status

    if not dist.is_available() or not dist.is_initialized():
        status = stopper.update(record)
        status.diagnostics.setdefault("telemetry_scope", "single_process")
        status.diagnostics.setdefault("telemetry_world_size", 1)
        return status

    schema_payload: list[list[str] | None] = [
        (
            list(stopper.distributed_schema)
            if rank == source_rank and stopper.distributed_schema is not None
            else (
                _distributed_decision_schema(record, stopper.metric)
                if rank == source_rank
                else None
            )
        )
    ]
    dist.broadcast_object_list(schema_payload, src=source_rank)
    if schema_payload[0] is None:
        raise RuntimeError("DDP early-stop telemetry schema broadcast failed.")
    stopper.distributed_schema = tuple(schema_payload[0])

    # Reduce finite values and finite counts separately. Any explicitly
    # non-finite value survives as NaN, making numerical failure visible
    # instead of being silently averaged away by healthy ranks.
    rows = []
    for path in stopper.distributed_schema:
        value, invalid = _finite_measurement(_get_nested(record, path))
        rows.append(
            [
                value if value is not None else 0.0,
                1.0 if value is not None else 0.0,
                1.0 if invalid else 0.0,
            ]
        )
    backend = str(dist.get_backend()).lower()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if "nccl" in backend
        else torch.device("cpu")
    )
    reduced = torch.tensor(rows, dtype=torch.float64, device=device)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)

    authoritative = _copy_summary_record(record)
    for path, (total, count, invalid_count) in zip(
        stopper.distributed_schema,
        reduced.cpu().tolist(),
    ):
        if invalid_count > 0.0:
            value: Any = float("nan")
        elif count > 0.0:
            value = total / count
        else:
            continue
        if path == "optimizer_step_skipped":
            value = bool(value > 0.0)
        _set_nested(authoritative, path, value)

    if rank == source_rank:
        authoritative_status = stopper.update(authoritative)
        authoritative_status.diagnostics.update(
            {
                "telemetry_scope": "finite_aware_global_rank_mean",
                "telemetry_world_size": dist.get_world_size(),
            }
        )
        serialized_status: dict[str, Any] | None = authoritative_status.as_dict()
    else:
        serialized_status = None
    payload: list[dict[str, Any] | None] = [serialized_status]
    dist.broadcast_object_list(payload, src=source_rank)
    if payload[0] is None:
        raise RuntimeError("DDP early-stop authority broadcast returned no status.")
    return EarlyStopStatus.from_dict(payload[0])


def handoff_contract(status: Mapping[str, Any] | None) -> tuple[bool, list[str]]:
    """Validate serialized termination telemetry before a downstream stage."""

    if not isinstance(status, Mapping):
        return False, ["missing_early_stop_status"]
    if status.get("reason") in {
        "nonfinite_primary_metric",
        "structural_zero_gradient",
        "confirmed_degeneration",
    }:
        return False, [str(status.get("reason"))]
    diagnostics = status.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    reasons = []
    if not bool(status.get("handoff_ready", False)):
        reasons.append("handoff_not_statistically_certified")
    if not bool(diagnostics.get("primary_learning_proven", False)):
        reasons.append("primary_learning_not_proven")
    if diagnostics.get("causal_available") and not diagnostics.get(
        "causal_significant"
    ):
        reasons.append("adapter_effect_below_numerical_resolution")
    return not reasons, reasons


def telemetry_health_report(record: Mapping[str, Any]) -> dict[str, Any]:
    """Audit what one terminal summary can establish without inventing history.

    A single snapshot can diagnose structural and causal failures, but it
    cannot identify a trend or an optimum: arbitrarily many loss trajectories
    share the same terminal value. Regime fields are therefore supplied only
    by :class:`EarlyStopper`'s persisted sufficient statistics.
    """

    health = _telemetry_health(record)
    return {
        "hard_failure": health.hard_failure,
        "causal_available": health.causal_available,
        "causal_significant": health.causal_significant,
        "gradient_available": health.gradient_available,
        "gradient_nonzero": health.gradient_nonzero,
        "warnings": list(health.warnings),
        "diagnostics": dict(health.diagnostics),
        "trajectory_identifiable_from_this_snapshot": False,
    }


def quarantine_legacy_best_checkpoint(
    best_path: Any,
    resume_state: Mapping[str, Any] | None,
) -> str | None:
    """Preserve an old patience-selected best before adaptive replacement."""

    if not isinstance(resume_state, Mapping):
        return None
    controller_state = resume_state.get("early_stopper")
    if (
        isinstance(controller_state, Mapping)
        and controller_state.get("protocol") == _PROTOCOL
    ):
        return None

    from pathlib import Path
    import os
    import shutil

    source = Path(best_path)
    if not source.is_file():
        return None
    target = source.with_name(
        source.name.removesuffix(".pt") + "_legacy_unverified.pt"
    )
    if target.is_file():
        return str(target)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)
    return str(target)


def add_early_stopping_args(parser):
    parser.add_argument("--early_stop", type=_str2bool, default=False)
    parser.add_argument("--early_stop_metric", type=str, default=None)
    parser.add_argument("--early_stop_mode", choices=["min", "max"], default="min")
    parser.add_argument("--max_train_hours", type=float, default=0.0)
    parser.add_argument("--save_best", type=_str2bool, default=True)
    # Patience/window/delta remain inert compatibility inputs. A minimum step is
    # honored only as a nonstationary-control exclusion boundary.
    parser.add_argument("--early_stop_patience", type=int, default=None, help="Deprecated; ignored.")
    parser.add_argument("--early_stop_min_delta", type=float, default=None, help="Deprecated; ignored.")
    parser.add_argument("--early_stop_relative_delta", type=_str2bool, default=None, help="Deprecated; ignored.")
    parser.add_argument("--early_stop_warmup_steps", type=int, default=None, help="Deprecated; ignored.")
    parser.add_argument("--early_stop_min_steps", type=int, default=None, help="Do not fit trends or intervene before this update.")
    parser.add_argument("--early_stop_ema", type=float, default=None, help="Deprecated; ignored.")
    parser.add_argument("--early_stop_window", type=int, default=None, help="Deprecated; ignored.")
    return parser


def _telemetry_health(record: Mapping[str, Any]) -> _Health:
    warnings: list[str] = []
    diagnostics: dict[str, Any] = {}
    optimizer_skipped = bool(record.get("optimizer_step_skipped", False))

    gradient_values: list[float] = []
    nonfinite_gradient = False
    nested = record.get("adapter_grad_norms")
    if isinstance(nested, Mapping):
        for item in nested.values():
            parsed, invalid = _finite_measurement(item)
            nonfinite_gradient = nonfinite_gradient or invalid
            if parsed is not None:
                gradient_values.append(parsed)
    global_gradient = _as_float(record.get("global_grad_norm"))
    if global_gradient is not None:
        gradient_values.append(global_gradient)
    elif record.get("global_grad_norm") is not None:
        _, invalid = _finite_measurement(record.get("global_grad_norm"))
        nonfinite_gradient = nonfinite_gradient or invalid
    gradient_available = bool(gradient_values)
    gradient_nonzero = any(value > 0.0 for value in gradient_values)
    hard_failure = None
    if optimizer_skipped:
        warnings.append("optimizer_step_skipped")
    elif gradient_available and not gradient_nonzero:
        hard_failure = "structural_zero_gradient"
    if nonfinite_gradient:
        hard_failure = "nonfinite_gradient"

    causal_pairs = (
        ("loss_raw_residual", "loss_effective_residual"),
        ("loss_slat_raw_residual", "loss_slat_effective_residual"),
    )
    causal_available = False
    causal_significant = False
    causal_gain = None
    causal_resolution = None
    for raw_name, effective_name in causal_pairs:
        raw, raw_invalid = _finite_measurement(record.get(raw_name))
        effective, effective_invalid = _finite_measurement(
            record.get(effective_name)
        )
        if raw_invalid or effective_invalid:
            hard_failure = "nonfinite_causal_signal"
            continue
        if raw is None or effective is None:
            continue
        causal_available = True
        scale = max(abs(raw), abs(effective), math.sqrt(sys.float_info.min))
        causal_gain = raw - effective
        causal_resolution = math.sqrt(_FLOAT32_EPSILON) * scale
        causal_significant = causal_gain > causal_resolution
        break
    diagnostics["causal_gain"] = causal_gain
    diagnostics["causal_numerical_resolution"] = causal_resolution

    if "occ_prob_min" in record and "occ_prob_max" in record:
        lower = _as_float(record.get("occ_prob_min"))
        upper = _as_float(record.get("occ_prob_max"))
        if lower is not None and upper is not None:
            dynamic_range = upper - lower
            scale = max(abs(lower), abs(upper), math.sqrt(sys.float_info.min))
            diagnostics["output_dynamic_range"] = dynamic_range
            if dynamic_range <= math.sqrt(_FLOAT32_EPSILON) * scale:
                hard_failure = "collapsed_output_dynamic_range"

    decoded_names = (
        "loss_decoded_asset",
        "loss_decoded_render",
        "loss_decoded_geometry",
    )
    decoded = [_as_float(record.get(name)) for name in decoded_names]
    if bool(record.get("decoder_enabled")) and all(
        value is not None and value == 0.0 for value in decoded
    ):
        warnings.append("decoded_objective_inactive")

    clipping = _as_float(record.get("clipping_ratio"))
    if clipping is not None and clipping > 0.0:
        active_shape = record.get("active_xyz")
        active_count = (
            _shape_product(active_shape[:-1])
            if isinstance(active_shape, (list, tuple)) and len(active_shape) > 1
            else None
        )
        diagnostics["clipping_ratio"] = clipping
        if active_count:
            diagnostics["expected_clipped_elements"] = clipping * active_count
            if clipping * active_count >= 1.0:
                warnings.append("trust_region_saturation_observed")

    return _Health(
        hard_failure=hard_failure,
        causal_available=causal_available,
        causal_significant=causal_significant,
        gradient_available=gradient_available,
        gradient_nonzero=gradient_nonzero,
        warnings=warnings,
        diagnostics=diagnostics,
    )


def _mdl_positive_evidence(gain: float, variance: float, comparisons: int) -> bool:
    if not math.isfinite(gain) or gain <= 0.0:
        return False
    variance = max(variance, _positive_floor(gain))
    # Gaussian log-likelihood gain versus an extra selected checkpoint; BIC
    # charges log(number of comparisons) nats for that additional choice.
    code_gain = 0.5 * gain * gain / variance
    return code_gain > math.log(max(comparisons, 2))


def _loss_components(
    record: Mapping[str, Any],
    *,
    primary: str,
) -> dict[str, float]:
    components: dict[str, float] = {}
    for name, value in record.items():
        if name == primary or not (
            name.startswith("loss_") or name in {"cfm_mse"}
        ):
            continue
        parsed = _as_float(value)
        if parsed is not None:
            components[name] = parsed
    return components


def _distributed_decision_schema(
    record: Mapping[str, Any],
    metric: str,
) -> list[str]:
    paths = {metric}
    for name in record:
        if name.startswith("loss_") or name in {
            "cfm_mse",
            "global_grad_norm",
            "optimizer_step_skipped",
        }:
            paths.add(name)
    gradients = record.get("adapter_grad_norms")
    if isinstance(gradients, Mapping):
        paths.update(f"adapter_grad_norms.{name}" for name in gradients)
    return sorted(paths)


def _copy_summary_record(record: Mapping[str, Any]) -> dict[str, Any]:
    # Summary telemetry contains only scalars and small containers; copying it
    # avoids mutating the rank-local JSON record during the all-rank reduction.
    return copy.deepcopy(dict(record))


def _set_nested(record: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current = record
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _effective_batch(record: Mapping[str, Any]) -> float:
    for name in ("effective_global_batch_size", "global_batch_size"):
        value = _as_float(record.get(name))
        if value is not None and value > 0.0:
            return value
    per_gpu = _as_float(record.get("per_gpu_batch_size"))
    world = _as_float(record.get("world_size"))
    if per_gpu is not None and per_gpu > 0.0:
        return per_gpu * max(world or 1.0, 1.0)
    return 1.0


def _positive_floor(*values: float) -> float:
    finite = [
        abs(float(value)) for value in values if math.isfinite(float(value))
    ]
    scale = max(finite) if finite else sys.float_info.min
    return max(math.ulp(scale) ** 2, sys.float_info.min)


def _shape_product(values: Any) -> Optional[int]:
    try:
        result = 1
        for value in values:
            result *= int(value)
        return result
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _finite_measurement(value: Any) -> tuple[float | None, bool]:
    """Return (finite value, explicitly present but non-finite/invalid)."""

    if value is None:
        return None, False
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None, True
    return (out, False) if math.isfinite(out) else (None, True)


def _get_nested(record: Mapping[str, Any], key: str) -> Any:
    if key in record:
        return record[key]
    current: Any = record
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in {"1", "true", "yes", "y", "on"}
