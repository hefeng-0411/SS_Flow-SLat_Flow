from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "should_stop": self.should_stop,
            "is_best": self.is_best,
            "metric": self.metric,
            "raw_value": self.raw_value,
            "score": self.score,
            "best_score": self.best_score,
            "bad_steps": self.bad_steps,
            "reason": self.reason,
        }


class EarlyStopper:
    """EMA-smoothed patience early stopping with relative min-delta and optional wall-time cap."""

    def __init__(
        self,
        *,
        enabled: bool,
        metric: str,
        mode: str = "min",
        patience: int = 5000,
        min_delta: float = 1e-4,
        relative_delta: bool = True,
        warmup_steps: int = 2000,
        min_steps: int = 5000,
        ema: float = 0.9,
        max_train_hours: float = 0.0,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("early stop mode must be 'min' or 'max'")
        self.enabled = bool(enabled)
        self.metric = metric
        self.mode = mode
        self.patience = max(1, int(patience))
        self.min_delta = max(0.0, float(min_delta))
        self.relative_delta = bool(relative_delta)
        self.warmup_steps = max(0, int(warmup_steps))
        self.min_steps = max(0, int(min_steps))
        self.ema = min(max(float(ema), 0.0), 0.999)
        self.max_train_seconds = max(0.0, float(max_train_hours)) * 3600.0
        self.start_time = time.time()
        self.best_score: float | None = None
        self.best_step: int | None = None
        self.bad_steps = 0
        self.smoothed: float | None = None
        self.seen_steps = 0

    @classmethod
    def from_args(cls, args, default_metric: str = "loss") -> "EarlyStopper":
        return cls(
            enabled=bool(getattr(args, "early_stop", False)),
            metric=getattr(args, "early_stop_metric", None) or default_metric,
            mode=getattr(args, "early_stop_mode", "min"),
            patience=getattr(args, "early_stop_patience", 5000),
            min_delta=getattr(args, "early_stop_min_delta", 1e-4),
            relative_delta=bool(getattr(args, "early_stop_relative_delta", True)),
            warmup_steps=getattr(args, "early_stop_warmup_steps", 2000),
            min_steps=getattr(args, "early_stop_min_steps", 5000),
            ema=getattr(args, "early_stop_ema", 0.9),
            max_train_hours=getattr(args, "max_train_hours", 0.0),
        )

    def update(self, record: Mapping[str, Any]) -> EarlyStopStatus:
        if not self.enabled:
            return EarlyStopStatus(False, False, False, self.metric, None, None, self.best_score, self.bad_steps, "disabled")

        step = _as_int(record.get("step"), default=0)
        raw = _as_float(_get_nested(record, self.metric))
        if raw is None:
            return EarlyStopStatus(True, False, False, self.metric, None, self.smoothed, self.best_score, self.bad_steps, "metric_missing")

        self.seen_steps = max(self.seen_steps, step)
        self.smoothed = raw if self.smoothed is None else self.ema * self.smoothed + (1.0 - self.ema) * raw
        is_best = self._is_improvement(self.smoothed)
        if is_best:
            self.best_score = self.smoothed
            self.best_step = step
            self.bad_steps = 0
        elif step >= self.warmup_steps and step >= self.min_steps:
            self.bad_steps += 1

        should_stop = False
        reason = "running"
        if step < self.warmup_steps:
            reason = "warmup"
        elif step < self.min_steps:
            reason = "below_min_steps"
        elif self.bad_steps >= self.patience:
            should_stop = True
            reason = f"plateau_patience_{self.patience}"

        if self.max_train_seconds > 0 and (time.time() - self.start_time) >= self.max_train_seconds:
            should_stop = True
            reason = f"max_train_hours_{self.max_train_seconds / 3600.0:.3g}"

        return EarlyStopStatus(True, should_stop, is_best, self.metric, raw, self.smoothed, self.best_score, self.bad_steps, reason)

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        delta = self.min_delta
        if self.relative_delta:
            delta *= max(1.0, abs(self.best_score))
        if self.mode == "min":
            return score < self.best_score - delta
        return score > self.best_score + delta

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_score": self.best_score,
            "best_step": self.best_step,
            "bad_steps": self.bad_steps,
            "smoothed": self.smoothed,
            "seen_steps": self.seen_steps,
            "metric": self.metric,
            "mode": self.mode,
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        if state.get("metric") and state.get("metric") != self.metric:
            return
        if state.get("mode") and state.get("mode") != self.mode:
            return
        self.best_score = _as_float(state.get("best_score"))
        self.best_step = _as_int(state.get("best_step"), default=0) if state.get("best_step") is not None else None
        self.bad_steps = _as_int(state.get("bad_steps"), default=0)
        self.smoothed = _as_float(state.get("smoothed"))
        self.seen_steps = _as_int(state.get("seen_steps"), default=0)


def add_early_stopping_args(parser):
    parser.add_argument("--early_stop", type=_str2bool, default=False)
    parser.add_argument("--early_stop_metric", type=str, default=None)
    parser.add_argument("--early_stop_mode", choices=["min", "max"], default="min")
    parser.add_argument("--early_stop_patience", type=int, default=5000)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--early_stop_relative_delta", type=_str2bool, default=True)
    parser.add_argument("--early_stop_warmup_steps", type=int, default=2000)
    parser.add_argument("--early_stop_min_steps", type=int, default=5000)
    parser.add_argument("--early_stop_ema", type=float, default=0.9)
    parser.add_argument("--max_train_hours", type=float, default=0.0)
    parser.add_argument("--save_best", type=_str2bool, default=True)
    return parser


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


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
