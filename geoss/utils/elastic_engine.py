from __future__ import annotations

import copy
import math
import os
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.distributed as dist


@dataclass
class OOMRetryResult:
    value: Any
    adjustment: Any
    retries: int


@dataclass
class TrainStepSnapshot:
    model: dict[str, torch.Tensor]
    optimizer: dict[str, Any]
    scaler: dict[str, Any] | None
    cpu_rng: torch.Tensor
    cuda_rng: list[torch.Tensor] | None
    sampler_state: dict[str, Any] | None


class FastPlateauDetector:
    """Aggressive, low-latency plateau detector for iteration-speed runs."""

    def __init__(
        self,
        *,
        enabled: bool,
        metric: str = "loss",
        mode: str = "min",
        window: int = 8,
        patience: int = 2,
        min_delta: float = 0.002,
        relative_delta: bool = True,
        warmup_steps: int = 200,
        min_steps: int = 300,
        max_train_hours: float = 0.0,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("plateau mode must be 'min' or 'max'")
        self.enabled = bool(enabled)
        self.metric = metric
        self.mode = mode
        self.window = max(1, int(window))
        self.patience = max(1, int(patience))
        self.min_delta = max(0.0, float(min_delta))
        self.relative_delta = bool(relative_delta)
        self.warmup_steps = max(0, int(warmup_steps))
        self.min_steps = max(0, int(min_steps))
        self.max_train_seconds = max(0.0, float(max_train_hours)) * 3600.0
        self.start_time = __import__("time").time()
        self.values: deque[float] = deque(maxlen=self.window)
        self.best_score: float | None = None
        self.best_step: int | None = None
        self.bad_steps = 0
        self.seen_steps = 0
        self.smoothed: float | None = None

    @classmethod
    def from_args(cls, args: Any, default_metric: str = "loss") -> "FastPlateauDetector":
        return cls(
            enabled=bool(getattr(args, "early_stop", False)),
            metric=getattr(args, "early_stop_metric", None) or default_metric,
            mode=getattr(args, "early_stop_mode", "min"),
            window=getattr(args, "early_stop_window", 8),
            patience=getattr(args, "early_stop_patience", 2),
            min_delta=getattr(args, "early_stop_min_delta", 0.002),
            relative_delta=bool(getattr(args, "early_stop_relative_delta", True)),
            warmup_steps=getattr(args, "early_stop_warmup_steps", 200),
            min_steps=getattr(args, "early_stop_min_steps", 300),
            max_train_hours=getattr(args, "max_train_hours", 0.0),
        )

    def update(self, record: Mapping[str, Any]):
        from geoss.utils.early_stopping import EarlyStopStatus

        if not self.enabled:
            return EarlyStopStatus(False, False, False, self.metric, None, None, self.best_score, self.bad_steps, "disabled")

        step = _as_int(record.get("step"), 0)
        raw = _as_float(_get_nested(record, self.metric))
        if raw is None:
            return EarlyStopStatus(True, False, False, self.metric, None, self.smoothed, self.best_score, self.bad_steps, "metric_missing")

        self.seen_steps = max(self.seen_steps, step)
        self.values.append(raw)
        self.smoothed = sum(self.values) / len(self.values)

        is_best = False
        should_stop = False
        reason = "running"
        if step < self.warmup_steps:
            reason = "warmup"
        elif step < self.min_steps:
            reason = "below_min_steps"
        elif len(self.values) < self.window:
            reason = "filling_window"
        else:
            is_best = self._is_improvement(self.smoothed)
            if is_best:
                self.best_score = self.smoothed
                self.best_step = step
                self.bad_steps = 0
            else:
                self.bad_steps += 1
                if self.bad_steps >= self.patience:
                    should_stop = True
                    reason = f"fast_plateau_patience_{self.patience}"

        if self.best_score is None and self.smoothed is not None:
            self.best_score = self.smoothed
            self.best_step = step
            is_best = True

        if self.max_train_seconds > 0 and (__import__("time").time() - self.start_time) >= self.max_train_seconds:
            should_stop = True
            reason = f"max_train_hours_{self.max_train_seconds / 3600.0:.3g}"

        return EarlyStopStatus(True, should_stop, is_best, self.metric, raw, self.smoothed, self.best_score, self.bad_steps, reason)

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        delta = self.min_delta * max(1.0, abs(self.best_score)) if self.relative_delta else self.min_delta
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
            "values": list(self.values),
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
        self.best_step = _as_int(state.get("best_step"), 0) if state.get("best_step") is not None else None
        self.bad_steps = _as_int(state.get("bad_steps"), 0)
        self.smoothed = _as_float(state.get("smoothed"))
        self.seen_steps = _as_int(state.get("seen_steps"), 0)
        self.values.clear()
        for value in state.get("values", [])[-self.window :]:
            parsed = _as_float(value)
            if parsed is not None:
                self.values.append(parsed)


class AsyncArtifactManager:
    """Single-process async checkpoint writer with atomic publish semantics."""

    def __init__(self, max_workers: int = 1, max_pending: int = 2) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="async_ckpt")
        self._lock = threading.Lock()
        self._futures: list[Future] = []
        self._max_pending = max(1, int(max_pending))

    def submit_checkpoint(self, path: str | Path, **payload: Any) -> Future:
        wait_for: Future | None = None
        with self._lock:
            self._futures = [item for item in self._futures if not item.done()]
            if len(self._futures) >= self._max_pending:
                wait_for = self._futures.pop(0)
        if wait_for is not None:
            wait_for.result()
        frozen = _freeze_payload_to_cpu(payload)
        future = self._pool.submit(write_checkpoint_atomic, Path(path), frozen)
        with self._lock:
            self._futures.append(future)
            self._futures = [item for item in self._futures if not item.done()]
        return future

    def drain(self) -> None:
        with self._lock:
            futures = list(self._futures)
            self._futures.clear()
        for future in futures:
            future.result()

    def close(self) -> None:
        self.drain()
        self._pool.shutdown(wait=True)


def write_checkpoint_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    torch.save(dict(payload), tmp_path)
    with tmp_path.open("ab") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def train_step_with_oom_retry(
    step_fn: Callable[[], Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any = None,
    sampler: Any = None,
    device: torch.device,
    batch_controller: Any,
    rebuild_after_adjustment: Callable[[Any], None] | None = None,
    max_retries: int = 8,
    log_oom: Callable[[dict[str, Any]], None] | None = None,
) -> OOMRetryResult:
    retries = 0
    while True:
        snapshot = capture_train_step_snapshot(model, optimizer, scaler, sampler)
        local_oom = torch.tensor(0, device=device, dtype=torch.int32)
        result = None
        caught_oom_message: str | None = None
        try:
            result = step_fn()
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            caught_oom_message = str(exc)
            local_oom.fill_(1)

        # Every rank must observe the same decision. If rank 0 succeeds while
        # rank 1 OOMs, rank 0 has already mutated parameters; it must roll back.
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_oom, op=dist.ReduceOp.MAX)

        if int(local_oom.item()) == 0:
            adjustment = batch_controller.update_after_success(device) if batch_controller is not None else None
            return OOMRetryResult(result, adjustment, retries)

        # Restore parameters, optimizer slots, AMP scale, RNG, and sampler epoch
        # before retrying. Otherwise replaying the same logical step would train
        # on a different stochastic graph or with partially mutated moments.
        restore_train_step_snapshot(snapshot, model, optimizer, scaler, sampler, device)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if dist.is_available() and dist.is_initialized():
            # DDP cannot safely re-enter the same reducer after a CUDA OOM that
            # may have happened after forward hooks were registered but before
            # every bucket received its backward gradient. Retrying in-process
            # can convert the original OOM into a misleading "unused parameter"
            # error on the next forward. Restore state, then force a process-
            # level retry so torchrun rebuilds the reducer from a clean graph.
            if log_oom is not None:
                log_oom({"event": "oom_restart_required", "retry": retries + 1, "distributed": True})
            raise RuntimeError(
                "CUDA OOM requires a DDP process restart after state rollback; "
                "the launcher should lower batch size and relaunch from the last checkpoint. "
                f"Original OOM: {caught_oom_message or 'observed on a peer rank'}"
            ) from None

        retries += 1
        if retries > max_retries:
            detail = f"after {max_retries} elastic retries"
            raise RuntimeError(
                f"CUDA OOM could not be recovered {detail}. "
                f"Original OOM: {caught_oom_message or 'observed on a peer rank'}"
            ) from None

        adjustment = batch_controller.update_after_oom(device)
        if rebuild_after_adjustment is not None:
            rebuild_after_adjustment(adjustment)
        if log_oom is not None:
            log_oom({"event": "oom_retry", "retry": retries, "adaptive_batch": adjustment.as_dict()})


def capture_train_step_snapshot(model, optimizer, scaler=None, sampler=None) -> TrainStepSnapshot:
    module = model.module if hasattr(model, "module") else model
    return TrainStepSnapshot(
        model={key: value.detach().cpu().clone() for key, value in module.state_dict().items()},
        optimizer=copy.deepcopy(optimizer.state_dict()),
        scaler=copy.deepcopy(scaler.state_dict()) if scaler is not None else None,
        cpu_rng=torch.get_rng_state().clone(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        sampler_state=_sampler_state_dict(sampler),
    )


def restore_train_step_snapshot(snapshot: TrainStepSnapshot, model, optimizer, scaler=None, sampler=None, device=None) -> None:
    module = model.module if hasattr(model, "module") else model
    module.load_state_dict(snapshot.model, strict=True)
    module.to(device) if device is not None else None
    optimizer.load_state_dict(snapshot.optimizer)
    if scaler is not None and snapshot.scaler is not None:
        scaler.load_state_dict(snapshot.scaler)
    torch.set_rng_state(snapshot.cpu_rng)
    if snapshot.cuda_rng is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot.cuda_rng)
    _load_sampler_state_dict(sampler, snapshot.sampler_state)


def _freeze_payload_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _freeze_payload_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_freeze_payload_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_freeze_payload_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def slice_batch_to_size(batch: Any, batch_size: int) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch[:batch_size] if batch.ndim > 0 and batch.shape[0] >= batch_size else batch
    if isinstance(batch, Mapping):
        return {key: slice_batch_to_size(value, batch_size) for key, value in batch.items()}
    if isinstance(batch, list):
        return batch[:batch_size]
    if isinstance(batch, tuple):
        return tuple(slice_batch_to_size(value, batch_size) for value in batch)
    return batch


def _sampler_state_dict(sampler: Any) -> dict[str, Any] | None:
    if sampler is None:
        return None
    if hasattr(sampler, "state_dict"):
        return copy.deepcopy(sampler.state_dict())
    state = {}
    if hasattr(sampler, "epoch"):
        state["epoch"] = int(sampler.epoch)
    return state or None


def _load_sampler_state_dict(sampler: Any, state: Mapping[str, Any] | None) -> None:
    if sampler is None or not state:
        return
    if hasattr(sampler, "load_state_dict"):
        sampler.load_state_dict(copy.deepcopy(dict(state)))
    elif "epoch" in state and hasattr(sampler, "epoch"):
        sampler.epoch = int(state["epoch"])


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "cuda out of memory" in text or "outofmemoryerror" in text or "out of memory" in text


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != math.inf else None


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_nested(record: Mapping[str, Any], key: str) -> Any:
    if key in record:
        return record[key]
    current: Any = record
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current
