from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any

_PREFIX = "KVPOOL_PERF_METRICS "
_MAX_SAMPLES_PER_EVENT = 8192
_DISABLED_MEASUREMENT: AbstractContextManager[None] = nullcontext()
_logger = logging.getLogger(__name__)


def _strict_enabled(value: str) -> bool:
    if value not in {"0", "1"}:
        raise ValueError(
            "VLLM_ASCEND_KVPOOL_PERF_METRICS must be either '0' or '1', "
            f"got {value!r}"
        )
    return value == "1"


def _positive_interval(value: str) -> int:
    try:
        interval = int(value)
    except ValueError as exc:
        raise ValueError(
            "VLLM_ASCEND_KVPOOL_PERF_METRICS_INTERVAL_SECONDS must be a "
            f"positive integer, got {value!r}"
        ) from exc
    if interval <= 0 or str(interval) != value:
        raise ValueError(
            "VLLM_ASCEND_KVPOOL_PERF_METRICS_INTERVAL_SECONDS must be a "
            f"positive integer, got {value!r}"
        )
    return interval


@dataclass(frozen=True)
class PerfMetricConfig:
    enabled: bool
    interval_seconds: int

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> PerfMetricConfig:
        if environ is None:
            from vllm_ascend import envs

            return cls(
                enabled=envs.VLLM_ASCEND_KVPOOL_PERF_METRICS,
                interval_seconds=(
                    envs.VLLM_ASCEND_KVPOOL_PERF_METRICS_INTERVAL_SECONDS
                ),
            )
        return cls(
            enabled=_strict_enabled(
                environ.get("VLLM_ASCEND_KVPOOL_PERF_METRICS", "0")
            ),
            interval_seconds=_positive_interval(
                environ.get(
                    "VLLM_ASCEND_KVPOOL_PERF_METRICS_INTERVAL_SECONDS",
                    "10",
                )
            ),
        )


@dataclass
class _EventStats:
    count: int = 0
    bytes: int = 0
    sum_ns: int = 0
    exclusive_sum_ns: int = 0
    max_ns: int = 0
    samples_ns: deque[int] = field(
        default_factory=lambda: deque(maxlen=_MAX_SAMPLES_PER_EVENT)
    )

    def add(
        self,
        elapsed_ns: int,
        exclusive_elapsed_ns: int,
        bytes_count: int,
    ) -> None:
        self.count += 1
        self.bytes += bytes_count
        self.sum_ns += elapsed_ns
        self.exclusive_sum_ns += exclusive_elapsed_ns
        self.max_ns = max(self.max_ns, elapsed_ns)
        self.samples_ns.append(elapsed_ns)


def _percentile(samples: list[int], percentile: float) -> int:
    if not samples:
        return 0
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile + 0.5)))
    return ordered[index]


class KVPoolPerfMetrics:
    def __init__(
        self,
        config: PerfMetricConfig,
        *,
        labels: Mapping[str, object] | None = None,
        emit_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.enabled = config.enabled
        self.interval_seconds = config.interval_seconds
        self._labels = dict(labels or {})
        self._emit_fn = emit_fn or _logger.info
        self._events: dict[str, _EventStats] = defaultdict(_EventStats)
        self._lock = threading.Lock()
        self._measurement_state = threading.local()
        self._stop = threading.Event()
        self._reporter: threading.Thread | None = None

    def configure_labels(self, **labels: object) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._labels.update(labels)

    @staticmethod
    def event_name(name: str, layer_id: int | None = None) -> str:
        return name if layer_id is None else f"{name}|layer={layer_id}"

    def record(
        self,
        name: str,
        elapsed_ns: int,
        *,
        bytes_count: int = 0,
        layer_id: int | None = None,
        exclusive_elapsed_ns: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        if exclusive_elapsed_ns is None:
            exclusive_elapsed_ns = elapsed_ns
        if elapsed_ns < 0 or exclusive_elapsed_ns < 0 or bytes_count < 0:
            raise ValueError(
                "elapsed_ns, exclusive_elapsed_ns, and bytes_count must be "
                "non-negative"
            )
        if exclusive_elapsed_ns > elapsed_ns:
            raise ValueError("exclusive_elapsed_ns cannot exceed elapsed_ns")
        event = self.event_name(name, layer_id)
        with self._lock:
            self._events[event].add(
                elapsed_ns,
                exclusive_elapsed_ns,
                bytes_count,
            )

    def measure(
        self,
        name: str,
        *,
        bytes_count: int = 0,
        layer_id: int | None = None,
    ) -> AbstractContextManager[None]:
        if not self.enabled:
            return _DISABLED_MEASUREMENT
        return self._measure_enabled(
            name,
            bytes_count=bytes_count,
            layer_id=layer_id,
        )

    @contextmanager
    def _measure_enabled(
        self,
        name: str,
        *,
        bytes_count: int,
        layer_id: int | None,
    ) -> Iterator[None]:
        stack = getattr(self._measurement_state, "stack", None)
        if stack is None:
            stack = []
            self._measurement_state.stack = stack
        started_ns = time.perf_counter_ns()
        frame = [started_ns, 0]
        stack.append(frame)
        try:
            yield
        finally:
            elapsed_ns = time.perf_counter_ns() - started_ns
            popped = stack.pop()
            if popped is not frame:
                raise RuntimeError("KVPool performance measurement stack corruption")
            child_elapsed_ns = int(frame[1])
            exclusive_elapsed_ns = max(0, elapsed_ns - child_elapsed_ns)
            try:
                self.record(
                    name,
                    elapsed_ns,
                    bytes_count=bytes_count,
                    layer_id=layer_id,
                    exclusive_elapsed_ns=exclusive_elapsed_ns,
                )
            finally:
                if stack:
                    stack[-1][1] += time.perf_counter_ns() - started_ns

    def snapshot(self, *, reset: bool = False) -> dict[str, Any]:
        with self._lock:
            labels = dict(self._labels)
            events = self._events
            if reset:
                self._events = defaultdict(_EventStats)

            result: dict[str, Any] = {}
            for name, stats in sorted(events.items()):
                samples = list(stats.samples_ns)
                result[name] = {
                    "count": stats.count,
                    "bytes": stats.bytes,
                    "sum_ms": stats.sum_ns / 1_000_000,
                    "exclusive_sum_ms": stats.exclusive_sum_ns / 1_000_000,
                    "p50_ms": _percentile(samples, 0.50) / 1_000_000,
                    "p95_ms": _percentile(samples, 0.95) / 1_000_000,
                    "max_ms": stats.max_ns / 1_000_000,
                    "sample_count": len(samples),
                }
        return {
            "schema_version": 2,
            "timestamp_ns": time.time_ns(),
            "interval_seconds": self.interval_seconds,
            "labels": labels,
            "events": result,
        }

    def emit(self, *, reset: bool = True) -> str | None:
        if not self.enabled:
            return None
        payload = self.snapshot(reset=reset)
        if not payload["events"]:
            return None
        line = _PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self._emit_fn(line)
        return line

    def start(self) -> None:
        if not self.enabled or self._reporter is not None:
            return

        def report() -> None:
            while not self._stop.wait(self.interval_seconds):
                self.emit(reset=True)

        self._reporter = threading.Thread(
            target=report,
            name="KVPoolPerfMetricsReporter",
            daemon=True,
        )
        self._reporter.start()

    def close(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        reporter = self._reporter
        if reporter is not None and reporter is not threading.current_thread():
            reporter.join(timeout=min(1.0, float(self.interval_seconds)))
        self.emit(reset=True)


_global_lock = threading.Lock()
_global_metrics: KVPoolPerfMetrics | None = None


def get_kvpool_perf_metrics() -> KVPoolPerfMetrics:
    global _global_metrics
    if _global_metrics is None:
        with _global_lock:
            if _global_metrics is None:
                _global_metrics = KVPoolPerfMetrics(PerfMetricConfig.from_env())
                _global_metrics.start()
    return _global_metrics


def reset_kvpool_perf_metrics_for_test() -> None:
    global _global_metrics
    with _global_lock:
        if _global_metrics is not None:
            _global_metrics.close()
        _global_metrics = None
