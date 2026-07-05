"""Metrics recorder interface + default no-op implementation.

All counters/histograms flow through a single recorder instance. Default is
NullRecorder (zero overhead). Users swap it via set_recorder() to integrate
with Prometheus, OpenTelemetry, StatsD, etc.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Protocol

logger = logging.getLogger(__name__)


class MetricsRecorder(Protocol):
    """Protocol for metrics backends.

    Implementations should be thread-safe and non-blocking. If a backend
    raises, exceptions must be caught inside the recorder so as not to
    affect business logic.
    """

    def record_counter(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Increment a counter by `value`."""

    def record_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record a histogram observation (typically ms for latency)."""


class NullRecorder:
    """Zero-overhead default."""

    def record_counter(self, name, value, labels=None):
        pass

    def record_histogram(self, name, value, labels=None):
        pass


class LoggingRecorder:
    """Debug recorder — logs every metric at DEBUG level.

    Useful in dev; not for production (too verbose).
    """

    def record_counter(self, name, value, labels=None):
        logger.debug("counter %s += %s %s", name, value, labels or {})

    def record_histogram(self, name, value, labels=None):
        logger.debug("histogram %s = %.2f %s", name, value, labels or {})


class InMemoryRecorder:
    """Aggregating recorder — stores metrics in memory.

    Useful for tests and short-lived scripts, and as the process-wide recorder
    behind the HTTP server's `/metrics` and `/status`. Writes come from recall
    running in FastAPI's sync threadpool while a probe reads concurrently, so
    every access to the underlying dicts is guarded by a lock — an unguarded
    read that iterated `counters` while a new key was being recorded could raise
    `RuntimeError: dictionary changed size during iteration`. Readers get a
    snapshot (`snapshot_counters` / `snapshot_histograms`); iterate those, never
    the live dicts.
    """

    def __init__(self):
        self.counters: dict[tuple, float] = {}
        self.histograms: dict[tuple, list[float]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(name: str, labels: dict[str, str] | None) -> tuple:
        if not labels:
            return (name,)
        return (name,) + tuple(sorted(labels.items()))

    def record_counter(self, name, value, labels=None):
        k = self._key(name, labels)
        with self._lock:
            self.counters[k] = self.counters.get(k, 0.0) + float(value)

    def record_histogram(self, name, value, labels=None):
        k = self._key(name, labels)
        with self._lock:
            self.histograms.setdefault(k, []).append(float(value))

    def counter_value(self, name: str, labels: dict[str, str] | None = None) -> float:
        with self._lock:
            return self.counters.get(self._key(name, labels), 0.0)

    def histogram_values(self, name: str, labels: dict[str, str] | None = None) -> list[float]:
        with self._lock:
            return list(self.histograms.get(self._key(name, labels), []))

    def snapshot_counters(self) -> dict[tuple, float]:
        """A point-in-time copy safe to iterate while writes continue."""
        with self._lock:
            return dict(self.counters)

    def snapshot_histograms(self) -> dict[tuple, list[float]]:
        """A point-in-time deep copy safe to iterate while writes continue."""
        with self._lock:
            return {k: list(v) for k, v in self.histograms.items()}

    def reset(self) -> None:
        with self._lock:
            self.counters.clear()
            self.histograms.clear()


# Module-level singleton recorder
_recorder: MetricsRecorder = NullRecorder()


def set_recorder(recorder: MetricsRecorder) -> None:
    """Install a new recorder as the global default."""
    global _recorder
    _recorder = recorder


def get_recorder() -> MetricsRecorder:
    return _recorder


def counter(name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
    """Increment a counter metric."""
    try:
        _recorder.record_counter(name, value, labels)
    except Exception as e:  # noqa: BLE001
        logger.debug("metrics recorder failed: %s", e)


@contextmanager
def histogram(name: str, labels: dict[str, str] | None = None):
    """Context manager that records elapsed milliseconds as a histogram observation.

    Usage:
        with histogram("mnemostack.recall.latency_ms", {"op": "search"}):
            results = do_work()
    """
    start = time.monotonic()
    try:
        yield
    finally:
        elapsed_ms = (time.monotonic() - start) * 1000.0
        try:
            _recorder.record_histogram(name, elapsed_ms, labels)
        except Exception as e:  # noqa: BLE001
            logger.debug("metrics recorder failed: %s", e)


def timed(name: str, labels: dict[str, str] | None = None):
    """Decorator variant of histogram() for timing a full function."""

    def decorator(func):
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs):
            with histogram(name, labels):
                return func(*args, **kwargs)

        return wrapper

    return decorator
