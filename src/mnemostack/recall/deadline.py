"""One elapsed-time budget for a recall request."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from concurrent.futures import Future, TimeoutError
from threading import Lock, Thread
from typing import TypeVar

T = TypeVar("T")

MAX_ABANDONED_WORKERS = 64
_worker_lock = Lock()
_abandoned_workers = 0


def abandoned_workers_full():
    with _worker_lock:
        return _abandoned_workers >= MAX_ABANDONED_WORKERS


class _DeadlineFuture(Future):
    _abandoned = False
    _finished = False

    def abandon(self):
        global _abandoned_workers
        with _worker_lock:
            if self._abandoned:
                return True
            if self._finished or _abandoned_workers >= MAX_ABANDONED_WORKERS:
                return False
            self._abandoned = True
            _abandoned_workers += 1
            return True


def submit_deadline_work(call):
    """Start a daemon worker unless abandoned work has filled the cap."""
    if abandoned_workers_full():
        return None
    future = _DeadlineFuture()

    def work():
        global _abandoned_workers
        try:
            future.set_result(call())
        except BaseException as exc:
            future.set_exception(exc)
        finally:
            with _worker_lock:
                future._finished = True
                if future._abandoned:
                    _abandoned_workers -= 1

    Thread(target=work, daemon=True).start()
    return future


def validate_deadline(value: float | None) -> float | None:
    """Accept None or a positive finite number of seconds."""
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            "recall.overall_deadline must be a positive finite number of seconds, "
            f"got {value!r}"
        )
    return value


def get_deadline(trace) -> Deadline | None:
    """Read the optional deadline from a duck-typed trace."""
    value = getattr(trace, "deadline", None)
    return value if isinstance(value, Deadline) else None


def start_deadline(recaller, trace, *, deadline: Deadline | None = None):
    """Start or reuse a request budget when the trace can carry it."""
    if get_deadline(trace) is not None:
        return trace
    seconds = getattr(recaller, "overall_deadline", None)
    if deadline is None and (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
        or seconds <= 0
    ):
        return trace
    if trace is None:
        from .trace import RecallTrace

        trace = RecallTrace()
    try:
        trace.deadline = deadline if deadline is not None else Deadline(seconds)
    except (AttributeError, TypeError):
        pass
    return trace


class Deadline:
    """Bound waits without interrupting synchronous work already running."""

    def __init__(self, seconds: float):
        validate_deadline(seconds)
        self._end = time.monotonic() + seconds

    def remaining(self) -> float:
        return max(0.0, self._end - time.monotonic())

    def expired(self) -> bool:
        return self.remaining() == 0.0

    def run(self, call: Callable[[], T], fallback: T, *, on_timeout=None) -> T:
        def abandoned():
            if on_timeout is not None:
                on_timeout()
            return fallback

        if self.expired():
            return abandoned()
        future = submit_deadline_work(call)
        if future is None:
            return call()
        try:
            return future.result(timeout=self.remaining())
        except TimeoutError:
            if future.done() or not future.abandon():
                return future.result()
            return abandoned()
