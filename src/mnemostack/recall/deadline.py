"""One elapsed-time budget for a recall request."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from concurrent.futures import Future, TimeoutError
from copy import deepcopy
from dataclasses import replace
from threading import Lock, Thread
from typing import TypeVar, cast

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


_MISSING = object()


class _TrialMemo(dict):
    """A ``deepcopy`` memo that reads through to ``base`` and keeps its own writes.

    A copy that fails leaves ``base`` untouched; a copy that succeeds is
    merged into it. The ``copy`` module reads its memo through ``get`` and
    ``[]`` only.
    """

    def __init__(self, base: dict):
        super().__init__()
        self.base = base

    def get(self, key, default=None):
        value = dict.get(self, key, _MISSING)
        return self.base.get(key, default) if value is _MISSING else value

    def __missing__(self, key):
        return self.base[key]

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def __contains__(self, key):
        return dict.__contains__(self, key) or key in self.base


def _copy_plain_containers(value, memo: dict):
    """Copy plain dicts and lists without recursion; keep every other object.

    The reduced copy for a payload that ``deepcopy`` cannot handle: it holds
    an object that cannot be copied (a lock, a client handle) or it is nested
    too deep for the stack.
    """
    pending: list[tuple[object, object]] = []
    keep = memo.setdefault(id(memo), [])

    def copy(item):
        kind = type(item)
        if kind is not dict and kind is not list:
            return memo.get(id(item), item)
        key = id(item)
        if key not in memo:
            memo[key] = kind()
            keep.append(item)
            pending.append((item, memo[key]))
        return memo[key]

    result = copy(value)
    while pending:
        source, target = pending.pop()
        if type(source) is dict:
            for name, member in source.items():
                cast(dict, target)[name] = copy(member)
        else:
            cast(list, target).extend(copy(member) for member in cast(list, source))
    return result


def isolate_results(results, memo: dict | None = None):
    """Copies of the results for work that may be abandoned.

    Each payload is deep-copied, so nothing mutable in it stays shared with
    the originals. Objects shared between the results stay shared between
    the copies, as they are without a deadline. Pass one ``memo`` to several
    calls to keep that sharing across them: a result that was copied before
    maps to the same copy.

    A payload that ``deepcopy`` cannot handle never fails the recall. Its
    plain dicts and lists are copied and its other objects stay shared.
    """
    if memo is None:
        memo = {}
    keep = memo.setdefault(id(memo), [])
    copies = []
    for result in results:
        key = ("result", id(result))
        if key not in memo:
            trial = _TrialMemo(memo)
            try:
                payload = deepcopy(result.payload, trial)
            except Exception:
                payload = _copy_plain_containers(result.payload, memo)
            else:
                keep.extend(trial.pop(id(trial), ()))
                memo.update(trial)
            memo[key] = (replace(result, sources=list(result.sources), payload=payload), result)
        copies.append(memo[key][0])
    return copies


def validate_deadline(value: object) -> float | None:
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
        trace.deadline = deadline if deadline is not None else Deadline(cast(float, seconds))
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
