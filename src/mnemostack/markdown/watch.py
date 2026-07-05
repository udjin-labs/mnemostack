"""Watch a markdown folder and apply changes incrementally.

Backs ``index-markdown --watch``. Uses ``watchdog`` when installed (native OS
filesystem events) and falls back to an mtime-polling scan otherwise, so the
core stays dependency-light. Both sources funnel through the same debouncer and
per-file dispatch, so behavior is identical either way.

Editors emit bursts (write to a temp file, rename, touch) and users save
rapidly, so events are **debounced**: repeated events for one path within a
quiet window collapse to a single apply, and the *latest* event kind wins (a
final delete removes; anything else re-indexes the current file state). A
failure on one file is reported and never stops the watch.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .sync import FileSyncResult, MarkdownSyncer

try:  # optional: native OS filesystem events
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    _WATCHDOG = True
except Exception:  # pragma: no cover - exercised only where watchdog is absent
    _WATCHDOG = False


UPSERT = "upsert"
REMOVE = "remove"

#: Floor for the watchdog idle/flush tick so --watch-debounce 0 doesn't busy-loop.
_MIN_IDLE_TICK = 0.1


def _is_markdown(path: str) -> bool:
    return path.lower().endswith(".md")


def _abspath(path: str | Path) -> str:
    """Absolute path WITHOUT resolving symlinks.

    The initial index keys chunks/links by a file's path relative to the corpus
    root as walked (a symlinked note keeps its link path, not the target). The
    watcher must address the same source names, so it must NOT resolve symlinks
    (``Path.resolve()`` would), or incremental sync would update the wrong source
    — or fail ``relative_to`` when the target is outside the root.
    """
    return os.path.abspath(str(path))


def _scan_mtimes(root: str | Path) -> dict[str, float]:
    """Map every markdown file under ``root`` to its mtime (symlinks unresolved).

    Case-insensitive suffix like the indexer (``.MD``/``.Md`` count), and shared
    with the watcher so a pre-index snapshot compares against the same key form.
    """
    snap: dict[str, float] = {}
    for p in Path(root).rglob("*"):
        if not _is_markdown(p.name):
            continue
        try:
            if p.is_file():
                snap[_abspath(p)] = p.stat().st_mtime
        except OSError:
            continue
    return snap


class _Debouncer:
    """Coalesces rapid events per path; the latest kind within the window wins."""

    def __init__(self, delay: float, clock: Callable[[], float]):
        self.delay = delay
        self.clock = clock
        self._pending: dict[str, list[Any]] = {}  # path -> [kind, deadline]
        self._lock = threading.Lock()

    def add(self, path: str, kind: str) -> None:
        with self._lock:
            self._pending[path] = [kind, self.clock() + self.delay]

    def drain(self, *, force: bool = False) -> list[tuple[str, str]]:
        """Pop (path, kind) whose quiet window elapsed (or everything if force)."""
        now = self.clock()
        with self._lock:
            ready = [
                (p, kd[0]) for p, kd in self._pending.items() if force or kd[1] <= now
            ]
            for p, _ in ready:
                del self._pending[p]
        return ready


class MarkdownWatcher:
    """Applies markdown filesystem changes to a live index via a ``MarkdownSyncer``.

    The event-handling core (``handle_event`` / ``flush`` / ``poll_once``) is
    filesystem-agnostic and directly testable; ``run`` wires it to watchdog or
    the polling fallback.
    """

    def __init__(
        self,
        syncer: MarkdownSyncer,
        root: str | Path,
        *,
        debounce: float = 0.5,
        poll_interval: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        on_result: Callable[[FileSyncResult], None] | None = None,
        on_error: Callable[[str, str, Exception], None] | None = None,
    ):
        self.syncer = syncer
        self.root = str(Path(root).resolve())
        self.poll_interval = poll_interval
        self._sleep = sleep
        self._debouncer = _Debouncer(debounce, clock)
        self._on_result = on_result
        self._on_error = on_error

    # --- event core (filesystem-agnostic, unit-testable) ---

    def handle_event(self, path: str, kind: str) -> None:
        """Queue a change. Non-markdown paths are ignored."""
        rpath = _abspath(path)  # do NOT resolve symlinks — see _abspath
        if not _is_markdown(rpath):
            return
        self._debouncer.add(rpath, kind)

    def _apply(self, path: str, kind: str) -> None:
        try:
            if kind == REMOVE:
                res = self.syncer.remove_file(path)
            else:
                res = self.syncer.index_file(path)
            if self._on_result is not None:
                self._on_result(res)
        except Exception as exc:  # noqa: BLE001 — one bad file must not stop the watch
            if self._on_error is not None:
                self._on_error(path, kind, exc)

    def flush(self, *, force: bool = False) -> int:
        """Apply all due (or, with force, all pending) debounced changes."""
        applied = 0
        for path, kind in self._debouncer.drain(force=force):
            self._apply(path, kind)
            applied += 1
        return applied

    # --- polling fallback ---

    def _scan(self) -> dict[str, float]:
        return _scan_mtimes(self.root)

    def poll_once(self, prev: dict[str, float]) -> dict[str, float]:
        """One mtime scan: emit upsert for new/changed, remove for vanished."""
        cur = self._scan()
        for path, mtime in cur.items():
            if prev.get(path) != mtime:
                self.handle_event(path, UPSERT)
        for path in prev:
            if path not in cur:
                self.handle_event(path, REMOVE)
        return cur

    # --- run loops ---

    def run(
        self,
        stop_event: threading.Event | None = None,
        *,
        baseline: dict[str, float] | None = None,
    ) -> None:
        """Block, applying changes until stop_event is set (or KeyboardInterrupt).

        ``baseline`` is an mtime snapshot taken *before* the initial index (see
        ``_scan_mtimes``); passing it makes the watcher reconcile any file that
        changed during that index (the window before it started observing),
        closing the startup gap for both backends.
        """
        if _WATCHDOG:
            self._run_watchdog(stop_event, baseline)
        else:
            self._run_polling(stop_event, baseline)

    def _catch_up(self, baseline: dict[str, float] | None) -> dict[str, float]:
        """Reconcile changes since ``baseline`` immediately; return the new snapshot."""
        if baseline is None:
            return self._scan()  # nothing to reconcile: current state is the baseline
        snap = self.poll_once(baseline)
        self.flush(force=True)  # apply the catch-up now, not after a debounce window
        return snap

    def _run_polling(
        self, stop_event: threading.Event | None, baseline: dict[str, float] | None = None
    ) -> None:
        snap = self._catch_up(baseline)
        while not (stop_event is not None and stop_event.is_set()):
            self._sleep(self.poll_interval)
            snap = self.poll_once(snap)
            # Honor --watch-debounce even in polling mode: a file still being
            # written re-stamps its mtime each tick, resetting the window, so it
            # is only applied once it has been quiet for the debounce interval.
            self.flush()

    def _run_watchdog(  # pragma: no cover - needs watchdog + real FS
        self, stop_event: threading.Event | None, baseline: dict[str, float] | None = None
    ) -> None:
        watcher = self

        class _Handler(FileSystemEventHandler):
            def _emit(self, path: str, kind: str) -> None:
                watcher.handle_event(path, kind)

            def on_created(self, event):
                if not event.is_directory:
                    self._emit(event.src_path, UPSERT)

            def on_modified(self, event):
                if not event.is_directory:
                    self._emit(event.src_path, UPSERT)

            def on_deleted(self, event):
                if not event.is_directory:
                    self._emit(event.src_path, REMOVE)

            def on_moved(self, event):
                if not event.is_directory:
                    self._emit(event.src_path, REMOVE)
                    self._emit(event.dest_path, UPSERT)

        observer = Observer()
        observer.schedule(_Handler(), self.root, recursive=True)
        observer.start()
        # Start observing BEFORE the catch-up so changes during reconciliation
        # are queued (the debouncer coalesces any duplicates).
        self._catch_up(baseline)
        # Idle sleep is decoupled from the debounce window: with --watch-debounce 0
        # (immediate) a raw sleep(delay) would spin a core, so floor the tick.
        idle = max(self._debouncer.delay, _MIN_IDLE_TICK)
        try:
            while not (stop_event is not None and stop_event.is_set()):
                self._sleep(idle)
                self.flush()
        finally:
            observer.stop()
            observer.join()
