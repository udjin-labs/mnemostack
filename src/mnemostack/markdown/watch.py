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


def _is_markdown(path: str) -> bool:
    return path.lower().endswith(".md")


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
        rpath = str(Path(path).resolve())
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
        snap: dict[str, float] = {}
        for p in Path(self.root).rglob("*.md"):
            try:
                snap[str(p.resolve())] = p.stat().st_mtime
            except OSError:
                continue
        return snap

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

    def run(self, stop_event: threading.Event | None = None) -> None:
        """Block, applying changes until stop_event is set (or KeyboardInterrupt)."""
        if _WATCHDOG:
            self._run_watchdog(stop_event)
        else:
            self._run_polling(stop_event)

    def _run_polling(self, stop_event: threading.Event | None) -> None:
        snap = self._scan()  # baseline: the folder is already indexed
        while not (stop_event is not None and stop_event.is_set()):
            self._sleep(self.poll_interval)
            snap = self.poll_once(snap)
            self.flush(force=True)  # a poll already batches, so apply immediately

    def _run_watchdog(self, stop_event: threading.Event | None) -> None:  # pragma: no cover - needs watchdog + real FS
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
        try:
            while not (stop_event is not None and stop_event.is_set()):
                self._sleep(self._debouncer.delay)
                self.flush()
        finally:
            observer.stop()
            observer.join()
