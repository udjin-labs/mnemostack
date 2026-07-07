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

from .sync import _is_within

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

#: Floor for the idle/flush tick so --watch-debounce/--watch-poll-interval 0
#: don't busy-loop on sleep(0).
_MIN_IDLE_TICK = 0.1

#: How often both backends run a safety-net reconcile — a store-vs-disk deletion
#: sweep (and, for watchdog, an mtime rescan for changes it can't emit events
#: for: directory moves, externally-edited symlink targets). Also the retry path
#: for a delete whose graph/vector write failed transiently.
_RESCAN_EVERY = 30.0


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
        # Floor both idle ticks so --watch-poll-interval 0 / --watch-debounce 0
        # (accepted as "as fast as possible") don't spin a core on sleep(0).
        self.poll_interval = max(poll_interval, _MIN_IDLE_TICK)
        self._sleep = sleep
        self._debouncer = _Debouncer(debounce, clock)
        self._on_result = on_result
        self._on_error = on_error
        #: Paths whose last upsert failed (e.g. a transient graph outage). Re-
        #: emitted on the next poll/rescan so a temporary failure self-heals
        #: instead of leaving vectors and links diverged until the next edit.
        self._failed: set[str] = set()

    # --- event core (filesystem-agnostic, unit-testable) ---

    def handle_event(self, path: str, kind: str) -> None:
        """Queue a change. Ignores non-markdown paths and paths outside root."""
        rpath = _abspath(path)  # do NOT resolve symlinks — see _abspath
        if not _is_markdown(rpath):
            return
        if not _is_within(rpath, self.root):
            return  # e.g. a move whose destination left the watched subtree
        self._debouncer.add(rpath, kind)

    def _apply(self, path: str, kind: str) -> None:
        from ..quotas import QuotaExceededError

        try:
            if kind == REMOVE:
                res = self.syncer.remove_file(path)
            else:
                res = self.syncer.index_file(path)
            self._failed.discard(path)
            if self._on_result is not None:
                self._on_result(res)
        except QuotaExceededError as exc:
            # An over-quota file can't fit the tenant's limit — don't queue it for
            # retry (it would just fail again on every flush/rescan and spam). Report
            # once and move on; it re-indexes if it changes or the quota is raised.
            self._failed.discard(path)
            if self._on_error is not None:
                self._on_error(path, kind, exc)
        except Exception as exc:  # noqa: BLE001 — one bad file must not stop the watch
            if kind != REMOVE:  # a failed delete is retried by reconcile_deletions
                self._failed.add(path)
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
        """One mtime scan: emit upsert for new/changed, remove for vanished.

        A file whose last upsert failed (``_failed``) is re-emitted even if its
        mtime is unchanged, so a transient failure retries each cycle until it
        succeeds (the mtime snapshot advances regardless of apply success).
        """
        cur = self._scan()
        for path, mtime in cur.items():
            if prev.get(path) != mtime or path in self._failed:
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
        """Queue changes since ``baseline`` (the startup gap); return new snapshot.

        Events are only *queued*, not force-applied: a file still being written
        as the watcher starts must wait out the debounce window like any other,
        so the loop's later ``flush()`` applies it once quiet.
        """
        if baseline is None:
            return self._scan()  # nothing to reconcile: current state is the baseline
        return self.poll_once(baseline)

    def reconcile(self) -> None:
        """Drop indexed sources whose file is gone (deletions events can miss).

        Scoped to the watched folder so a narrower ``--index-root`` watch never
        prunes siblings outside the observed subtree.
        """
        try:
            removed = self.syncer.reconcile_deletions(within=self.root)
        except Exception as exc:  # noqa: BLE001 — reconcile is best-effort, never fatal
            if self._on_error is not None:
                self._on_error("<reconcile>", REMOVE, exc)
            return
        if self._on_result is not None:
            from .sync import FileSyncResult

            for source in removed:
                self._on_result(FileSyncResult(source=source, pruned=1))

    def _run_polling(
        self, stop_event: threading.Event | None, baseline: dict[str, float] | None = None
    ) -> None:
        snap = self._catch_up(baseline)
        self.reconcile()  # drop sources deleted during the initial index
        next_reconcile = self._debouncer.clock() + _RESCAN_EVERY
        while not (stop_event is not None and stop_event.is_set()):
            self._sleep(self.poll_interval)
            snap = self.poll_once(snap)
            # Honor --watch-debounce even in polling mode: a file still being
            # written re-stamps its mtime each tick, resetting the window, so it
            # is only applied once it has been quiet for the debounce interval.
            self.flush()
            # Periodic reconcile: retry a delete whose graph/vector write failed
            # (poll_once won't re-emit a REMOVE for a path already gone from the
            # snapshot), plus sweep any deletion the diff missed.
            now = self._debouncer.clock()
            if now >= next_reconcile:
                next_reconcile = now + _RESCAN_EVERY
                self.reconcile()

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
        snap = self._catch_up(baseline)
        self.reconcile()  # drop sources deleted during the initial index
        # Idle sleep is decoupled from the debounce window: with --watch-debounce 0
        # (immediate) a raw sleep(delay) would spin a core, so floor the tick.
        idle = max(self._debouncer.delay, _MIN_IDLE_TICK)
        next_rescan = self._debouncer.clock() + _RESCAN_EVERY
        try:
            while not (stop_event is not None and stop_event.is_set()):
                self._sleep(idle)
                self.flush()
                # Periodic safety-net rescan: watchdog can't emit events for
                # everything — a directory rename/move (its children), or a
                # symlinked note whose target is edited via its real path outside
                # the tree. An mtime poll (stat follows the symlink) catches these,
                # and reconcile drops whatever disappeared.
                now = self._debouncer.clock()
                if now >= next_rescan:
                    next_rescan = now + _RESCAN_EVERY
                    snap = self.poll_once(snap)
                    self.reconcile()
        finally:
            observer.stop()
            observer.join()
