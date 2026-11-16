"""Sliding-window chunks for memories written through the remote surfaces.

``Ingestor(window_size=N)`` slides over a whole conversation in one call. A
remote caller (HTTP ``POST /memories``, MCP ``mnemostack_remember``) sends one
message or a small batch per request, so the neighbours of a new memory are
already in the store. This module builds the same window chunks from the
store: a window is N memories of one tenant and one ``source`` that are
neighbours by ``offset``.

One reconciliation serves the write path, the lifecycle calls and the
``rebuild-windows`` command (see :func:`sync_source_windows`):

- a POSITION is an offset held by at least one live (not retracted) base
  memory of the source;
- a position is usable when exactly one live memory holds it and that memory
  carries no validity bound and no ``index_root``. An offset held by two live
  memories has no defined order, and a memory with its own validity period
  must not become readable through a window that outlives it;
- a window is valid when its members are consecutive usable positions. A
  position that is not usable is never bridged.

Windows are derived data. A window that stops being valid is deleted, not
retracted, and is built again when its members are valid again.

Concurrency: callers hold the per-tenant write lock of this process. Writers
in other processes are not serialized; a write re-validates the source after
it stored its windows, and a retraction re-validates after it took effect, so
either order of two interleaved calls converges to valid windows only. A
window that such a race leaves MISSING is restored by ``rebuild-windows``.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections.abc import Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from mnemostack.embeddings.roles import EmbeddingSpaceError
from mnemostack.ingest import (
    DEFAULT_WINDOW_SEPARATOR,
    IngestItem,
    Ingestor,
    IngestStats,
    RemoteMemoryResult,
    _make_window_item,
    _scroll_keywords,
    _tenant_write_lock,
)
from mnemostack.ingest_policy import IngestPolicyError, SecretPolicy
from mnemostack.observability.recorder import counter
from mnemostack.quotas import QuotaExceededError, enforce_points_quota
from mnemostack.vector.patch import PayloadPatch
from mnemostack.vector.qdrant import TENANT_ID_KEY

log = logging.getLogger(__name__)

#: Window payload key: the ids of the member memories, in offset order.
WINDOW_MEMBERS_KEY = "_window_members"
#: Base payload key: set once a memory became a member of a window, so a
#: by-id retraction can tell without a scan whether windows may hold it.
WINDOWED_KEY = "_windowed"

_STATE_KEYS = (
    "source",
    "offset",
    "chunk_kind",
    "invalidated_at",
    "valid_from",
    "valid_until",
    "index_root",
    TENANT_ID_KEY,
    WINDOW_MEMBERS_KEY,
    WINDOWED_KEY,
)

_STATE_PAGE = 4096

#: Payload keys of a stored memory that are not caller metadata: the
#: pipeline writes them itself, so a window must not inherit them.
_SERVER_PAYLOAD_KEYS = frozenset(
    {
        "text",
        "source",
        "offset",
        "tags",
        "indexed_at",
        "invalidated_at",
        "access_count",
        "last_accessed",
        TENANT_ID_KEY,
    }
)

SyncMode = Literal["write", "rebuild", "prune"]


@dataclass
class WindowSync:
    """Outcome of one window reconciliation."""

    created: int = 0
    removed: int = 0
    kept: int = 0
    #: Offsets held by more than one live memory. No window uses them.
    ambiguous_offsets: int = 0
    #: Memories kept out of windows by a validity bound or an ``index_root``.
    restricted: int = 0
    #: Sources left alone because they hold window chunks that another
    #: writer (``Ingestor`` / ``mnemostack index``) built.
    foreign_sources: int = 0
    #: Windows not built, or removed by a rebuild, because the joined text
    #: trips the secret policy.
    policy_blocked: int = 0
    #: False when a window that should exist was not written. The reason is
    #: "quota", "embedding", "unsupported" or "error"; ``rebuild-windows``
    #: (or a repeat of the write) repairs it.
    complete: bool = True
    reason: str | None = None

    def add(self, other: WindowSync) -> None:
        self.created += other.created
        self.removed += other.removed
        self.kept += other.kept
        self.ambiguous_offsets += other.ambiguous_offsets
        self.restricted += other.restricted
        self.foreign_sources += other.foreign_sources
        self.policy_blocked += other.policy_blocked
        if not other.complete and self.complete:
            self.complete = False
            self.reason = other.reason


@dataclass
class _Base:
    id: Any
    offset: int
    blocked: bool
    flagged: bool


@dataclass
class _Window:
    id: Any
    members: tuple[str, ...]
    live: bool


@dataclass
class _SourceState:
    at: dict[int, list[_Base]] = field(default_factory=dict)
    windows: list[_Window] = field(default_factory=list)
    #: Window chunks without the member list: another writer manages them.
    foreign: int = 0
    _runs: dict[int, set[tuple[str, ...]]] = field(default_factory=dict)

    def runs(self, size: int) -> set[tuple[str, ...]]:
        """Member ids of every valid window of ``size``."""
        cached = self._runs.get(size)
        if cached is not None:
            return cached
        offsets = sorted(self.at)
        out: set[tuple[str, ...]] = set()
        for start in range(0, len(offsets) - size + 1):
            holders = [self.at[o] for o in offsets[start : start + size]]
            if all(len(h) == 1 and not h[0].blocked for h in holders):
                out.add(tuple(str(h[0].id) for h in holders))
        self._runs[size] = out
        return out

    def valid(self, window: _Window) -> bool:
        size = len(window.members)
        return window.live and size >= 2 and window.members in self.runs(size)

    def bases(self) -> dict[str, _Base]:
        return {str(h.id): h for holders in self.at.values() for h in holders}


def _owned(payload: dict[str, Any], tenant: str | None) -> bool:
    # Unscoped callers own only unstamped points: a window must never join
    # the memories of different tenants that share a source name.
    return payload.get(TENANT_ID_KEY) == tenant


def _read_source(store: Any, source: str, tenant: str | None) -> _SourceState | None:
    """Positions and remote-built windows of one source, or None when the
    store cannot be scanned (a duck store without ``scroll``)."""
    scroll = getattr(store, "scroll", None)
    if not callable(scroll):
        return None
    kwargs: dict[str, Any] = {}
    supported = _scroll_keywords(scroll)
    if tenant is not None:
        kwargs["tenant"] = tenant
    if supported is None or "payload_keys" in supported:
        kwargs["payload_keys"] = _STATE_KEYS
    if supported is None or "batch_size" in supported:
        # The projected payloads are small, and the cost of the scan is its
        # round trips: a long conversation is read in few large pages.
        kwargs["batch_size"] = _STATE_PAGE
    state = _SourceState()
    for hit in scroll(filters={"source": source}, **kwargs):
        payload = dict(getattr(hit, "payload", None) or {})
        if payload.get("source") != source or not _owned(payload, tenant):
            continue  # array/partial match, or another tenant's point
        if payload.get("chunk_kind") == "sliding_window":
            members = payload.get(WINDOW_MEMBERS_KEY)
            # Windows without the member list were written by an operator
            # path (Ingestor / `mnemostack index`) and are not managed here.
            if isinstance(members, list):
                state.windows.append(
                    _Window(
                        id=hit.id,
                        members=tuple(str(m) for m in members),
                        live=not payload.get("invalidated_at"),
                    )
                )
            else:
                state.foreign += 1
            continue
        offset = payload.get("offset")
        if payload.get("invalidated_at") or not isinstance(offset, int) or isinstance(offset, bool):
            continue
        state.at.setdefault(offset, []).append(
            _Base(
                id=hit.id,
                offset=offset,
                blocked=_restricted(payload),
                flagged=bool(payload.get(WINDOWED_KEY)),
            )
        )
    return state


def _restricted(payload: dict[str, Any]) -> bool:
    return any(payload.get(key) is not None for key in ("valid_from", "valid_until", "index_root"))


def _member_item(payload: dict[str, Any], text_key: str) -> IngestItem:
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in _SERVER_PAYLOAD_KEYS and not key.startswith("_") and key != text_key
    }
    return IngestItem(
        text=payload["text"],
        source=payload["source"],
        offset=payload["offset"],
        metadata=metadata,
        tags=list(payload.get("tags") or []),
    )


def _window_item(
    members: Sequence[dict[str, Any]],
    member_ids: tuple[str, ...],
    text_key: str,
    separator: str,
) -> IngestItem:
    item = _make_window_item(
        [_member_item(payload, text_key) for payload in members], len(members), separator
    )
    item.metadata[WINDOW_MEMBERS_KEY] = list(member_ids)
    if text_key != "text":
        item.metadata[text_key] = item.text
    return item


def _delete(store: Any, ids: list[Any], tenant: str | None) -> None:
    if ids:
        store.delete_points(ids, **({"tenant": tenant} if tenant is not None else {}))


def sync_source_windows(
    store: Any,
    source: str,
    *,
    tenant: str | None,
    window_size: int,
    mode: SyncMode,
    embedding: Any = None,
    focus_ids: Iterable[Any] = (),
    max_points: int | None = None,
    text_key: str = "text",
    separator: str = DEFAULT_WINDOW_SEPARATOR,
    dry_run: bool = False,
    secret_policy: str = "off",
) -> WindowSync:
    """Bring the remote-built windows of one source in line with its memories.

    Every mode deletes the windows that are no longer valid. ``write`` also
    creates the missing windows of ``window_size`` that hold one of
    ``focus_ids``, so the embedding work of a request is bounded by the
    request. ``rebuild`` creates every missing window and deletes valid ones
    of another size. ``prune`` creates nothing and needs no ``embedding``.
    """
    result = WindowSync()
    state = _read_source(store, source, tenant)
    if state is None:
        if mode != "prune":
            result.complete, result.reason = False, "unsupported"
        return result
    result.ambiguous_offsets = sum(len(holders) > 1 for holders in state.at.values())
    result.restricted = sum(h.blocked for holders in state.at.values() for h in holders)
    stale = [w for w in state.windows if not state.valid(w)]
    kept = [w for w in state.windows if state.valid(w)]
    wanted: set[tuple[str, ...]] = set()
    if state.foreign:
        # Another writer keeps window chunks in this source. A remote-built
        # window next to them would be a second chunk over the same members,
        # so none is built, and those already stored give way.
        stale, kept = list(state.windows), []
        if mode != "prune":
            result.foreign_sources = 1
    elif mode != "prune" and window_size > 1:
        wanted = state.runs(window_size)
        if mode == "write":
            focus = {str(pid) for pid in focus_ids}
            wanted = {run for run in wanted if focus.intersection(run)}
    if mode == "rebuild":
        stale += [w for w in kept if len(w.members) != window_size]
        kept = [w for w in kept if len(w.members) == window_size]
    policy = SecretPolicy(secret_policy)
    blocked: set[tuple[str, ...]] = set()
    if mode == "rebuild" and secret_policy != "off" and kept:
        # A window built under another policy setting: the rebuild is where
        # the current policy is applied to what is already stored.
        blocked = _blocked_by_policy(store, kept, policy)
        stale += [w for w in kept if w.members in blocked]
        kept = [w for w in kept if w.members not in blocked]
        result.policy_blocked += len(blocked)
    bases = state.bases()
    missing = sorted(
        wanted - {w.members for w in kept} - blocked, key=lambda run: bases[run[0]].offset
    )
    result.kept = len(kept)
    result.removed = len(stale)
    # Planning reads the member texts and applies the policy. It writes
    # nothing, so a dry run reports what the real run would do.
    planned = (
        _plan(store, bases, missing, result, tenant=tenant, text_key=text_key,
              separator=separator, policy=policy)
        if missing
        else []
    )
    if dry_run:
        result.created = len(planned)
        return result
    _delete(store, [w.id for w in stale], tenant)
    if planned:
        _create(
            store, embedding, source, bases, planned, result,
            tenant=tenant, max_points=max_points, secret_policy=secret_policy,
        )
    return result


def window_chunk_id(source: str, offset: int, text: str, *, tenant: str | None = None) -> str:
    """Deterministic id of a remote-built window chunk.

    The same inputs as :func:`stable_chunk_id`, and the same digest, but the
    OTHER half of it. A memory with the source, the middle offset and the
    joined text of a window would otherwise get the id of that window, and
    one of the two would replace the other. No choice of tenant, source or
    text maps a memory onto a window id, short of a hash coincidence.
    """
    base = f"{source}|{offset}|{text}"
    if tenant is not None:
        base = f"{tenant}\x00{base}"
    digest = hashlib.sha256(base.encode()).hexdigest()
    return str(uuid.UUID(digest[32:]))


def _blocked_by_policy(
    store: Any, windows: list[_Window], policy: SecretPolicy
) -> set[tuple[str, ...]]:
    """Member tuples of the stored windows that the policy refuses. A failure
    of the check itself propagates: it must not read as a pass."""
    by_id = {str(w.id): w for w in windows}
    blocked: set[tuple[str, ...]] = set()
    ids = [w.id for w in windows]
    for start in range(0, len(ids), _STATE_PAGE):
        for point in store.client.retrieve(
            collection_name=store.collection,
            ids=ids[start : start + _STATE_PAGE],
            with_payload=True,
            with_vectors=False,
        ):
            try:
                policy.check(dict(point.payload or {}))
            except IngestPolicyError:
                blocked.add(by_id[str(point.id)].members)
    return blocked


def _plan(
    store: Any,
    bases: dict[str, _Base],
    missing: list[tuple[str, ...]],
    result: WindowSync,
    *,
    tenant: str | None,
    text_key: str,
    separator: str,
    policy: SecretPolicy,
) -> list[tuple[str, IngestItem]]:
    """The window chunks to write for ``missing``, with their ids. Reads only."""
    wanted_ids = list(dict.fromkeys(mid for run in missing for mid in run))
    payloads = {
        str(point.id): dict(point.payload or {})
        for point in store.client.retrieve(
            collection_name=store.collection,
            ids=[bases[mid].id for mid in wanted_ids],
            with_payload=True,
            with_vectors=False,
        )
    }
    planned: list[tuple[str, IngestItem]] = []
    for run in missing:
        members = [payloads.get(mid) for mid in run]
        # Re-checked on the full payload: the scan is a snapshot, and a
        # member may have been retracted or restricted since. Such a window
        # is no longer wanted, so skipping it is not a failure.
        if any(
            m is None
            or not _owned(m, tenant)
            or m.get("invalidated_at")
            or _restricted(m)
            or not isinstance(m.get("text"), str)
            or m.get("offset") != bases[mid].offset
            for m, mid in zip(members, run, strict=True)
        ):
            continue
        item = _window_item([m for m in members if m is not None], run, text_key, separator)
        try:
            # Members pass the policy one by one; the joined text can still
            # complete a blocked format (a key split over two messages).
            policy.check_item(item)
        except IngestPolicyError:
            result.policy_blocked += 1
            continue
        planned.append((window_chunk_id(item.source, item.offset, item.text, tenant=tenant), item))
    return planned


def _create(
    store: Any,
    embedding: Any,
    source: str,
    bases: dict[str, _Base],
    planned: list[tuple[str, IngestItem]],
    result: WindowSync,
    *,
    tenant: str | None,
    max_points: int | None,
    secret_policy: str,
) -> None:
    tkw: dict[str, Any] = {"tenant": tenant} if tenant is not None else {}
    items = [item for _wid, item in planned]
    window_ids = [wid for wid, _item in planned]
    if tenant is not None and max_points is not None:
        existing = store.retrieve_existing_ids(list(window_ids), tenant=tenant)
        try:
            enforce_points_quota(
                tenant, store.count(tenant=tenant), len(set(window_ids) - existing), max_points
            )
        except QuotaExceededError:
            result.complete, result.reason = False, "quota"
            return
    # Flag the members BEFORE the windows exist: a window whose member lacks
    # the flag would be invisible to a later by-id retraction of that member.
    unflagged = [
        bases[mid].id
        for mid in dict.fromkeys(m for i in items for m in i.metadata[WINDOW_MEMBERS_KEY])
        if not bases[mid].flagged
    ]
    if unflagged:
        store.apply_payload_patches(
            [PayloadPatch(id=pid, set_values={WINDOWED_KEY: True}) for pid in unflagged], **tkw
        )
    ingestor = Ingestor(
        embedding, store, skip_seen=False, tenant=tenant, max_points=max_points,
        secret_policy=secret_policy,
    )
    stats = IngestStats()
    try:
        # `_flush`, not `ingest`: the ids are those of the window id space.
        for start in range(0, len(planned), ingestor.batch_size):
            ingestor._flush(list(planned[start : start + ingestor.batch_size]), stats)
    except QuotaExceededError:
        result.complete, result.reason = False, "quota"
        _meter(tenant, items)
        return
    except Exception as exc:
        if not isinstance(exc, EmbeddingSpaceError):
            _meter(tenant, items)  # the provider was called; see ingest._meter_spend
        raise
    _meter(tenant, items)
    result.created = len(stats.ids)
    if result.created < len(items):
        result.complete, result.reason = False, "embedding"
    # A writer in another process may have changed the source between the
    # scan and the upsert (a retraction, a memory between two members).
    after = sync_source_windows(store, source, tenant=tenant, window_size=1, mode="prune")
    result.removed += after.removed


def _meter(tenant: str | None, items: list[IngestItem]) -> None:
    if tenant is None:
        return
    counter("mnemostack.tenant.embedded_chunks", len(items), labels={"tenant": tenant})
    counter(
        "mnemostack.tenant.embedded_chars",
        sum(len(item.text) for item in items),
        labels={"tenant": tenant},
    )


def sync_written(
    embedding: Any,
    store: Any,
    items: Sequence[IngestItem],
    results: Sequence[RemoteMemoryResult],
    *,
    tenant: str | None,
    window_size: int,
    max_points: int | None,
    text_key: str,
    secret_policy: str = "off",
) -> WindowSync:
    """Window step of one remote write. Never raises: the memories of the
    request are already stored, and a window failure must not report them
    as failed. The caller holds the tenant write lock."""
    by_source: dict[str, list[str]] = {}
    for item, outcome in zip(items, results, strict=True):
        # Duplicates count too, so a repeat of a request whose window step
        # failed repairs it.
        if item.source and outcome.status != "failed":
            by_source.setdefault(item.source, []).append(outcome.id)
    total = WindowSync()
    for source, ids in by_source.items():
        try:
            total.add(
                sync_source_windows(
                    store, source, tenant=tenant, window_size=window_size, mode="write",
                    embedding=embedding, focus_ids=ids, max_points=max_points, text_key=text_key,
                    secret_policy=secret_policy,
                )
            )
        except Exception:  # noqa: BLE001 — see docstring
            log.exception("window step failed after the memories were stored")
            total.add(WindowSync(complete=False, reason="error"))
    counter("mnemostack.ingest.remote_windows", total.created)
    return total


def _member_scopes(
    store: Any, ids: Sequence[Any], tenant: str | None
) -> dict[tuple[str | None, str], bool]:
    """``(owner tenant, source)`` of the named points, mapped to whether one
    of them carries the member flag. The owner comes from the point, not from
    the caller: an unscoped caller can retract a tenant's memory, and the
    windows to remove are that tenant's."""
    fetch = getattr(store, "retrieve_payload_fields", None)
    if not callable(fetch) or not ids:
        return {}
    tkw: dict[str, Any] = {"tenant": tenant} if tenant is not None else {}
    scopes: dict[tuple[str | None, str], bool] = {}
    for f in fetch(list(ids), ["source", WINDOWED_KEY, TENANT_ID_KEY], **tkw).values():
        source, owner = f.get("source"), f.get(TENANT_ID_KEY)
        # A legacy collection may hold any value under `tenant_id`. Only a
        # string or an absent owner can have remote-built windows.
        if isinstance(source, str) and source and (owner is None or isinstance(owner, str)):
            key = (owner, source)
            scopes[key] = scopes.get(key, False) or bool(f.get(WINDOWED_KEY))
    return scopes


def _check_after(store: Any, source: str, owner: str | None) -> None:
    try:
        sync_source_windows(store, source, tenant=owner, window_size=1, mode="prune")
    except Exception:  # noqa: BLE001
        # The retraction itself took effect, so the call must not report a
        # failure. The windows known before it are gone; this pass only
        # covers a concurrent writer.
        log.exception("window check after a retraction failed")


@contextmanager
def member_retraction(
    store: Any, ids: Sequence[Any], *, tenant: str | None, active: bool = False
) -> Iterator[None]:
    """Wrap a by-id retraction or deletion of memories.

    The windows that hold one of ``ids`` are deleted BEFORE the body runs, so
    a failure of the body cannot leave retracted text readable through a
    window. The pass after the body removes a window that a writer in another
    process stored in between.

    ``active`` says that this deployment builds windows now. The member flag
    is then not trusted as the trigger of the pass after the body: another
    process may set it, and store its window, after the flag was read here.
    With the setting off no writer builds windows and the flag alone decides,
    so a deployment that never used the feature pays one read by id.
    """
    scopes = _member_scopes(store, ids, tenant)
    wanted = {str(pid) for pid in ids}
    with ExitStack() as stack:
        for attempt in range(3):
            owners = sorted(
                {owner for owner, _ in scopes} | {tenant}, key=lambda t: (t is not None, str(t))
            )
            for owner in owners:
                stack.enter_context(_tenant_write_lock(owner))
            # Read again under the locks: a writer of this process may have
            # stored a named memory, flagged it, and stored its window, since
            # the read before.
            fresh = _member_scopes(store, ids, tenant)
            for key, flagged in fresh.items():
                scopes[key] = scopes.get(key, False) or flagged
            if {owner for owner, _ in fresh} <= set(owners) or attempt == 2:
                # After the last attempt a new owner stays unlocked. Its
                # sources are still visited below, like those of a writer in
                # another process.
                break
            # A new owner: its lock must be taken in the common order.
            stack.close()
        for (owner, source), flagged in scopes.items():
            state = _read_source(store, source, owner) if flagged else None
            if state is not None:
                _delete(store, [w.id for w in state.windows if wanted.intersection(w.members)], owner)
        try:
            yield
        finally:
            # Also after a failed body: it may have taken effect in part.
            for (owner, source), flagged in scopes.items():
                if flagged or active:
                    _check_after(store, source, owner)


class _SourceRetraction:
    def __init__(self, store: Any, source: str, active: bool):
        self._store, self._source = store, source
        self.needed = active
        self.owners: set[str | None] = set()

    def note(self, payloads: Sequence[dict[str, Any]]) -> list[bool]:
        """Record the selected points; True for each remote-built window."""
        marks: list[bool] = []
        for payload in payloads:
            owner = payload.get(TENANT_ID_KEY)
            window = (
                (owner is None or isinstance(owner, str))
                and payload.get("chunk_kind") == "sliding_window"
                and isinstance(payload.get(WINDOW_MEMBERS_KEY), list)
            )
            if owner is None or isinstance(owner, str):
                self.owners.add(owner)
                self.needed = self.needed or window or bool(payload.get(WINDOWED_KEY))
            marks.append(window)
        return marks

    def take_windows(self, ids: Sequence[Any], payloads: Sequence[dict[str, Any]]) -> list[Any]:
        """Delete the remote-built windows among the selected points and
        return the ids of the other points, in order."""
        marks = self.note(payloads)
        windows: dict[str | None, list[Any]] = {}
        for pid, payload, window in zip(ids, payloads, marks, strict=True):
            if window:
                windows.setdefault(payload.get(TENANT_ID_KEY), []).append(pid)
        for owner, window_ids in windows.items():
            _delete(self._store, window_ids, owner)
        return [pid for pid, window in zip(ids, marks, strict=True) if not window]


@contextmanager
def source_retraction(
    store: Any, source: str, *, tenant: str | None, active: bool = False
) -> Iterator[_SourceRetraction]:
    """Wrap a source-scoped retraction or deletion: selection and effect.

    The caller selects the points of the source INSIDE the block and passes
    them to ``take_windows``. The tenant write lock is held from the
    selection to the effect, so a write of this process cannot store a window
    over the selected memories in between. The pass after the body removes
    the windows that the selection did not hold: those of a later batch, and
    those of a writer in another process.
    """
    guard = _SourceRetraction(store, source, active)
    with _tenant_write_lock(tenant):
        try:
            yield guard
        finally:
            if guard.needed:
                for owner in guard.owners:
                    _check_after(store, source, owner)


__all__ = [
    "WINDOWED_KEY",
    "WINDOW_MEMBERS_KEY",
    "WindowSync",
    "member_retraction",
    "source_retraction",
    "sync_source_windows",
    "sync_written",
    "window_chunk_id",
]
