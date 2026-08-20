"""Server-side access accounting — `access_count` / `last_accessed`.

The recall pipeline's freshness stage already reads these two payload keys:
each recorded access extends a memory's effective half-life, so a fact that
keeps being retrieved decays more slowly than one that never is
(:func:`mnemostack.recall.pipeline.stages.compute_decay`). Until now nothing
in the stack ever WROTE them — a deployment that wanted reinforcement had to
have every client stamp the payloads itself, which means each client
reimplements the same read-modify-write, and a client that forgets silently
turns the whole stage off for everyone sharing the collection.

The service is the only party that sees every retrieval, so this records the
access where the retrieval happens. It is opt-in (`serve --record-access`):
it turns reads into writes, which is a cost and a data change an operator
must choose.

Contract:

- **Fail-open, always.** Bookkeeping must never fail a recall the caller
  already has the results of. Every failure is swallowed here, logged once,
  and counted — the recall response is identical either way.
- **Best-effort counting.** Qdrant has no atomic increment, so the new count
  is derived from the payload this recall already read. Two concurrent
  recalls of the same point can therefore record one increment instead of
  two. The reader clamps the reinforcement at 10 accesses, so a lost
  increment changes a half-life by at most a few percent — a lock or a
  read-back per hit would cost far more than it buys.
- **Tenant-scoped.** Patches go through the store's tenant-aware batch hook,
  so a foreign-owned point is skipped by the store itself, not by trust.
- **Point ids only.** A recall's results can include hits that are not
  vector points (a knowledge-graph node is named, not addressed by point
  id), and handing such an id to the store would fail the WHOLE batch. Ids
  are validated against the store's id domain first, so one graph hit cannot
  cost every other hit its bookkeeping.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .observability.recorder import counter
from .vector.patch import PayloadPatch, apply_patches_via

log = logging.getLogger(__name__)

#: Payload keys the freshness stage reads. Kept as literals here (the same
#: way the vector layer keeps its own marker keys) so this module does not
#: import the recall pipeline just to name two strings.
ACCESS_COUNT_KEY = "access_count"
LAST_ACCESSED_KEY = "last_accessed"

#: A stored counter is a payload field, not a metric: past the reader's
#: clamp it carries no information, so it is bounded rather than left to
#: grow without limit on a hot memory.
MAX_STORED_ACCESS_COUNT = 1_000_000


def _current_count(payload: dict[str, Any]) -> int:
    """The point's recorded access count, treating anything else as zero.

    The key is client-writable on deployments that stamped it themselves,
    so it can hold a string, a float, a bool, or nothing at all. None of
    those may raise here — this runs after a successful recall.
    """
    value = payload.get(ACCESS_COUNT_KEY)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _recordable_ids(results: Any) -> list[tuple[Any, dict[str, Any]]]:
    """(id, payload) for the hits this can safely patch, deduplicated."""
    from .ingest import coerce_point_ids, validate_remote_ids

    out: list[tuple[Any, dict[str, Any]]] = []
    seen: set[str] = set()
    for result in results:
        raw = getattr(result, "id", None)
        if raw is None or isinstance(raw, bool) or not isinstance(raw, (str, int)):
            continue
        if validate_remote_ids([raw]) is not None:
            continue  # not a point id (a graph node's name, for instance)
        pid = coerce_point_ids([raw])[0]
        key = str(pid)
        if key in seen:
            continue  # one recall, one increment per point
        seen.add(key)
        out.append((pid, dict(getattr(result, "payload", None) or {})))
    return out


def record_access(
    store: Any,
    results: Any,
    *,
    tenant: str | None = None,
    now: datetime | None = None,
) -> int:
    """Stamp an access on every point this recall returned.

    Returns the number of points the store confirmed patching (0 when
    disabled by having nothing to do, or when the write failed — see the
    fail-open rule in the module docstring).
    """
    entries = _recordable_ids(results)
    if not entries:
        return 0
    stamp = (now or datetime.now(timezone.utc)).isoformat()
    patches = [
        PayloadPatch(
            id=pid,
            set_values={
                ACCESS_COUNT_KEY: min(_current_count(payload) + 1, MAX_STORED_ACCESS_COUNT),
                LAST_ACCESSED_KEY: stamp,
            },
        )
        for pid, payload in entries
    ]
    try:
        applied = int(apply_patches_via(store, patches, tenant=tenant))
    except Exception:  # noqa: BLE001 — bookkeeping must not fail the recall
        log.warning("access recording failed for %d point(s)", len(patches), exc_info=True)
        counter("mnemostack.access.record_failed", 1)
        return 0
    counter("mnemostack.access.recorded", applied)
    return applied
