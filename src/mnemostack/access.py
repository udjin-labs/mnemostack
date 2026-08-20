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

**What turning it on changes about RANKING.** Not just a counter: because
``compute_decay`` returns 1.0 when ``last_accessed`` is missing, and nothing
in the stack wrote that key before, confidence decay has been inert on every
deployment that did not stamp the keys itself. Recording it makes the stage
live, and the effect is asymmetric in a way the "reinforcement" framing
hides — a point recalled ONCE and then left cold starts decaying from that
moment, while a point NEVER recalled keeps the undecayed ceiling forever. At
the preset 30-day half-life, one access and then silence scores 0.87 after a
week, 0.56 after a month, and floors at 0.10 after roughly half a year,
against a flat 1.0 for a memory nothing ever found. The factor multiplies the
whole blended score, so ``freshness_weight`` does not scale it down.

That is the deliberate trade of an Ebbinghaus model — recency of USE is the
signal, and a memory nobody has retrieved has no use to be recent — but it
is a real change in what the ranking means, and an operator should turn the
flag on knowing it. It only bites recalls that run the pipeline
(``full_pipeline``, the default on `/recall` and `/answer`); raw RRF output
is unaffected.

Contract:

- **Fail-open, always.** Bookkeeping must never fail a recall the caller
  already has the results of. Every failure is swallowed here, logged once,
  and counted — the recall response is identical either way.
- **The counter is read from the STORE, not from the hit.** A result's
  payload is only current if it came from the vector arm: a lexical
  (BM25) hit carries the in-process corpus SNAPSHOT taken at startup, so
  incrementing that would write 1 forever — and a fused result whose
  payload came from the stale arm could overwrite a higher stored count
  with a lower one. The current values are fetched for the whole batch in
  one round-trip before the write; the hit's own payload is only a
  fallback for a store without the batch reader, and the larger of the two
  always wins so a stale snapshot can never walk a counter backwards.
- **Best-effort counting.** Qdrant has no atomic increment, so read and
  write are still two steps: two concurrent recalls of the same point can
  record one increment instead of two. The reader clamps the reinforcement
  at 10 accesses, so a lost increment changes a half-life by at most a few
  percent — a lock per hit would cost far more than it buys.
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


def _stored_counts(store: Any, ids: list[Any], tenant: str | None) -> dict[str, int]:
    """Authoritative access counts for these ids, in one round-trip.

    Empty when the store has no batch reader or the read fails — the caller
    then falls back to the hit's own payload, which is right for a vector
    hit and merely stale for a lexical one. Never raises: this is
    bookkeeping.
    """
    reader = getattr(store, "retrieve_payload_fields", None)
    if not callable(reader):
        return {}
    try:
        rows = reader(ids, [ACCESS_COUNT_KEY], tenant=tenant)
    except Exception:  # noqa: BLE001 — fall back to the hit payloads
        log.warning("access counter read failed for %d point(s)", len(ids), exc_info=True)
        return {}
    return {key: _current_count(value) for key, value in rows.items()}


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
    stored = _stored_counts(store, [pid for pid, _ in entries], tenant)
    patches = [
        PayloadPatch(
            id=pid,
            set_values={
                # max(): the stored value is authoritative, the hit's payload
                # is the fallback — and taking the larger means a stale
                # lexical snapshot can never walk a counter backwards.
                ACCESS_COUNT_KEY: min(
                    max(stored.get(str(pid), 0), _current_count(payload)) + 1,
                    MAX_STORED_ACCESS_COUNT,
                ),
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
