"""Server-side access accounting — `access_count` / `last_accessed`.

The recall pipeline's freshness stage already reads these two payload keys:
each recorded access earns a memory a bounded ranking bonus, so a fact that
keeps being retrieved outranks an equally similar one that never is
(:func:`mnemostack.recall.pipeline.stages.compute_access_boost`). Until now nothing
in the stack ever WROTE them — a deployment that wanted reinforcement had to
have every client stamp the payloads itself, which means each client
reimplements the same read-modify-write, and a client that forgets silently
turns the whole stage off for everyone sharing the collection.

The service is the only party that sees every retrieval, so this records the
access where the retrieval happens. It is opt-in (`serve --record-access`):
it turns reads into writes, which is a cost and a data change an operator
must choose.

**What turning it on changes about RANKING.** Not just a counter. The
freshness stage multiplies each blended score by a bounded reinforcement
bonus computed from these two keys
(:func:`mnemostack.recall.pipeline.stages.compute_access_boost`), so
recording them makes that term live. The bonus can only RAISE a used
memory's rank — never lower it — and it decays back to exactly 1.0 with
time since the last access, so a memory nothing has retrieved lately ends
up where it would have been with no access signal at all. A deployment
that leaves this flag off sees no ranking change whatsoever.

That direction is deliberate and was not always so. This term used to be
a DECAY measured from ``last_accessed``, which made being used strictly
punishing: recalled once and then left cold, a memory fell to 0.56 within
a month and hit a 0.1 floor within four, while a memory nothing had ever
found kept a flat 1.0 forever — junk nobody wanted outranked a useful
fact nobody had needed lately. Ageing by AGE is not lost as a result: it
is the same stage's ``freshness`` term, computed from the memory's own
timestamp. Folding age in here as well would count it twice.

The one thing to weigh before turning it on is that this is the only term
in the pipeline that a recall's own output feeds back into. Three things
bound it: the ceiling (``access_bonus_max``, 0.25 by default, and 0
removes the signal from ranking entirely), saturation in the counter, and
the fact that only points actually handed to the caller are recorded.

**Where it does not take effect.** Under ``text_search=qdrant_bm25`` the
lexical arm serves each point's payload from the corpus snapshot taken at
startup. Recording writes the store correctly — the counter accumulates,
and the values are right for anything the vector arm returns — but a point
returned ONLY by that arm keeps presenting its startup payload to the
freshness stage, so its own reinforcement does not reach ranking until the
service restarts. This is the same startup-snapshot property already
documented for invalidation and deletion on that arm, not a new one, and
refreshing the in-process corpus per recall is deliberately not attempted:
the snapshot is what makes that arm cheap. A deployment that wants
reinforcement to steer a purely lexical recall has to restart to pick it
up.

It only affects recalls that run the pipeline (``full_pipeline``, the
default on `/recall` and `/answer`); raw RRF output is unaffected.

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
  always wins so a stale snapshot can never walk a counter backwards. Cost
  is constant in the number of hits — that read, the store's own
  existence/ownership pre-check, and one batched write — never a request
  per point.
- **Best-effort counting.** Qdrant has no atomic increment, so read and
  write are still two steps: two concurrent recalls of the same point can
  record one increment instead of two. The reader saturates the bonus at 10
  accesses, so a lost increment moves the multiplier by at most a fraction
  of a percent — a lock per hit would cost far more than it buys.
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


def _stored_counts(store: Any, ids: list[Any], tenant: str | None) -> dict[str, int] | None:
    """Authoritative access counts for these ids, in one round-trip.

    Three outcomes, and they are NOT interchangeable:

    - a mapping — the authoritative values (a point absent from it is
      absent from the store or owned by another tenant);
    - ``{}`` from a store with no batch reader — the caller may fall back
      to the hit's own payload, since nothing better exists;
    - ``None`` — the read FAILED. The caller must not fall back here: a
      lexical hit's payload is a startup snapshot, so treating a transient
      read failure as "count 0" would overwrite a stored 7 with 1 and walk
      the counter backwards, which is exactly what reading the store was
      introduced to prevent.

    Never raises: this is bookkeeping.
    """
    reader = getattr(store, "retrieve_payload_fields", None)
    if not callable(reader):
        return {}
    try:
        rows = reader(ids, [ACCESS_COUNT_KEY], tenant=tenant)
    except Exception:  # noqa: BLE001 — reported through the return value
        log.warning("access counter read failed for %d point(s)", len(ids), exc_info=True)
        counter("mnemostack.access.count_read_failed", 1)
        return None
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
    try:
        return _record(store, results, tenant=tenant, now=now)
    except Exception:  # noqa: BLE001 — bookkeeping must not fail the recall
        # The WHOLE body, not just the write. Every step in _record is safe
        # by construction today, but "must never raise" is a contract for
        # the function, and a future edit adding a fallible step inside it
        # would otherwise break that contract with nothing to catch it.
        log.warning("access recording failed", exc_info=True)
        counter("mnemostack.access.record_failed", 1)
        return 0


def _record(
    store: Any,
    results: Any,
    *,
    tenant: str | None = None,
    now: datetime | None = None,
) -> int:
    entries = _recordable_ids(results)
    if not entries:
        return 0
    stamp = (now or datetime.now(timezone.utc)).isoformat()
    stored = _stored_counts(store, [pid for pid, _ in entries], tenant)
    patches = [
        PayloadPatch(
            id=pid,
            set_values=(
                # The read failed: stamp the TIME and leave the counter
                # alone. Half the bookkeeping beats a wrong number — the
                # decay stage still gets its input, and a stale lexical
                # snapshot cannot decrease a count that was never touched.
                {LAST_ACCESSED_KEY: stamp}
                if stored is None
                else {
                    # max(): the stored value is authoritative, the hit's
                    # payload is the fallback for a store that cannot be
                    # read — the larger of the two means a stale lexical
                    # snapshot can never walk a counter backwards.
                    ACCESS_COUNT_KEY: min(
                        max(stored.get(str(pid), 0), _current_count(payload)) + 1,
                        MAX_STORED_ACCESS_COUNT,
                    ),
                    LAST_ACCESSED_KEY: stamp,
                }
            ),
        )
        for pid, payload in entries
    ]
    applied = int(apply_patches_via(store, patches, tenant=tenant))
    counter("mnemostack.access.recorded", applied)
    return applied
