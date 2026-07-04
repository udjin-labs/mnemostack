"""Validity model for stale-fact invalidation (bi-temporal, vector side).

Three optional ISO-8601 payload keys, all absent by default so existing
chunks and existing behavior are unchanged. This is the vector-side twin of
the graph's ``valid_from``/``valid_until`` model (see ``GraphStore``):

- ``valid_from``     — world-time: the fact became true at this instant.
- ``valid_until``    — world-time: the fact stopped being true at this instant.
- ``invalidated_at`` — system-time: when *we recorded* the fact as stale.

A fact is **current** iff ``invalidated_at`` is absent. Default recall returns
only current facts; point-in-time recall (``as_of``) ignores ``invalidated_at``
and reconstructs the world-time window instead, matching
``GraphStore.query_triples(as_of=...)``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .recaller import RecallResult

VALID_FROM = "valid_from"
VALID_UNTIL = "valid_until"
INVALIDATED_AT = "invalidated_at"


def is_current(payload: dict[str, Any] | None) -> bool:
    """True when the payload carries no ``invalidated_at`` marker."""
    if not payload:
        return True
    return not payload.get(INVALIDATED_AT)


def _to_instant(value: Any) -> datetime | None:
    """Parse an ISO-8601 value to an aware UTC datetime, or None if unparseable.

    Naive timestamps are assumed UTC; a trailing ``Z`` is accepted. Returning
    None lets callers fall back to a lexicographic compare for values that are
    not full ISO instants (e.g. a bare ``2026-03-01`` date).
    """
    try:
        text = str(value)
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _le(a: Any, b: Any) -> bool:
    """``a <= b`` comparing ISO instants when both parse, else lexicographically.

    Instant comparison is what makes timezone offsets correct: a bare string
    compare misreads ``2026-07-04T00:00:00+02:00`` (= 2026-07-03T22:00Z) as
    later than ``2026-07-03T23:00:00Z`` because its text starts with the next
    calendar day.
    """
    ia, ib = _to_instant(a), _to_instant(b)
    if ia is not None and ib is not None:
        return ia <= ib
    return str(a) <= str(b)


def valid_at(payload: dict[str, Any] | None, as_of: str) -> bool:
    """World-time point-in-time test: was this fact true at ``as_of``?

    ``valid_from <= as_of AND (valid_until absent OR valid_until > as_of)``.
    Absent bounds are open (indefinite past / indefinite future), matching the
    graph's legacy-NULL-is-current handling. Timezone-aware ISO instants are
    compared as instants (see ``_le``); non-instant strings fall back to a
    lexicographic compare, correct for same-format ISO dates.
    """
    if not payload:
        return True
    start = payload.get(VALID_FROM)
    end = payload.get(VALID_UNTIL)
    if start is not None and not _le(start, as_of):  # start > as_of
        return False
    if end is not None and _le(end, as_of):  # end <= as_of (exclusive)
        return False
    return True


def keep_payload(
    payload: dict[str, Any] | None,
    *,
    include_invalidated: bool = False,
    as_of: str | None = None,
) -> bool:
    """Whether a single payload should surface under the validity settings.

    The per-payload primitive behind ``filter_by_validity`` — used directly on
    retriever hits (``Hit`` / ``BM25Doc``) so stale facts are dropped *before*
    fusion cuts to top-K, not after.

    - ``as_of`` set: keep facts valid at that world-time instant (invalidated
      facts included — point-in-time reconstruction wants what was true then).
    - else ``include_invalidated=False`` (default): drop invalidated facts.
    - else: keep everything.
    """
    if as_of is not None:
        return valid_at(payload, as_of)
    if not include_invalidated:
        return is_current(payload)
    return True


def filter_by_validity(
    results: list[RecallResult],
    *,
    include_invalidated: bool = False,
    as_of: str | None = None,
) -> list[RecallResult]:
    """Drop results that should not surface given the validity settings.

    See ``keep_payload`` for the per-item predicate.
    """
    if include_invalidated and as_of is None:
        return results
    return [
        r
        for r in results
        if keep_payload(r.payload, include_invalidated=include_invalidated, as_of=as_of)
    ]
