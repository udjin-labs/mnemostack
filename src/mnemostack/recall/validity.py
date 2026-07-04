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

import re
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .recaller import RecallResult

# A trailing ISO timezone suffix: ``Z`` or ``±HH:MM`` / ``±HHMM`` (4 offset
# digits, so a date's ``-15`` never matches). Stripped only to test whether the
# remainder is a bare calendar date — i.e. a date carrying a zone but no time.
_ZONE_SUFFIX_RE = re.compile(r"([Zz]|[+-]\d{2}:?\d{2})$")

VALID_FROM = "valid_from"
VALID_UNTIL = "valid_until"
INVALIDATED_AT = "invalidated_at"


def is_current(payload: dict[str, Any] | None) -> bool:
    """True when the payload carries no ``invalidated_at`` marker."""
    if not payload:
        return True
    return not payload.get(INVALIDATED_AT)


def _parse_iso(value: Any) -> datetime | None:
    """Parse an ISO-8601 value with ``datetime.fromisoformat``, preserving tz.

    Returns the datetime exactly as parsed — **naive stays naive, aware stays
    aware** — or None if unparseable. A trailing ``Z``/``z`` is accepted. The
    preserved tzinfo is what lets ``to_utc_iso`` tell an offset-bearing datetime
    (which must be UTC-normalized) from a naive one, without a second parse.
    """
    try:
        text = str(value)
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _to_instant(value: Any) -> datetime | None:
    """Parse an ISO-8601 value to an aware UTC datetime, or None if unparseable.

    Naive timestamps are assumed UTC; a trailing ``Z`` is accepted. Returning
    None lets callers fall back to a lexicographic compare for values that are
    not full ISO instants (e.g. a bare ``2026-03-01`` date).
    """
    dt = _parse_iso(value)
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _is_bare_date(text: str) -> bool:
    """True when ``text`` is a pure calendar date (no time-of-day, no zone).

    Uses ``date.fromisoformat``: it accepts ``2024-01-15`` (and, on 3.11+, the
    basic ``20240115``) but rejects anything with a time or a zone suffix. This
    is what separates a bare date — left untouched so date-only graph data keeps
    its format — from a datetime that must be canonicalized, and it does so
    without guessing which separator character ``datetime.fromisoformat`` allowed
    (it permits ``T``, ``t``, a space, or any other single character).
    """
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def _utc_z(instant: datetime) -> str:
    """Format an aware datetime as a UTC ``…Z`` string, preserving its precision.

    Emits ``Z`` (not ``+00:00``) because graph predicates compare these as raw
    strings and ``Z`` is the common form of existing UTC rows. Precision is
    **not** widened: a whole-second instant stays ``…00:00Z`` so it still
    compares equal to bounds written by the previous normalizer (forcing
    microseconds would make ``…00Z`` and ``…00.000000Z`` — the same instant —
    sort unequal, silently dropping facts at their exact boundary).
    """
    return instant.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_date_with_zone(text: str) -> bool:
    """True when ``text`` is a calendar date carrying a zone suffix but no time.

    ``2024-01-15+02:00`` / ``2024-01-15Z`` — ``datetime.fromisoformat`` parses
    these into a *naive* datetime (it reads the offset digits as a time), so
    they must not be treated as a real time-of-day. Stripping the zone leaves a
    bare date, which marks them as date-only and leaves them unchanged.
    """
    return _is_bare_date(_ZONE_SUFFIX_RE.sub("", text))


def to_utc_iso(value: Any) -> Any:
    """Canonicalize a timezone-bearing ISO instant to UTC ISO; pass others through.

    Graph validity predicates compare timestamps as raw strings in Cypher (no
    instant parsing available there), so an offset-bearing datetime must be
    normalized to UTC on both the write and the query side. A value is
    normalized when it is a genuine datetime — ``datetime.fromisoformat`` parses
    it, and it is not a pure date (``2024-01-15``) or a date carrying only a
    zone suffix (``2024-01-15+02:00``). That still catches every separator
    ``fromisoformat`` accepts (``T``, ``t``, space, ``_``, …) and separatorless
    basic forms. Bare dates, offset-suffixed dates, the ``current`` marker, and
    ``None`` are returned unchanged, so date-only graph data keeps its format.
    Precision is preserved (see ``_utc_z``) so bounds written by the previous
    normalizer still compare equal.
    """
    if value is None:
        return None
    text = str(value)
    raw = _parse_iso(text)
    if raw is None or _is_bare_date(text) or _is_date_with_zone(text):
        # Not a real datetime: a marker like "current", a pure calendar date,
        # or a date with only a zone suffix and no time-of-day.
        return text
    # A naive datetime is presumed UTC (matching ``_to_instant``) before
    # formatting — ``astimezone`` on a naive value would assume the system zone.
    instant = raw if raw.tzinfo is not None else raw.replace(tzinfo=timezone.utc)
    return _utc_z(instant)


def to_utc_instant(value: Any) -> Any:
    """Normalize an ``as_of`` query value to a full UTC instant string.

    Like ``to_utc_iso`` but also expands a **date-only** value (``2026-03-01``)
    to a full midnight-UTC instant (``2026-03-01T00:00:00Z``). Used for the
    ``as_of`` bound in graph Cypher, where bounds written from datetimes are
    stored as full instants: a bare-date ``as_of`` would otherwise be shorter
    than (and sort before) ``valid_from = '...T00:00:00Z'`` at the exact start,
    dropping a fact the vector-side ``valid_at`` treats as valid at midnight.
    Non-instant markers (``current``) and ``None`` pass through unchanged.
    """
    if value is None:
        return None
    dt = _to_instant(value)
    if dt is None:
        return str(value)
    # Same ``…Z`` form as ``to_utc_iso`` (precision preserved) so an ``as_of``
    # and a stored bound are directly string-comparable in Cypher.
    return _utc_z(dt)


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
