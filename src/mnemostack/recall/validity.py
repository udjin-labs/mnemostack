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

from datetime import date, datetime, timezone
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
    """Parse an ISO-8601 date or datetime to an aware UTC datetime.

    Returns None only when the value is not an ISO instant at all (e.g. the
    ``current`` marker), letting callers fall back to a raw compare. A bare date
    maps to midnight; a naive datetime is assumed UTC; a trailing ``Z`` is
    accepted. Every separator ``datetime.fromisoformat`` allows (``T``, ``t``,
    space, ``_``, the ``+`` from ``isoformat(sep=...)``, …), basic and ISO-week
    forms, and comma-decimal fractions are handled by the parser itself.
    """
    try:
        text = str(value)
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _bare_date_iso(value: Any) -> str | None:
    """The value as an extended ISO date string iff it is a pure calendar date.

    ``2024-01-01`` stays ``2024-01-01``; a basic ``20240101`` is canonicalized to
    the extended form (on 3.11+, where ``date.fromisoformat`` accepts it).
    Anything with a time-of-day, offset, or marker returns None.
    """
    try:
        return date.fromisoformat(str(value)).isoformat()
    except ValueError:
        return None


def to_utc_iso(value: Any) -> Any:
    """Canonicalize a validity **bound** for storage; graph-comparable as a string.

    Graph validity predicates compare timestamps as raw strings in Cypher (no
    instant parsing available there), so bounds and ``as_of`` must share a
    canonical form. A **bare calendar date** is kept as an extended ISO date
    (``2024-01-01`` stays a date; a basic ``20240101`` is normalized to
    extended) — dates stay dates, which keeps date-only graph data in its long-
    standing format and comparable (as a prefix) with a midnight-expanded
    ``as_of``. A **datetime** — any separator ``datetime.fromisoformat`` accepts,
    a basic/ISO-week form, or an offset — is rewritten to the UTC instant
    ``YYYY-MM-DDTHH:MM:SS[.ffffff]Z``. The ``current`` marker, ``None``, and
    anything unparseable pass through unchanged.

    Precision is preserved (a whole-second instant stays ``…00Z``), not widened
    to microseconds, so bounds written by the previous normalizer still compare
    equal. Sub-second lexical ordering across *mixed* precision (a fractional
    bound vs. a whole-second ``as_of``) remains a known limitation of the raw-
    string compare; a full fix needs parsed-instant comparison in the graph
    query layer.
    """
    if value is None:
        return None
    bare = _bare_date_iso(value)
    if bare is not None:  # a pure date stays a date (not expanded to an instant)
        return bare
    dt = _to_instant(value)
    if dt is None:  # not an ISO instant (e.g. the "current" marker)
        return str(value)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def to_utc_instant(value: Any) -> Any:
    """Normalize an ``as_of`` **query** value to a full UTC instant string.

    Like :func:`to_utc_iso` but a bare date is **expanded to midnight UTC**
    (``2026-03-01`` → ``2026-03-01T00:00:00Z``): the as_of must be a full instant
    so it sorts correctly against full-instant stored bounds (a bare-date as_of
    would otherwise be shorter than, and sort before, ``valid_from`` at the exact
    start). A date-only stored *bound* stays a date (see :func:`to_utc_iso`) and
    still compares correctly as a prefix of the expanded as_of.
    """
    if value is None:
        return None
    dt = _to_instant(value)
    if dt is None:
        return str(value)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


#: Cypher regex (Java syntax, used with ``=~``) matching a canonical ISO date or
#: date-time bound — the only shapes Memgraph's ``datetime()`` can parse. A stored
#: bound not matching this (e.g. free text an LLM put in ``valid_from`` via
#: TripleExtractor: "early 2024", or an impossible date "2024-02-31", or junk
#: "2024-01-01TBD") falls back to the old raw-string comparison instead of
#: reaching ``datetime()``, which would raise and abort the whole query.
#:
#: Validates per-month day ranges and a strict time/offset suffix so impossible
#: calendar dates and malformed times are rejected. Uses ``[0-9]``/``[.]`` (no
#: backslashes) to avoid Cypher string-escape ambiguity. Every value this matches
#: is a date ``datetime()`` can parse — so the datetime() branch never raises.
#: February is capped at 28 because regex can't know leap years: a non-leap
#: ``…-02-29`` would abort ``datetime()``, so ALL ``-02-29`` fall to the
#: raw-string branch instead (a genuine leap-year Feb-29 bound thus compares as a
#: raw string — correct for a date, losing only the ultra-rare Feb-29 +
#: sub-second-precision fix). Fractional seconds are restricted to 3 or 6 digits
#: — the only counts Memgraph's ``datetime()`` accepts (milli/micro); any other
#: count raises, so e.g. ``.5`` or ``.123456789`` falls back. ``to_utc_iso``
#: only ever emits 0 or 6 fractional digits, so canonical bounds always match.
_GRAPH_TS_RE = (
    "([0-9]{4}-(0[13578]|1[02])-(0[1-9]|[12][0-9]|3[01])"
    "|[0-9]{4}-(0[469]|11)-(0[1-9]|[12][0-9]|30)"
    "|[0-9]{4}-02-(0[1-9]|1[0-9]|2[0-8]))"
    "(T([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]([.]([0-9]{3}|[0-9]{6}))?"
    "(Z|[+-]([01][0-9]|2[0-3]):?[0-5][0-9]))?"
)


def _graph_instant_expr(field: str) -> str:
    """Cypher expr parsing a (regex-vetted) bound string to a ZonedDateTime.

    A bare calendar date (no ``T``) has midnight-UTC appended first — Memgraph's
    ``datetime()`` rejects a date without a timezone (``"Timezone is not
    designated"``). Only reached for values matching :data:`_GRAPH_TS_RE`.
    """
    return (
        f"datetime(CASE WHEN {field} CONTAINS 'T' THEN {field} "
        f"ELSE {field} + 'T00:00:00Z' END)"
    )


def _graph_bound_clause(field: str, op: str) -> str:
    """One bound comparison: parsed-instant for canonical values, raw-string else.

    ``field <op> as_of`` where a canonical ISO value is compared as a parsed
    instant (fixes mixed sub-second precision) and anything else falls back to
    the pre-existing raw-string compare — which never raises, so a single
    malformed bound excludes/includes only its own fact instead of aborting the
    whole graph query.
    """
    return (
        f"CASE WHEN {field} =~ '{_GRAPH_TS_RE}' "
        f"THEN {_graph_instant_expr(field)} {op} datetime($as_of) "
        f"ELSE {field} {op} $as_of END"
    )


def graph_as_of_predicate(var: str) -> str:
    """Cypher point-in-time validity predicate comparing PARSED instants.

    ``valid_from <= as_of AND (valid_until absent/'current' OR valid_until >
    as_of)``, but a canonical ISO bound is parsed with ``datetime()`` instead of
    compared as a raw string, so mixed sub-second precision orders correctly (a
    raw compare misreads ``…00.5Z`` as *before* ``…00Z`` because ``.`` < ``Z``).
    Guards, in order:

    - The markers (``valid_from`` NULL, ``valid_until`` ``'current'``/NULL) are
      handled by ``CASE`` *before* any ``datetime()`` call — Memgraph raises on
      ``datetime(NULL)`` and ``datetime('current')`` and does not guarantee
      ``OR`` short-circuits.
    - A non-canonical bound (unparseable free text stored via an unvalidated LLM
      path) falls back to the old raw-string compare, so it never reaches
      ``datetime()`` and can't abort the query (see :func:`_graph_bound_clause`).

    Bind a ``datetime()``-parseable ``$as_of`` (a full instant, via
    :func:`to_utc_instant`). Verified against Memgraph.
    """
    vf, vu = f"{var}.valid_from", f"{var}.valid_until"
    return (
        f"(CASE WHEN {vf} IS NULL THEN true "
        f"ELSE {_graph_bound_clause(vf, '<=')} END) AND "
        f"(CASE WHEN {vu} = 'current' OR {vu} IS NULL THEN true "
        f"ELSE {_graph_bound_clause(vu, '>')} END)"
    )


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
