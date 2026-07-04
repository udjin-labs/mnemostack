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


def valid_at(payload: dict[str, Any] | None, as_of: str) -> bool:
    """World-time point-in-time test: was this fact true at ``as_of``?

    ``valid_from <= as_of AND (valid_until absent OR valid_until > as_of)``.
    Absent bounds are open (indefinite past / indefinite future), matching the
    graph's legacy-NULL-is-current handling. Comparison is lexicographic on
    ISO-8601 strings, which is correct for that format.
    """
    if not payload:
        return True
    start = payload.get(VALID_FROM)
    end = payload.get(VALID_UNTIL)
    if start is not None and str(start) > as_of:
        return False
    if end is not None and str(end) <= as_of:
        return False
    return True


def filter_by_validity(
    results: list[RecallResult],
    *,
    include_invalidated: bool = False,
    as_of: str | None = None,
) -> list[RecallResult]:
    """Drop results that should not surface given the validity settings.

    - ``as_of`` set: keep facts valid at that world-time instant (invalidated
      facts included — point-in-time reconstruction wants what was true then).
    - else ``include_invalidated=False`` (default): drop invalidated facts.
    - else: return the list unchanged.
    """
    if as_of is not None:
        return [r for r in results if valid_at(r.payload, as_of)]
    if not include_invalidated:
        return [r for r in results if is_current(r.payload)]
    return results
