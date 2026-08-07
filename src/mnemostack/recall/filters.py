"""In-Python payload filter matching, mirroring VectorStore filter semantics.

Qdrant applies filters natively inside the vector store. Retrievers that
hold their corpus in memory (BM25) need the same semantics applied locally,
otherwise a fused recall with `filters=` would mix filtered vector hits with
unfiltered candidates from other sources — in multi-tenant deployments that
is a data-isolation leak, not just a ranking bug.
"""

from __future__ import annotations

from typing import Any


def payload_matches(
    payload: dict[str, Any] | None,
    filters: dict[str, Any] | None,
    *,
    timestamp_key: str = "timestamp",
    numeric_unit: str = "auto",
) -> bool:
    """True when *payload* satisfies every condition in *filters*.

    Mirrors `VectorStore._build_filter`: a plain value is an exact match; a
    `{"gte": ..., "lte": ...}` dict is an inclusive range (ISO timestamp
    strings compare lexicographically, which is correct for ISO-8601).
    Array-valued payload fields match when ANY element satisfies the
    condition — the same semantics Qdrant applies to arrays. A missing key
    never matches — a point that cannot be attributed to the filtered scope
    must not pass it.

    ``timestamp_key`` names the ONE field whose range comparisons may cross
    timestamp domains (an epoch-int payload vs ISO-string bounds, or vice
    versa, on a foreign collection): only that field falls back to comparing
    on the time line. Every other field keeps strict Qdrant semantics —
    incomparable types exclude — so a numeric-looking string in an unrelated
    field can never sneak past a numeric range as an accidental "instant".
    """
    if not filters:
        return True
    payload = payload or {}
    for key, condition in filters.items():
        if key not in payload:
            return False
        value = payload[key]
        candidates = value if isinstance(value, list) else [value]
        instant_ok = key == timestamp_key
        if isinstance(condition, dict) and ("gte" in condition or "lte" in condition):
            if not any(
                _in_range(c, condition, instant_ok=instant_ok, numeric_unit=numeric_unit)
                for c in candidates
            ):
                return False
        elif instant_ok and not _exact_instant_match(condition, candidates, numeric_unit):
            return False
        elif not instant_ok and condition not in candidates:
            return False
    return True


def _exact_instant_match(condition: Any, candidates: list[Any], numeric_unit: str) -> bool:
    """Exact match for the timestamp key: equal INSTANTS count as equal even
    across domains (an ISO condition vs an epoch payload names one moment).
    When either side doesn't parse, plain equality applies as before."""
    if condition in candidates:
        return True
    from .validity import parse_payload_instant

    want = parse_payload_instant(condition, numeric_unit=numeric_unit)
    if want is None:
        return False
    return any(
        parse_payload_instant(c, numeric_unit=numeric_unit) == want for c in candidates
    )


def _in_range(
    value: Any,
    condition: dict[str, Any],
    *,
    instant_ok: bool = False,
    numeric_unit: str = "auto",
) -> bool:
    gte = condition.get("gte")
    lte = condition.get("lte")
    if instant_ok:
        # The timestamp field compares ON THE TIME LINE whenever value and
        # bounds all parse as instants — not merely on TypeError: two STRINGS
        # from different domains (an ISO payload vs a numeric-string bound)
        # compare lexicographically without raising, silently passing scope
        # they shouldn't. Unparseable pieces fall through to the native
        # compare below (preserving plain string/number semantics).
        from .validity import parse_payload_instant

        v = parse_payload_instant(value, numeric_unit=numeric_unit)
        g = parse_payload_instant(gte, numeric_unit=numeric_unit) if gte is not None else None
        t = parse_payload_instant(lte, numeric_unit=numeric_unit) if lte is not None else None
        if (
            v is not None
            and (gte is None or g is not None)
            and (lte is None or t is not None)
        ):
            if g is not None and v < g:
                return False
            if t is not None and v > t:
                return False
            return True
    try:
        if gte is not None and value < gte:
            return False
        if lte is not None and value > lte:
            return False
    except TypeError:
        # Incomparable types: cannot be proven inside the range — exclude.
        return False
    return True


#: Payload key stamped by the graph retriever on a hit whose scope membership
#: was PROVEN via chunk attribution (see MemgraphRetriever._attributed). The
#: post-pipeline filter backstop honors it for graph-sourced results only —
#: their payloads are retriever-constructed, never user data — and the key is
#: reserved from enrichers/frontmatter on the vector write path.
ATTRIBUTED_FILTERS_KEY = "_attributed_filters"


def result_passes_filters(
    result: Any,
    filters: dict[str, Any] | None,
    *,
    timestamp_key: str = "timestamp",
    numeric_unit: str = "auto",
) -> bool:
    """The post-pipeline filter check for a RECALL RESULT.

    ``payload_matches`` semantics, plus one honest exception: a graph hit
    whose membership in EXACTLY this filter scope was proven through its
    chunks carries the proof marker — its payload legitimately lacks the
    filtered fields (graph nodes have no chunk payload), and re-requiring
    them would silently drop every attributed graph hit right after the
    retriever proved it. The marker is honored only for results sourced
    solely from the graph arm, and only when it records the SAME filters
    this call is enforcing.
    """
    payload = getattr(result, "payload", None) or {}
    if payload_matches(
        payload, filters, timestamp_key=timestamp_key, numeric_unit=numeric_unit
    ):
        return True
    if not filters:
        return True
    attributed = payload.get(ATTRIBUTED_FILTERS_KEY)
    sources = list(getattr(result, "sources", []) or [])
    return sources == ["memgraph"] and attributed == filters
