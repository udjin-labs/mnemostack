"""In-Python payload filter matching, mirroring VectorStore filter semantics.

Qdrant applies filters natively inside the vector store. Retrievers that
hold their corpus in memory (BM25) need the same semantics applied locally,
otherwise a fused recall with `filters=` would mix filtered vector hits with
unfiltered candidates from other sources — in multi-tenant deployments that
is a data-isolation leak, not just a ranking bug.
"""

from __future__ import annotations

from typing import Any


def payload_matches(payload: dict[str, Any] | None, filters: dict[str, Any] | None) -> bool:
    """True when *payload* satisfies every condition in *filters*.

    Mirrors `VectorStore._build_filter`: a plain value is an exact match; a
    `{"gte": ..., "lte": ...}` dict is an inclusive range (ISO timestamp
    strings compare lexicographically, which is correct for ISO-8601).
    Array-valued payload fields match when ANY element satisfies the
    condition — the same semantics Qdrant applies to arrays. A missing key
    never matches — a point that cannot be attributed to the filtered scope
    must not pass it.
    """
    if not filters:
        return True
    payload = payload or {}
    for key, condition in filters.items():
        if key not in payload:
            return False
        value = payload[key]
        candidates = value if isinstance(value, list) else [value]
        if isinstance(condition, dict) and ("gte" in condition or "lte" in condition):
            if not any(_in_range(c, condition) for c in candidates):
                return False
        elif condition not in candidates:
            return False
    return True


def _in_range(value: Any, condition: dict[str, Any]) -> bool:
    gte = condition.get("gte")
    lte = condition.get("lte")
    try:
        if gte is not None and value < gte:
            return False
        if lte is not None and value > lte:
            return False
    except TypeError:
        # Incomparable types — most commonly a TIME range crossing domains (an
        # epoch-int payload vs an ISO-string bound, or vice versa, on a foreign
        # collection). Before excluding, try to read all three as instants and
        # compare on the time line; only when that fails too is the value
        # genuinely unprovable inside the range.
        from .validity import parse_payload_instant

        v = parse_payload_instant(value)
        g = parse_payload_instant(gte) if gte is not None else None
        t = parse_payload_instant(lte) if lte is not None else None
        if v is None or (gte is not None and g is None) or (lte is not None and t is None):
            return False
        if g is not None and v < g:
            return False
        if t is not None and v > t:
            return False
    return True
