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
    A missing key never matches — a point that cannot be attributed to the
    filtered scope must not pass it.
    """
    if not filters:
        return True
    payload = payload or {}
    for key, condition in filters.items():
        if key not in payload:
            return False
        value = payload[key]
        if isinstance(condition, dict) and ("gte" in condition or "lte" in condition):
            gte = condition.get("gte")
            lte = condition.get("lte")
            try:
                if gte is not None and value < gte:
                    return False
                if lte is not None and value > lte:
                    return False
            except TypeError:
                # Incomparable types (e.g. str payload vs numeric bound):
                # cannot be proven inside the range — exclude.
                return False
        elif value != condition:
            return False
    return True
