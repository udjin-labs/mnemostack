"""Reciprocal Rank Fusion (RRF) — merges ranked lists from multiple retrievers."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[tuple[Any, float]]],
    k: int = 60,
    limit: int | None = None,
) -> list[tuple[Any, float]]:
    """Merge multiple ranked lists into a single fused ranking via RRF.

    RRF: score(item) = sum over lists of 1 / (k + rank_in_list)

    The `k` parameter (default 60) dampens the influence of low-ranked items.
    Items are identified by equality on the first tuple element.

    Scores from the input lists are IGNORED — RRF uses only rank positions.
    This is a feature: it tolerates score-scale differences between BM25,
    semantic similarity, and graph traversal scores.

    Args:
        ranked_lists: iterable of ranked results, each a list of (item, original_score)
        k: RRF dampening constant (higher k = flatter reward curve)
        limit: if set, return only top-N fused results

    Returns:
        Fused list of (item, rrf_score) sorted by score descending.
    """
    fused: dict[Any, float] = {}
    item_map: dict[Any, Any] = {}

    for ranked in ranked_lists:
        for rank, (item, _score) in enumerate(ranked, start=1):
            # Use id or item itself as key
            key = _get_key(item)
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
            if key not in item_map:
                item_map[key] = item

    merged = [(item_map[key], fused[key]) for key in fused]
    merged.sort(key=lambda x: -x[1])
    if limit is not None:
        return merged[:limit]
    return merged


def _get_key(item: Any) -> Any:
    """Extract a hashable key for RRF deduplication."""
    if hasattr(item, "id"):
        return item.id
    if isinstance(item, dict) and "id" in item:
        return item["id"]
    return item
