"""Reciprocal Rank Fusion (RRF) — merges ranked lists from multiple retrievers."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[tuple[Any, float]]],
    k: int = 60,
    limit: int | None = None,
    weights: list[float] | None = None,
) -> list[tuple[Any, float]]:
    """Merge multiple ranked lists into a single fused ranking via RRF.

    RRF: score(item) = sum over lists of weight_list / (k + rank_in_list)

    The `k` parameter (default 60) dampens the influence of low-ranked items.
    Items are identified by equality on the first tuple element.

    Scores from the input lists are IGNORED — RRF uses only rank positions.
    This is a feature: it tolerates score-scale differences between BM25,
    semantic similarity, and graph traversal scores.

    Weights let callers express that some retrievers are more trustworthy
    than others for a given query shape. For example, on exact-token queries
    (IP addresses, tickers, IDs) a BM25 exact match is a stronger signal than
    a semantically-nearby vector hit. Pass `weights=[w_vector, w_bm25, ...]`
    in the same order as `ranked_lists`. Default (None) keeps every list at
    weight 1.0 — equivalent to classical RRF.

    Args:
        ranked_lists: iterable of ranked results, each a list of (item, original_score)
        k: RRF dampening constant (higher k = flatter reward curve)
        limit: if set, return only top-N fused results
        weights: optional per-list weights. Length must match `ranked_lists`
            (materialised if it is an iterator). Non-positive weights are
            clamped to 0. Missing entries default to 1.0.

    Returns:
        Fused list of (item, rrf_score) sorted by score descending.
    """
    # Materialise to a list so we can index with weights without consuming
    # the iterator twice.
    lists = [list(rl) for rl in ranked_lists]
    if weights is None:
        weight_seq = [1.0] * len(lists)
    else:
        weight_seq = list(weights)
        if len(weight_seq) < len(lists):
            weight_seq.extend([1.0] * (len(lists) - len(weight_seq)))
        weight_seq = [max(0.0, float(w)) for w in weight_seq]

    fused: dict[Any, float] = {}
    item_map: dict[Any, Any] = {}

    for list_idx, ranked in enumerate(lists):
        w = weight_seq[list_idx]
        if w == 0.0:
            continue
        for rank, (item, _score) in enumerate(ranked, start=1):
            # Use id or item itself as key
            key = _get_key(item)
            fused[key] = fused.get(key, 0.0) + w / (k + rank)
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
