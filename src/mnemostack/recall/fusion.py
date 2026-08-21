"""Reciprocal Rank Fusion (RRF) — merges ranked lists from multiple retrievers."""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterable
from typing import Any

from .identity import memory_key

logger = logging.getLogger(__name__)


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

    One side effect worth knowing: when two entries turn out to be the same
    memory, the survivor's `sources` is replaced with the union of both —
    the dropped copy's arms are not lost. Nothing else about the items is
    touched, and items carrying no `sources` list are left exactly as they
    came in.

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
        # ONE LIST, ONE VOTE. RRF adds `1/(k+rank)` once per list an item
        # appears in, so a list that names the same memory twice — `1` at
        # one rank and `"1"` at another — would hand it two contributions
        # and let a single arm out-vote genuine agreement between two.
        # Normalising the key alone does not prevent that: the loop below
        # scores whatever it is handed, so the repeat has to be dropped
        # here, where the list is read.
        voted: set[Any] = set()
        for rank, (item, _score) in enumerate(ranked, start=1):
            # Use id or item itself as key
            key = _get_key(item)
            if key in voted:
                # Still pool what this copy knew, then move on: it is the
                # same memory, and its arms are not a second vote.
                item_map[key] = _pool_sources(item_map[key], item)
                continue
            voted.add(key)
            fused[key] = fused.get(key, 0.0) + w / (k + rank)
            if key not in item_map:
                item_map[key] = item
            else:
                # Collapsing two objects for one memory must not throw away
                # what the second one knew. Only the first survives as the
                # representative, so an arm that found this memory under
                # the other id representation would vanish from `sources`
                # — a wrong answer in the response and wrong evidence for
                # the reinforcement that reads it. The recaller pools this
                # explicitly on its own merge path; the generic callers
                # (query expansion, inference retry) have no such step, so
                # it belongs here, where the collapse happens.
                item_map[key] = _pool_sources(item_map[key], item)

    merged = [(item_map[key], fused[key]) for key in fused]
    merged.sort(key=lambda x: -x[1])
    if limit is not None:
        return merged[:limit]
    return merged


def _pool_sources(keeper: Any, dropped: Any) -> Any:
    """Union the arms of a memory that arrived twice.

    Returns a REPLACEMENT rather than editing anything. The caller's own
    objects are what arrive here — `inference_retry.merge_results` fuses
    the caller's memories directly — so editing the keeper, or even just
    its `sources` list, reaches back into data the caller still holds.
    (`dataclasses.replace()` is no protection either: it is shallow, and a
    "copy" shares the same `sources` list.) It reached back:
    `inference_retry.merge_results` fuses the caller's own memories with a
    retry's sub-results, and `answer.py` returns those originals unchanged
    when it REJECTS the retry, to represent what the served draft was
    actually generated from. Pooling in place put a discarded retry's arms
    on them anyway — wrong evidence for the very reinforcement this
    pooling exists to keep honest.

    Duck-typed on both sides: an item without a `sources` list (a bare id,
    a plain value) is left alone, and a `dropped` whose `sources` is not a
    list is ignored rather than iterated — a string would otherwise be
    walked character by character.

    Only `sources` is pooled, deliberately. The recaller's own merge also
    carries the larger `raw_vector_score`, but that field is read only by
    the vector floor, and neither caller this exists for — query expansion
    and inference retry — ever applies the floor to a fused list.
    """
    # Both item shapes `_get_key` accepts: an object with attributes, and
    # a mapping with an "id". Handling only the first meant a pair of dicts
    # collapsed under the identity rule and silently lost the second one's
    # arms — the same defect as for objects, in the shape the key function
    # explicitly supports.
    as_mapping = isinstance(keeper, dict)
    keeper_sources = keeper.get("sources") if as_mapping else getattr(keeper, "sources", None)
    if not isinstance(keeper_sources, list):
        return keeper
    dropped_sources = (
        dropped.get("sources") if isinstance(dropped, dict) else getattr(dropped, "sources", None)
    )
    if not isinstance(dropped_sources, list):
        return keeper
    added = [source for source in dropped_sources if source not in keeper_sources]
    if not added:
        return keeper
    try:
        merged = copy.copy(keeper)
        if as_mapping:
            # A copied dict is already the caller's data left alone; the
            # nested payload still needs its own copy, same as below.
            merged["sources"] = [*keeper_sources, *added]
            if isinstance(merged.get("payload"), dict):
                merged["payload"] = dict(merged["payload"])
            return merged
        merged.sources = [*keeper_sources, *added]
        payload = getattr(merged, "payload", None)
        if isinstance(payload, dict):
            # The copy is shallow, so `payload` still points at the
            # keeper's dict — and the fuse loops downstream write into
            # `payload` (the legacy path stamps `raw_vector_score` on
            # whatever fusion hands back). Left shared, that write lands on
            # an object this function promised not to touch. Same lesson as
            # `sources`, one field over.
            merged.payload = dict(payload)
    except Exception as exc:  # noqa: BLE001 — an unpoolable item must not fail a recall
        # The ASSIGNMENTS are inside the guard too, not just the copy: a
        # frozen dataclass copies happily and then raises on the very next
        # line, which would break a public function for input it used to
        # accept. Degrading costs one observability field, not the request
        # — but say it happened, or nobody learns that a custom item type
        # is quietly losing its second copy's arms.
        logger.debug("could not pool sources for %r: %s", type(keeper).__name__, exc)
        return keeper
    return merged


def _get_key(item: Any) -> Any:
    """Extract a hashable key for RRF deduplication.

    An id goes through `memory_key`, because one memory can carry `1` from
    one arm and `"1"` from another and this dict decides whether the two
    are the same thing. Keying them apart does not merely list a memory
    twice: the `limit` cut below then spends two of the caller's slots on
    it and drops a genuinely different memory to make room.

    An item that carries no id is returned as-is. Its identity is the
    caller's business — stringifying an arbitrary object here would merge
    things this function has no business merging.
    """
    if hasattr(item, "id"):
        return memory_key(item.id)
    if isinstance(item, dict) and "id" in item:
        return memory_key(item["id"])
    return item
