"""The canonical post-recall flow shared by every entry point.

CLI, HTTP and MCP used to assemble recall → pipeline → rerank → top-K →
vector-floor each in their own way, so the same query ranked differently
depending on the surface. This module is the single implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .trace import RecallTrace, apply_rerank_safe

if TYPE_CHECKING:
    from .pipeline import Pipeline
    from .recaller import Recaller, RecallResult
    from .reranker import Reranker


def recall_flow(
    recaller: Recaller,
    query: str,
    limit: int = 10,
    *,
    pipeline: Pipeline | None = None,
    reranker: Reranker | None = None,
    filters: dict[str, object] | None = None,
    trace: RecallTrace | None = None,
) -> list[RecallResult]:
    """Run hybrid recall plus the canonical post-processing chain.

    With a pipeline, recall fetches a wider raw pool (3x the requested
    limit, at least 30) so the ranking stages have candidates to work
    with; the reranker is applied fail-open via `apply_rerank_safe`; the
    result is cut to `limit` and vector-floor guarantees are re-applied
    after the cut. Without a pipeline this degrades to plain recall.

    `reranker=None` means "no reranking requested" and leaves the trace
    untouched. A caller that *wanted* a reranker but could not build one
    should mark `reranker:unavailable` on the trace itself.
    """
    raw_limit = max(limit * 3, 30) if pipeline is not None else limit
    recalled = recaller.recall(query, limit=raw_limit, filters=filters, trace=trace)
    results = recalled
    if pipeline is not None:
        results = pipeline.apply(query, results)
    if reranker is not None:
        results = apply_rerank_safe(reranker, query, results, trace)
    results = results[:limit]
    apply_floor = getattr(recaller, "apply_vector_floor_after_rerank", None)
    if apply_floor is not None:
        results = apply_floor(results, recalled)
    return results
