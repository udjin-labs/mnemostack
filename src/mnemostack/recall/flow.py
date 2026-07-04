"""The canonical post-recall flow shared by every entry point.

CLI, HTTP and MCP used to assemble recall → pipeline → rerank → top-K →
vector-floor each in their own way, so the same query ranked differently
depending on the surface. This module is the single implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .filters import payload_matches
from .tokens import TokenCounter, apply_token_budget
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
    token_budget: int | None = None,
    token_counter: TokenCounter | None = None,
    include_invalidated: bool = False,
    as_of: str | None = None,
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

    `token_budget` trims the *final* order (after rerank, top-K cut and
    vector floor) to the ranked prefix whose total text tokens fit the
    budget — see `apply_token_budget` for the exact contract.

    `include_invalidated` / `as_of` control validity: by default stale
    facts are dropped from the candidate pool (before the pipeline) so
    they never influence ranking — see `Recaller.recall`.
    """
    raw_limit = max(limit * 3, 30) if pipeline is not None else limit
    recalled = recaller.recall(
        query,
        limit=raw_limit,
        filters=filters,
        trace=trace,
        include_invalidated=include_invalidated,
        as_of=as_of,
    )
    results = recalled
    if pipeline is not None:
        results = pipeline.apply(query, results, as_of=as_of)
        if filters:
            # Pipeline stages may append candidates that never passed the
            # filtered retrievers (e.g. graph resurrection injects records
            # with no tenant/timestamp payload). Enforce the caller's scope
            # on the pipeline output too: anything that cannot be attributed
            # to the scope is dropped, not leaked.
            results = [r for r in results if payload_matches(r.payload, filters)]
    if reranker is not None:
        results = apply_rerank_safe(reranker, query, results, trace)
    results = results[:limit]
    apply_floor = getattr(recaller, "apply_vector_floor_after_rerank", None)
    if apply_floor is not None:
        results = apply_floor(results, recalled)
    if token_budget is not None:
        results, _ = apply_token_budget(results, token_budget, token_counter)
    return results


async def recall_flow_async(
    recaller: Recaller,
    query: str,
    limit: int = 10,
    *,
    pipeline: Pipeline | None = None,
    reranker: Reranker | None = None,
    filters: dict[str, object] | None = None,
    trace: RecallTrace | None = None,
    token_budget: int | None = None,
    token_counter: TokenCounter | None = None,
    include_invalidated: bool = False,
    as_of: str | None = None,
) -> list[RecallResult]:
    """Async wrapper around `recall_flow`.

    Runs the blocking recall stack (embedding calls, Qdrant HTTP, Memgraph
    Bolt, CPU-bound BM25, pipeline stages, LLM rerank) in a worker thread so
    asyncio services are not blocked. Same contract and results as
    `recall_flow`; the public signature stays stable if the internals ever
    switch to native-async retrievers.
    """
    import asyncio

    return await asyncio.to_thread(
        recall_flow,
        recaller,
        query,
        limit,
        pipeline=pipeline,
        reranker=reranker,
        filters=filters,
        trace=trace,
        token_budget=token_budget,
        token_counter=token_counter,
        include_invalidated=include_invalidated,
        as_of=as_of,
    )
