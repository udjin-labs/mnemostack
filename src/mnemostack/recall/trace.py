"""Per-call recall trace — observability on top of fail-open recall.

Recall deliberately degrades instead of failing: a broken retriever
contributes nothing, a broken reranker leaves the original order. This
module makes those degradations visible without changing the behavior.

Usage:

    trace = RecallTrace()
    results = recaller.recall(query, trace=trace)
    results = apply_rerank_safe(reranker, query, results, trace)
    trace.degraded      # e.g. ["retriever:bm25:failed", "reranker:fallback"]
    trace.to_dict()     # JSON-friendly dump for debug responses

A trace object is per-call: never share one between concurrent requests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .recaller import RecallResult
    from .reranker import Reranker

logger = logging.getLogger(__name__)


@dataclass
class RetrieverTrace:
    """One retriever's contribution to a recall call, pre-fusion."""

    name: str
    ranked: list[tuple[str, float]] = field(default_factory=list)
    error: str | None = None
    latency_ms: float = 0.0
    query: str | None = None  # set when query expansion produced variants

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "ranked": [[rid, round(score, 6)] for rid, score in self.ranked],
            "latency_ms": round(self.latency_ms, 2),
        }
        if self.error:
            d["error"] = self.error
        if self.query is not None:
            d["query"] = self.query
        return d


@dataclass
class RecallTrace:
    """Trace of one recall call: per-retriever inputs, fused output, degradations.

    `fused` is the order recall returned (post-fusion, post-vector-floor);
    `post_rerank` is the reranker's order when a reranker ran. The final
    response list may still differ if vector-floor re-appends items after
    rerank. `degraded` tags are stable strings: "retriever:<name>:failed",
    "reranker:fallback", "reranker:unavailable", "temporal:no_parse".
    """

    retrievers: list[RetrieverTrace] = field(default_factory=list)
    fused: list[tuple[str, float]] = field(default_factory=list)
    post_rerank: list[tuple[str, float]] | None = None
    degraded: list[str] = field(default_factory=list)

    def mark(self, tag: str) -> None:
        if tag not in self.degraded:
            self.degraded.append(tag)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "retrievers": [rt.to_dict() for rt in self.retrievers],
            "fused": [[rid, round(score, 6)] for rid, score in self.fused],
            "degraded": list(self.degraded),
        }
        if self.post_rerank is not None:
            d["post_rerank"] = [[rid, round(score, 6)] for rid, score in self.post_rerank]
        return d


def apply_rerank_safe(
    reranker: Reranker | None,
    query: str,
    results: list[RecallResult],
    trace: RecallTrace | None = None,
) -> list[RecallResult]:
    """Rerank with the fail-open contract, but leave a trace of the fallback."""
    if reranker is None:
        return results
    try:
        out = reranker.rerank(query, results)
    except Exception as exc:  # noqa: BLE001 — fail-open by design
        logger.warning("reranker failed (%s) — returning pre-rerank order", exc)
        if trace is not None:
            trace.mark("reranker:fallback")
        return results
    if trace is not None:
        trace.post_rerank = [(str(r.id), r.score) for r in out]
    return out
