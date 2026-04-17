"""High-level Recaller — orchestrates embedding + vector + BM25 + RRF fusion."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..embeddings.base import EmbeddingProvider
from ..vector.qdrant import Hit, VectorStore
from .bm25 import BM25, BM25Doc
from .fusion import reciprocal_rank_fusion


@dataclass
class RecallResult:
    """Unified result from hybrid recall.

    Contains merged ranking + which sources contributed + fused score.
    """

    id: str | int
    text: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)  # ['bm25', 'vector']


class Recaller:
    """Hybrid recall: BM25 + semantic search + RRF fusion.

    Does not include reranker or answer layer — those are separate, optional stages.
    Use this directly for raw hybrid retrieval, or wrap it in higher-level pipelines.

    Usage:
        recaller = Recaller(
            embedding_provider=gemini_provider,
            vector_store=qdrant_store,
            bm25_docs=[BM25Doc(...), ...],
        )
        results = recaller.recall("query", limit=10)
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        bm25_docs: list[BM25Doc] | None = None,
        rrf_k: int = 60,
    ):
        self.embedding = embedding_provider
        self.vector = vector_store
        self.bm25 = BM25(bm25_docs) if bm25_docs else None
        self.rrf_k = rrf_k

    def recall(
        self,
        query: str,
        limit: int = 10,
        vector_limit: int = 20,
        bm25_limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[RecallResult]:
        """Run hybrid recall and return fused top-K results."""
        # Vector search
        vector_hits: list[Hit] = []
        query_vec = self.embedding.embed(query)
        if query_vec:
            vector_hits = self.vector.search(
                query_vec, limit=vector_limit, filters=filters
            )

        # BM25 search
        bm25_hits: list[tuple[BM25Doc, float]] = []
        if self.bm25:
            bm25_hits = self.bm25.search(query, limit=bm25_limit)

        # Build id→source map and per-list tuples for RRF
        vector_list = [(hit, hit.score) for hit in vector_hits]
        bm25_list = [(doc, score) for doc, score in bm25_hits]

        # Two separate fusions — RRF by rank
        fused = reciprocal_rank_fusion(
            [vector_list, bm25_list],
            k=self.rrf_k,
            limit=limit,
        )

        # Convert to unified RecallResult
        results = []
        for item, rrf_score in fused:
            if isinstance(item, Hit):
                results.append(
                    RecallResult(
                        id=item.id,
                        text=item.payload.get("text", ""),
                        score=rrf_score,
                        payload=item.payload,
                        sources=self._sources_for(item, vector_hits, bm25_hits),
                    )
                )
            elif isinstance(item, BM25Doc):
                results.append(
                    RecallResult(
                        id=item.id,
                        text=item.text,
                        score=rrf_score,
                        payload=item.payload or {},
                        sources=self._sources_for(item, vector_hits, bm25_hits),
                    )
                )
        return results

    @staticmethod
    def _sources_for(
        item: Any,
        vector_hits: list[Hit],
        bm25_hits: list[tuple[BM25Doc, float]],
    ) -> list[str]:
        item_id = getattr(item, "id", item)
        sources = []
        if any(h.id == item_id for h in vector_hits):
            sources.append("vector")
        if any(d.id == item_id for d, _ in bm25_hits):
            sources.append("bm25")
        return sources
