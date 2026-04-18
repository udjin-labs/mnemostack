"""High-level Recaller — orchestrates embedding + vector + BM25 + RRF fusion."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..embeddings.base import EmbeddingProvider
from ..observability import counter, histogram
from ..observability.recorder import get_recorder
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
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        bm25_docs: list[BM25Doc] | None = None,
        rrf_k: int = 60,
        retrievers: list["Retriever"] | None = None,
    ):
        """Two modes:

        1. Legacy constructor (backward compat): embedding_provider + vector_store
           (+ optional bm25_docs). Fuses Vector + BM25 via RRF.
        2. Retrievers mode: pass `retrievers=[...]` with any number of
           Retriever instances (Vector, BM25, Memgraph, Temporal, custom).
           All are fused via RRF. This matches the legacy enhanced-recall.py
           architecture where Memgraph and Temporal are first-class RRF
           sources, not post-stages.
        """
        self.embedding = embedding_provider
        self.vector = vector_store
        self.bm25 = BM25(bm25_docs) if bm25_docs else None
        self.rrf_k = rrf_k
        self.retrievers = retrievers or []

    async def recall_async(
        self,
        query: str,
        limit: int = 10,
        vector_limit: int = 20,
        bm25_limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[RecallResult]:
        """Async wrapper around `recall`.

        Default implementation runs the sync path in a worker thread so event
        loops (FastAPI / asyncio services) are not blocked by embedding calls,
        Qdrant HTTP, Memgraph Bolt, or CPU-bound BM25. When mnemostack grows
        native-async retrievers this can switch to real concurrent gathering;
        the public signature stays stable.
        """
        import asyncio

        return await asyncio.to_thread(
            self.recall, query, limit, vector_limit, bm25_limit, filters
        )

    def recall(
        self,
        query: str,
        limit: int = 10,
        vector_limit: int = 20,
        bm25_limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[RecallResult]:
        """Run hybrid recall and return fused top-K results."""
        counter("mnemostack.recall.calls", 1)
        # Retrievers mode: fuse N arbitrary ranked lists
        if self.retrievers:
            return self._recall_via_retrievers(query, limit, vector_limit, filters)
        with histogram("mnemostack.recall.latency_ms"):
            # Vector search
            vector_hits: list[Hit] = []
            with histogram("mnemostack.recall.embed_latency_ms"):
                query_vec = self.embedding.embed(query)
            if query_vec:
                with histogram("mnemostack.recall.vector_latency_ms"):
                    vector_hits = self.vector.search(
                        query_vec, limit=vector_limit, filters=filters
                    )
                counter(
                    "mnemostack.recall.vector_hits", len(vector_hits)
                )

            # BM25 search
            bm25_hits: list[tuple[BM25Doc, float]] = []
            if self.bm25:
                with histogram("mnemostack.recall.bm25_latency_ms"):
                    bm25_hits = self.bm25.search(query, limit=bm25_limit)
                counter("mnemostack.recall.bm25_hits", len(bm25_hits))

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
            counter("mnemostack.recall.results", len(results))
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

    def _recall_via_retrievers(
        self,
        query: str,
        limit: int,
        per_source_limit: int,
        filters: dict[str, Any] | None,
    ) -> list[RecallResult]:
        """Fuse N retrievers' ranked lists via RRF. Preserves source tags.

        Retrievers run in a threadpool in parallel so a slow one (e.g. a
        Memgraph bolt roundtrip) doesn't serialise in front of a fast one
        (BM25). This is the pragmatic path to concurrency while Retriever
        instances are still synchronous. True per-retriever async support
        lands later via `Retriever.search_async`.
        """
        import concurrent.futures
        import time

        def _run(retr):
            start = time.monotonic()
            try:
                hits = retr.search(query, limit=per_source_limit, filters=filters)
            except Exception:
                hits = []
            elapsed_ms = (time.monotonic() - start) * 1000.0
            # Per-retriever latency — exposed in /metrics as
            # mnemostack_recall_<name>_latency_ms{...}.
            try:
                get_recorder().record_histogram(
                    f"mnemostack.recall.{retr.name}_latency_ms", elapsed_ms
                )
            except Exception:
                pass
            return retr, hits

        with histogram("mnemostack.recall.latency_ms"):
            all_lists: list[list[tuple[Any, float]]] = []
            id_to_result: dict[Any, RecallResult] = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(len(self.retrievers), 1)
            ) as ex:
                retriever_hits = list(ex.map(_run, self.retrievers))
            for retr, hits in retriever_hits:
                counter(f"mnemostack.recall.{retr.name}_hits", len(hits))
                ranked: list[tuple[Any, float]] = []
                for r in hits:
                    if r.id in id_to_result:
                        existing = id_to_result[r.id]
                        for s in r.sources:
                            if s not in existing.sources:
                                existing.sources.append(s)
                    else:
                        id_to_result[r.id] = r
                    ranked.append((r.id, r.score))
                all_lists.append(ranked)
            fused = reciprocal_rank_fusion(all_lists, k=self.rrf_k, limit=limit)
            results: list[RecallResult] = []
            for key, rrf_score in fused:
                r = id_to_result.get(key)
                if not r:
                    continue
                r.score = rrf_score
                results.append(r)
            counter("mnemostack.recall.results", len(results))
            return results
