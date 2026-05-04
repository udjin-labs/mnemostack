"""High-level Recaller — orchestrates embedding + vector + BM25 + RRF fusion."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..embeddings.base import EmbeddingProvider
from ..observability import counter, histogram
from ..observability.recorder import get_recorder
from ..vector.qdrant import Hit, VectorStore
from .bm25 import BM25, BM25Doc
from .fusion import reciprocal_rank_fusion
from .mca_prefilter import mca_prefilter as run_mca_prefilter
from .query_expansion import expand_query

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..llm.base import LLMProvider
    from .retrievers import Retriever


@dataclass
class RecallResult:
    """Unified result from hybrid recall.

    Contains merged ranking + which sources contributed + fused score.

    The ``id`` field is a stable, citable identifier for the underlying chunk.
    For most retrievers it is derived from chunk content (e.g. a UUID-shaped
    content hash) and is therefore consistent across recall invocations and
    across processes as long as the chunk itself does not change. Callers may
    use it as a citation handle (``[id:<...>]``) and later resolve it back to
    the full record via storage-specific helpers.

    The ``sources`` list records which retrievers contributed this result
    (e.g. ``['bm25', 'vector']``) and is useful for observability and for
    rendering compact result indexes.
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

    # Default weight profiles per detected query shape. Picked conservatively
    # so that switching `adaptive_weights=True` cannot lower recall@K by more
    # than 1-2 percentage points on mixed workloads in our measurements, while
    # materially improving the cases the profile targets.
    _WEIGHT_PROFILES: dict[str, dict[str, float]] = {
        # Queries mentioning specific IP / port / version / UUID / ID markers.
        # Exact-token retrievers (BM25, graph with structural matches) are
        # more reliable than pure vector similarity here.
        "exact_token": {"bm25": 1.4, "memgraph": 1.4, "vector": 1.0, "temporal": 0.9},
        # Queries with a when/date shape. The temporal retriever is date-aware
        # — let it lead, but don't zero out the others (dates often require
        # entity context too).
        "temporal": {"temporal": 1.4, "vector": 1.0, "bm25": 1.0, "memgraph": 1.0},
        # Queries asking about a person by name, handle, or numeric contact ID.
        # Graph lookups are authoritative in that case.
        "person": {"memgraph": 1.5, "vector": 1.0, "bm25": 1.0, "temporal": 0.9},
        # Fall-through — classical equal-weight RRF.
        "general": {},
    }

    # Heuristics for picking a profile. Intentionally light (regex + word-
    # boundary scan) so that wrapping the call in adaptive-weights mode costs
    # almost nothing. Markers are matched as whole words to avoid false-positives
    # like "pipeline" → "ip" or "important" → "port".
    _EXACT_TOKEN_RE = re.compile(
        r"\b\d{1,3}(?:\.\d{1,3}){3}\b|"       # IPv4
        r"\b\d{4,5}\b|"                        # port / numeric code
        r"\b\d{4}\.\d+\.\d+\b|"                 # version
        r"\b[A-Za-z]+[-_]\d+[A-Za-z0-9-]*\b"  # id/code
    )
    _EXACT_MARKERS = {"ip", "порт", "port", "версия", "version", "uuid", "api"}
    _PERSON_MARKERS = {
        "кто", "who", "telegram", "handle", "username",
        "contact", "контакт",
    }
    _TEMPORAL_MARKERS = {
        "когда", "when", "дата", "date", "вчера", "yesterday",
        "сегодня", "today", "завтра", "tomorrow",
    }

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        bm25_docs: list[BM25Doc] | None = None,
        rrf_k: int = 60,
        retrievers: list[Retriever] | None = None,
        retriever_weights: dict[str, float] | None = None,
        adaptive_weights: bool = False,
        query_expansion: bool = False,
        expansion_llm: LLMProvider | None = None,
        fallback_threshold: float = 0.45,
        mca_prefilter: bool = False,
    ):
        """Two modes:

        1. Legacy constructor (backward compat): embedding_provider + vector_store
           (+ optional bm25_docs). Fuses Vector + BM25 via RRF.
        2. Retrievers mode: pass `retrievers=[...]` with any number of
           Retriever instances (Vector, BM25, Memgraph, Temporal, custom).
           All are fused via RRF. This matches the legacy enhanced-recall.py
           architecture where Memgraph and Temporal are first-class RRF
           sources, not post-stages.

        Three weight modes, in priority order:

        - **`retriever_weights=dict(...)`** — static override. Always applied.
        - **`adaptive_weights=True`** — per-query shape detection: if the
          query looks like an exact-token / person / temporal question, lift
          the retrievers that are authoritative for that shape (e.g. graph
          and BM25 for numeric IDs, temporal for "when did X happen").
          Falls back to classical equal-weight RRF on general queries.
        - **neither set** — classical equal-weight RRF.

        Static weights win over adaptive when both are given: operators who
        have already tuned their workload shouldn't be overridden implicitly.
        """
        self.embedding = embedding_provider
        self.vector = vector_store
        self.bm25 = BM25(bm25_docs) if bm25_docs else None
        self.rrf_k = rrf_k
        self.retrievers = retrievers or []
        self.retriever_weights = dict(retriever_weights) if retriever_weights else {}
        self.adaptive_weights = adaptive_weights
        self.query_expansion = query_expansion
        self.expansion_llm = expansion_llm
        self.fallback_threshold = fallback_threshold
        self.mca_prefilter_enabled = mca_prefilter
        self._query_expansion_cache: dict[str, list[str]] = {}

    # --- adaptive weight helpers ---

    @classmethod
    def _detect_query_shape(cls, query: str) -> str:
        q_lower = query.lower()
        # Word-boundary tokens for marker matching
        tokens = set(re.findall(r"[\w@]+", q_lower))
        if cls._EXACT_TOKEN_RE.search(q_lower) or tokens & cls._EXACT_MARKERS:
            return "exact_token"
        # Telegram-style @handle is a person signal too
        if (
            (tokens & cls._PERSON_MARKERS)
            or re.search(r"@\w+", q_lower)
        ):
            return "person"
        if tokens & cls._TEMPORAL_MARKERS:
            return "temporal"
        return "general"

    def _weight_for(self, retriever_name: str, query: str) -> float:
        # Static override wins — explicit is explicit.
        if retriever_name in self.retriever_weights:
            return float(self.retriever_weights[retriever_name])
        if not self.adaptive_weights:
            return 1.0
        shape = self._detect_query_shape(query)
        profile = self._WEIGHT_PROFILES.get(shape, {})
        return float(profile.get(retriever_name, 1.0))

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
            self.recall,
            query,
            limit,
            vector_limit,
            bm25_limit,
            filters,
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
        if self.query_expansion:
            return self._recall_with_query_expansion(
                query=query,
                limit=limit,
                vector_limit=vector_limit,
                bm25_limit=bm25_limit,
                filters=filters,
            )
        return self._recall_once(query, limit, vector_limit, bm25_limit, filters)

    def _recall_once(
        self,
        query: str,
        limit: int,
        vector_limit: int,
        bm25_limit: int,
        filters: dict[str, Any] | None,
    ) -> list[RecallResult]:
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
            mca_hits = self._mca_hits(query, bm25_limit) if self.mca_prefilter_enabled else []
            mca_by_id = {hit.id: hit for hit in mca_hits}
            ranked_lists = [vector_list, bm25_list]
            if mca_hits:
                ranked_lists.insert(0, [(hit, hit.score) for hit in mca_hits])

            # Two separate fusions — RRF by rank
            fused = reciprocal_rank_fusion(
                ranked_lists,
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
                elif isinstance(item, RecallResult):
                    for source in self._sources_for(item, vector_hits, bm25_hits):
                        if source not in item.sources:
                            item.sources.append(source)
                    item.score = rrf_score
                    results.append(item)
                else:
                    mca_result = mca_by_id.get(item)
                    if mca_result is not None:
                        mca_result.score = rrf_score
                        results.append(mca_result)
            results = self._maybe_apply_fallback(
                query, results, limit=limit, vector_limit=vector_limit, filters=filters
            )
            counter("mnemostack.recall.results", len(results))
            return results


    def search_many(self, vectors: list[list[float]], limit: int) -> list[RecallResult]:
        """Search Qdrant for multiple vectors and RRF-merge the ranked hits."""
        if not self.vector:
            return []

        ranked_lists: list[list[tuple[Any, float]]] = []
        id_to_hit: dict[Any, Hit] = {}
        for vector in vectors:
            if not vector:
                continue
            try:
                hits = self.vector.search(vector, limit=limit)
            except Exception:
                hits = []
            ranked: list[tuple[Any, float]] = []
            for hit in hits:
                id_to_hit.setdefault(hit.id, hit)
                ranked.append((hit.id, hit.score))
            ranked_lists.append(ranked)

        fused = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k, limit=limit)
        results: list[RecallResult] = []
        for key, rrf_score in fused:
            hit = id_to_hit.get(key)
            if hit is None:
                continue
            results.append(
                RecallResult(
                    id=hit.id,
                    text=hit.payload.get("text", ""),
                    score=rrf_score,
                    payload=hit.payload,
                    sources=["vector"],
                )
            )
        counter("mnemostack.recall.results", len(results))
        return results

    def _expanded_queries(self, query: str, n_variants: int = 3) -> list[str]:
        if not self.expansion_llm:
            raise ValueError("query_expansion=True requires expansion_llm")
        if query not in self._query_expansion_cache:
            self._query_expansion_cache[query] = expand_query(
                query, self.expansion_llm, n_variants=n_variants
            )
        seen = {query.strip()}
        queries = [query]
        for variant in self._query_expansion_cache[query]:
            key = variant.strip()
            if key and key not in seen:
                queries.append(key)
                seen.add(key)
        return queries

    def _recall_with_query_expansion(
        self,
        query: str,
        limit: int,
        vector_limit: int,
        bm25_limit: int,
        filters: dict[str, Any] | None,
    ) -> list[RecallResult]:
        ranked_lists: list[list[tuple[Any, float]]] = []
        id_to_result: dict[Any, RecallResult] = {}
        for expanded_query in self._expanded_queries(query):
            results = self._recall_once(
                expanded_query,
                limit=max(limit, vector_limit, bm25_limit),
                vector_limit=vector_limit,
                bm25_limit=bm25_limit,
                filters=filters,
            )
            ranked: list[tuple[Any, float]] = []
            for result in results:
                if result.id in id_to_result:
                    existing = id_to_result[result.id]
                    for source in result.sources:
                        if source not in existing.sources:
                            existing.sources.append(source)
                else:
                    id_to_result[result.id] = result
                ranked.append((result.id, result.score))
            ranked_lists.append(ranked)

        fused = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k, limit=limit)
        merged: list[RecallResult] = []
        for key, rrf_score in fused:
            result = id_to_result.get(key)
            if result is None:
                continue
            result.score = rrf_score
            merged.append(result)
        counter("mnemostack.recall.results", len(merged))
        return merged

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
            per_list_weights: list[float] = []
            id_to_result: dict[Any, RecallResult] = {}
            if self.mca_prefilter_enabled:
                mca_hits = self._mca_hits(query, per_source_limit)
                if mca_hits:
                    all_lists.append([(hit, hit.score) for hit in mca_hits])
                    per_list_weights.append(self._weight_for("mca", query))
                    for hit in mca_hits:
                        id_to_result[hit.id] = hit
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
                per_list_weights.append(self._weight_for(retr.name, query))
            fused = reciprocal_rank_fusion(
                all_lists, k=self.rrf_k, limit=limit, weights=per_list_weights
            )
            results: list[RecallResult] = []
            for item, rrf_score in fused:
                key = getattr(item, "id", item)
                r = id_to_result.get(key)
                if not r:
                    continue
                r.score = rrf_score
                results.append(r)
            results = self._maybe_apply_fallback(
                query, results, limit=limit, vector_limit=per_source_limit, filters=filters
            )
            counter("mnemostack.recall.results", len(results))
            return results

    def _mca_hits(self, query: str, limit: int) -> list[RecallResult]:
        bm25 = self.bm25
        if bm25 is None:
            for retriever in self.retrievers:
                bm25 = getattr(retriever, "bm25", None)
                if bm25 is not None:
                    break
        if bm25 is None:
            return []
        return run_mca_prefilter(query, bm25, limit=limit)

    def _vector_fallback_hits(
        self,
        query: str,
        *,
        limit: int,
        filters: dict[str, Any] | None,
    ) -> list[RecallResult]:
        if self.embedding and self.vector:
            query_vec = self.embedding.embed(query)
            if not query_vec:
                return []
            hits = self.vector.search(query_vec, limit=limit, filters=filters)
            return [
                RecallResult(
                    id=hit.id,
                    text=hit.payload.get("text", ""),
                    score=hit.score,
                    payload=hit.payload,
                    sources=["vector"],
                )
                for hit in hits
            ]

        for retriever in self.retrievers:
            if getattr(retriever, "name", None) == "vector":
                return retriever.search(query, limit=limit, filters=filters)
        return []

    def _maybe_apply_fallback(
        self,
        query: str,
        results: list[RecallResult],
        *,
        limit: int,
        vector_limit: int,
        filters: dict[str, Any] | None,
    ) -> list[RecallResult]:
        top_score = max((result.score for result in results), default=0.0)
        if top_score >= self.fallback_threshold:
            return results

        fallback_hits = self._vector_fallback_hits(
            query, limit=max(limit, vector_limit), filters=filters
        )
        if not fallback_hits:
            return results

        counter("mnemostack.recall.fallback_triggered", 1)
        logger.info(
            "Low-confidence fallback triggered: top_score=%.3f < %.3f",
            top_score,
            self.fallback_threshold,
        )
        by_id: dict[Any, RecallResult] = {result.id: result for result in results}
        for fallback in fallback_hits:
            existing = by_id.get(fallback.id)
            if existing is None:
                by_id[fallback.id] = fallback
                continue
            for source in fallback.sources:
                if source not in existing.sources:
                    existing.sources.append(source)
            if fallback.score > existing.score:
                fallback.sources = existing.sources
                by_id[fallback.id] = fallback

        merged = list(by_id.values())
        merged.sort(key=lambda result: result.score, reverse=True)
        return merged[:limit]
