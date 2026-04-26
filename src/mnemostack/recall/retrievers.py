"""Retriever abstraction — pluggable ranked-list sources for RRF fusion.

A Retriever takes a query and returns a ranked list of RecallResult-like items
(id, text, score, payload, sources). Multiple retrievers are fused via RRF in
Recaller. This matches the legacy enhanced-recall.py architecture where
Vector / BM25 / Memgraph / Temporal are all first-class ranked sources, not
post-ranking stages.

Built-in retrievers:
- VectorRetriever     — Qdrant semantic search (embedding-based)
- BM25Retriever       — exact token match
- MemgraphRetriever   — knowledge graph exact/contains match on node names
- TemporalRetriever   — vector search inside a date range extracted from query
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from ..embeddings.base import EmbeddingProvider
from ..llm.base import LLMProvider
from ..vector import VectorStore

try:
    from qdrant_client.models import DatetimeRange, FieldCondition, Filter
except ImportError:  # pragma: no cover - qdrant-client is a runtime dependency
    DatetimeRange = FieldCondition = Filter = None  # type: ignore[assignment]
from .bm25 import BM25, BM25Doc
from .recaller import RecallResult

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase
    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False


class Retriever(ABC):
    """A ranked-list source. Called by Recaller for each query."""

    name: str = "retriever"

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[RecallResult]:
        """Return ranked results. May be empty. Must not raise on expected misses."""
        ...


class VectorRetriever(Retriever):
    """Semantic search via embedding + vector store (e.g. Qdrant)."""

    name = "vector"

    def __init__(self, embedding: EmbeddingProvider, vector_store: VectorStore):
        self.embedding = embedding
        self.vector_store = vector_store

    def search(self, query, limit=20, filters=None):
        vec = self.embedding.embed(query)
        if not vec:
            return []
        hits = self.vector_store.search(vec, limit=limit, filters=filters)
        return [
            RecallResult(
                id=h.id,
                text=h.payload.get("text", ""),
                score=h.score,
                payload=h.payload,
                sources=["vector"],
            )
            for h in hits
        ]


def _with_timestamp_range_filter(
    scroll_filter: Any | None,
    *,
    newer_than: str | None,
    older_than: str | None,
) -> Any | None:
    if newer_than is None and older_than is None:
        return scroll_filter
    if Filter is None or FieldCondition is None or DatetimeRange is None:
        raise RuntimeError("qdrant-client is required for timestamp range filters")

    timestamp_condition = FieldCondition(
        key="timestamp",
        range=DatetimeRange(gte=newer_than, lte=older_than),
    )
    if scroll_filter is None:
        return Filter(must=[timestamp_condition])
    if isinstance(scroll_filter, Filter):
        must = list(scroll_filter.must or [])
        return scroll_filter.model_copy(update={"must": [*must, timestamp_condition]})
    raise TypeError("newer_than/older_than can only be combined with a qdrant Filter scroll_filter")


def bm25_docs_from_qdrant(
    client: Any,
    collection_name: str,
    *,
    scroll_filter: Any | None = None,
    limit: int | None = None,
    batch_size: int = 1_000,
    text_key: str = "text",
    id_prefix: str | None = None,
    payload_filter: Callable[[dict[str, Any]], bool] | None = None,
    newer_than: str | None = None,
    older_than: str | None = None,
) -> list[BM25Doc]:
    """Create BM25 documents from Qdrant payload text.

    This is useful when the canonical memory corpus is already stored in
    Qdrant. It keeps lexical BM25 search aligned with vector search and avoids
    a common failure mode where exact tokens in transcripts (message IDs,
    commit hashes, filenames, quoted phrases) are invisible to BM25 because the
    lexical corpus was built only from local markdown files.

    Args:
        client: ``qdrant_client.QdrantClient``-compatible object.
        collection_name: Qdrant collection to scroll.
        scroll_filter: Optional Qdrant filter passed to ``scroll``.
        limit: Maximum BM25 docs to load. ``None`` (default) means unbounded:
            scroll until the collection/filter is exhausted.
        batch_size: Number of points per scroll call.
        text_key: Payload key containing searchable text.
        id_prefix: Optional prefix for generated BM25Doc IDs.
        payload_filter: Optional predicate to skip payloads client-side.
        newer_than: Optional ISO timestamp; keep chunks with
            ``payload["timestamp"] >= newer_than``.
        older_than: Optional ISO timestamp; keep chunks with
            ``payload["timestamp"] <= older_than``.

    Warning:
        For very large collections (>1M chunks), consider passing a Filter to
        scope the BM25 corpus, or using newer_than/older_than for a rolling
        window. BM25 build memory is O(n). For example:
        ``bm25_docs_from_qdrant(client, "memory", newer_than="2026-04-01T00:00:00Z")``.
    """
    docs: list[BM25Doc] = []
    offset = None
    effective_filter = _with_timestamp_range_filter(
        scroll_filter, newer_than=newer_than, older_than=older_than
    )
    while limit is None or len(docs) < limit:
        current_limit = batch_size if limit is None else min(batch_size, limit - len(docs))
        if current_limit <= 0:
            break
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=effective_filter,
            limit=current_limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            if payload_filter and not payload_filter(payload):
                continue
            text = payload.get(text_key) or ""
            if not isinstance(text, str) or not text.strip():
                continue
            qdrant_id = str(getattr(point, "id", len(docs)))
            doc_payload = dict(payload)
            doc_payload.setdefault("qdrant_id", qdrant_id)
            doc_payload.setdefault("source", payload.get("source_file") or payload.get("source"))
            doc_id = qdrant_id if id_prefix is None else f"bm25:{id_prefix}:{qdrant_id}"
            docs.append(
                BM25Doc(
                    id=doc_id,
                    text=text,
                    payload=doc_payload,
                )
            )
            if limit is not None and len(docs) >= limit:
                break
        if offset is None:
            break
    return docs


class BM25Retriever(Retriever):
    """Exact token match via BM25."""

    name = "bm25"

    def __init__(self, docs: list[BM25Doc]):
        self.bm25 = BM25(docs)

    @classmethod
    def from_qdrant(
        cls,
        client: Any,
        collection_name: str,
        *,
        scroll_filter: Any | None = None,
        limit: int | None = None,
        batch_size: int = 1_000,
        text_key: str = "text",
        id_prefix: str | None = None,
        payload_filter: Callable[[dict[str, Any]], bool] | None = None,
        newer_than: str | None = None,
        older_than: str | None = None,
    ) -> BM25Retriever:
        """Build a BM25 retriever from Qdrant payload text.

        Vector search catches semantic similarity, but exact-token recall
        (message IDs, commit hashes, quoted words, filenames) needs a lexical
        retriever over the same corpus. If memories/transcripts already live in
        Qdrant payloads, this helper builds the BM25 corpus directly from those
        payloads without requiring a duplicate markdown export.

        Args:
            client: ``qdrant_client.QdrantClient``-compatible object.
            collection_name: Qdrant collection to scroll.
            scroll_filter: Optional Qdrant filter passed to ``scroll``.
            limit: Maximum BM25 docs to load. ``None`` (default) means unbounded:
                scroll until the collection/filter is exhausted.
            batch_size: Number of points per scroll call.
            text_key: Payload key containing searchable text.
            id_prefix: Optional prefix for generated BM25Doc IDs. By default,
                Qdrant point IDs are reused so vector and BM25 hits fuse as the
                same memory. Set a prefix to namespace docs when mixing corpora.
            payload_filter: Optional predicate to skip payloads client-side.
            newer_than: Optional ISO timestamp; keep chunks with
                ``payload["timestamp"] >= newer_than``.
            older_than: Optional ISO timestamp; keep chunks with
                ``payload["timestamp"] <= older_than``.

        Warning:
            For very large collections (>1M chunks), consider passing a Filter
            to scope the BM25 corpus, or using newer_than/older_than for a
            rolling window. BM25 build memory is O(n). Example:
            ``BM25Retriever.from_qdrant(client, "memory", newer_than="2026-04-01T00:00:00Z")``.

        Returns:
            BM25Retriever over the collected Qdrant payload chunks.
        """
        docs = bm25_docs_from_qdrant(
            client,
            collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            batch_size=batch_size,
            text_key=text_key,
            id_prefix=id_prefix,
            payload_filter=payload_filter,
            newer_than=newer_than,
            older_than=older_than,
        )
        return cls(docs=docs)

    def search(self, query, limit=20, filters=None):
        hits = self.bm25.search(query, limit=limit)
        return [
            RecallResult(
                id=d.id,
                text=d.text,
                score=s,
                payload=d.payload or {},
                sources=["bm25"],
            )
            for d, s in hits
        ]


class HyDERetriever(Retriever):
    """Hypothetical Document Embeddings retriever (opt-in).

    Instead of embedding the raw query, we ask an LLM to sketch what a good
    answer would look like, then embed *that*, then search for memories
    similar to the hypothetical answer:

        query:  "what fields would she likely pursue in her education?"
        hypo:   "Caroline is considering psychology, counselling and mental
                 health degrees, aiming to work with teenagers."

    When it helps vs when it doesn't (measured, not guessed):

    - **Helps** when the query vocabulary differs a lot from how the answer
      is stored — typically structured technical corpora (code, API docs,
      schemas) where questions are abstract and stored content is concrete.

    - **Does not reliably help** on dialogue-backed memory (transcripts,
      chat logs, markdown notes). Our own LoCoMo smoke showed +1 correct
      on the hardest cat_3 reasoning sample (14 questions, 14.3% → 21.4%)
      at the cost of ~1 extra LLM roundtrip per query. On everyday real-
      corpus probes it traded slightly lower top-1 score for marginal
      diversity gains — not a clear win.

    - **Always costs** one LLM call per `search()` (latency + $) before
      the vector search even starts.

    Use this when your workload has a question↔answer vocabulary gap large
    enough to justify the extra LLM call. For general-purpose dialogue
    memory, the built-in Vector + BM25 + Memgraph + Temporal combo is
    usually enough and cheaper.

    Graceful: if the LLM errors or returns empty, returns [] so the rest
    of the retrieval stack is unaffected.
    """

    name = "hyde"

    _PROMPT = (
        "Imagine a short factual answer to this question, written as if it "
        "were a note in someone's memory. Do not hedge. One or two sentences. "
        "If you have no information, invent a plausible answer — this is only "
        "used to seed a vector search, not returned to the user.\n\n"
        "Question: {query}\n\n"
        "Answer:"
    )

    def __init__(
        self,
        llm: LLMProvider,
        embedding: EmbeddingProvider,
        vector_store: VectorStore,
        max_tokens: int = 120,
    ):
        self.llm = llm
        self.embedding = embedding
        self.vector_store = vector_store
        self.max_tokens = max_tokens

    def _generate_hypothetical(self, query: str) -> str | None:
        try:
            resp = self.llm.generate(
                self._PROMPT.format(query=query),
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            text = (getattr(resp, "text", None) or "").strip()
            return text or None
        except Exception:
            return None

    def search(self, query, limit=20, filters=None):
        hypo = self._generate_hypothetical(query)
        if not hypo:
            return []
        vec = self.embedding.embed(hypo)
        if not vec:
            return []
        hits = self.vector_store.search(vec, limit=limit, filters=filters)
        return [
            RecallResult(
                id=h.id,
                text=h.payload.get("text", ""),
                score=h.score,
                payload=h.payload,
                sources=["hyde"],
            )
            for h in hits
        ]


class MemgraphRetriever(Retriever):
    """Knowledge-graph retriever — exact/contains match on node names.

    Mirrors legacy enhanced-recall.py:fetch_memgraph. Each word >=3 chars in
    the query becomes a probe; nodes matched by multiple probes get higher
    counts (used as score).
    """

    name = "memgraph"

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "",
        password: str = "",
        min_word: int = 3,
        contains_min: int = 5,
        max_nodes: int = 10,
        max_rels: int = 5,
        driver: Any = None,
        timeout: float = 5.0,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.min_word = min_word
        self.contains_min = contains_min
        self.max_nodes = max_nodes
        self.max_rels = max_rels
        self.timeout = timeout
        self._driver = driver
        self._own_driver = driver is None

    def _get_driver(self):
        if self._driver is not None:
            return self._driver
        if not _NEO4J_AVAILABLE:
            return None
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password) if self.user else None,
                connection_timeout=self.timeout,
                connection_acquisition_timeout=self.timeout,
            )
            return self._driver
        except Exception:
            return None

    def close(self) -> None:
        if self._driver is not None and self._own_driver:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None

    def search(self, query, limit=20, filters=None):
        driver = self._get_driver()
        if driver is None:
            return []
        words = [w.lower() for w in query.split() if len(w) >= self.min_word]
        if not words:
            return []
        counts: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "type": "", "mc": ""}
        )
        try:
            with driver.session() as session:
                for w in words:
                    # Probe 1: numeric-looking tokens may be contact IDs
                    # (Telegram, Discord, etc). If a canonical Person node has
                    # a matching contact_id property, surface it directly —
                    # most reliable entity-resolution signal we have.
                    rows: list[dict] = []
                    if w.isdigit() and len(w) >= 6:
                        rows = session.run(
                            "MATCH (n) WHERE (n.telegram_id = $w OR n.contact_id = $w) "
                            "AND coalesce(n.valid_until, 'current') = 'current' "
                            "RETURN n.name AS name, labels(n)[0] AS type, "
                            "n.memory_class AS mc LIMIT 5",
                            w=w,
                        ).data()

                    # Probe 2: exact name match.
                    # `name_lower` is a precomputed lower-case copy of n.name
                    # — Memgraph's toLower() only lower-cases ASCII characters,
                    # so relying on it silently loses hits on non-ASCII names.
                    # For graphs that haven't backfilled name_lower yet we fall
                    # back to toLower() so the retriever still works on ASCII.
                    if not rows:
                        rows = session.run(
                            "MATCH (n) WHERE coalesce(n.name_lower, toLower(n.name)) = $w "
                            "AND coalesce(n.valid_until, 'current') = 'current' "
                            "RETURN n.name AS name, labels(n)[0] AS type, "
                            "n.memory_class AS mc LIMIT 5",
                            w=w,
                        ).data()
                    # Probe 3: also match by handle/username (e.g. @alice)
                    if not rows and len(w) >= 3:
                        rows = session.run(
                            "MATCH (n) WHERE toLower(coalesce(n.telegram_username, '')) = $w "
                            "AND coalesce(n.valid_until, 'current') = 'current' "
                            "RETURN n.name AS name, labels(n)[0] AS type, "
                            "n.memory_class AS mc LIMIT 5",
                            w=w,
                        ).data()
                    # Probe 4: substring fallback for longer tokens.
                    if not rows and len(w) >= self.contains_min:
                        rows = session.run(
                            "MATCH (n) WHERE coalesce(n.name_lower, toLower(n.name)) CONTAINS $w "
                            "AND coalesce(n.valid_until, 'current') = 'current' "
                            "RETURN n.name AS name, labels(n)[0] AS type, "
                            "n.memory_class AS mc LIMIT 5",
                            w=w,
                        ).data()
                    for n in rows:
                        name = n.get("name")
                        if not name:
                            continue
                        counts[name]["count"] += 1
                        counts[name]["type"] = n.get("type", "") or ""
                        counts[name]["mc"] = n.get("mc", "") or ""
                # Fetch relationships for top-N
                ranked = sorted(counts.items(), key=lambda kv: -kv[1]["count"])[: self.max_nodes]
                results: list[RecallResult] = []
                for name, info in ranked:
                    rel_rows = session.run(
                        "MATCH (n {name: $name})-[r]->(m) "
                        "WHERE coalesce(n.valid_until, 'current') = 'current' "
                        "AND coalesce(r.valid_until, 'current') = 'current' "
                        "RETURN n.name AS from_n, type(r) AS rel, m.name AS to_n "
                        "LIMIT $lim",
                        name=name, lim=self.max_rels,
                    ).data()
                    rel_text = "; ".join(
                        f"{r['from_n']}-[{r['rel']}]->{r['to_n']}" for r in rel_rows
                    )
                    content = (
                        f"{info['type']}: {name}. {rel_text}"
                        if rel_text
                        else f"{info['type']}: {name}"
                    )
                    results.append(
                        RecallResult(
                            id=f"graph:{name}",
                            text=content[:300],
                            score=float(info["count"]),
                            payload={
                                "text": content[:300],
                                "source": "memgraph",
                                "memory_class": info.get("mc", ""),
                                "name": name,
                                "type": info["type"],
                            },
                            sources=["memgraph"],
                        )
                    )
                return results[:limit]
        except Exception:
            return []


# --- Temporal extraction ---
# Port of legacy temporal_extractor.extract_temporal (minimal inline version).

_MONTHS = {
    "январ": 1, "january": 1, "jan": 1,
    "феврал": 2, "february": 2, "feb": 2,
    "март": 3, "march": 3, "mar": 3,
    "апрел": 4, "april": 4, "apr": 4,
    "май": 5, "may": 5,
    "июн": 6, "june": 6, "jun": 6,
    "июл": 7, "july": 7, "jul": 7,
    "август": 8, "august": 8, "aug": 8,
    "сентябр": 9, "september": 9, "sep": 9,
    "октябр": 10, "october": 10, "oct": 10,
    "ноябр": 11, "november": 11, "nov": 11,
    "декабр": 12, "december": 12, "dec": 12,
}


def extract_temporal(query: str) -> tuple[str, str] | None:
    """Best-effort date range extraction. Returns (start_iso, end_iso) or None."""
    q = query.lower()
    # "<month> <year>" or Russian stem
    for stem, month in _MONTHS.items():
        if stem in q:
            y_m = re.search(r"\b(20\d{2})\b", q)
            y = int(y_m.group(1)) if y_m else datetime.now(timezone.utc).year
            start = datetime(y, month, 1, tzinfo=timezone.utc)
            end_month = month + 1
            end_year = y + (1 if end_month > 12 else 0)
            end_month = end_month if end_month <= 12 else 1
            end = datetime(end_year, end_month, 1, tzinfo=timezone.utc)
            return start.isoformat(), end.isoformat()
    # "YYYY" — full year
    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        y = int(m.group(1))
        return (
            datetime(y, 1, 1, tzinfo=timezone.utc).isoformat(),
            datetime(y + 1, 1, 1, tzinfo=timezone.utc).isoformat(),
        )
    return None


class TemporalRetriever(Retriever):
    """Vector search filtered by date range extracted from query.

    If no date range can be extracted, returns empty. When a range is found,
    runs semantic search with a timestamp payload filter.
    """

    name = "temporal"

    def __init__(
        self,
        embedding: EmbeddingProvider,
        vector_store: VectorStore,
        extractor=extract_temporal,
    ):
        self.embedding = embedding
        self.vector_store = vector_store
        self.extractor = extractor

    def search(self, query, limit=10, filters=None):
        window = self.extractor(query)
        if not window:
            return []
        start, end = window
        vec = self.embedding.embed(query)
        if not vec:
            return []
        # Flat filter shape understood by VectorStore._build_filter. Preserve
        # caller filters (workspace/source/tenant scope) and add the temporal
        # timestamp constraint instead of replacing the whole filter map.
        temporal_filter = dict(filters or {})
        temporal_filter["timestamp"] = {"gte": start, "lte": end}
        try:
            hits = self.vector_store.search(vec, limit=limit, filters=temporal_filter)
        except Exception as exc:  # noqa: BLE001 — defensive; log instead of silent
            logger.warning(
                "TemporalRetriever: vector_store.search failed (window=%s..%s): %s",
                start, end, exc,
            )
            return []
        return [
            RecallResult(
                id=h.id,
                text=h.payload.get("text", ""),
                score=h.score,
                payload={**h.payload, "temporal_match": True},
                sources=["temporal"],
            )
            for h in hits
        ]
