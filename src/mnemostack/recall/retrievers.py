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
import threading
import time
from abc import ABC, abstractmethod
from calendar import monthrange
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, cast
from urllib.parse import quote

from ..embeddings.base import EmbeddingProvider
from ..embeddings.roles import SpaceGuard, embed_document_via, embed_query_via
from ..llm.base import LLMProvider
from ..observability import counter
from ..vector import VectorStore

try:
    from qdrant_client.models import DatetimeRange, FieldCondition, Filter, Range
except ImportError:  # pragma: no cover - qdrant-client is a runtime dependency
    DatetimeRange = FieldCondition = Filter = Range = None  # type: ignore[assignment,misc]
from .bm25 import BM25, BM25Doc, Tokenizer, tokenize
from .filters import payload_matches
from .recaller import RecallResult
from .validity import (
    graph_as_of_predicate,
    numeric_unit_for,
    parse_payload_instant,
    parses_as_iso,
    to_utc_instant,
)

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase

    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False


#: The built-in retriever family names. A ``name=`` override may carry its
#: OWN family's name (suffixed or not) but never another family's: the
#: recaller branches on exact identities (``vector`` drives the vector floor
#: and the no-vector fallback), and the source filters partition by base
#: name — a BM25 arm named "vector" would masquerade through both.
_RESERVED_RETRIEVER_NAMES = frozenset(
    {"vector", "bm25", "sparse", "qdrant_text", "memgraph", "temporal", "hyde", "mca"}
)


class Retriever(ABC):
    """A ranked-list source. Called by Recaller for each query."""

    name: str = "retriever"

    def _set_name(self, name: str | None) -> None:
        """Instance-level name override (shadows the class attribute).

        Everything downstream keys on ``retriever.name``: fusion weights,
        the adaptive (Q-learning) profile, latency/hit metrics, trace marks
        and degraded tags. Two arms of the same class (e.g. lexical gates
        over different payload fields) therefore NEED distinct names — with
        a shared name they get one merged weight, one merged learning
        profile, and indistinguishable telemetry."""
        if name is None:
            return
        if not isinstance(name, str) or not name.strip():
            raise ValueError("retriever name must be a non-empty string")
        base = name.partition(":")[0]
        if base in _RESERVED_RETRIEVER_NAMES and base != type(self).name:
            raise ValueError(
                f"retriever name {name!r} collides with the reserved "
                f"{base!r} family — the recaller and source filters key "
                "semantics on that identity"
            )
        self.name = name

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

    def __init__(
        self,
        embedding: EmbeddingProvider,
        vector_store: VectorStore,
        text_key: str = "text",
        *,
        timestamp_key: str = "timestamp",
        timestamp_format: str = "iso",
    ):
        self.embedding = embedding
        self.vector_store = vector_store
        # Every vector-backed retriever carries its own revalidating guard so
        # DIRECT use (synthesis, library callers) is protected too, not only
        # the Recaller path. TTL-cached: negligible per-search cost.
        self._space_guard = SpaceGuard(vector_store, embedding)
        #: Payload schema of the collection — configurable so recall works
        #: over a pre-existing collection's own field names and timestamp
        #: domain (same knobs as ``bm25_docs_from_qdrant``).
        self.text_key = text_key
        self.timestamp_key = timestamp_key
        self.timestamp_format = timestamp_format

    #: Advertise validity-awareness so the recaller passes as_of/include_invalidated.
    #: The default view (hide invalidated) is pushed into Qdrant; as_of stays in
    #: the client-side filter (see `_hide_invalidated_condition`).
    accepts_as_of = True
    accepts_include_invalidated = True
    #: Can enforce a tenant filter server-side (see VectorStore.search(tenant=)).
    accepts_tenant = True

    def search(
        self, query, limit=20, filters=None, as_of=None, include_invalidated=False, tenant=None
    ):
        self._space_guard.ensure()
        vec = embed_query_via(self.embedding, query)
        if not vec:
            return []
        hide_invalidated = as_of is None and not include_invalidated
        # A caller timestamp constraint is converted into the collection's own
        # domain first — Qdrant only matches when bound type == field type.
        filters = convert_timestamp_filter(
            filters, timestamp_key=self.timestamp_key, timestamp_format=self.timestamp_format
        )
        # Pass tenant only when set, so a custom store without the parameter
        # still works on the single-tenant path.
        tkw = {"tenant": tenant} if tenant is not None else {}
        hits = self.vector_store.search(
            vec, limit=limit, filters=filters, hide_invalidated=hide_invalidated, **tkw
        )
        results: list[RecallResult] = []
        for h in hits:
            payload = dict(h.payload or {})
            payload["raw_vector_score"] = h.score
            results.append(
                RecallResult(
                    id=h.id,
                    text=payload.get(self.text_key, ""),
                    score=h.score,
                    payload=payload,
                    sources=["vector"],
                )
            )
        return results


def _with_timestamp_range_filter(
    scroll_filter: Any | None,
    *,
    newer_than: str | int | float | None,
    older_than: str | int | float | None,
    timestamp_key: str = "timestamp",
) -> Any | None:
    if newer_than is None and older_than is None:
        return scroll_filter
    if Filter is None or FieldCondition is None or DatetimeRange is None:
        raise RuntimeError("qdrant-client is required for timestamp range filters")

    # Numeric bounds → numeric Range (an epoch payload field never matches a
    # DatetimeRange); string bounds → DatetimeRange, as before. Mirrors
    # VectorStore._build_filter's dispatch.
    if isinstance(newer_than, (int, float)) or isinstance(older_than, (int, float)):
        timestamp_condition = FieldCondition(
            key=timestamp_key,
            range=Range(
                gte=cast("float | None", newer_than),
                lte=cast("float | None", older_than),
            ),
        )
    else:
        timestamp_condition = FieldCondition(
            key=timestamp_key,
            # ISO strings are accepted by Qdrant at runtime; the stubs only admit
            # datetime/date, hence the casts.
            range=DatetimeRange(
                gte=cast("datetime | None", newer_than),
                lte=cast("datetime | None", older_than),
            ),
        )
    if scroll_filter is None:
        return Filter(must=[timestamp_condition])
    if isinstance(scroll_filter, Filter):
        must = list(scroll_filter.must or [])
        return scroll_filter.model_copy(update={"must": [*must, timestamp_condition]})
    raise TypeError("newer_than/older_than can only be combined with a qdrant Filter scroll_filter")


def _emit_epoch_bound(dt: datetime, timestamp_format: str) -> str | float:
    """A bound in the collection's own domain. Numeric domains emit the EXACT
    float — integral rounding in either direction is lossy against fractional
    ``time.time()`` payloads (truncation widens; inward rounding excludes a
    valid ``23:59:59.5`` point at a ``.999999`` window edge)."""
    if timestamp_format == "epoch":
        return dt.timestamp()
    if timestamp_format == "epoch_ms":
        return dt.timestamp() * 1000
    return dt.isoformat()


def convert_timestamp_filter(
    filters: dict[str, Any] | None,
    *,
    timestamp_key: str,
    timestamp_format: str,
) -> dict[str, Any] | None:
    """A copy of ``filters`` with the timestamp condition's bounds converted
    into the collection's own domain.

    The caller's constraint arrives in whatever domain the caller thinks in
    (ISO strings, epoch numbers); Qdrant's range condition only matches when
    the bound's type matches the FIELD's type — an ISO ``DatetimeRange`` over
    a numeric field silently returns nothing. Bounds that don't parse as
    instants are left untouched (strict Qdrant semantics, same as today).
    Non-timestamp keys pass through verbatim.
    """
    if not filters or timestamp_key not in filters:
        return filters
    unit = numeric_unit_for(timestamp_format)

    def _conv(v: Any) -> Any:
        # A value ALREADY in the collection's domain passes through verbatim:
        # rewriting a same-domain value can only break things — an exact ISO
        # MatchValue is string equality ("...Z" != "...+00:00"), and an int
        # payload need not equal a float rewrite of itself.
        is_num = isinstance(v, (int, float)) and not isinstance(v, bool)
        if timestamp_format in ("epoch", "epoch_ms"):
            if is_num:
                return v
            dt = parse_payload_instant(v, numeric_unit=unit)
            if dt is None:
                return v
            ts = dt.timestamp() * (1000 if timestamp_format == "epoch_ms" else 1)
            # Integral instants emit as int so exact matches against int
            # payloads stay exact; fractional ones keep their float precision.
            return int(ts) if float(ts).is_integer() else ts
        # iso collection: a numeric value OR a numeric STRING ("1776254400")
        # is cross-domain — an ISO-shaped string is native and stays verbatim.
        if is_num or (isinstance(v, str) and not parses_as_iso(v)):
            dt = parse_payload_instant(v, numeric_unit="auto")
            return dt.isoformat() if dt is not None else v
        return v

    cond = filters[timestamp_key]
    if isinstance(cond, dict) and ("gte" in cond or "lte" in cond):
        new_cond: Any = dict(cond)
        for side in ("gte", "lte"):
            if cond.get(side) is not None:
                new_cond[side] = _conv(cond[side])
    else:
        converted = _conv(cond)
        if (converted is not cond and converted != cond) or isinstance(converted, float):
            # A CROSS-domain exact value becomes a degenerate range: scalar
            # MatchValue equality is representation-sensitive (a stored
            # "...Z" string never equals a generated "...+00:00"; an int
            # payload need not equal a float), while a range condition
            # compares as instants/numbers on the server. A same-domain FLOAT
            # gets the same treatment — Qdrant's MatchValue admits ints,
            # strings and bools but rejects floats outright.
            new_cond = {"gte": converted, "lte": converted}
        else:
            new_cond = cond
    out = dict(filters)
    out[timestamp_key] = new_cond
    return out


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
    newer_than: str | int | float | None = None,
    older_than: str | int | float | None = None,
    tokenizer: Tokenizer | None = None,
    timestamp_key: str = "timestamp",
    timestamp_format: str = "iso",
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
        newer_than: Optional ISO timestamp — or a numeric epoch, for a
            collection that stores numeric timestamps; keep chunks with
            ``payload[timestamp_key] >= newer_than``.
        older_than: Same shapes; keep chunks with
            ``payload[timestamp_key] <= older_than``.
        timestamp_key: Payload key the newer_than/older_than window filters
            on (a pre-existing collection's own schema).
        tokenizer: Optional analyzer used to pre-tokenize document text. If you
            pass these docs to ``BM25``/``BM25Retriever`` with the same
            tokenizer directly, use ``retokenize=False`` there to avoid a
            second corpus pass. ``BM25Retriever.from_qdrant`` handles this
            automatically.

    Warning:
        For very large collections (>1M chunks), consider passing a Filter to
        scope the BM25 corpus, or using newer_than/older_than for a rolling
        window. BM25 build memory is O(n). For example:
        ``bm25_docs_from_qdrant(client, "memory", newer_than="2026-04-01T00:00:00Z")``.
    """
    docs: list[BM25Doc] = []
    offset = None
    if newer_than is not None or older_than is not None:
        # The window bounds must live in the collection's own domain — an ISO
        # window over a numeric field builds a DatetimeRange that matches
        # nothing and silently loads an EMPTY corpus.
        converted = convert_timestamp_filter(
            {timestamp_key: {"gte": newer_than, "lte": older_than}},
            timestamp_key=timestamp_key,
            timestamp_format=timestamp_format,
        )
        window = (converted or {})[timestamp_key]
        newer_than, older_than = window.get("gte"), window.get("lte")
    effective_filter = _with_timestamp_range_filter(
        scroll_filter, newer_than=newer_than, older_than=older_than, timestamp_key=timestamp_key
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
            # Keep the NATIVE id (int stays int): the recaller fuses and
            # dedups by exact id equality, so a stringified copy of a dense
            # hit's integer id would double the memory instead of boosting it.
            native_id = getattr(point, "id", len(docs))
            qdrant_id = str(native_id)
            doc_payload = dict(payload)
            doc_payload.setdefault("qdrant_id", qdrant_id)
            # Only set a source when one actually exists — Qdrant points often
            # carry neither key, and writing source=None poisons every
            # consumer that does payload.get("source", "") expecting a string.
            src = payload.get("source_file") or payload.get("source")
            if src is not None:
                doc_payload.setdefault("source", src)
            doc_id = native_id if id_prefix is None else f"bm25:{id_prefix}:{qdrant_id}"
            docs.append(
                BM25Doc(
                    id=doc_id,
                    text=text,
                    payload=doc_payload,
                    tokens=(tokenizer or tokenize)(text),
                )
            )
            if limit is not None and len(docs) >= limit:
                break
        if offset is None:
            break
    return docs


class QdrantTextRetriever(Retriever):
    """Lexical-gated dense search — the scalable replacement for in-process BM25.

    Instead of holding a lexical corpus in client memory (the in-process BM25's
    documented "<100K documents" ceiling), the GATE runs inside Qdrant: the
    query's salient tokens become a full-text ``MatchText`` should-clause, and
    dense similarity ranks *within* the lexically-matching subset. Exact-token
    recall (IDs, error codes, names) at any collection size, no reindex — the
    one server-side prerequisite is a full-text payload index on the text
    field (``VectorStore.ensure_text_index()`` / ``mnemostack text-index``);
    without it a real server rejects MatchText filters.

    Scoring note: the gate FILTERS, it does not score (Qdrant's full-text
    match carries no relevance weight) — ranking inside the gate is dense
    similarity. For server-side lexical *scoring*, see
    ``QdrantSparseRetriever``.
    """

    name = "qdrant_text"

    accepts_as_of = True
    accepts_include_invalidated = True
    accepts_tenant = True

    def __init__(
        self,
        embedding: EmbeddingProvider,
        vector_store: VectorStore,
        *,
        text_key: str = "text",
        gate_key: str | None = None,
        timestamp_key: str = "timestamp",
        timestamp_format: str = "iso",
        max_gate_tokens: int = 8,
        min_token_len: int = 3,
        name: str | None = None,
    ):
        self.embedding = embedding
        self.vector_store = vector_store
        self._space_guard = SpaceGuard(vector_store, embedding)
        self.text_key = text_key
        # The field the MatchText gate runs on. Distinct from text_key on
        # purpose: an arm gating on a title/heading field must still RETURN
        # the chunk body (text_key) — otherwise a point surfaced only by the
        # title arm would carry its title as the result text into fusion and
        # synthesis. Needs a full-text index on THIS field on real servers.
        self.gate_key = gate_key or text_key
        self.timestamp_key = timestamp_key
        self.timestamp_format = timestamp_format
        self.max_gate_tokens = max_gate_tokens
        self.min_token_len = min_token_len
        self._set_name(name)

    def _gate_tokens(self, query: str) -> list[str]:
        """Salient tokens for the lexical gate: exact technical tokens first
        (IDs, IPs, versions — the queries this arm exists for), then content
        words, longest first (a cheap rarity proxy with no corpus statistics),
        capped so a verbose query can't turn the gate into match-anything."""
        from .mca_prefilter import extract_exact_tokens
        from .pipeline.stages import STOPWORDS

        exact = extract_exact_tokens(query)
        words = [
            t
            for t in tokenize(query)
            if len(t) >= self.min_token_len and t not in STOPWORDS
        ]
        # Deterministic across processes: length desc (rarity proxy), then
        # lexicographic — a set() sort keyed on length alone would let hash
        # seeds decide which same-length tokens make the capped gate.
        rest = sorted(set(words) - set(exact), key=lambda t: (-len(t), t))
        return (exact + rest)[: self.max_gate_tokens]

    def explain_empty(self, query: str) -> str | None:
        # Instance name, not the class literal: each multi-field arm must
        # report ITS OWN empty-gate verdict in traces.
        return f"{self.name}:no_tokens" if not self._gate_tokens(query) else None

    def search(
        self, query, limit=20, filters=None, as_of=None, include_invalidated=False, tenant=None
    ):
        tokens = self._gate_tokens(query)
        if not tokens:
            return []
        self._space_guard.ensure()
        vec = embed_query_via(self.embedding, query)
        if not vec:
            return []
        filters = convert_timestamp_filter(
            filters, timestamp_key=self.timestamp_key, timestamp_format=self.timestamp_format
        )
        tkw = {"tenant": tenant} if tenant is not None else {}
        hits = self.vector_store.search(
            vec,
            limit=limit,
            filters=filters,
            hide_invalidated=as_of is None and not include_invalidated,
            text_any=tokens,
            text_any_key=self.gate_key,
            **tkw,
        )
        return [
            RecallResult(
                id=h.id,
                text=(h.payload or {}).get(self.text_key, ""),
                score=h.score,
                payload=dict(h.payload or {}),
                # The instance name, not the class literal: multi-field arms
                # must stay distinguishable in result sources and telemetry.
                sources=[self.name],
            )
            for h in hits
        ]


class _SharedQueryEmbedding(EmbeddingProvider):
    """Memoizing wrapper shared by the arms of one multi-field lexical set.

    The arms differ only in their MatchText gate field — the query vector is
    identical, yet each ``search()`` would embed the query independently
    (concurrently, from the recaller's thread pool): N fields = N identical
    billable provider requests per recall. One bounded memo collapses them to
    one. Scoped to the factory's arm set on purpose — nothing outside the
    multi-field feature changes behavior.

    Role note: the arms reach this wrapper through ``embed_query``. For a
    declaratively-profiled inner provider that inherits the base role
    methods, the transform applies exactly once (in the inherited
    ``embed_query``) and lands in the memoized ``embed`` — the memo key is
    the already-transformed inference input. An inner provider that
    OVERRIDES a role method (native backend query/document task types) is
    delegated to directly instead, memoized under a role-tagged key —
    otherwise the base implementation would route around its override and
    embed queries in the wrong mode.
    """

    _MAX_ENTRIES = 32
    #: Memo entries expire on the same bounded-freshness contract as
    #: SpaceGuard verdicts: after a tag repoint or collection recreation, a
    #: recurring query's cached vector (from the OLD weights) must not be
    #: searched against the new vectors indefinitely — recomputing the
    #: fingerprint on the hot path would be far costlier than re-embedding
    #: one query every few minutes.
    _MEMO_TTL_S = 300.0

    def __init__(self, embedding: EmbeddingProvider):
        self._embedding = embedding
        self._lock = threading.Lock()
        self._memo: dict[Any, tuple[float, list[float]]] = {}
        self._inflight: dict[Any, tuple[threading.Event, list[list[float]]]] = {}

    def embed(self, text: str) -> list[float]:
        return self._single_flight(text, lambda: self._embedding.embed(text))

    def _inner_overrides(self, name: str) -> bool:
        method = getattr(type(self._embedding), name, None)
        return method is not None and method is not getattr(EmbeddingProvider, name)

    def embed_query(self, text: str) -> list[float]:
        if self._inner_overrides("embed_query"):
            return self._single_flight(
                ("native", "query", text), lambda: self._embedding.embed_query(text)
            )
        return super().embed_query(text)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        if self._inner_overrides("embed_queries"):
            return self._embedding.embed_queries(texts)
        return super().embed_queries(texts)

    def embed_document(self, text: str) -> list[float]:
        if self._inner_overrides("embed_document"):
            return self._single_flight(
                ("native", "document", text), lambda: self._embedding.embed_document(text)
            )
        return super().embed_document(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._inner_overrides("embed_documents"):
            return self._embedding.embed_documents(texts)
        return super().embed_documents(texts)

    def _single_flight(self, key: Any, compute: Callable[[], list[float]]) -> list[float]:
        # Single-flight: the arms run CONCURRENTLY from the recaller's thread
        # pool, so a plain check-then-compute memo lets every arm miss before
        # any result lands — N identical billable requests on every new query,
        # exactly what this wrapper exists to prevent. The first caller for a
        # key becomes the leader; concurrent callers wait on its event.
        with self._lock:
            memo_entry = self._memo.get(key)
            if memo_entry is not None:
                stamp, cached = memo_entry
                if time.monotonic() - stamp <= self._MEMO_TTL_S:
                    return cached
                # Expired: recompute so a repointed tag / recreated
                # collection stops being served a stale vector.
                self._memo.pop(key, None)
            entry = self._inflight.get(key)
            if entry is None:
                entry = (threading.Event(), [])
                self._inflight[key] = entry
                leader = True
            else:
                leader = False
        event, box = entry
        if not leader:
            event.wait()
            return box[0] if box else []
        vec: list[float] = []  # empty = the providers' own failure contract
        try:
            vec = compute()
        finally:
            with self._lock:
                box.append(vec)
                # Failures (empty vectors) are NOT memoized: a transient
                # provider outage must not pin the lexical arms dead for
                # this query until 32 other queries evict the entry.
                if vec:
                    if len(self._memo) >= self._MAX_ENTRIES:
                        # FIFO eviction — a tiny bound is plenty; this only
                        # exists to stop growth.
                        self._memo.pop(next(iter(self._memo)))
                    self._memo[key] = (time.monotonic(), vec)
                self._inflight.pop(key, None)
            event.set()
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._embedding.embed_batch(texts)

    @property
    def dimension(self) -> int:
        return self._embedding.dimension

    @property
    def name(self) -> str:
        return self._embedding.name

    @property
    def profile(self):
        # Delegate instead of re-resolving from `name`: an instance-level
        # profile override on the wrapped provider must win here too. A
        # duck-typed inner without profiles keeps today's identity behavior.
        prof = getattr(self._embedding, "profile", None)
        if prof is not None:
            return prof
        from ..embeddings.profiles import IDENTITY_PROFILE

        return IDENTITY_PROFILE

    @profile.setter
    def profile(self, value) -> None:
        self._embedding.profile = value

    def document_space_fingerprint(self) -> str:
        # Forwarded so the wrapped provider's own fingerprint extras
        # (e.g. HuggingFace pooling) are not lost; a duck-typed inner falls
        # back to the base computation over the delegated name/profile.
        method = getattr(self._embedding, "document_space_fingerprint", None)
        if method is not None:
            return method()
        return super().document_space_fingerprint()

    def query_profile_fingerprint(self) -> str:
        method = getattr(self._embedding, "query_profile_fingerprint", None)
        if method is not None:
            return method()
        return super().query_profile_fingerprint()

    def health_check(self) -> tuple[bool, str]:
        return self._embedding.health_check()


def build_qdrant_text_arms(
    embedding: EmbeddingProvider,
    vector_store: VectorStore,
    *,
    text_key: str = "text",
    timestamp_key: str = "timestamp",
    timestamp_format: str = "iso",
    fields: dict[str, float] | None = None,
) -> tuple[list[QdrantTextRetriever], dict[str, float]]:
    """Construct the lexical-gated arm set for ``text_search: lexical``.

    The single shared builder for every surface (HTTP server, MCP, CLI).
    Without ``fields`` (the default) it returns the one historical arm gating
    on ``text_key``. With ``fields`` ({payload_field: fusion_weight}) each
    field becomes its own arm: gating on that field, returning chunk text
    from ``text_key``, named ``qdrant_text`` for the ``text_key`` field
    (continuity: an existing deployment adding a title arm keeps the body
    arm's learned profile) and ``qdrant_text:<field>`` otherwise.

    Returns ``(arms, weights)`` where ``weights`` maps arm name -> fusion
    weight for ``Recaller(retriever_weights=...)``. Weight 1.0 entries are
    OMITTED on purpose: a static override always beats the adaptive
    (Q-learning) profile, so default-weight arms stay adaptive.
    """
    from ..config import parse_text_search_fields

    # Validate at THIS public boundary too, not only in Config.load: a
    # programmatic caller (ServerConfig/build_server/library use) passing
    # weight 0/-1/NaN would satisfy the type annotation while RRF clamps the
    # arm to zero — a silently omitted arm instead of a loud config error.
    fields = parse_text_search_fields(fields)
    if not fields:
        fields = {text_key: 1.0}
    if len(fields) > 1:
        # The arms run concurrently and embed the SAME query — share one
        # memoized provider so N fields cost one embedding call per recall.
        embedding = _SharedQueryEmbedding(embedding)
    arms: list[QdrantTextRetriever] = []
    weights: dict[str, float] = {}
    for gate_field, weight in fields.items():
        arm_name = (
            QdrantTextRetriever.name
            if gate_field == text_key
            else f"{QdrantTextRetriever.name}:{gate_field}"
        )
        arms.append(
            QdrantTextRetriever(
                embedding=embedding,
                vector_store=vector_store,
                text_key=text_key,
                gate_key=gate_field,
                timestamp_key=timestamp_key,
                timestamp_format=timestamp_format,
                name=arm_name,
            )
        )
        if weight != 1.0:
            weights[arm_name] = float(weight)
    return arms, weights


class QdrantSparseRetriever(Retriever):
    """Server-side lexical scoring over Qdrant sparse vectors.

    The counterpart to ``QdrantTextRetriever`` for deployments that CAN
    (re)index: writes maintain a sparse tf encoding of each chunk's text
    (``VectorStore(sparse_text=True)``), Qdrant's ``Modifier.IDF`` supplies
    collection-wide term weighting at query time, and this retriever ranks by
    that tf·idf-like score — true lexical *scoring* (not just a gate) with no
    client-side corpus, at any collection size. Requires the collection to
    carry the sparse space (created by ``ensure_collection`` on a
    ``sparse_text=True`` store; existing collections need a re-index or a
    server that accepts the config update).
    """

    name = "sparse"

    accepts_as_of = True
    accepts_include_invalidated = True
    accepts_tenant = True

    def __init__(
        self,
        vector_store: VectorStore,
        *,
        text_key: str = "text",
        timestamp_key: str = "timestamp",
        timestamp_format: str = "iso",
        name: str | None = None,
    ):
        if not getattr(vector_store, "sparse_text", False):
            raise ValueError(
                "QdrantSparseRetriever needs a VectorStore(sparse_text=True) — "
                "this store does not maintain the sparse text space"
            )
        self.vector_store = vector_store
        self.text_key = text_key
        self.timestamp_key = timestamp_key
        self.timestamp_format = timestamp_format
        self._set_name(name)

    def search(
        self, query, limit=20, filters=None, as_of=None, include_invalidated=False, tenant=None
    ):
        filters = convert_timestamp_filter(
            filters, timestamp_key=self.timestamp_key, timestamp_format=self.timestamp_format
        )
        tkw = {"tenant": tenant} if tenant is not None else {}
        hits = self.vector_store.sparse_search(
            query,
            limit=limit,
            filters=filters,
            hide_invalidated=as_of is None and not include_invalidated,
            **tkw,
        )
        return [
            RecallResult(
                id=h.id,
                text=(h.payload or {}).get(self.text_key, ""),
                score=h.score,
                payload=dict(h.payload or {}),
                sources=[self.name],
            )
            for h in hits
        ]


class BM25Retriever(Retriever):
    """Exact token match via BM25."""

    name = "bm25"

    #: File-corpus BM25 has no tenant metadata, so it cannot be tenant-scoped
    #: (the recaller skips it under a tenant). A corpus scrolled from Qdrant
    #: payloads DOES carry tenant_id — from_qdrant flips this on.
    accepts_tenant = False

    def __init__(
        self,
        docs: list[BM25Doc],
        tokenizer: Tokenizer = tokenize,
        *,
        retokenize: bool | None = None,
        timestamp_key: str = "timestamp",
        timestamp_format: str = "iso",
        tenant_aware: bool = False,
        name: str | None = None,
    ):
        self.bm25 = BM25(docs, tokenizer=tokenizer, retokenize=retokenize)
        self._set_name(name)
        #: The one payload key whose range filters may cross timestamp domains
        #: (see payload_matches) — a foreign collection's own schema — and how
        #: its numeric values are read.
        self.timestamp_key = timestamp_key
        self.timestamp_format = timestamp_format
        if tenant_aware:
            # Instance-level capability marker: the recaller only routes a
            # tenant to retrievers that advertise it.
            self.accepts_tenant = True

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
        newer_than: str | int | float | None = None,
        older_than: str | int | float | None = None,
        tokenizer: Tokenizer | None = None,
        timestamp_key: str = "timestamp",
        timestamp_format: str = "iso",
        name: str | None = None,
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
            tokenizer: Optional analyzer applied consistently to corpus and
                query text.

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
            tokenizer=tokenizer,
            timestamp_key=timestamp_key,
            timestamp_format=timestamp_format,
        )
        return cls(
            docs=docs,
            tokenizer=tokenizer or tokenize,
            retokenize=False,
            timestamp_key=timestamp_key,
            timestamp_format=timestamp_format,
            name=name,
            # Qdrant payloads carry tenant_id — this corpus CAN be scoped.
            tenant_aware=True,
        )

    def search(self, query, limit=20, filters=None, tenant=None):
        # Same filter semantics the vector store applies natively. Without
        # this, a fused recall with filters= mixed unfiltered BM25 candidates
        # into the output — in multi-tenant deployments that is an isolation
        # leak. The candidate set is restricted before the top-K cut. A tenant
        # (only ever routed here when tenant_aware) is a STRICT gate: a doc
        # without the tenant marker never matches a scoped search.
        predicate = None
        if filters or tenant is not None:

            def predicate(d: BM25Doc) -> bool:
                if tenant is not None:
                    from ..vector.qdrant import TENANT_ID_KEY

                    if (d.payload or {}).get(TENANT_ID_KEY) != tenant:
                        return False
                return payload_matches(
                    d.payload,
                    filters,
                    timestamp_key=self.timestamp_key,
                    numeric_unit=numeric_unit_for(self.timestamp_format),
                )

        hits = self.bm25.search(query, limit=limit, predicate=predicate)
        return [
            RecallResult(
                id=d.id,
                text=d.text,
                score=s,
                payload=d.payload or {},
                sources=[self.name],
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
        text_key: str = "text",
        *,
        timestamp_key: str = "timestamp",
        timestamp_format: str = "iso",
    ):
        self.llm = llm
        self.embedding = embedding
        self.vector_store = vector_store
        self._space_guard = SpaceGuard(vector_store, embedding)
        self.max_tokens = max_tokens
        self.text_key = text_key
        self.timestamp_key = timestamp_key
        self.timestamp_format = timestamp_format

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
        # Guard BEFORE the LLM call — no point paying for a hypothetical
        # that cannot be embedded into a compatible space.
        self._space_guard.ensure()
        hypo = self._generate_hypothetical(query)
        if not hypo:
            return []
        # HyDE's whole premise is that the hypothetical lives in DOCUMENT
        # space (it imitates a stored memory), so it takes the document role.
        vec = embed_document_via(self.embedding, hypo)
        if not vec:
            return []
        hits = self.vector_store.search(
            vec,
            limit=limit,
            filters=convert_timestamp_filter(
                filters,
                timestamp_key=self.timestamp_key,
                timestamp_format=self.timestamp_format,
            ),
        )
        return [
            RecallResult(
                id=h.id,
                text=h.payload.get(self.text_key, ""),
                score=h.score,
                payload=h.payload,
                sources=["hyde"],
            )
            for h in hits
        ]


def graph_valid_clause(var: str, as_of: str | None, include_invalidated: bool = False) -> str:
    """Cypher validity predicate for a node/rel variable.

    - ``include_invalidated`` and no ``as_of``: no filter (``true``) — the
      caller asked for stale facts too, so don't suppress closed graph edges.
    - No ``as_of`` (default): "currently valid" (``valid_until`` = the
      ``'current'`` marker, or legacy NULL).
    - ``as_of`` set: the point-in-time predicate ``GraphStore.query_triples``
      uses, referencing the bound ``$as_of`` parameter (``as_of`` wins over
      ``include_invalidated`` — point-in-time reconstruction is explicit).

    Shared by ``MemgraphRetriever`` and the pipeline's graph-resurrection stage
    so both filter graph facts consistently. Point-in-time bounds are compared
    as **parsed instants** (``datetime(...)``), not raw strings, so mixed
    sub-second precision orders correctly (a raw compare misreads ``…00.5Z`` as
    before ``…00Z`` because ``.`` < ``Z``). A ``CASE`` guards the
    ``'current'``/NULL markers so ``datetime()`` is never evaluated on them;
    ``datetime(NULL)`` is null (not an error) for an open ``valid_from``. Bind a
    ``datetime()``-parseable ``$as_of`` (see ``to_utc_instant``).
    """
    if as_of is None:
        if include_invalidated:
            return "true"
        return f"coalesce({var}.valid_until, 'current') = 'current'"
    return graph_as_of_predicate(var)


def graph_result_id(node_id: str, tenant: str | None) -> str:
    """Stable id for a graph hit, tenant-namespaced (collision-free) when scoped.

    The stateful pipeline (IoR / Q-learning / feedback) keys on ``str(result.id)``,
    so two tenants' same-named graph nodes must not share an id. Tenants and node
    names are arbitrary strings that can contain ``:``, so the tenant is
    percent-encoded (``safe=""``) — otherwise tenant ``a:b`` + node ``c`` and
    tenant ``a`` + node ``b:c`` would both render ``graph:a:b:c``. Unscoped
    (``tenant is None``) keeps the historical ``graph:<node_id>`` form unchanged.
    """
    if tenant is None:
        return f"graph:{node_id}"
    return f"graph:{quote(tenant, safe='')}:{node_id}"


class MemgraphRetriever(Retriever):
    """Knowledge-graph retriever — exact/contains match on node names.

    Mirrors legacy enhanced-recall.py:fetch_memgraph. Each word >=3 chars in
    the query becomes a probe; nodes matched by multiple probes get higher
    counts (used as score).
    """

    name = "memgraph"
    #: The recall path passes each validity kwarg only to retrievers that
    #: advertise it, so a custom retriever that accepts only `as_of` (or
    #: neither) is never handed an argument its `search` doesn't take. Graph
    #: facts carry no validity payload, so both are filtered at query time.
    accepts_as_of = True
    accepts_include_invalidated = True
    #: Graph nodes/edges now carry a server-owned ``tenant`` property (see
    #: ``GraphStore``), so the retriever can honor a tenant scope — the recall
    #: path pushes ``tenant`` here instead of skipping the graph under auth.
    accepts_tenant = True

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
        # database appended at the tail to preserve positional back-compat for
        # existing callers (a mid-signature insert would shift min_word etc.).
        database: str | None = None,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
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

    _valid_clause = staticmethod(graph_valid_clause)

    def search(
        self, query, limit=20, filters=None, as_of=None, include_invalidated=False, tenant=None
    ):
        if filters:
            # Caller payload `filters` (source, arbitrary keys, time range) can't
            # be proven against graph nodes, which carry no chunk payload — under
            # the isolation contract anything unattributable is excluded, not
            # leaked. The dedicated `tenant` scope below is different: it's a
            # server-owned graph property, so it IS honored (not via `filters`).
            return []
        driver = self._get_driver()
        if driver is None:
            return []
        words = [w.lower() for w in query.split() if len(w) >= self.min_word]
        if not words:
            return []
        # Tenant scope: confine every probe/relationship match to nodes+edges
        # carrying `tenant`. Unscoped (tenant=None) adds nothing, so a legacy
        # single-tenant graph is queried exactly as before.
        tnode = " AND n.tenant = $tenant" if tenant is not None else ""
        trel = (
            " AND n.tenant = $tenant AND m.tenant = $tenant AND r.tenant = $tenant"
            if tenant is not None
            else ""
        )
        # Normalize `as_of` to a full UTC instant (expanding a bare date to
        # midnight-Z) so it parses via Cypher datetime() in the point-in-time
        # predicate (see graph_as_of_predicate).
        as_of = to_utc_instant(as_of)
        node_valid = self._valid_clause("n", as_of, include_invalidated)
        rel_valid = self._valid_clause("r", as_of, include_invalidated)
        target_valid = self._valid_clause("m", as_of, include_invalidated)
        # Only bind $as_of when the predicate references it.
        validity_active = as_of is not None or not include_invalidated
        # Cap each probe at the candidate budget rather than an arbitrary 5:
        # with the (name, index_root) grouping one filename can legitimately
        # match many nodes — one per root — and a hardcoded LIMIT would silently
        # drop roots before the grouping/ranking below could consider them.
        node_budget = self.max_nodes * 3 if validity_active else self.max_nodes
        extra: dict[str, Any] = {"probe_lim": node_budget}
        if as_of is not None:
            extra["as_of"] = as_of
        if tenant is not None:
            extra["tenant"] = tenant
        counts: dict[tuple[str, str], dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "type": "", "mc": ""}
        )
        # Only pass database= when a non-default DB is configured, so an injected
        # driver whose session() takes no args (fakes/wrappers) still works.
        session_kwargs = {"database": self.database} if self.database else {}
        try:
            with driver.session(**session_kwargs) as session:
                for w in words:
                    # Probe 1: numeric-looking tokens may be contact IDs
                    # (Telegram, Discord, etc). If a canonical Person node has
                    # a matching contact_id property, surface it directly —
                    # most reliable entity-resolution signal we have.
                    rows: list[dict] = []
                    if w.isdigit() and len(w) >= 6:
                        rows = session.run(
                            "MATCH (n) WHERE (n.telegram_id = $w OR n.contact_id = $w) "
                            f"AND {node_valid}{tnode} "
                            "RETURN n.name AS name, labels(n)[0] AS type, "
                            "n.memory_class AS mc, n.index_root AS index_root "
                            "LIMIT $probe_lim",
                            w=w,
                            **extra,
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
                            f"AND {node_valid}{tnode} "
                            "RETURN n.name AS name, labels(n)[0] AS type, "
                            "n.memory_class AS mc, n.index_root AS index_root "
                            "LIMIT $probe_lim",
                            w=w,
                            **extra,
                        ).data()
                    # Probe 3: also match by handle/username (e.g. @alice)
                    if not rows and len(w) >= 3:
                        rows = session.run(
                            "MATCH (n) WHERE toLower(coalesce(n.telegram_username, '')) = $w "
                            f"AND {node_valid}{tnode} "
                            "RETURN n.name AS name, labels(n)[0] AS type, "
                            "n.memory_class AS mc, n.index_root AS index_root "
                            "LIMIT $probe_lim",
                            w=w,
                            **extra,
                        ).data()
                    # Probe 4: substring fallback for longer tokens.
                    if not rows and len(w) >= self.contains_min:
                        rows = session.run(
                            "MATCH (n) WHERE coalesce(n.name_lower, toLower(n.name)) CONTAINS $w "
                            f"AND {node_valid}{tnode} "
                            "RETURN n.name AS name, labels(n)[0] AS type, "
                            "n.memory_class AS mc, n.index_root AS index_root "
                            "LIMIT $probe_lim",
                            w=w,
                            **extra,
                        ).data()
                    for n in rows:
                        name = n.get("name")
                        if not name:
                            continue
                        # Key by (name, index_root) so same-named :File nodes
                        # from different markdown roots stay distinct — the read
                        # side must honor the (name, index_root) scoping the write
                        # side (sync_file_links) uses. :Entity nodes have no
                        # index_root, so their key root is "" (unchanged).
                        key = (name, n.get("index_root") or "")
                        counts[key]["count"] += 1
                        counts[key]["type"] = n.get("type", "") or ""
                        counts[key]["mc"] = n.get("mc", "") or ""
                # Over-fetch node candidates so a window of stale-only nodes
                # doesn't hide valid ones below it before the bare-node skip.
                ranked = sorted(counts.items(), key=lambda kv: -kv[1]["count"])[:node_budget]
                results: list[RecallResult] = []
                for (name, root_key), info in ranked:
                    if len(results) >= self.max_nodes:
                        break
                    rel_rows = session.run(
                        # Undirected: a target-only node (e.g. `Team A` in
                        # `Alice -[MEMBER_OF]-> Team A`) has a valid incoming
                        # edge but no outgoing one, so only checking `->` would
                        # wrongly drop it under the bare-node skip. startNode/
                        # endNode render the true direction. Filter the other
                        # endpoint `m` too so a stale/future neighbor isn't
                        # serialized into rel_text.
                        # Pin the node to its (name, index_root): without this an
                        # undifferentiated name match would follow LINKS_TO edges
                        # of every same-named :File across roots. coalesce(...,'')
                        # matches :Entity nodes (no index_root) when root_key="".
                        "MATCH (n {name: $name})-[r]-(m) "
                        f"WHERE coalesce(n.index_root, '') = $root_key "
                        f"AND {node_valid} AND {rel_valid} AND {target_valid}{trel} "
                        "RETURN startNode(r).name AS from_n, type(r) AS rel, "
                        "endNode(r).name AS to_n LIMIT $lim",
                        name=name,
                        root_key=root_key,
                        lim=self.max_rels,
                        **extra,
                    ).data()
                    # Nodes carry no valid_from and invalidate() closes edges,
                    # not nodes — so under a validity view a node can pass its
                    # own probe while all its incident facts are out of window.
                    # Don't emit a bare entity with no valid incident fact then.
                    if not rel_rows and validity_active:
                        continue
                    rel_text = "; ".join(
                        f"{r['from_n']}-[{r['rel']}]->{r['to_n']}" for r in rel_rows
                    )
                    content = (
                        f"{info['type']}: {name}. {rel_text}"
                        if rel_text
                        else f"{info['type']}: {name}"
                    )
                    # Qualify the id with the root so same-named :File nodes from
                    # different roots don't collide into one result. :Entity nodes
                    # (and files with no index_root) keep the plain graph:<name> id.
                    node_id = f"{root_key}:{name}" if root_key else name
                    payload: dict[str, Any] = {
                        "text": content[:300],
                        "source": "memgraph",
                        "memory_class": info.get("mc", ""),
                        "name": name,
                        "type": info["type"],
                        "index_root": root_key or None,
                    }
                    # Stamp the tenant so this graph hit carries the same
                    # `tenant_id` the vector hits do and survives the recall
                    # `filter_by_tenant` backstop (which drops any result lacking
                    # a matching tenant_id). Only every node we matched is already
                    # tenant-scoped by the probes above.
                    # Namespace the id by tenant when scoped so two tenants with the
                    # same graph node don't share an id — the stateful pipeline
                    # (IoR / Q-learning / feedback) keys on str(result.id), so a
                    # shared id would let one tenant's clicks move the other's
                    # ranking. Vector ids are already tenant-scoped (stable_chunk_id).
                    if tenant is not None:
                        payload["tenant_id"] = tenant
                    result_id = graph_result_id(node_id, tenant)
                    results.append(
                        RecallResult(
                            id=result_id,
                            text=content[:300],
                            score=float(info["count"]),
                            payload=payload,
                            sources=["memgraph"],
                        )
                    )
                return results[:limit]
        except Exception:
            # Fail open (graph is optional), but log — a bad query or a malformed
            # stored bound blanks graph recall for this call, which was silent.
            logger.warning("graph recall failed, returning no graph hits", exc_info=True)
            return []


# --- Temporal extraction ---
# Port of legacy temporal_extractor.extract_temporal, extended for date-focused recall.

_MONTHS = {
    "январ": 1,
    "january": 1,
    "jan": 1,
    "феврал": 2,
    "february": 2,
    "feb": 2,
    "март": 3,
    "march": 3,
    "mar": 3,
    "апрел": 4,
    "april": 4,
    "apr": 4,
    "май": 5,
    "мая": 5,
    "мае": 5,
    "may": 5,
    "июн": 6,
    "june": 6,
    "jun": 6,
    "июл": 7,
    "july": 7,
    "jul": 7,
    "август": 8,
    "august": 8,
    "aug": 8,
    "сентябр": 9,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "октябр": 10,
    "october": 10,
    "oct": 10,
    "ноябр": 11,
    "november": 11,
    "nov": 11,
    "декабр": 12,
    "december": 12,
    "dec": 12,
}

_MONTH_NAME_PATTERN = "|".join(sorted(map(re.escape, _MONTHS), key=len, reverse=True))
_GENERIC_DATE_QUERY_WORDS = {
    "что",
    "делали",
    "делал",
    "делала",
    "было",
    "случилось",
    "произошло",
    "события",
    "событие",
    "работа",
    "работали",
    "задачи",
    "задача",
    "дела",
    "итоги",
    "дневник",
    "what",
    "happened",
    "happen",
    "did",
    "do",
    "done",
    "worked",
    "work",
    "tasks",
    "task",
    "events",
    "event",
    "notes",
    "note",
    "diary",
    "log",
    "logs",
    "anything",
    "from",
    "on",
    "in",
    "at",
    "the",
    "a",
    "an",
    # "around <date>" qualifiers must not make a date query look specific
    "around",
    "about",
    "approximately",
    "circa",
    "примерно",
    "около",
    "где-то",
    "за",
    "на",
    "в",
    "во",
    "и",
    "по",
}


@dataclass(frozen=True)
class TemporalQuery:
    """Parsed temporal intent for recall queries."""

    start_iso: str
    end_iso: str
    target_date: date | None = None
    date_focused: bool = False

    @property
    def window(self) -> tuple[str, str]:
        return self.start_iso, self.end_iso


def _now_date(now: datetime | None = None) -> date:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc).date()


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _day_window(day: date, slack: int = 1) -> tuple[str, str]:
    start = datetime.combine(day - timedelta(days=slack), datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(day + timedelta(days=slack), datetime.max.time(), tzinfo=timezone.utc)
    return start.isoformat(), end.isoformat()


_AROUND_QUALIFIER_RE = re.compile(
    r"\b(?:around|about|approximately|circa|примерно|около|где-то)\s*$",
    re.IGNORECASE,
)


def _around_slack(q: str, match_start: int) -> int:
    """±3 day window when the date is qualified ('around April 30'), else ±1."""
    prefix = q[max(0, match_start - 16) : match_start]
    return 3 if _AROUND_QUALIFIER_RE.search(prefix) else 1


def _intersect_instant_window(
    start: datetime,
    end: datetime,
    caller_condition: Any,
    *,
    numeric_unit: str = "auto",
) -> tuple[datetime, datetime] | None:
    """Intersect the parsed query window with a caller timestamp filter.

    The caller condition may be a ``{"gte"/"lte"}`` range or an exact value
    (a degenerate range), in ANY timestamp domain — ISO string, datetime, or
    epoch number — since the comparison runs on the time line, not on raw
    values (a mixed-domain compare used to TypeError the whole window away).
    Returns None when the intersection is empty or a caller bound doesn't
    parse as an instant — the temporal retriever must then contribute nothing
    rather than hits outside a scope it couldn't understand.
    """
    if caller_condition is None:
        return start, end
    if isinstance(caller_condition, dict) and (
        "gte" in caller_condition or "lte" in caller_condition
    ):
        raw_gte = caller_condition.get("gte")
        raw_lte = caller_condition.get("lte")
    else:
        raw_gte = raw_lte = caller_condition
    gte = parse_payload_instant(raw_gte, numeric_unit=numeric_unit) if raw_gte is not None else None
    lte = parse_payload_instant(raw_lte, numeric_unit=numeric_unit) if raw_lte is not None else None
    if (raw_gte is not None and gte is None) or (raw_lte is not None and lte is None):
        return None
    if gte is not None and gte > start:
        start = gte
    if lte is not None and lte < end:
        end = lte
    if start > end:
        return None
    return start, end


def _month_window(year: int, month: int) -> tuple[str, str]:
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end_month = month + 1
    end_year = year + (1 if end_month > 12 else 0)
    end_month = end_month if end_month <= 12 else 1
    end = datetime(end_year, end_month, 1, tzinfo=timezone.utc)
    return start.isoformat(), end.isoformat()


def _year_window(year: int) -> tuple[str, str]:
    return (
        datetime(year, 1, 1, tzinfo=timezone.utc).isoformat(),
        datetime(year + 1, 1, 1, tzinfo=timezone.utc).isoformat(),
    )


def _date_focused(query: str, consumed: list[tuple[int, int]], *, has_target_date: bool) -> bool:
    if not has_target_date:
        return False
    remainder = query.lower()
    for start, end in sorted(consumed, reverse=True):
        remainder = remainder[:start] + " " + remainder[end:]
    words = re.findall(r"[\wёЁ-]+", remainder, flags=re.IGNORECASE)
    content_words = [w.strip("-") for w in words if w.strip("-")]
    specific = [w for w in content_words if w not in _GENERIC_DATE_QUERY_WORDS and not w.isdigit()]
    return not specific


def _month_from_text(month_text: str) -> int | None:
    m = month_text.lower()
    for stem, month in _MONTHS.items():
        if m.startswith(stem):
            return month
    return None


def extract_temporal_query(query: str, now: datetime | None = None) -> TemporalQuery | None:
    """Parse absolute/relative temporal intent from a query.

    Single-day expressions return a ±1 day timestamp window and mark vague
    generic questions as ``date_focused`` so retrieval can use a plain date
    filter instead of depending on semantic overlap with diary text.
    """
    q = query.lower()
    today = _now_date(now)

    # ISO absolute date: 2026-04-30
    m = re.search(r"\b(20\d{2})-(0?[1-9]|1[0-2])-(0?[1-9]|[12]\d|3[01])\b", q)
    if m:
        target = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if target is not None:
            start, end = _day_window(target, slack=_around_slack(q, m.start()))
            return TemporalQuery(
                start, end, target, _date_focused(query, [m.span()], has_target_date=True)
            )

    # Relative single-date expressions.
    relative_patterns: list[tuple[str, int]] = [
        (r"\bпозавчера\b", 2),
        (r"\bвчера\b", 1),
        (r"\bсегодня\b", 0),
        (r"\bday\s+before\s+yesterday\b", 2),
        (r"\byesterday\b", 1),
        (r"\btoday\b", 0),
        (r"\bнеделю\s+назад\b", 7),
        (r"\bна\s+прошлой\s+неделе\b", 7),
        (r"\b(?:a|one)\s+day\s+ago\b", 1),
        (r"\b(?:a|one)\s+week\s+ago\b", 7),
        (r"\blast\s+week\b", 7),
        (r"\b(\d+)\s+days?\s+ago\b", -1),
        (r"\b(\d+)\s+weeks?\s+ago\b", -7),
        (r"\b(\d+)\s+(?:день|дн(?:я|ей|ь)?)\s+назад\b", -1),
        (r"\b(\d+)\s+(?:неделя|недел(?:ю|и|ь)?)\s+назад\b", -7),
    ]
    for pattern, days in relative_patterns:
        m = re.search(pattern, q)
        if not m:
            continue
        delta_days = int(m.group(1)) * abs(days) if days < 0 else days
        target = today - timedelta(days=delta_days)
        start, end = _day_window(target, slack=_around_slack(q, m.start()))
        return TemporalQuery(
            start, end, target, _date_focused(query, [m.span()], has_target_date=True)
        )

    # Absolute day/month: "30 апреля", "April 30", optional year.
    day_month = re.search(rf"\b([0-3]?\d)\s+({_MONTH_NAME_PATTERN})\w*\s*(20\d{{2}})?\b", q)
    if day_month:
        day = int(day_month.group(1))
        month = _month_from_text(day_month.group(2))
        if month is not None:
            year = int(day_month.group(3)) if day_month.group(3) else today.year
            target = _safe_date(year, month, day)
            if target is not None:
                start, end = _day_window(target, slack=_around_slack(q, day_month.start()))
                return TemporalQuery(
                    start,
                    end,
                    target,
                    _date_focused(query, [day_month.span()], has_target_date=True),
                )

    month_day = re.search(
        rf"\b({_MONTH_NAME_PATTERN})\w*\s+([0-3]?\d)(?:st|nd|rd|th)?\s*(20\d{{2}})?\b", q
    )
    if month_day:
        month = _month_from_text(month_day.group(1))
        day = int(month_day.group(2))
        if month is not None:
            year = int(month_day.group(3)) if month_day.group(3) else today.year
            target = _safe_date(year, month, day)
            if target is not None:
                start, end = _day_window(target, slack=_around_slack(q, month_day.start()))
                return TemporalQuery(
                    start,
                    end,
                    target,
                    _date_focused(query, [month_day.span()], has_target_date=True),
                )

    # Part-of-month expressions: "early April", "mid-April", "late April 2026",
    # "в начале апреля", "в конце апреля". Must run before the month-only
    # fallback, which would otherwise match the same text with a wider window.
    part_m = re.search(
        rf"\b(?P<qual>early|beginnings?\s+of|start\s+of|middle\s+of|mid|late|end\s+of|"
        rf"начал[еоа]|середин[аеуы]|кон(?:це|ца|ец))(?:\s+|-)(?:of\s+)?"
        rf"(?P<month>{_MONTH_NAME_PATTERN})\w*\s*(?P<year>20\d{{2}})?\b",
        q,
    )
    if part_m:
        month = _month_from_text(part_m.group("month"))
        if month is not None:
            year = int(part_m.group("year")) if part_m.group("year") else today.year
            qual = part_m.group("qual")
            if qual.startswith(("early", "beginning", "start", "начал")):
                lo, hi = 1, 10
            elif qual.startswith(("mid", "middle", "середин")):
                lo, hi = 11, 20
            else:  # late / end of / конце
                lo, hi = 21, monthrange(year, month)[1]
            part_start = datetime(year, month, lo, tzinfo=timezone.utc)
            part_end = datetime.combine(
                date(year, month, hi), datetime.max.time(), tzinfo=timezone.utc
            )
            return TemporalQuery(part_start.isoformat(), part_end.isoformat(), None, False)

    # Month-only expressions keep the legacy month-wide semantic path.
    # Match whole month words/stems, not arbitrary substrings. The English
    # month "may" is also a common modal verb, so require either a nearby
    # temporal preposition or an explicit year before treating it as a month.
    for stem, month in _MONTHS.items():
        month_m = re.search(rf"\b{re.escape(stem)}\w*\b", q)
        if not month_m:
            continue
        y_m = re.search(r"\b(20\d{2})\b", q)
        if stem == "may" and not y_m:
            prefix = q[max(0, month_m.start() - 12) : month_m.start()]
            if not re.search(r"\b(in|from|during)\s+$", prefix):
                continue
        y = int(y_m.group(1)) if y_m else today.year
        start, end = _month_window(y, month)
        return TemporalQuery(start, end, None, False)

    # "YYYY" — full year
    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        start, end = _year_window(int(m.group(1)))
        return TemporalQuery(start, end, None, False)
    return None


def extract_temporal(query: str) -> tuple[str, str] | None:
    """Best-effort date range extraction. Returns (start_iso, end_iso) or None."""
    parsed = extract_temporal_query(query)
    return parsed.window if parsed else None


class TemporalRetriever(Retriever):
    """Vector search filtered by date range extracted from query.

    If no date range can be extracted, returns empty. When a range is found,
    runs semantic search with a timestamp payload filter.
    """

    name = "temporal"
    #: Can enforce a tenant filter (its store search/scroll take tenant=).
    accepts_tenant = True
    DATE_FOCUSED_SCROLL_BUFFER_MIN = 25
    DATE_FOCUSED_SCROLL_BUFFER_MULTIPLIER = 5

    #: Bound-emission formats for the Qdrant window filter. "iso" (default)
    #: emits RFC3339 strings (DatetimeRange semantics); "epoch"/"epoch_ms"
    #: emit numbers so the filter compares against a NUMERIC payload field —
    #: a DatetimeRange never matches an epoch-int field, so a collection that
    #: stores numeric timestamps would silently get zero temporal hits.
    TIMESTAMP_FORMATS = ("iso", "epoch", "epoch_ms")

    def __init__(
        self,
        embedding: EmbeddingProvider,
        vector_store: VectorStore,
        extractor=extract_temporal_query,
        *,
        text_key: str = "text",
        timestamp_key: str = "timestamp",
        timestamp_format: str = "iso",
    ):
        if timestamp_format not in self.TIMESTAMP_FORMATS:
            raise ValueError(
                f"timestamp_format must be one of {self.TIMESTAMP_FORMATS}, "
                f"got {timestamp_format!r}"
            )
        self.embedding = embedding
        self.vector_store = vector_store
        self._space_guard = SpaceGuard(vector_store, embedding)
        self.extractor = extractor
        self.text_key = text_key
        self.timestamp_key = timestamp_key
        self.timestamp_format = timestamp_format
        #: How NUMERIC payload/caller timestamps are read: the configured
        #: format beats the magnitude heuristic (early ms epochs are only
        #: unambiguous this way).
        self._numeric_unit = numeric_unit_for(timestamp_format)

    def _bound(self, dt: datetime) -> str | float:
        """A window bound in the collection's own timestamp domain — EXACT
        (floats for numeric domains): rounding in either direction is lossy
        (truncation widens the scope; inward integral rounding excludes a
        fractional ``time.time()`` payload at the window edge)."""
        return _emit_epoch_bound(dt, self.timestamp_format)

    def explain_empty(self, query: str) -> str | None:
        """Why an empty result is a degradation, not an absence of data.

        Recognized by the recaller's trace collection (duck-typed): when the
        retriever returned nothing without raising, this names the reason.
        """
        return "temporal:no_parse" if not self.extractor(query) else None

    def search(self, query, limit=10, filters=None, tenant=None):
        # Pass tenant only when set (a custom store may lack the parameter).
        tkw = {"tenant": tenant} if tenant is not None else {}
        parsed_or_window = self.extractor(query)
        if not parsed_or_window:
            # Visible even without tracing — temporal boost silently lost.
            counter("mnemostack.recall.temporal_no_parse", 1)
            return []
        if isinstance(parsed_or_window, TemporalQuery):
            start, end = parsed_or_window.window
            target_date = parsed_or_window.target_date
            date_focused = parsed_or_window.date_focused
        else:
            start, end = parsed_or_window
            target_date = None
            date_focused = False

        # Flat filter shape understood by VectorStore._build_filter. Preserve
        # caller filters (workspace/source/tenant scope) and INTERSECT any
        # caller timestamp constraint with the parsed query window — replacing
        # it would return hits outside the caller's advertised scope. The
        # intersection runs on the TIME LINE (datetimes), so a caller bound in
        # a different domain than the window (an epoch number vs the
        # extractor's ISO strings) still intersects instead of erroring out;
        # the final filter bounds are emitted in the collection's own domain
        # (see _bound), so the Qdrant condition matches the field's real type.
        temporal_filter = dict(filters or {})
        start_dt = parse_payload_instant(start)
        end_dt = parse_payload_instant(end)
        if start_dt is None or end_dt is None:  # defensive: the extractor emits ISO
            return []
        window = _intersect_instant_window(
            start_dt,
            end_dt,
            temporal_filter.get(self.timestamp_key),
            numeric_unit=self._numeric_unit,
        )
        if window is None:
            # The query's date range lies entirely outside the caller's
            # timestamp scope (or that scope is unparseable — honoring an
            # unknown constraint loosely could leak out-of-scope hits).
            return []
        start_dt, end_dt = window
        temporal_filter[self.timestamp_key] = {
            "gte": self._bound(start_dt),
            "lte": self._bound(end_dt),
        }

        strict_bounds = None
        if date_focused and target_date is not None:
            # The target day must itself fall inside the (already intersected)
            # window; otherwise skip the date-focused pass and keep only the
            # in-scope neighborhood search below.
            day_start = datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc)
            day_end = datetime.combine(target_date, datetime.max.time(), tzinfo=timezone.utc)
            strict_bounds = _intersect_instant_window(
                day_start, day_end, {"gte": start_dt, "lte": end_dt}
            )

        if strict_bounds is not None and hasattr(self.vector_store, "scroll"):
            try:
                strict_filter = dict(temporal_filter)
                strict_filter[self.timestamp_key] = {
                    "gte": self._bound(strict_bounds[0]),
                    "lte": self._bound(strict_bounds[1]),
                }

                buffer_limit = max(
                    limit * self.DATE_FOCUSED_SCROLL_BUFFER_MULTIPLIER,
                    self.DATE_FOCUSED_SCROLL_BUFFER_MIN,
                )
                exact_hits = self._collect_sorted_date_focused_hits(
                    self.vector_store.scroll(
                        batch_size=max(limit, 100),
                        filters=strict_filter,
                        with_vectors=False,
                        **tkw,
                    ),
                    target_date,
                    max_hits=buffer_limit,
                )

                hits = exact_hits[:limit]
                if len(hits) < limit:
                    seen = {hit.id for hit in hits}
                    neighbor_hits = self._collect_sorted_date_focused_hits(
                        self.vector_store.scroll(
                            batch_size=max(limit, 100),
                            filters=temporal_filter,
                            with_vectors=False,
                            **tkw,
                        ),
                        target_date,
                        max_hits=buffer_limit,
                        skip_ids=seen,
                        exclude_target_date=True,
                    )
                    hits.extend(neighbor_hits[: limit - len(hits)])

                return self._to_results(hits)
            except Exception as exc:  # noqa: BLE001 — defensive; log then fall back
                logger.warning(
                    "TemporalRetriever: vector_store.scroll failed (window=%s..%s): %s",
                    start_dt,
                    end_dt,
                    exc,
                )

        self._space_guard.ensure()
        vec = embed_query_via(self.embedding, query)
        if not vec:
            return []
        try:
            hits = self.vector_store.search(vec, limit=limit, filters=temporal_filter, **tkw)
        except Exception as exc:  # noqa: BLE001 — defensive; log instead of silent
            logger.warning(
                "TemporalRetriever: vector_store.search failed (window=%s..%s): %s",
                start_dt,
                end_dt,
                exc,
            )
            return []
        return self._to_results(hits)

    def _hit_date(self, hit) -> date | None:
        timestamp = (hit.payload or {}).get(self.timestamp_key)
        # Any domain the collection stores; numerics read in the CONFIGURED
        # unit (an early ms epoch is ambiguous under the heuristic).
        dt = parse_payload_instant(timestamp, numeric_unit=self._numeric_unit)
        if dt is not None:
            return dt.astimezone(timezone.utc).date()
        if isinstance(timestamp, str) and len(timestamp) >= 10:
            # Historical fallback: a date-prefixed non-ISO string still yields
            # its calendar date.
            try:
                return date.fromisoformat(timestamp[:10])
            except ValueError:
                return None
        return None

    @staticmethod
    def _is_diary_source_for_date(hit, target_date: date) -> bool:
        payload = hit.payload or {}
        expected = f"{target_date.isoformat()}.md"
        for key in ("source_file", "source", "file", "path"):
            value = payload.get(key)
            if isinstance(value, str) and value.replace("\\", "/").endswith(expected):
                return True
        return False

    def _collect_sorted_date_focused_hits(
        self,
        hits,
        target_date: date,
        *,
        max_hits: int,
        skip_ids: set[Any] | None = None,
        exclude_target_date: bool = False,
    ):
        collected = []
        skip_ids = skip_ids or set()
        for hit in hits:
            if hit.id in skip_ids:
                continue
            if exclude_target_date and self._hit_date(hit) == target_date:
                continue
            collected.append(hit)
            if len(collected) >= max_hits:
                break
        return self._sort_date_focused_hits(collected, target_date)

    def _sort_date_focused_hits(self, hits, target_date: date):
        noon = datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc) + timedelta(
            hours=12
        )

        def timestamp_distance(hit) -> float:
            # Any stored domain; unparseable sorts last, as before.
            dt = parse_payload_instant(
                (hit.payload or {}).get(self.timestamp_key), numeric_unit=self._numeric_unit
            )
            if dt is None:
                return float("inf")
            return abs((dt.astimezone(timezone.utc) - noon).total_seconds())

        return sorted(
            hits,
            key=lambda hit: (
                self._hit_date(hit) != target_date,
                not self._is_diary_source_for_date(hit, target_date),
                timestamp_distance(hit),
            ),
        )

    def _to_results(self, hits) -> list[RecallResult]:
        return [
            RecallResult(
                id=h.id,
                text=(h.payload or {}).get(self.text_key, ""),
                score=h.score,
                payload={**(h.payload or {}), "temporal_match": True},
                sources=["temporal"],
            )
            for h in hits
        ]
