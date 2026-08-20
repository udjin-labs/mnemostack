"""FastAPI service wrapper for mnemostack.

Exposes `/recall`, `/answer`, `/memories`, `/triples`, `/feedback`, `/health`,
and `/metrics` over HTTP so callers in any language (Node, Go, Rust, curl) can
use mnemostack without a Python SDK — reads AND writes: `/memories` embeds
server-side and stores under the caller's tenant.

Start from the CLI:

    mnemostack serve --provider gemini --collection memory

Or programmatically:

    from mnemostack.server import build_app
    app = build_app(provider_name="gemini", collection="memory")
    # pass `app` to uvicorn / gunicorn / etc.

The server is opt-in: install with `pip install 'mnemostack[server]'`.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from pydantic import BaseModel, Field, StrictInt
except ImportError as e:  # pragma: no cover - import guard
    raise ImportError(
        "FastAPI is not installed. Install the optional server extra: "
        "`pip install 'mnemostack[server]'`."
    ) from e

from mnemostack import __version__
from mnemostack.config import (
    Config,
    ensure_text_fields_mode,
    model_kwargs,
    provider_kwargs,
    resolve_text_search_mode,
)
from mnemostack.embeddings import get_provider
from mnemostack.embeddings.roles import EmbeddingSpaceError
from mnemostack.feedback import apply_feedback, record_recall_events
from mnemostack.ingest import (
    REMOTE_CHUNK_SIZE,
    REMOTE_LIST_PAGE,
    REMOTE_MAX_DOC_CHARS,
    REMOTE_MAX_IDS,
    REMOTE_MAX_INDEX_ROOT_CHARS,
    REMOTE_MAX_ITEMS,
    REMOTE_MAX_OFFSET,
    REMOTE_MAX_SOURCE_CHARS,
    REMOTE_MAX_TAGS,
    REMOTE_MAX_TEXT_CHARS,
    REMOTE_MAX_TIMESTAMP_CHARS,
    REMOTE_SOURCE_BATCH,
    IngestItem,
    RemoteRequestTooLarge,
    coerce_point_ids,
    ensure_remote_schema_keys,
    expand_remote_items,
    find_source_points,
    ingest_remote_items,
    validate_invalidate_options,
    validate_remote_cursor,
    validate_remote_invalidate,
    validate_remote_item,
    validate_remote_source,
    validate_remote_triple,
)
from mnemostack.llm import get_llm
from mnemostack.observability.recorder import (
    InMemoryRecorder,
    counter,
    get_recorder,
    set_recorder,
)
from mnemostack.provenance import SOURCE_HASH_KEY
from mnemostack.quotas import QuotaExceededError
from mnemostack.recall import (
    DEGRADED_COUNTER,
    RERANK_MODES,
    AnswerGenerator,
    BM25Retriever,
    MemgraphRetriever,
    QdrantSparseRetriever,
    Recaller,
    RecallTrace,
    Reranker,
    Retriever,
    TemporalRetriever,
    VectorRetriever,
    build_full_pipeline,
    build_qdrant_text_arms,
    chunk_filter_probe_via,
    recall_flow,
    sum_tokens,
)
from mnemostack.recall.pipeline import FileStateStore, default_state_path
from mnemostack.vector import VectorStore

log = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ----- Request / response models -----


class RecallRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query.")
    limit: int = Field(10, ge=1, le=100, description="Top-K memories to return.")
    full_pipeline: bool = Field(
        True,
        description="Apply the 8-stage recall pipeline. Set to False for raw RRF output.",
    )
    include_trace: bool = Field(
        False,
        description="Return the per-retriever recall trace (debug; verbose).",
    )
    filters: dict[str, Any] | None = Field(
        None,
        description=(
            "Payload filters applied inside every retriever (exact match, or "
            '{"gte"/"lte"} ranges), e.g. {"tenant": "a"} or '
            '{"timestamp": {"gte": "2026-01-01"}}. Results never include '
            "points outside the filtered scope."
        ),
    )
    token_budget: int | None = Field(
        None,
        ge=1,
        description=(
            "Hard cap on the total (estimated) text tokens of the returned "
            "results: the final ranking is cut to the prefix that fits. "
            "Unset falls back to the server-wide recall.token_budget config."
        ),
    )
    include_invalidated: bool = Field(
        False,
        description="Include facts marked stale (default: invalidated facts are hidden).",
    )
    as_of: str | None = Field(
        None,
        description=(
            "Point-in-time recall (ISO-8601): return facts valid at this "
            "world-time instant, ignoring later invalidation."
        ),
    )


class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(10, ge=1, le=100)
    full_pipeline: bool = Field(True)
    include_trace: bool = Field(False)
    filters: dict[str, Any] | None = Field(None)
    token_budget: int | None = Field(
        None,
        ge=1,
        description=(
            "Hard cap on the total (estimated) text tokens of the memories "
            "fed to the answer LLM (same contract as /recall)."
        ),
    )
    include_invalidated: bool = Field(
        False, description="Include facts marked stale (same contract as /recall)."
    )
    as_of: str | None = Field(
        None, description="Point-in-time recall (ISO-8601); same contract as /recall."
    )


class FeedbackRequest(BaseModel):
    hit_id: str = Field(..., min_length=1, description="Memory/result id the feedback refers to.")
    signal: Literal["useful", "irrelevant", "clicked"] = Field(
        ...,
        description="User feedback signal. Useful/clicked are positive, irrelevant is negative.",
    )
    query: str | None = Field(None, description="Original query; used to infer query_type.")
    query_type: str | None = Field(None, description="Explicit query type override.")
    source: str | None = Field(None, description="Single retriever/source label.")
    sources: list[str] = Field(
        default_factory=list,
        description="Retriever/source labels from the recalled hit.",
    )
    reward: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Optional reward override in [0, 1].",
    )


class MemoryItemIn(BaseModel):
    """One memory a remote client asks the service to store."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=REMOTE_MAX_DOC_CHARS,
        description=(
            "The memory content. Embedded server-side. Plain items are "
            f"capped at {REMOTE_MAX_TEXT_CHARS} characters; longer documents "
            "must set `chunk: true`."
        ),
    )
    chunk: bool = Field(
        False,
        description=(
            "Split this text server-side into fixed character windows "
            f"({REMOTE_CHUNK_SIZE} chars) — the split `mnemostack index` "
            "applies to prose files at its default --chunk-size. In "
            "unscoped (single-tenant) deployments with default settings "
            "the chunk ids are identical across both paths; tenant-scoped "
            "writes use tenant-prefixed ids, and a custom --chunk-size or "
            "the markdown indexer produce different boundaries. Requires "
            "a non-empty `source`."
        ),
    )
    source: str = Field(
        "",
        max_length=REMOTE_MAX_SOURCE_CHARS,
        description=(
            "Logical origin (e.g. 'chat/2026-08-19'). With `offset` it forms "
            "the deterministic id, so re-sending the same (source, offset, "
            "text) is a no-cost duplicate, not a second copy."
        ),
    )
    offset: int = Field(
        0,
        ge=0,
        le=REMOTE_MAX_OFFSET,
        description="Position within `source` for multi-chunk documents.",
    )
    timestamp: str | None = Field(
        None,
        max_length=REMOTE_MAX_TIMESTAMP_CHARS,
        description="Event time of the content (ISO-8601); drives temporal recall.",
    )
    tags: list[str] = Field(default_factory=list, max_length=REMOTE_MAX_TAGS)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Free payload fields, filterable at recall. Server-reserved keys "
            "(underscore-prefixed and structural ones like tenant_id/source, "
            "plus tags/timestamp — use their dedicated fields) are rejected "
            "with 400."
        ),
    )


class MemoriesRequest(BaseModel):
    items: list[MemoryItemIn] = Field(..., min_length=1, max_length=REMOTE_MAX_ITEMS)


class MemoryResultOut(BaseModel):
    id: str = Field(description="Deterministic chunk id (also the /resolve handle).")
    status: Literal["stored", "duplicate", "failed"]
    item: int = Field(
        description=(
            "Index of the request item this result came from — a `chunk: "
            "true` document yields one result per chunk."
        )
    )
    offset: int = Field(0, description="Character offset within the item's source.")


class MemoriesResponse(BaseModel):
    results: list[MemoryResultOut] = Field(
        description="Per-chunk outcome, in request order."
    )
    stored: int
    duplicates: int
    failed: int


class TripleIn(BaseModel):
    subject: str = Field(..., min_length=1, max_length=512)
    predicate: str = Field(..., min_length=1, max_length=512)
    object: str = Field(..., min_length=1, max_length=512)
    valid_from: str | None = Field(None, max_length=64)
    valid_until: str | None = Field(None, max_length=64)


class TriplesRequest(BaseModel):
    triples: list[TripleIn] = Field(..., min_length=1, max_length=REMOTE_MAX_ITEMS)


class TripleResultOut(BaseModel):
    status: Literal["added", "failed"]
    error: str | None = None


class TriplesResponse(BaseModel):
    results: list[TripleResultOut]
    added: int
    failed: int


class InvalidateRequest(BaseModel):
    # StrictInt, not int: the lax union coerces JSON true/false to 1/0,
    # which would silently target numeric point ids 1/0 instead of a 422.
    ids: list[str | StrictInt] | None = Field(
        None, min_length=1, max_length=REMOTE_MAX_IDS
    )
    source: str | None = Field(
        None,
        max_length=REMOTE_MAX_SOURCE_CHARS,
        description=(
            "Retract every memory from this source instead of listing ids "
            "— the lifecycle operation a client actually has (a file was "
            "rewritten or a session reset). Exactly one of ids/source."
        ),
    )
    invalidated_at: str | None = Field(
        None,
        max_length=64,
        description="System-time stamp (ISO-8601); default: now (UTC).",
    )
    valid_until: str | None = Field(
        None,
        max_length=64,
        description=(
            "World-time the fact stopped being true (ISO-8601); optional, "
            "separate from the system-time invalidation stamp."
        ),
    )
    index_root: str | None = Field(
        None,
        max_length=REMOTE_MAX_INDEX_ROOT_CHARS,
        description=(
            "Owner guard: when set, points owned by a different index_root "
            "are skipped, so one root cannot invalidate another's chunks in "
            "a shared collection."
        ),
    )


class InvalidateResponse(BaseModel):
    requested: int = Field(
        description=(
            "Ids submitted, or points matched in this batch for a "
            "source-scoped call."
        )
    )
    complete: bool = Field(
        True,
        description=(
            "False when a source-scoped call hit its per-request batch cap "
            "— repeat the identical call until it reports true."
        ),
    )
    invalidated: int = Field(
        description=(
            "Points actually updated. Ids that do not exist, belong to "
            "another tenant, or fail the index_root owner guard are skipped "
            "— indistinguishably, so the count is not an oracle for foreign "
            "ids."
        )
    )


class DeleteMemoriesRequest(BaseModel):
    # StrictInt: see InvalidateRequest.
    ids: list[str | StrictInt] | None = Field(
        None, min_length=1, max_length=REMOTE_MAX_IDS
    )
    source: str | None = Field(
        None,
        max_length=REMOTE_MAX_SOURCE_CHARS,
        description=(
            "Erase every memory from this source instead of listing ids. "
            "Exactly one of ids/source."
        ),
    )
    index_root: str | None = Field(
        None,
        max_length=REMOTE_MAX_INDEX_ROOT_CHARS,
        description=(
            "Owner guard, matching /invalidate: when set, points owned by a "
            "different index_root are skipped. Points carrying no index_root "
            "tag are NOT protected by the guard."
        ),
    )


class DeleteMemoriesResponse(BaseModel):
    requested: int = Field(
        description=(
            "Ids submitted, or points matched in this batch for a "
            "source-scoped call."
        )
    )
    complete: bool = Field(
        True,
        description=(
            "False when a source-scoped call hit its per-request batch cap "
            "— repeat the identical call until it reports true."
        ),
    )
    deleted: int = Field(
        description=(
            "Points actually deleted. Ids that do not exist or belong to "
            "another tenant are skipped — indistinguishably, so the count "
            "is not an oracle for foreign ids."
        )
    )


class MemoryListItem(BaseModel):
    id: str
    content_hash: str | None = Field(
        None,
        description=(
            "Snapshot hash of the SOURCE document at ingest, when the "
            "point carries one (points ingested before provenance shipped "
            "return null). Lets a client detect divergence without the "
            "service returning any memory text."
        ),
    )
    timestamp: str | int | float | None = Field(
        None,
        description=(
            "Event time of the memory, when it carries one, in the "
            "deployment's own timestamp domain: an ISO-8601 string under "
            "the default `recall.timestamp_format: iso`, a number under "
            "`epoch` / `epoch_ms`. Returned as stored so a client can "
            "compare it against its own record without a lossy conversion."
        ),
    )
    indexed_at: str | None = Field(None, description="When the point was written.")


def _list_text(value: Any) -> str | None:
    """A listing metadata field, or None when the payload holds something
    unexpected there. These fields are integrity hints; a single point with
    a foreign value under a reserved key must not fail the WHOLE page with
    a response-validation 500."""
    return value if isinstance(value, str) else None


def _list_timestamp(value: Any) -> str | int | float | None:
    """The point's event time in the deployment's own domain — ISO string
    under `timestamp_format: iso`, a number under `epoch`/`epoch_ms`.
    Returned as stored (a bool is not a timestamp; int subclass or not)."""
    if isinstance(value, bool):
        return None
    return value if isinstance(value, (str, int, float)) else None


class MemoryListResponse(BaseModel):
    items: list[MemoryListItem]
    complete: bool = Field(
        description=(
            "False when more points match than fit this page — repeat with "
            "`after` set to the last id returned."
        )
    )


class Memory(BaseModel):
    id: str
    text: str
    score: float
    source: str | None = None
    retrievers: list[str] = Field(
        default_factory=list,
        description="Which retrievers produced this hit (vector/bm25/memgraph/temporal).",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecallResponse(BaseModel):
    query: str
    results: list[Memory]
    degraded: list[str] = Field(
        default_factory=list,
        description=(
            "Components that actually fell back while serving this call "
            "(e.g. retriever:bm25:failed, reranker:fallback) — plus, "
            "DEPRECATED until the next major, a back-compat duplicate of the "
            "routine tags from `notes`. An entry absent from `notes` is a "
            "real fault."
        ),
    )
    notes: list[str] = Field(
        default_factory=list,
        description=(
            "AUTHORITATIVE list of routine signals: stages that did not "
            "apply to this query (e.g. temporal:no_parse on any query "
            "without a parseable date). Not a fault; these tags also appear "
            "in `degraded` for back-compat until the next major."
        ),
    )
    trace: dict[str, Any] | None = Field(
        None,
        description=(
            "Recall trace (per-retriever ranked lists, fused and post-rerank "
            "order). Present only when include_trace=true. The results list is "
            "the final order; it may differ from post_rerank when vector-floor "
            "re-appends items."
        ),
    )
    tokens_estimate: int = Field(
        0,
        description=(
            "Estimated total text tokens of the returned results (heuristic, "
            "not a tokenizer). This is what token_budget is enforced against."
        ),
    )


class AnswerResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    sources: list[str]
    memories: list[Memory]
    degraded: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    trace: dict[str, Any] | None = None
    tokens_estimate: int = Field(
        0, description="Estimated total text tokens of the memories used as context."
    )
    tokens_used: int | None = Field(
        None,
        description=(
            "Token usage reported by the LLM provider for the answer "
            "generation call (provider-specific semantics; null when the "
            "provider reports nothing)."
        ),
    )


class ResolveResponse(BaseModel):
    """One citation verified against its current source (see mnemostack.provenance)."""

    chunk_id: str
    verdict: str
    supported: bool
    source: str
    resolved_path: str | None
    snapshot: str
    stored_offset: int | None
    found_offset: int | None
    fragment: str | None
    detail: str
    captured_at: str | None


class FeedbackResponse(BaseModel):
    ok: bool
    hit_id: str
    signal: str
    reward: float
    query_type: str
    ior_recorded: bool
    q_learning_updates: int


class HealthResponse(BaseModel):
    status: str
    version: str
    provider: str
    collection: str
    qdrant: bool
    memgraph: bool


class LiveResponse(BaseModel):
    status: str  # always "ok" — the process is up
    version: str


class ReadyResponse(BaseModel):
    status: str  # "ready" | "not_ready"
    version: str
    qdrant: bool  # readiness gate: recall needs the vector store
    # readiness gate: recall can't embed a query without the provider. Checked
    # via a cached, bounded reachability probe so /readyz neither hangs on a slow
    # provider nor pays a live embedding call on every probe.
    embedding: bool
    # NOTE: the graph is deliberately absent. It's optional/fail-soft, and a live
    # graph ping on the readiness hot path would let a slow/blackholed Memgraph
    # add up to graph_health_timeout of latency to /readyz and trip a probe
    # timeout — gating readiness on the graph through the back door. Graph
    # reachability is on /health and /status instead.


class StatusResponse(BaseModel):
    version: str
    provider: str
    llm: str
    collection: str
    qdrant_url: str
    qdrant: bool
    embedding: bool
    memgraph: bool
    # total recall invocations recorded, including the extra sub-recalls the
    # answer generator issues on expansion/inference retries — so this counts
    # recall calls, not user-facing recall requests served.
    recall_calls: float
    # total degradation events across the recall/answer serving path (fallbacks,
    # rewrite failures, answer errors) — see _DEGRADED_METRICS.
    degraded_events: float


# Counters that represent a real degradation of the recall/answer serving path.
# An explicit allowlist rather than a substring match: substring matching both
# under-counts (e.g. `answer.errors` contains none of "fail"/"degrad"/"fallback")
# and, if the recorder is ever shared across processes, leaks unrelated failures
# (e.g. `ingest.failed`) into an operator's serving-degradation signal.
# `temporal.no_parse` is deliberately excluded — a query with no parseable date
# is routine, not a degradation.
#
# `mnemostack.recall.degraded` is the counter the recall trace mirrors its
# per-call degradation tags into (reranker unavailable/fallback, retriever
# failures) — those live only on the trace and have no other counter, so
# without it a reranker that's unavailable at startup would leave this at 0
# while every /recall reports degraded service.
_DEGRADED_METRICS = frozenset(
    {
        DEGRADED_COUNTER,
        "mnemostack.recall.fallback_triggered",
        "mnemostack.recall.followup_rewrite_failed",
        "mnemostack.query_expansion.errors",
        "mnemostack.answer.errors",
        "mnemostack.answer.unavailable",
        "mnemostack.answer.list_finalize_fallback",
    }
)


# Embedding readiness is cached so /readyz doesn't pay a live embedding call on
# every probe (embeddings are billed/rate-limited for hosted providers) and so
# a slow/hung provider can't add latency to the probe.
_EMBED_HEALTH_TTL_S = 30.0  # refresh the cached result in the background this often
#: Ceiling for a rate-limit Retry-After header (seconds). Caps an absurdly large
#: (or non-finite) computed backoff so a tiny rate can't crash header building.
_MAX_RETRY_AFTER = 86400  # 1 day


def _make_probe_client(url: str, timeout: int) -> Any:
    """A short-timeout Qdrant client dedicated to liveness/readiness pings.

    Separate from the recall store's client (whose timeout is sized for recall,
    not probes): a slow or blackholed Qdrant makes /readyz return 503 within
    ``timeout`` instead of tying up a FastAPI worker thread for the store's full
    recall timeout. Construction is lazy (no connection until first call), so
    building the app never blocks on Qdrant. Isolated as a function so tests can
    substitute a fake without a live Qdrant.
    """
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, timeout=timeout)


# ----- Server construction -----


@dataclass
class ServerConfig:
    provider_name: str = "gemini"
    embedding_model: str | None = None
    llm_name: str = "gemini"
    llm_model: str | None = None
    collection: str = "mnemostack"
    qdrant_url: str = "http://localhost:6333"
    graph_uri: str | None = "bolt://localhost:7687"
    graph_health_timeout: float = 1.0
    graph_timeout: float = 5.0
    bm25_paths: list[str] | None = None  # optional markdown dirs for BM25 corpus
    vector_floor: int = 0
    rerank_mode: str = "relevant_only"
    token_budget: int | None = None  # default recall token budget; requests may override
    state_path: str = field(default_factory=default_state_path)
    auto_record_ior: bool = False
    # graph auth appended at the tail to preserve positional back-compat.
    graph_user: str = ""
    graph_password: str = ""
    graph_database: str | None = None
    # Short timeout (whole seconds — qdrant-client's client timeout is integer)
    # for the dedicated liveness/readiness Qdrant ping, so a slow or blackholed
    # Qdrant makes /readyz return 503 promptly instead of tying up a worker
    # thread for the recall client's full timeout. Separate from recall's own.
    qdrant_health_timeout: int = 2
    # Multi-tenant auth. When enabled, every data endpoint (/recall, /answer,
    # /feedback) requires a valid service key (default-deny); the key resolves
    # the tenant and scopes. Off by default — single-tenant deployments serve
    # unauthenticated, tenant-less recall exactly as before.
    auth_enabled: bool = False
    keys_file: str | None = None  # None = FileKeyStore default path
    # Per-tenant request rate limiting (only meaningful under auth, which resolves
    # the tenant). A tenant with a `max_rps` quota is throttled; others are not.
    quotas_file: str | None = None  # None = FileQuotaStore default path
    # Payload schema of the collection recall reads (a pre-existing collection
    # keeps its own field names; timestamps may be numeric epochs).
    text_key: str = "text"
    timestamp_key: str = "timestamp"
    timestamp_format: str = "iso"
    text_search: str = "auto"
    # Multi-field lexical arms: payload field -> fusion weight (lexical only).
    text_search_fields: dict[str, float] = field(default_factory=dict)
    # Citation resolution allowlist (GET /resolve): the corpus directories
    # this process may read. The stored index_root is payload data and cannot
    # be its own security boundary — EMPTY (default) disables resolution on
    # the HTTP surface entirely (fail closed). Env: MNEMOSTACK_RESOLVE_ROOTS
    # (os.pathsep-separated).
    resolve_roots: list[str] = field(default_factory=list)
    # Provider knobs resolved by the shared config/env precedence — passed to
    # get_provider() so a configured Ollama host/timeout actually reaches the
    # provider on this surface too (they used to silently stop at the config).
    # Appended at the TAIL on purpose: ServerConfig is documented stable and
    # may be constructed positionally — inserting mid-signature would shift
    # every later argument.
    ollama_host: str | None = None
    embedding_timeout: int | None = None

    def __post_init__(self) -> None:
        if self.rerank_mode not in RERANK_MODES:
            allowed = ", ".join(sorted(RERANK_MODES))
            raise ValueError(f"rerank_mode must be one of: {allowed}")
        # 0 or negative means "no budget" — apply_token_budget requires >= 1
        # and a bad value here would 500 every request.
        if self.token_budget is not None and self.token_budget <= 0:
            self.token_budget = None

    @classmethod
    def from_env(cls) -> ServerConfig:
        cfg = Config.load()
        return cls(
            provider_name=cfg.embedding.provider,
            embedding_model=cfg.embedding.model,
            ollama_host=cfg.embedding.ollama_host,
            embedding_timeout=cfg.embedding.timeout,
            llm_name=cfg.llm.provider,
            llm_model=cfg.llm.model,
            collection=cfg.vector.collection,
            qdrant_url=cfg.vector.host,
            qdrant_health_timeout=cfg.vector.health_timeout,
            graph_uri=cfg.graph.uri or "bolt://localhost:7687",
            graph_user=cfg.graph.user,
            graph_password=cfg.graph.password,
            graph_database=cfg.graph.database,
            graph_health_timeout=cfg.graph.health_timeout,
            graph_timeout=cfg.graph.timeout,
            bm25_paths=list(cfg.recall.bm25_paths) or None,
            vector_floor=max(0, int(cfg.recall.vector_floor)),
            rerank_mode=cfg.recall.rerank_mode,
            token_budget=cfg.recall.token_budget,
            auto_record_ior=_env_bool("MNEMOSTACK_AUTO_RECORD_IOR"),
            auth_enabled=_env_bool("MNEMOSTACK_AUTH_ENABLED"),
            keys_file=os.environ.get("MNEMOSTACK_KEYS_FILE") or None,
            quotas_file=os.environ.get("MNEMOSTACK_QUOTAS_FILE") or None,
            text_key=cfg.recall.text_key,
            timestamp_key=cfg.recall.timestamp_key,
            timestamp_format=cfg.recall.timestamp_format,
            text_search=cfg.recall.text_search,
            text_search_fields=dict(cfg.recall.text_search_fields),
            resolve_roots=_resolve_roots_env(),
        )


def _resolve_roots_env() -> list[str]:
    """MNEMOSTACK_RESOLVE_ROOTS as a list (os.pathsep-separated)."""
    return [p for p in os.environ.get("MNEMOSTACK_RESOLVE_ROOTS", "").split(os.pathsep) if p]


def _build_bm25_docs(paths: list[str] | None):
    if not paths:
        return []
    from pathlib import Path

    from mnemostack.recall import BM25Doc

    docs = []
    for root in paths:
        p = Path(root)
        if not p.exists():
            log.warning("bm25 path %s does not exist — skipping", root)
            continue
        targets = [p] if p.is_file() else sorted(p.rglob("*.md")) + sorted(p.rglob("*.txt"))
        for f in targets:
            text = f.read_text(encoding="utf-8", errors="ignore")
            for i in range(0, len(text), 800):
                chunk = text[i : i + 800]
                if chunk.strip():
                    docs.append(
                        BM25Doc(
                            id=f"{f}:{i}",
                            text=chunk,
                            payload={"source": str(f), "offset": i},
                        )
                    )
    return docs


def _memory_of(result) -> Memory:
    """Convert a mnemostack.recall.RecallResult into the HTTP response shape.

    Looks in .payload (the real field on RecallResult) first, with a fallback
    to .metadata for callers that pass a bare dict-like object (used in tests).
    """
    payload = getattr(result, "payload", None)
    if not payload:
        payload = getattr(result, "metadata", None) or {}
    # Internal recall mechanics stay internal: the vector-floor working set
    # and the graph filter-attribution proof marker are not user metadata.
    payload = {
        key: value
        for key, value in payload.items()
        if key not in ("_vector_floor_candidates", "_attributed_filters", "_enrich_keys")
    }
    # Common source fields populated by our indexers. Order matters: explicit
    # 'source' wins, then the workspace conventions, finally nothing.
    source = (
        payload.get("source")
        or payload.get("source_file")
        or payload.get("file")
        or getattr(result, "source", None)
    )
    retrievers = list(getattr(result, "sources", []) or [])
    return Memory(
        id=str(result.id),
        text=result.text,
        score=float(getattr(result, "score", 0.0)),
        source=str(source) if source else None,
        retrievers=retrievers,
        metadata=payload,
    )


def _prometheus_dump(rec: InMemoryRecorder) -> str:
    """Render an AggregatingRecorder as Prometheus text exposition format.

    We emit one counter per metric name and, for histograms, a count + sum
    + a small set of summary quantiles (p50/p90/p99/max). No external
    `prometheus_client` dependency — the exposition format is stable and
    trivial to produce by hand.
    """

    def _fmt_labels(labels: dict[str, str] | None) -> str:
        if not labels:
            return ""
        # Label VALUES can carry operator-configured text (e.g. a multi-field
        # arm name inside a degraded reason). The Prometheus text format
        # requires backslash, double-quote and newline escaped — unescaped, a
        # newline would inject extra exposition lines and malform the scrape.
        def esc(v: Any) -> str:
            return (
                str(v)
                .replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
            )

        parts = [f'{k}="{esc(v)}"' for k, v in labels.items()]
        return "{" + ",".join(parts) + "}"

    def _from_key(key: tuple):
        name = key[0]
        labels = dict(key[1:]) if len(key) > 1 else None
        return name, labels

    def _safe_name(name: str) -> str:
        # Prometheus metric names allow [a-zA-Z_:][a-zA-Z0-9_:]*
        return name.replace(".", "_").replace("-", "_")

    lines: list[str] = []
    seen_help: set[str] = set()

    # Iterate snapshots — the live dicts are mutated by recall running in the
    # sync threadpool while this scrape runs.
    counters = rec.snapshot_counters()
    histograms = rec.snapshot_histograms()

    for key, val in sorted(counters.items()):
        name, labels = _from_key(key)
        prom = _safe_name(name) + "_total"
        if prom not in seen_help:
            lines.append(f"# HELP {prom} mnemostack counter: {name}")
            lines.append(f"# TYPE {prom} counter")
            seen_help.add(prom)
        lines.append(f"{prom}{_fmt_labels(labels)} {val}")

    for key, obs in sorted(histograms.items()):
        name, labels = _from_key(key)
        prom = _safe_name(name)
        if not obs:
            continue
        obs_sorted = sorted(obs)
        n = len(obs_sorted)

        def pct(p: float, _n: int = n, _obs_sorted: list[float] = obs_sorted) -> float:
            idx = min(_n - 1, max(0, int(round(p * (_n - 1)))))
            return _obs_sorted[idx]

        if prom not in seen_help:
            lines.append(f"# HELP {prom} mnemostack histogram: {name} (ms)")
            lines.append(f"# TYPE {prom} summary")
            seen_help.add(prom)
        base = _fmt_labels(labels)
        for quant, label in ((0.5, "0.5"), (0.9, "0.9"), (0.99, "0.99")):
            combined = _fmt_labels({**(labels or {}), "quantile": label})
            lines.append(f"{prom}{combined} {pct(quant)}")
        lines.append(f"{prom}_sum{base} {sum(obs_sorted)}")
        lines.append(f"{prom}_count{base} {n}")

    return "\n".join(lines) + "\n"


def build_app(config: ServerConfig | None = None) -> FastAPI:
    cfg = config or ServerConfig.from_env()

    # Install a process-wide in-memory recorder so /metrics has something
    # to show. Safe to replace an existing one — the old counters are simply
    # discarded. For multi-worker deployments, pair this with uvicorn --workers
    # 1 or a shared aggregator (Redis/Statsd); Prometheus scrape-then-diff
    # handles single-worker fine.
    set_recorder(InMemoryRecorder())

    # Fail a schema-key misconfiguration at BOOT, before any provider cost.
    ensure_remote_schema_keys(cfg.text_key, cfg.timestamp_key)
    provider = get_provider(
        cfg.provider_name,
        **provider_kwargs(
            cfg.provider_name,
            model=cfg.embedding_model,
            ollama_host=cfg.ollama_host,
            timeout=cfg.embedding_timeout,
        ),
    )
    text_mode = resolve_text_search_mode(cfg.text_search, cfg.bm25_paths)
    ensure_text_fields_mode(text_mode, cfg.text_search_fields)
    store = VectorStore(
        collection=cfg.collection,
        dimension=provider.dimension,
        host=cfg.qdrant_url,
        # sparse mode: recall queries the sparse space AND /memories writes
        # through this store — the flag keeps every remotely written point
        # carrying the named sparse vector the lexical arm searches.
        sparse_text=text_mode == "sparse",
        text_key=cfg.text_key,
    )
    # Dedicated short-timeout client for health/readiness pings — never the
    # recall store's client, whose timeout is sized for recall.
    probe_client = _make_probe_client(cfg.qdrant_url, cfg.qdrant_health_timeout)

    def _graph_ok() -> bool:
        if not cfg.graph_uri:
            return False
        try:
            from neo4j import GraphDatabase

            d = GraphDatabase.driver(
                cfg.graph_uri,
                auth=(cfg.graph_user, cfg.graph_password) if cfg.graph_user else None,
                connection_timeout=cfg.graph_health_timeout,
                connection_acquisition_timeout=cfg.graph_health_timeout,
            )
            with d.session(database=cfg.graph_database) as s:
                s.run("RETURN 1").single()
            d.close()
            return True
        except Exception:
            return False

    def _qdrant_ok() -> bool:
        # Ping-level reachability: only that the client can reach Qdrant, not
        # that the collection exists (a fresh deploy before any ingest is still
        # reachable). Used by /health, /readyz and /status. Uses the dedicated
        # short-timeout probe client so a slow Qdrant can't hang the probe.
        try:
            probe_client.get_collections()
            return True
        except Exception:
            return False

    # Embedding reachability for readiness. Recall can't embed a query without
    # the provider, so it's a hard readiness dependency — but a live embedding
    # call per probe would cost money / hit rate limits, and a slow or hung
    # provider would add latency to (or hang) the probe. So this is a
    # serve-stale-while-revalidate cache: probes never call the provider inline;
    # they read the last known value and, when it's older than the TTL, trigger
    # a single background refresh (single-flight — at most one in flight). The
    # refresh runs on a daemon thread, so a hung health_check can neither block a
    # probe nor delay process shutdown; the worst case is readiness reporting the
    # last known value until the background check completes. Starts not-ready
    # (ts sentinel) until the first refresh lands — correct readiness semantics.
    _embed_health = {"ok": False, "ts": -1e18, "refreshing": False}
    _embed_lock = threading.Lock()

    def _refresh_embedding_health() -> None:
        ok = False
        try:
            healthy, _msg = provider.health_check()
            ok = bool(healthy)
        except Exception:
            ok = False
        with _embed_lock:
            _embed_health["ok"] = ok
            _embed_health["ts"] = time.monotonic()
            _embed_health["refreshing"] = False

    def _embedding_ok() -> bool:
        now = time.monotonic()
        spawn = False
        with _embed_lock:
            ok = bool(_embed_health["ok"])
            stale = now - _embed_health["ts"] >= _EMBED_HEALTH_TTL_S
            if stale and not _embed_health["refreshing"]:
                _embed_health["refreshing"] = True
                spawn = True
        if spawn:
            threading.Thread(
                target=_refresh_embedding_health, name="embed-health", daemon=True
            ).start()
        return ok

    lexical_arms: list[Retriever] = []
    lexical_weights: dict[str, float] = {}
    if text_mode == "bm25":
        bm25_docs = _build_bm25_docs(cfg.bm25_paths)
        if bm25_docs:
            lexical_arms.append(
                BM25Retriever(
                    docs=bm25_docs,
                    timestamp_key=cfg.timestamp_key,
                    timestamp_format=cfg.timestamp_format,
                )
            )
    elif text_mode == "qdrant_bm25":
        # In-process BM25 whose corpus scrolls straight out of the collection's
        # payloads — fine to ~100K chunks (the in-process ceiling), no files.
        # On a FRESH deployment the collection may not exist yet (the write
        # surface bootstraps it lazily): skip the arm instead of crashing the
        # boot — the corpus is a startup snapshot anyway, so the arm joins on
        # the restart after first ingest. Qdrant-down at boot keeps its
        # historical behavior (this mode always required a reachable server).
        if store.collection_exists():
            lexical_arms.append(
                BM25Retriever.from_qdrant(
                    store.client,
                    cfg.collection,
                    text_key=cfg.text_key,
                    timestamp_key=cfg.timestamp_key,
                    timestamp_format=cfg.timestamp_format,
                )
            )
        else:
            log.info(
                "text_search=qdrant_bm25: collection %r does not exist yet — "
                "starting without the BM25 arm (it loads on the next restart "
                "after first ingest)",
                cfg.collection,
            )
    elif text_mode == "lexical":
        arms, lexical_weights = build_qdrant_text_arms(
            embedding=provider,
            vector_store=store,
            text_key=cfg.text_key,
            timestamp_key=cfg.timestamp_key,
            timestamp_format=cfg.timestamp_format,
            fields=cfg.text_search_fields,
        )
        lexical_arms.extend(arms)
    elif text_mode == "sparse":
        lexical_arms.append(
            QdrantSparseRetriever(
                vector_store=store,
                text_key=cfg.text_key,
                timestamp_key=cfg.timestamp_key,
                timestamp_format=cfg.timestamp_format,
            )
        )
    maybe_retrievers = [
        VectorRetriever(
            embedding=provider,
            vector_store=store,
            text_key=cfg.text_key,
            timestamp_key=cfg.timestamp_key,
            timestamp_format=cfg.timestamp_format,
        ),
        *lexical_arms,
        MemgraphRetriever(
            uri=cfg.graph_uri,
            user=cfg.graph_user,
            password=cfg.graph_password,
            database=cfg.graph_database,
            timeout=cfg.graph_timeout,
            chunk_filter_probe=chunk_filter_probe_via(
                store,
                timestamp_key=cfg.timestamp_key,
                timestamp_format=cfg.timestamp_format,
            ),
        )
        if cfg.graph_uri
        else None,
        TemporalRetriever(
            embedding=provider,
            vector_store=store,
            text_key=cfg.text_key,
            timestamp_key=cfg.timestamp_key,
            timestamp_format=cfg.timestamp_format,
        ),
    ]
    retrievers: list[Retriever] = [r for r in maybe_retrievers if r is not None]
    recaller = Recaller(
        retrievers=retrievers,
        vector_floor=cfg.vector_floor,
        retriever_weights=lexical_weights or None,
        text_key=cfg.text_key,
        timestamp_key=cfg.timestamp_key,
        timestamp_format=cfg.timestamp_format,
    )

    from pathlib import Path

    state_path = Path(cfg.state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline = build_full_pipeline(
        state_store=FileStateStore(state_path),
        graph_uri=cfg.graph_uri,
        graph_user=cfg.graph_user,
        graph_password=cfg.graph_password,
        graph_database=cfg.graph_database,
        graph_timeout=cfg.graph_timeout,
        text_key=cfg.text_key,
        timestamp_key=cfg.timestamp_key,
        timestamp_format=cfg.timestamp_format,
    )

    try:
        llm = get_llm(cfg.llm_name, **model_kwargs(cfg.llm_model))
        answer_gen: AnswerGenerator | None = AnswerGenerator(
            llm=llm,
            recaller=recaller,
            timestamp_key=cfg.timestamp_key,
            timestamp_format=cfg.timestamp_format,
        )
        reranker: Reranker | None = Reranker(
            llm=llm,
            max_items=20,
            rerank_mode=cfg.rerank_mode,
        )
    except Exception as exc:  # pragma: no cover - missing provider key
        log.warning("LLM init failed (%s); /answer and reranker disabled.", exc)
        answer_gen = None
        reranker = None

    app = FastAPI(
        title="mnemostack",
        version=__version__,
        description="Hybrid memory stack for AI agents. See https://github.com/udjin-labs/mnemostack.",
    )

    import asyncio

    # ----- Auth (multi-tenant): resolve the tenant + scopes from a service key.
    key_store = None
    rate_limiter = None
    if cfg.auth_enabled:
        from mnemostack.auth import make_key_store
        from mnemostack.quotas import FileQuotaStore, RateLimitExceededError
        from mnemostack.ratelimit import RateLimiter

        # Backend selected by MNEMOSTACK_KEYSTORE (file default, openbao optional);
        # a selected-but-misconfigured backend raises at boot (fail loud).
        key_store = make_key_store(cfg.keys_file)
        # Per-tenant request rate limiting reads each tenant's max_rps from the
        # quota store (shared with the storage quota). Only tenants with a rate
        # quota are throttled; a live `quota set` is picked up within the cache TTL.
        quota_store = FileQuotaStore(cfg.quotas_file)
        rate_limiter = RateLimiter(quota_store)
    else:
        quota_store = None

    def _extract_key(authorization: str | None, x_api_key: str | None) -> str | None:
        if x_api_key:
            return x_api_key.strip()
        if authorization and authorization.lower().startswith("bearer "):
            return authorization[7:].strip()
        return None

    from fastapi.encoders import jsonable_encoder
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse

    # The 422 echo is a DIAGNOSTIC, not a mirror: it runs synchronously on
    # the event loop for bodies that failed schema validation — i.e. before
    # any business-rule cap — so both its total size and the collision
    # loop's work below must be bounded here, not trusted to upstream caps.
    _ECHO_MAX_ITEMS = 64
    _ECHO_MAX_CHARS = 2048

    def _strip_surrogates(value: Any) -> Any:
        if isinstance(value, str):
            # backslashreplace, not "replace": U+FFFD would collapse keys
            # that differ only in their surrogate, silently dropping
            # entries from the echoed input. Truncate AFTER the escape
            # expansion (each surrogate becomes 6 chars) — truncating the
            # raw string first would let an all-surrogate string overshoot
            # the cap ~6x.
            value = value.encode("utf-8", "backslashreplace").decode("utf-8")
            if len(value) > _ECHO_MAX_CHARS:
                value = value[:_ECHO_MAX_CHARS] + "…[truncated]"
            return value
        if isinstance(value, dict):
            # KEYS too: a surrogate metadata key echoed into the error body
            # would crash the response encoder exactly like a value. Keys
            # can still collide post-sanitization (a literal backslash-u
            # sequence vs a real surrogate produce the same text) — suffix
            # U+FFFD until unique so no entry is ever silently dropped.
            # The item cap bounds the disambiguation loop: unbounded, a
            # batch of crafted all-colliding keys is quadratic on the
            # event loop.
            out: dict[Any, Any] = {}

            def _put(key: Any, val: Any) -> None:
                while isinstance(key, str) and key in out:
                    key += "�"
                out[key] = val

            for i, (k, v) in enumerate(value.items()):
                if i >= _ECHO_MAX_ITEMS:
                    _put("…", f"{len(value) - _ECHO_MAX_ITEMS} more entries omitted")
                    break
                _put(_strip_surrogates(k), _strip_surrogates(v))
            return out
        if isinstance(value, list):
            if len(value) > _ECHO_MAX_ITEMS:
                return [_strip_surrogates(v) for v in value[:_ECHO_MAX_ITEMS]] + [
                    f"…{len(value) - _ECHO_MAX_ITEMS} more items omitted"
                ]
            return [_strip_surrogates(v) for v in value]
        return value

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(request, exc):  # noqa: ANN001 — FastAPI signature
        # The default 422 renderer echoes the raw offending input into the
        # response body; a lone surrogate in it (valid JSON escape) crashes
        # the UTF-8 response encoder — the client would see a bare 500 for
        # caller-fixable input. Sanitize before serializing.
        detail = _strip_surrogates(jsonable_encoder(exc.errors()))
        return JSONResponse(status_code=422, content={"detail": detail})

    def _route_label(request) -> str:
        # The route TEMPLATE ("/resolve/{chunk_id}"), never the raw path —
        # a per-id label would explode metric cardinality. Starlette sets
        # scope["route"] before dependencies run on every route this app
        # registers; the fallback is pure defense, not a live case.
        route = request.scope.get("route")
        path = getattr(route, "path", None) or "unmatched"
        return f"{request.method} {path}"

    def _require(scope: str):
        """FastAPI dependency: enforce a valid key with `scope` and return the
        Principal (or None when auth is disabled → single-tenant, tenant=None).

        Also the per-tenant metering choke point: every authenticated call
        increments `mnemostack.tenant.requests{tenant,endpoint}` (and 429s
        `mnemostack.tenant.rate_limited{tenant}`) here, so a tenant burning
        the deployment's budget is visible in /metrics without instrumenting
        each endpoint. Tenants are few by design — label cardinality is
        bounded. Auth-off mode emits no tenant metrics (the global counters
        already cover single-tenant deployments)."""

        def _dep(
            request: Request,
            authorization: str | None = Header(default=None),
            x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        ):
            if not cfg.auth_enabled:
                return None  # auth off: unauthenticated, tenant-less (legacy)
            key = _extract_key(authorization, x_api_key)
            if not key:
                raise HTTPException(status_code=401, detail="missing service key")
            principal = key_store.verify(key) if key_store else None
            if principal is None:
                raise HTTPException(status_code=401, detail="invalid service key")
            if not principal.can(scope):
                raise HTTPException(status_code=403, detail=f"key lacks '{scope}' scope")
            # Rate limit AFTER auth/authz so an unauthenticated flood can't
            # consume a tenant's tokens (an invalid key is rejected above, free).
            # Count BEFORE the rate limiter so `requests` really is every
            # authenticated request (as documented) — the 429 rejection
            # ratio is then rate_limited / requests.
            counter(
                "mnemostack.tenant.requests",
                labels={"tenant": principal.tenant, "endpoint": _route_label(request)},
            )
            if rate_limiter is not None:
                try:
                    rate_limiter.check(principal.tenant)
                except RateLimitExceededError as exc:
                    counter(
                        "mnemostack.tenant.rate_limited",
                        labels={"tenant": principal.tenant},
                    )
                    # A vanishingly small (but valid) rate makes 1/rate overflow to
                    # inf, so guard/cap retry_after — never ceil(inf) → 500. A day is
                    # a sane ceiling for "come back later".
                    ra = exc.retry_after
                    secs = _MAX_RETRY_AFTER if not math.isfinite(ra) else max(
                        1, min(_MAX_RETRY_AFTER, math.ceil(ra))
                    )
                    raise HTTPException(
                        status_code=429,
                        detail=str(exc),
                        headers={"Retry-After": str(secs)},
                    ) from exc
            return principal

        return _dep

    def _tenant_of(principal) -> str | None:
        return principal.tenant if principal is not None else None

    def _run_recall_sync(
        query: str,
        limit: int,
        full_pipeline: bool,
        filters: dict[str, Any] | None = None,
        token_budget: int | None = None,
        include_invalidated: bool = False,
        as_of: str | None = None,
        tenant: str | None = None,
    ):
        trace = RecallTrace()
        # Reranking is part of the full pipeline; if it was requested but the
        # LLM never initialized, that is a degradation worth surfacing. With
        # full_pipeline=False the reranker is off by request, not degraded.
        if full_pipeline and reranker is None:
            trace.mark("reranker:unavailable")
        results = recall_flow(
            recaller,
            query,
            limit,
            pipeline=pipeline if full_pipeline else None,
            reranker=reranker if full_pipeline else None,
            filters=filters,
            trace=trace,
            # Per-request budget wins; unset falls back to the server-wide
            # default so operators can cap every client uniformly.
            token_budget=token_budget if token_budget is not None else cfg.token_budget,
            include_invalidated=include_invalidated,
            as_of=as_of,
            tenant=tenant,
        )
        if cfg.auto_record_ior:
            # Record into the caller's tenant partition so auto-IoR is per-tenant.
            record_recall_events(pipeline, results, tenant)
        return results, trace

    async def _run_recall(
        query: str,
        limit: int,
        full_pipeline: bool,
        filters: dict[str, Any] | None = None,
        token_budget: int | None = None,
        include_invalidated: bool = False,
        as_of: str | None = None,
        tenant: str | None = None,
    ):
        """Offload the blocking recall stack to a worker thread.

        Recaller/pipeline/reranker all do blocking I/O or CPU work. Running
        them inline in the event loop would serialise every HTTP request
        behind the slowest retriever.
        """
        return await asyncio.to_thread(
            _run_recall_sync,
            query,
            limit,
            full_pipeline,
            filters,
            token_budget,
            include_invalidated,
            as_of,
            tenant,
        )

    @app.get("/", include_in_schema=False)
    def root():
        return {"name": "mnemostack", "version": __version__, "docs": "/docs"}

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        from fastapi.responses import PlainTextResponse

        rec = get_recorder()
        if not isinstance(rec, InMemoryRecorder):
            # Can happen if someone swapped in a null recorder externally.
            return PlainTextResponse(
                "# mnemostack: aggregating recorder not installed\n",
                media_type="text/plain; version=0.0.4",
            )
        body = _prometheus_dump(rec)
        return PlainTextResponse(body, media_type="text/plain; version=0.0.4")

    @app.get("/health", response_model=HealthResponse)
    def health():
        qdrant_ok = _qdrant_ok()
        return HealthResponse(
            status="ok" if qdrant_ok else "degraded",
            version=__version__,
            provider=cfg.provider_name,
            collection=cfg.collection,
            qdrant=qdrant_ok,
            memgraph=_graph_ok(),
        )

    @app.get("/healthz", response_model=LiveResponse)
    def healthz():
        # Liveness: is the process up? Deliberately checks NO dependencies — a
        # transient Qdrant/Memgraph outage must not make an orchestrator kill an
        # otherwise-healthy process (that's what /readyz is for).
        return LiveResponse(status="ok", version=__version__)

    @app.get(
        "/readyz",
        response_model=ReadyResponse,
        responses={503: {"model": ReadyResponse}},
    )
    def readyz():
        # Readiness: can we serve traffic? The hard dependencies are Qdrant (the
        # vector store recall reads) and the embedding provider (recall can't
        # embed a query without it); either being down gates readiness with a
        # 503. Both checks are bounded so a slow backend can't hang the probe.
        # The graph is optional/fail-soft and is intentionally not pinged here —
        # see ReadyResponse — so a slow graph can never add latency to /readyz.
        qdrant_ok = _qdrant_ok()
        embedding_ok = _embedding_ok()
        ready = qdrant_ok and embedding_ok
        body = ReadyResponse(
            status="ready" if ready else "not_ready",
            version=__version__,
            qdrant=qdrant_ok,
            embedding=embedding_ok,
        )
        if not ready:
            from fastapi.responses import JSONResponse

            # mode="json" so the body stays JSON-safe if a non-native field type
            # is ever added — returning a Response bypasses response_model.
            return JSONResponse(status_code=503, content=body.model_dump(mode="json"))
        return body

    @app.get("/status", response_model=StatusResponse)
    def status():
        # Operator snapshot: config, live dependency reachability, and headline
        # counters (recall volume + total degradation events) from the recorder.
        rec = get_recorder()
        recall_calls = 0.0
        degraded = 0.0
        if isinstance(rec, InMemoryRecorder):
            # Iterate a snapshot — the live dict is mutated by concurrent recall.
            # key = (name, labels...); sum across label sets for each metric.
            for key, value in rec.snapshot_counters().items():
                name = key[0]
                if name == "mnemostack.recall.calls":
                    recall_calls += value
                elif name in _DEGRADED_METRICS:
                    degraded += value
        return StatusResponse(
            version=__version__,
            provider=cfg.provider_name,
            llm=cfg.llm_name,
            collection=cfg.collection,
            qdrant_url=cfg.qdrant_url,
            qdrant=_qdrant_ok(),
            embedding=_embedding_ok(),
            memgraph=_graph_ok(),
            recall_calls=recall_calls,
            degraded_events=degraded,
        )

    @app.post("/recall", response_model=RecallResponse)
    async def recall_endpoint(req: RecallRequest, principal=Depends(_require("read"))):  # noqa: B008 — FastAPI DI pattern
        try:
            results, trace = await _run_recall(
                req.query,
                req.limit,
                req.full_pipeline,
                req.filters,
                req.token_budget,
                req.include_invalidated,
                req.as_of,
                _tenant_of(principal),
            )
        except Exception as exc:
            log.exception("recall endpoint failed")
            raise HTTPException(status_code=500, detail="recall failed") from exc
        return RecallResponse(
            query=req.query,
            results=[_memory_of(r) for r in results],
            degraded=trace.degraded,
            notes=trace.notes,
            trace=trace.to_dict() if req.include_trace else None,
            tokens_estimate=sum_tokens(results),
        )

    @app.get("/resolve/{chunk_id}", response_model=ResolveResponse)
    async def resolve_endpoint(chunk_id: str, principal=Depends(_require("read"))):  # noqa: B008 — FastAPI DI pattern
        """Verify a citation: resolve a chunk id back to its source document.

        Runs OUTSIDE the recall path (recall latency is untouched). Verdicts
        are honest about what this process can see: a server without the
        source tree mounted reports missing/unresolvable rather than
        pretending. Under auth the lookup is tenant-scoped — another tenant's
        point is indistinguishable from an absent one.
        """
        from mnemostack.provenance import resolve_citation

        try:
            res = await asyncio.to_thread(
                partial(
                    resolve_citation,
                    store,
                    chunk_id,
                    tenant=_tenant_of(principal),
                    text_key=cfg.text_key,
                    allowed_roots=list(cfg.resolve_roots),
                )
            )
        except Exception as exc:
            log.exception("resolve endpoint failed")
            raise HTTPException(status_code=500, detail="resolve failed") from exc
        return res.to_dict()

    @app.post("/answer", response_model=AnswerResponse)
    async def answer_endpoint(req: AnswerRequest, principal=Depends(_require("read"))):  # noqa: B008 — FastAPI DI pattern
        if answer_gen is None:
            # Count the outage: this returns before AnswerGenerator.generate()
            # can emit mnemostack.answer.errors, so without this the operator's
            # /status.degraded_events would read 0 during a full answer outage.
            counter("mnemostack.answer.unavailable", 1)
            raise HTTPException(
                status_code=503,
                detail="answer generator unavailable (LLM not configured)",
            )
        # Resolve the effective budget once: the generator needs it too, or
        # its retry paths would prompt over fresh unbudgeted sub-recalls.
        token_budget = req.token_budget if req.token_budget is not None else cfg.token_budget
        tenant = _tenant_of(principal)
        try:
            results, trace = await _run_recall(
                req.query,
                req.limit,
                req.full_pipeline,
                req.filters,
                token_budget,
                req.include_invalidated,
                req.as_of,
                tenant,
            )
            # recall_filters keeps the answer generator's retry sub-recalls
            # inside the same filtered scope; the validity view AND the tenant
            # must reach the generator too, or its retry sub-recalls would ignore
            # as_of / pull cross-tenant evidence.
            ans = await asyncio.to_thread(
                partial(
                    answer_gen.generate,
                    req.query,
                    results,
                    recall_filters=req.filters,
                    token_budget=token_budget,
                    include_invalidated=req.include_invalidated,
                    as_of=req.as_of,
                    tenant=tenant,
                )
            )
        except Exception as exc:
            log.exception("answer endpoint failed")
            raise HTTPException(status_code=500, detail="answer failed") from exc
        # Prefer the generator's own estimate: its retry paths can swap in a
        # freshly recalled context pool, and the primary recall results would
        # then misreport what the answer prompt actually contained.
        tokens_estimate = getattr(ans, "context_tokens_estimate", None)
        if tokens_estimate is None:
            tokens_estimate = sum_tokens(results)
        return AnswerResponse(
            query=req.query,
            answer=ans.text,
            confidence=float(getattr(ans, "confidence", 0.0)),
            sources=list(getattr(ans, "sources", []) or []),
            memories=[_memory_of(r) for r in results],
            degraded=trace.degraded,
            notes=trace.notes,
            trace=trace.to_dict() if req.include_trace else None,
            tokens_estimate=tokens_estimate,
            tokens_used=getattr(ans, "tokens_used", None),
        )

    @app.post("/feedback", response_model=FeedbackResponse)
    async def feedback_endpoint(req: FeedbackRequest, principal=Depends(_require("write"))):  # noqa: B008 — FastAPI DI pattern
        """Record explicit feedback for stateful recall stages.

        Q-learning updates require retriever/source labels. Pass the `retrievers`
        field returned by /recall or /answer as `sources`.

        Multi-tenant: under `--auth` the `write` scope gates access AND the
        learning state (Q-table, IoR log) it updates is partitioned by the key's
        tenant, so one tenant's feedback can't shift another tenant's ranking.
        """
        try:
            outcome = apply_feedback(
                pipeline,
                hit_id=req.hit_id,
                signal=req.signal,
                query=req.query,
                query_type=req.query_type,
                source=req.source,
                sources=req.sources,
                reward=req.reward,
                tenant=_tenant_of(principal),
            )
        except Exception as exc:
            log.exception("feedback endpoint failed")
            raise HTTPException(status_code=500, detail="feedback failed") from exc
        return FeedbackResponse(**outcome.to_dict())

    @app.post("/memories", response_model=MemoriesResponse)
    def memories_endpoint(req: MemoriesRequest, principal=Depends(_require("write"))):  # noqa: B008 — FastAPI DI pattern
        """Store memories for the caller's tenant (the remote write surface).

        Texts are embedded SERVER-side (clients need no provider key), ids
        are deterministic from (source, offset, text) so retries and re-sent
        content come back `duplicate` at zero embedding cost, and under
        `--auth` everything lands stamped with the key's tenant — a client
        cannot write into another tenant. Work is bounded: request caps
        (items/text/metadata), the tenant's storage quota (507 when
        exceeded), and the shared per-tenant rate limit. Declared as a sync
        route on purpose: FastAPI runs it on the threadpool, so the
        embedding round-trips never block the event loop.

        Mode note: under `text_search=qdrant_bm25` the in-process BM25
        corpus is a startup snapshot — new memories join the vector arm
        immediately but that lexical arm only after a restart (the same
        pre-existing behavior as CLI ingest while the server runs).
        """
        # Pydantic enforces shapes/caps; the reserved-namespace rule and
        # cross-field constraints live in the shared validator so the MCP
        # tool enforces the identical contract. The deployment's configured
        # text/timestamp keys are reserved too — metadata must not shadow
        # what recall actually reads.
        schema_reserved = {cfg.text_key, cfg.timestamp_key} - {"text", "timestamp"}
        for i, item in enumerate(req.items):
            problem = validate_remote_item(
                item.text,
                item.source,
                item.timestamp,
                item.tags,
                item.metadata,
                offset=item.offset,
                chunk=item.chunk,
                reserved_extra=schema_reserved or None,
            )
            if problem:
                raise HTTPException(status_code=400, detail=f"items[{i}]: {problem}")
        tenant = _tenant_of(principal)
        max_points = None
        if tenant is not None and quota_store is not None:
            try:
                q = quota_store.get(tenant)
                max_points = q.max_points if q is not None else None
            except Exception:  # noqa: BLE001 — quota store is fail-open by contract
                # Fail-open is the quota contract (guardrail, not a security
                # boundary) — but never silently: this warning is the only
                # signal that per-tenant caps stopped applying.
                log.warning(
                    "quota store lookup failed; storage quota not enforced "
                    "for this /memories request",
                    exc_info=True,
                )
                max_points = None
        entries = [
            (
                IngestItem(
                    text=it.text,
                    source=it.source,
                    offset=it.offset,
                    metadata=dict(it.metadata),
                    tags=list(it.tags),
                    timestamp=it.timestamp,
                ),
                it.chunk,
            )
            for it in req.items
        ]
        try:
            flat_items, origins = expand_remote_items(entries)
        except RemoteRequestTooLarge as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            results = ingest_remote_items(
                provider,
                store,
                flat_items,
                tenant=tenant,
                max_points=max_points,
                text_key=cfg.text_key,
                timestamp_key=cfg.timestamp_key,
                timestamp_format=cfg.timestamp_format,
            )
        except QuotaExceededError as exc:
            if tenant is not None:
                counter("mnemostack.tenant.quota_rejected", labels={"tenant": tenant})
            raise HTTPException(status_code=507, detail=str(exc)) from exc
        except EmbeddingSpaceError as exc:
            # Server-side configuration fault (repointed model tag / mixed
            # space), not a client error — loud and retriable after the
            # operator fixes the deployment.
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("memories endpoint failed")
            raise HTTPException(status_code=500, detail="ingest failed") from exc
        # (Per-tenant embedding-spend meters are emitted inside
        # ingest_remote_items with exception-aware semantics: post-embed
        # failures bill, a pre-embed space-guard abort does not.)
        failed_n = sum(r.status == "failed" for r in results)
        if results and failed_n == len(results):
            # Same honesty rule as /triples: per-item isolation is for
            # PARTIAL failure — when EVERY item failed to embed, the write
            # path is down and a 200 would hide it from status-code checks.
            raise HTTPException(
                status_code=502,
                detail=f"embedding failed for all {failed_n} item(s)",
            )
        return MemoriesResponse(
            results=[
                MemoryResultOut(
                    id=r.id, status=r.status, item=origin, offset=flat.offset
                )
                for r, origin, flat in zip(results, origins, flat_items, strict=True)
            ],
            stored=sum(r.status == "stored" for r in results),
            duplicates=sum(r.status == "duplicate" for r in results),
            failed=failed_n,
        )

    @app.post("/triples", response_model=TriplesResponse)
    def triples_endpoint(req: TriplesRequest, principal=Depends(_require("write"))):  # noqa: B008 — FastAPI DI pattern
        """Add temporal facts to the knowledge graph (HTTP parity with the
        MCP `mnemostack_graph_add_triple` tool).

        Structured writes only — subject/predicate/object with optional
        validity bounds; nodes are created on demand. Under `--auth` every
        node and edge is stamped with the key's tenant (the graph-side
        mirror of the vector `tenant_id`), so a client cannot write into
        another tenant's subgraph. Requires a configured graph; 503 when
        the deployment runs without Memgraph.
        """
        # Input validation FIRST (a 400 the caller can fix), availability
        # second (a 503 the operator owns). Contract shared with the MCP
        # graph tool via validate_remote_triple.
        for i, t in enumerate(req.triples):
            problem = validate_remote_triple(
                t.subject, t.predicate, t.object, t.valid_from, t.valid_until
            )
            if problem:
                raise HTTPException(status_code=400, detail=f"triples[{i}]: {problem}")
        if not cfg.graph_uri:
            raise HTTPException(status_code=503, detail="graph is not configured")
        tenant = _tenant_of(principal)
        from mnemostack.graph.factory import make_graph_store

        try:
            gs = make_graph_store(
                cfg.graph_uri,
                timeout=cfg.graph_timeout,
                user=cfg.graph_user,
                password=cfg.graph_password,
                database=cfg.graph_database,
            )
        except Exception as exc:
            log.exception("triples endpoint: graph store unavailable")
            raise HTTPException(status_code=503, detail="graph unavailable") from exc
        results: list[TripleResultOut] = []
        try:
            for t in req.triples:
                try:
                    gs.add_triple(
                        subject=t.subject,
                        predicate=t.predicate,
                        obj=t.object,
                        valid_from=t.valid_from,
                        valid_until=t.valid_until,
                        tenant=tenant,
                    )
                    results.append(TripleResultOut(status="added"))
                except Exception as exc:  # noqa: BLE001 — per-triple isolation
                    log.warning("triples endpoint: add_triple failed: %s", exc)
                    results.append(
                        TripleResultOut(status="failed", error=type(exc).__name__)
                    )
        finally:
            try:
                gs.close()
            except Exception:  # noqa: BLE001
                pass
        counter("mnemostack.server.triples", len(req.triples))
        added = sum(r.status == "added" for r in results)
        failed = sum(r.status == "failed" for r in results)
        if results and added == 0:
            # Per-triple isolation is for PARTIAL failure; when every triple
            # failed the write path itself is broken — a 200 here would hide
            # a dead graph from any caller that checks only the status code.
            raise HTTPException(
                status_code=502, detail=f"all {failed} triple write(s) failed"
            )
        return TriplesResponse(results=results, added=added, failed=failed)

    def _selector_problem(ids: Any, source: Any) -> str | None:
        """Exactly one selector: a request naming both is ambiguous about
        which one bounds the operation, and one naming neither would be an
        unscoped lifecycle call."""
        if (ids is None) == (source is None):
            return "exactly one of ids/source is required"
        if source is not None:
            return validate_remote_source(source)
        return None

    @app.get("/memories", response_model=MemoryListResponse)
    def list_memories(
        source: str,
        after: str | None = None,
        limit: int = REMOTE_LIST_PAGE,
        index_root: str | None = None,
        principal=Depends(_require("read")),  # noqa: B008 — FastAPI DI pattern
    ):
        """List what this tenant holds from one source, for reconciliation.

        Ids and integrity metadata only — NEVER memory text: a client
        needs to detect that its own state and the service's have
        diverged (a file it re-indexed, points it thinks it deleted), and
        that check must not become a second read channel for memory
        content with its own scope semantics. `content_hash` is the
        source snapshot written at ingest; points stored before that
        feature return null.

        Pagination follows the store's own iteration order and `after`
        is an opaque cursor: pass back the last id you received. A page
        costs the page, not a full scan of the source. Under `--auth`
        only the key's tenant is visible.

        `index_root` is the same owner guard the lifecycle endpoints
        take, and for the same reason: one source name can exist under
        several indexing roots, and a client reconciling ONE root must
        not be shown (or, on the sibling endpoints, act on) another's
        points. Identical semantics to those endpoints, including the
        edge: a point carrying NO `index_root` tag is still listed under
        any guard — the guard excludes a DIFFERENT root, it does not
        require one ("points carrying no index_root tag are not protected
        by it").
        """
        problem = (
            validate_remote_source(source)
            or validate_remote_cursor(after)
            or validate_invalidate_options(None, None, index_root)
        )
        if problem:
            raise HTTPException(status_code=400, detail=problem)
        if limit < 1 or limit > REMOTE_LIST_PAGE:
            raise HTTPException(
                status_code=400, detail=f"limit must be 1..{REMOTE_LIST_PAGE}"
            )
        tenant = _tenant_of(principal)
        # The cursor comes back through JSON as the string this endpoint
        # printed, but a collection may use INTEGER point ids, and the store
        # distinguishes 7 from "7" — an uncoerced cursor would resume from
        # an id that does not exist, silently returning an empty page and
        # reporting the listing complete while points remain. Same coercion
        # the lifecycle selectors apply to caller-supplied ids.
        cursor = coerce_point_ids([after])[0] if after is not None else None
        try:
            if not store.collection_exists():
                return MemoryListResponse(items=[], complete=True)
            ids, payloads, more = find_source_points(
                store,
                source,
                tenant=tenant,
                index_root=index_root,
                limit=limit,
                start_after=cursor,
            )
        except Exception as exc:
            log.exception("list memories failed")
            raise HTTPException(status_code=500, detail="listing failed") from exc
        page = [
            MemoryListItem(
                id=str(pid),
                content_hash=_list_text(payload.get(SOURCE_HASH_KEY)),
                # `is None`, not `or`: under `timestamp_format: epoch` the
                # Unix epoch itself is a VALID timestamp stored as numeric
                # 0, and a falsy test would silently substitute the legacy
                # mirror for it.
                timestamp=_list_timestamp(
                    payload[cfg.timestamp_key]
                    if payload.get(cfg.timestamp_key) is not None
                    else payload.get("timestamp")
                ),
                indexed_at=_list_text(payload.get("indexed_at")),
            )
            for pid, payload in zip(ids, payloads, strict=True)
        ]
        counter("mnemostack.server.list_memories", len(page))
        return MemoryListResponse(items=page, complete=not more)

    @app.post("/invalidate", response_model=InvalidateResponse)
    def invalidate_endpoint(req: InvalidateRequest, principal=Depends(_require("write"))):  # noqa: B008 — FastAPI DI pattern
        """Mark memories stale, non-destructively.

        The `ids` path is HTTP parity with the MCP `mnemostack_invalidate`
        tool (same shared validator, same anti-oracle skip semantics). The
        `source` path has NO MCP twin: source-scoped retraction, the
        reconciliation listing and `DELETE /memories` are HTTP-only for
        now — an agent tool that can retract a whole document from a
        conversation is a bigger blast radius than one that names ids, so
        that surface is deliberately operator-side until asked for.

        Sets `invalidated_at` (and optionally `valid_until`) on each
        point's payload without deleting or re-embedding it; invalidated
        facts drop out of default recall but stay reachable via
        `include_invalidated` / point-in-time `as_of`. Under `--auth` only
        the key's tenant's points are touched — foreign or unknown ids are
        skipped, indistinguishably from missing ones (the response count
        is not an existence oracle). Re-sending the same content to
        `/memories` reactivates an invalidated memory.

        Scope note: vector and temporal recall honor the marker
        immediately; an in-process BM25 corpus built from Qdrant payloads
        (`text_search=qdrant_bm25`) reflects it only after a restart.

        Source scope is FIRST-RETRACTION-WINS: a point this source already
        had invalidated is not restamped by a later source-scoped call
        (and is not counted in `requested`), because `invalidated_at`
        records when the memory was actually retracted. Retract by id to
        change an existing retraction's bounds.
        """
        problem = _selector_problem(req.ids, req.source)
        if problem is None:
            problem = (
                validate_remote_invalidate(
                    req.ids, req.invalidated_at, req.valid_until, req.index_root
                )
                if req.ids is not None
                else validate_invalidate_options(
                    req.invalidated_at, req.valid_until, req.index_root
                )
            )
        if problem:
            raise HTTPException(status_code=400, detail=problem)
        tenant = _tenant_of(principal)
        tkw: dict[str, Any] = {"tenant": tenant} if tenant is not None else {}
        # Dedup AFTER coercion ("7" and 7 are the same point): the store
        # counts per requested id, so duplicates would inflate `invalidated`
        # past the number of points actually touched.
        complete = True
        try:
            if not store.collection_exists():
                # Fresh deployment before the first write: the collection is
                # created lazily by /memories. Nothing exists — the same
                # "unknown ids are skipped" semantics, not a 500.
                return InvalidateResponse(
                    requested=len(req.ids or []), invalidated=0, complete=True
                )
            if req.source is not None:
                ids, _payloads, more = find_source_points(
                    store,
                    req.source,
                    tenant=tenant,
                    index_root=req.index_root,
                    limit=REMOTE_SOURCE_BATCH,
                    # Without this the next call re-selects the same first
                    # batch forever: an invalidated point keeps its source.
                    # Consequence, documented rather than papered over:
                    # source retraction is FIRST-RETRACTION-WINS. A point
                    # already invalidated keeps the `invalidated_at` and
                    # `valid_until` of the call that retracted it — a later
                    # source-scoped call does not restamp it, because
                    # `invalidated_at` records WHEN the memory was actually
                    # retracted and rewriting it would falsify that. To
                    # change an existing retraction's world-time bound, name
                    # the point by id.
                    skip_invalidated=True,
                )
                complete = not more
                requested = len(ids)
                if not ids:
                    return InvalidateResponse(
                        requested=0, invalidated=0, complete=complete
                    )
            else:
                ids = list(dict.fromkeys(coerce_point_ids(req.ids or [])))
                requested = len(req.ids or [])
            updated = store.invalidate(
                ids,
                invalidated_at=req.invalidated_at,
                valid_until=req.valid_until,
                index_root=req.index_root,
                **tkw,
            )
        except Exception as exc:
            log.exception("invalidate endpoint failed")
            raise HTTPException(status_code=500, detail="invalidate failed") from exc
        counter("mnemostack.server.invalidate", updated)
        return InvalidateResponse(
            requested=requested, invalidated=updated, complete=complete
        )

    @app.delete("/memories", response_model=DeleteMemoriesResponse)
    def delete_memories_endpoint(req: DeleteMemoriesRequest, principal=Depends(_require("write"))):  # noqa: B008 — FastAPI DI pattern
        """Hard-delete memories by id (right-to-erasure).

        Unlike `/invalidate`, this REMOVES the points — vector, payload,
        and text — irreversibly; re-sending the same content to
        `/memories` afterwards stores a fresh copy. Under `--auth` the
        ownership check runs server-side: only points stamped with the
        key's tenant are deleted, and foreign or unknown ids are skipped
        indistinguishably (the count is not an existence oracle). Without
        auth every requested existing id is deleted (single-tenant
        legacy). `index_root` narrows the operation to one indexing
        root's points, matching `/invalidate` — points carrying no
        `index_root` tag are not protected by the guard.

        Erasure scope: the deletion is immediate for vector and temporal
        recall (they read Qdrant live). An in-process BM25 corpus built
        from Qdrant payloads (`text_search=qdrant_bm25`) is a startup
        snapshot and can keep serving the deleted text until the server
        restarts — for erasure requests on such deployments, restart (or
        rebuild the corpus) to complete the erasure. Graph facts written
        via `/triples` are separate records and are not deleted by this
        endpoint.
        """
        problem = _selector_problem(req.ids, req.source)
        if problem is None:
            problem = (
                validate_remote_invalidate(req.ids, None, None, req.index_root)
                if req.ids is not None
                else validate_invalidate_options(None, None, req.index_root)
            )
        if problem:
            raise HTTPException(status_code=400, detail=problem)
        tenant = _tenant_of(principal)
        tkw: dict[str, Any] = {"tenant": tenant} if tenant is not None else {}
        # Dedup AFTER coercion — duplicates would inflate `deleted`.
        ids = list(dict.fromkeys(coerce_point_ids(req.ids or [])))
        requested = len(req.ids or [])
        complete = True
        source_scoped = req.source is not None
        try:
            if not store.collection_exists():
                # See /invalidate: pre-bootstrap there is nothing to erase.
                return DeleteMemoriesResponse(
                    requested=requested, deleted=0, complete=True
                )
            if source_scoped:
                # The finder already re-validated payload ownership and the
                # index_root guard, so no second precheck is needed here.
                ids, _payloads, more = find_source_points(
                    store,
                    req.source or "",
                    tenant=tenant,
                    index_root=req.index_root,
                    limit=REMOTE_SOURCE_BATCH,
                )
                complete = not more
                requested = len(ids)
            elif req.index_root is not None or tenant is None:
                # One retrieve serves two jobs. (1) The index_root owner
                # guard, matching /invalidate: points owned by a DIFFERENT
                # root are skipped; untagged points pass (documented). (2)
                # In unscoped mode delete_points performs no existence
                # check — without this precheck `deleted` would echo the
                # request instead of counting actual removals.
                found = store.client.retrieve(
                    store.collection, ids=ids, with_payload=["index_root"]
                )
                keep = {
                    str(p.id)
                    for p in found
                    if req.index_root is None
                    or (getattr(p, "payload", None) or {}).get("index_root")
                    in (None, req.index_root)
                }
                ids = [pid for pid in ids if str(pid) in keep]
            deleted = store.delete_points(ids, **tkw) if ids else 0
        except Exception as exc:
            log.exception("delete memories endpoint failed")
            raise HTTPException(status_code=500, detail="delete failed") from exc
        counter("mnemostack.server.deleted", deleted)
        return DeleteMemoriesResponse(
            requested=requested, deleted=deleted, complete=complete
        )

    return app
