"""FastAPI service wrapper for mnemostack.

Exposes `/recall`, `/answer`, `/feedback`, `/health`, and `/metrics` over HTTP
so callers in any language (Node, Go, Rust, curl) can use mnemostack without a
Python SDK.

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
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as e:  # pragma: no cover - import guard
    raise ImportError(
        "FastAPI is not installed. Install the optional server extra: "
        "`pip install 'mnemostack[server]'`."
    ) from e

from mnemostack import __version__
from mnemostack.config import Config, model_kwargs
from mnemostack.embeddings import get_provider
from mnemostack.feedback import apply_feedback, record_recall_events
from mnemostack.llm import get_llm
from mnemostack.observability.recorder import (
    InMemoryRecorder,
    get_recorder,
    set_recorder,
)
from mnemostack.recall import (
    RERANK_MODES,
    AnswerGenerator,
    BM25Retriever,
    MemgraphRetriever,
    Recaller,
    RecallTrace,
    Reranker,
    Retriever,
    TemporalRetriever,
    VectorRetriever,
    build_full_pipeline,
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
            "Degradations that occurred while serving this call "
            "(e.g. retriever:bm25:failed, reranker:fallback). Empty when healthy."
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
_DEGRADED_METRICS = frozenset(
    {
        "mnemostack.recall.fallback_triggered",
        "mnemostack.recall.followup_rewrite_failed",
        "mnemostack.query_expansion.errors",
        "mnemostack.answer.errors",
        "mnemostack.answer.list_finalize_fallback",
    }
)


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
            llm_name=cfg.llm.provider,
            llm_model=cfg.llm.model,
            collection=cfg.vector.collection,
            qdrant_url=cfg.vector.host,
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
        )


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
    payload = {key: value for key, value in payload.items() if key != "_vector_floor_candidates"}
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
        parts = [f'{k}="{v}"' for k, v in labels.items()]
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

    provider = get_provider(cfg.provider_name, **model_kwargs(cfg.embedding_model))
    store = VectorStore(
        collection=cfg.collection, dimension=provider.dimension, host=cfg.qdrant_url
    )

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
        # reachable). Used by /health, /readyz and /status.
        try:
            store.client.get_collections()
            return True
        except Exception:
            return False

    bm25_docs = _build_bm25_docs(cfg.bm25_paths)
    maybe_retrievers = [
        VectorRetriever(embedding=provider, vector_store=store),
        BM25Retriever(docs=bm25_docs) if bm25_docs else None,
        MemgraphRetriever(
            uri=cfg.graph_uri,
            user=cfg.graph_user,
            password=cfg.graph_password,
            database=cfg.graph_database,
            timeout=cfg.graph_timeout,
        )
        if cfg.graph_uri
        else None,
        TemporalRetriever(embedding=provider, vector_store=store),
    ]
    retrievers: list[Retriever] = [r for r in maybe_retrievers if r is not None]
    recaller = Recaller(retrievers=retrievers, vector_floor=cfg.vector_floor)

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
    )

    try:
        llm = get_llm(cfg.llm_name, **model_kwargs(cfg.llm_model))
        answer_gen: AnswerGenerator | None = AnswerGenerator(llm=llm, recaller=recaller)
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

    def _run_recall_sync(
        query: str,
        limit: int,
        full_pipeline: bool,
        filters: dict[str, Any] | None = None,
        token_budget: int | None = None,
        include_invalidated: bool = False,
        as_of: str | None = None,
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
        )
        if cfg.auto_record_ior:
            record_recall_events(pipeline, results)
        return results, trace

    async def _run_recall(
        query: str,
        limit: int,
        full_pipeline: bool,
        filters: dict[str, Any] | None = None,
        token_budget: int | None = None,
        include_invalidated: bool = False,
        as_of: str | None = None,
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
        # Readiness: can we serve traffic? Qdrant is the only hard dependency
        # (recall needs it), so it alone gates readiness with a 503. The graph is
        # optional/fail-soft and is intentionally not pinged here — see
        # ReadyResponse — so a slow graph can never add latency to this probe.
        qdrant_ok = _qdrant_ok()
        body = ReadyResponse(
            status="ready" if qdrant_ok else "not_ready",
            version=__version__,
            qdrant=qdrant_ok,
        )
        if not qdrant_ok:
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
            memgraph=_graph_ok(),
            recall_calls=recall_calls,
            degraded_events=degraded,
        )

    @app.post("/recall", response_model=RecallResponse)
    async def recall_endpoint(req: RecallRequest):
        try:
            results, trace = await _run_recall(
                req.query,
                req.limit,
                req.full_pipeline,
                req.filters,
                req.token_budget,
                req.include_invalidated,
                req.as_of,
            )
        except Exception as exc:
            log.exception("recall endpoint failed")
            raise HTTPException(status_code=500, detail="recall failed") from exc
        return RecallResponse(
            query=req.query,
            results=[_memory_of(r) for r in results],
            degraded=trace.degraded,
            trace=trace.to_dict() if req.include_trace else None,
            tokens_estimate=sum_tokens(results),
        )

    @app.post("/answer", response_model=AnswerResponse)
    async def answer_endpoint(req: AnswerRequest):
        if answer_gen is None:
            raise HTTPException(
                status_code=503,
                detail="answer generator unavailable (LLM not configured)",
            )
        # Resolve the effective budget once: the generator needs it too, or
        # its retry paths would prompt over fresh unbudgeted sub-recalls.
        token_budget = req.token_budget if req.token_budget is not None else cfg.token_budget
        try:
            results, trace = await _run_recall(
                req.query,
                req.limit,
                req.full_pipeline,
                req.filters,
                token_budget,
                req.include_invalidated,
                req.as_of,
            )
            # recall_filters keeps the answer generator's retry sub-recalls
            # inside the same filtered scope; the validity view must reach the
            # generator too, or its retry sub-recalls would ignore as_of.
            ans = await asyncio.to_thread(
                partial(
                    answer_gen.generate,
                    req.query,
                    results,
                    recall_filters=req.filters,
                    token_budget=token_budget,
                    include_invalidated=req.include_invalidated,
                    as_of=req.as_of,
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
            trace=trace.to_dict() if req.include_trace else None,
            tokens_estimate=tokens_estimate,
            tokens_used=getattr(ans, "tokens_used", None),
        )

    @app.post("/feedback", response_model=FeedbackResponse)
    async def feedback_endpoint(req: FeedbackRequest):
        """Record explicit feedback for stateful recall stages.

        Q-learning updates require retriever/source labels. Pass the `retrievers`
        field returned by /recall or /answer as `sources`.
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
            )
        except Exception as exc:
            log.exception("feedback endpoint failed")
            raise HTTPException(status_code=500, detail="feedback failed") from exc
        return FeedbackResponse(**outcome.to_dict())

    return app
