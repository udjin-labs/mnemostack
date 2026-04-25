"""FastAPI service wrapper for mnemostack.

Exposes `/recall`, `/answer`, `/health`, and `/metrics` over HTTP so callers in
any language (Node, Go, Rust, curl) can use mnemostack without a Python SDK.

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
from dataclasses import dataclass
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as e:  # pragma: no cover - import guard
    raise ImportError(
        "FastAPI is not installed. Install the optional server extra: "
        "`pip install 'mnemostack[server]'`."
    ) from e

from mnemostack import __version__
from mnemostack.embeddings import get_provider
from mnemostack.llm import get_llm
from mnemostack.observability.recorder import (
    InMemoryRecorder,
    get_recorder,
    set_recorder,
)
from mnemostack.recall import (
    AnswerGenerator,
    BM25Retriever,
    MemgraphRetriever,
    Recaller,
    Reranker,
    TemporalRetriever,
    VectorRetriever,
    build_full_pipeline,
)
from mnemostack.recall.pipeline import FileStateStore
from mnemostack.vector import VectorStore

log = logging.getLogger(__name__)


# ----- Request / response models -----

class RecallRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query.")
    limit: int = Field(10, ge=1, le=100, description="Top-K memories to return.")
    full_pipeline: bool = Field(
        True,
        description="Apply the 8-stage recall pipeline. Set to False for raw RRF output.",
    )


class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(10, ge=1, le=100)
    full_pipeline: bool = Field(True)


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


class AnswerResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    sources: list[str]
    memories: list[Memory]


class HealthResponse(BaseModel):
    status: str
    version: str
    provider: str
    collection: str
    qdrant: bool
    memgraph: bool


# ----- Server construction -----

@dataclass
class ServerConfig:
    provider_name: str = "gemini"
    llm_name: str = "gemini"
    collection: str = "mnemostack"
    qdrant_url: str = "http://localhost:6333"
    graph_uri: str = "bolt://localhost:7687"
    graph_health_timeout: float = 1.0
    bm25_paths: list[str] | None = None  # optional markdown dirs for BM25 corpus
    state_path: str = "/tmp/mnemostack-server-state.json"

    @classmethod
    def from_env(cls) -> ServerConfig:
        return cls(
            provider_name=os.environ.get("MNEMOSTACK_PROVIDER", "gemini"),
            llm_name=os.environ.get("MNEMOSTACK_LLM", "gemini"),
            collection=os.environ.get("MNEMOSTACK_COLLECTION", "mnemostack"),
            qdrant_url=os.environ.get("MNEMOSTACK_QDRANT_URL", "http://localhost:6333"),
            graph_uri=os.environ.get("MNEMOSTACK_GRAPH_URI", "bolt://localhost:7687"),
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
                    docs.append(BM25Doc(
                        id=f"{f}:{i}",
                        text=chunk,
                        payload={"source": str(f), "offset": i},
                    ))
    return docs


def _memory_of(result) -> Memory:
    """Convert a mnemostack.recall.RecallResult into the HTTP response shape.

    Looks in .payload (the real field on RecallResult) first, with a fallback
    to .metadata for callers that pass a bare dict-like object (used in tests).
    """
    payload = getattr(result, "payload", None)
    if not payload:
        payload = getattr(result, "metadata", None) or {}
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

    for key, val in sorted(rec.counters.items()):
        name, labels = _from_key(key)
        prom = _safe_name(name) + "_total"
        if prom not in seen_help:
            lines.append(f"# HELP {prom} mnemostack counter: {name}")
            lines.append(f"# TYPE {prom} counter")
            seen_help.add(prom)
        lines.append(f"{prom}{_fmt_labels(labels)} {val}")

    for key, obs in sorted(rec.histograms.items()):
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

    provider = get_provider(cfg.provider_name)
    store = VectorStore(collection=cfg.collection, dimension=provider.dimension, host=cfg.qdrant_url)

    def _graph_ok() -> bool:
        if not cfg.graph_uri:
            return False
        try:
            from neo4j import GraphDatabase

            d = GraphDatabase.driver(
                cfg.graph_uri,
                connection_timeout=cfg.graph_health_timeout,
                connection_acquisition_timeout=cfg.graph_health_timeout,
            )
            with d.session() as s:
                s.run("RETURN 1", timeout=cfg.graph_health_timeout).single()
            d.close()
            return True
        except Exception:
            return False

    bm25_docs = _build_bm25_docs(cfg.bm25_paths)
    retrievers = [
        VectorRetriever(embedding=provider, vector_store=store),
        BM25Retriever(docs=bm25_docs) if bm25_docs else None,
        MemgraphRetriever(uri=cfg.graph_uri),
        TemporalRetriever(embedding=provider, vector_store=store),
    ]
    retrievers = [r for r in retrievers if r is not None]
    recaller = Recaller(retrievers=retrievers)

    from pathlib import Path

    state_path = Path(cfg.state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline = build_full_pipeline(
        state_store=FileStateStore(state_path),
        graph_uri=cfg.graph_uri,
    )

    try:
        llm = get_llm(cfg.llm_name)
        answer_gen: AnswerGenerator | None = AnswerGenerator(llm=llm)
        reranker: Reranker | None = Reranker(llm=llm, max_items=20)
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

    def _run_recall_sync(query: str, limit: int, full_pipeline: bool):
        raw_limit = max(limit * 3, 30) if full_pipeline else limit
        results = recaller.recall(query, limit=raw_limit)
        if full_pipeline:
            results = pipeline.apply(query, results)
            if reranker is not None:
                try:
                    results = reranker.rerank(query, results)
                except Exception as exc:  # pragma: no cover
                    log.warning("reranker failed (%s) — returning pre-rerank order", exc)
        return results[:limit]

    async def _run_recall(query: str, limit: int, full_pipeline: bool):
        """Offload the blocking recall stack to a worker thread.

        Recaller/pipeline/reranker all do blocking I/O or CPU work. Running
        them inline in the event loop would serialise every HTTP request
        behind the slowest retriever.
        """
        return await asyncio.to_thread(_run_recall_sync, query, limit, full_pipeline)

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
        # Ping-level reachability. Avoid `store.count()` because it requires
        # the collection to exist — a fresh deployment before any ingest
        # would be reported unhealthy. Use the underlying client so we only
        # check the HTTP endpoint is alive.
        qdrant_ok = False
        try:
            store.client.get_collections()
            qdrant_ok = True
        except Exception:
            qdrant_ok = False
        return HealthResponse(
            status="ok" if qdrant_ok else "degraded",
            version=__version__,
            provider=cfg.provider_name,
            collection=cfg.collection,
            qdrant=qdrant_ok,
            memgraph=_graph_ok(),
        )

    @app.post("/recall", response_model=RecallResponse)
    async def recall_endpoint(req: RecallRequest):
        try:
            results = await _run_recall(req.query, req.limit, req.full_pipeline)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"recall failed: {exc}") from exc
        return RecallResponse(query=req.query, results=[_memory_of(r) for r in results])

    @app.post("/answer", response_model=AnswerResponse)
    async def answer_endpoint(req: AnswerRequest):
        if answer_gen is None:
            raise HTTPException(
                status_code=503,
                detail="answer generator unavailable (LLM not configured)",
            )
        try:
            results = await _run_recall(req.query, req.limit, req.full_pipeline)
            ans = await asyncio.to_thread(answer_gen.generate, req.query, results)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"answer failed: {exc}") from exc
        return AnswerResponse(
            query=req.query,
            answer=ans.text,
            confidence=float(getattr(ans, "confidence", 0.0)),
            sources=list(getattr(ans, "sources", []) or []),
            memories=[_memory_of(r) for r in results],
        )

    return app
