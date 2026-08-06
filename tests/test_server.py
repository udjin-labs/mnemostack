"""Smoke tests for the FastAPI server wrapper.

We don't spin up real Qdrant/Memgraph — we stub the retrieval layer so the
tests stay pure and fast. The goal is to catch wiring and contract bugs:
request/response shapes, HTTP codes, and graceful degradation when an
LLM / graph backend is missing.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient

from mnemostack.server import (
    Memory,
    RecallResponse,
    ServerConfig,
    _memory_of,
    build_app,
)


@dataclass
class FakeResult:
    id: str
    text: str
    score: float = 0.42
    payload: dict[str, Any] | None = None
    source: str | None = None
    sources: list[str] | None = None


def test_memory_of_translates_results():
    r = FakeResult(
        id="1",
        text="hello",
        score=0.5,
        payload={"source": "notes/a.md"},
        sources=["vector", "bm25"],
    )
    m = _memory_of(r)
    assert isinstance(m, Memory)
    assert m.id == "1"
    assert m.text == "hello"
    assert m.score == 0.5
    assert m.source == "notes/a.md"
    assert m.retrievers == ["vector", "bm25"]
    assert m.metadata == {"source": "notes/a.md"}


def test_memory_of_prefers_source_file_when_source_absent():
    r = FakeResult(id="2", text="t", payload={"source_file": "transcript:abc.jsonl"})
    m = _memory_of(r)
    assert m.source == "transcript:abc.jsonl"


def test_recall_response_round_trip():
    # Pydantic contract: RecallResponse accepts a list of Memory and serialises.
    r = RecallResponse(
        query="q",
        results=[Memory(id="a", text="t", score=0.1, source="s", metadata={"k": 1})],
    )
    data = r.model_dump()
    assert data["query"] == "q"
    assert data["results"][0]["id"] == "a"


class _FakeRecaller:
    def __init__(self):
        self.calls = []
        self.last_filters = None
        self.last_tenant = "__unset__"

    def recall(self, query, limit=10, filters=None, tenant=None, **_):
        self.calls.append((query, limit))
        self.last_filters = filters
        self.last_tenant = tenant
        return [FakeResult(id=str(i), text=f"m{i}") for i in range(min(limit, 3))]

    def apply_vector_floor_after_rerank(self, results, recalled_results):
        return results


class _FakePipeline:
    def __init__(self, stages=None):
        self.stages = list(stages or [])

    def apply(self, query, results, **_):
        return results

    def __iter__(self):
        return iter(self.stages)


def _patched_app(
    monkeypatch,
    with_answer: bool = True,
    pipeline: _FakePipeline | None = None,
    config: ServerConfig | None = None,
    qdrant_reachable: bool = False,
    embedding_healthy: bool = True,
):
    """Build the FastAPI app with the heavy retrieval layers mocked out.

    `qdrant_reachable` controls the dedicated health/readiness probe client:
    False (default) -> the probe raises, so /readyz is 503 and /health degraded;
    True -> the probe succeeds. Readiness keys off this probe, not the store.
    `embedding_healthy` controls the provider's health_check (both gate /readyz).
    """
    import mnemostack.server as srv

    monkeypatch.setattr(
        srv, "VectorStore", lambda **_: type("VS", (), {"count": lambda self: 0, "dimension": 3})()
    )

    def _fake_probe(*_a, **_k):
        class _Probe:
            def get_collections(self):
                if not qdrant_reachable:
                    raise ConnectionError("qdrant down")
                return object()

        return _Probe()

    monkeypatch.setattr(srv, "_make_probe_client", _fake_probe)

    class _FakeProvider:
        dimension = 3

        def embed(self, text):  # unused; kept for shape
            return [0.0, 0.0, 0.0]

        def health_check(self):
            if embedding_healthy:
                return True, "ok"
            return False, "embedding backend down"

    monkeypatch.setattr(srv, "get_provider", lambda _name, **_kwargs: _FakeProvider())

    fake_recaller = _FakeRecaller()
    monkeypatch.setattr(srv, "Recaller", lambda **_: fake_recaller)
    monkeypatch.setattr(srv, "VectorRetriever", lambda **_: object())
    monkeypatch.setattr(srv, "BM25Retriever", lambda **_: object())
    monkeypatch.setattr(srv, "MemgraphRetriever", lambda **_: object())
    monkeypatch.setattr(srv, "TemporalRetriever", lambda **_: object())
    fake_pipeline = pipeline or _FakePipeline()
    monkeypatch.setattr(srv, "build_full_pipeline", lambda **_: fake_pipeline)
    monkeypatch.setattr(srv, "FileStateStore", lambda path: object())

    if with_answer:

        class _FakeAns:
            text = "42"
            confidence = 0.9
            sources = ["notes/a.md"]

        class _FakeAnswerGen:
            last_kwargs = None
            last_generate_kwargs = None

            def __init__(self, **kwargs):
                type(self).last_kwargs = kwargs

            def generate(self, q, memories, **kwargs):
                type(self).last_generate_kwargs = kwargs
                return _FakeAns()

        class _FakeLLM:
            def generate(self, *a, **kw):
                from mnemostack.llm.base import LLMResponse

                return LLMResponse(text="ok")

        monkeypatch.setattr(srv, "AnswerGenerator", _FakeAnswerGen)

        class _FakeReranker:
            def __init__(self, **_):
                pass

            def rerank(self, q, rs):
                return list(rs)  # successful no-op rerank returns a new list

        monkeypatch.setattr(srv, "Reranker", _FakeReranker)
        monkeypatch.setattr(srv, "get_llm", lambda _n, **_kwargs: _FakeLLM())
    else:

        def _raise(*_a, **_kw):
            raise RuntimeError("no llm")

        monkeypatch.setattr(srv, "get_llm", _raise)

    app = build_app(
        config
        or ServerConfig(
            provider_name="fake",
            llm_name="fake",
            graph_uri=None,
        )
    )
    return app, fake_recaller


def test_build_app_wires_recaller_into_answer_generator(monkeypatch):
    import mnemostack.server as srv

    app, recaller = _patched_app(monkeypatch)

    assert app is not None
    assert srv.AnswerGenerator.last_kwargs["recaller"] is recaller


def test_health_endpoint(monkeypatch):
    app, _ = _patched_app(monkeypatch)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["provider"] == "fake"
    assert "qdrant" in data and "memgraph" in data


def _readyz_until(client, want_embedding: bool, tries: int = 200):
    """Poll /readyz until embedding readiness settles.

    Embedding readiness is refreshed in a background thread (serve-stale-while-
    revalidate), so the first probe reports the sentinel `embedding=False` before
    the (instant, faked) refresh lands. Poll until it reaches the expected state.
    """
    import time

    r = client.get("/readyz")
    for _ in range(tries):
        if r.json().get("embedding") is want_embedding:
            return r
        time.sleep(0.01)
        r = client.get("/readyz")
    return r


def test_healthz_liveness_always_ok(monkeypatch):
    # Liveness must not depend on Qdrant/graph — up even with no backends.
    app, _ = _patched_app(monkeypatch)
    r = TestClient(app).get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["version"]


def test_readyz_503_when_qdrant_unreachable(monkeypatch):
    # Probe client raises (as a real outage would) -> get_collections fails -> not ready.
    app, _ = _patched_app(monkeypatch, qdrant_reachable=False)
    r = TestClient(app).get("/readyz")
    assert r.status_code == 503
    assert r.json()["status"] == "not_ready"
    assert r.json()["qdrant"] is False
    # Graph is intentionally absent from readiness — no field, no ping.
    assert "memgraph" not in r.json()


def test_readyz_200_when_qdrant_reachable(monkeypatch):
    app, _ = _patched_app(monkeypatch, qdrant_reachable=True)
    r = _readyz_until(TestClient(app), want_embedding=True)
    assert r.status_code == 200
    assert r.json()["status"] == "ready"
    assert r.json()["qdrant"] is True
    assert r.json()["embedding"] is True


def test_readyz_503_when_embedding_down(monkeypatch):
    # Qdrant is reachable but the embedding provider is down -> not ready, since
    # recall can't embed a query. Embedding is a hard readiness dependency.
    app, _ = _patched_app(monkeypatch, qdrant_reachable=True, embedding_healthy=False)
    # embedding stays False (down); the background refresh confirms not-ready.
    r = _readyz_until(TestClient(app), want_embedding=False)
    assert r.status_code == 503
    assert r.json()["status"] == "not_ready"
    assert r.json()["qdrant"] is True
    assert r.json()["embedding"] is False


def test_status_reports_config_and_counters(monkeypatch):
    app, _ = _patched_app(monkeypatch)
    r = TestClient(app).get("/status")
    assert r.status_code == 200
    d = r.json()
    assert d["provider"] == "fake"
    assert d["collection"]
    assert "recall_calls" in d and "degraded_events" in d
    # graph_configured is intentionally not exposed (config conflates unset with
    # the localhost default, so the field would be misleading).
    assert "graph_configured" not in d


def test_status_degraded_events_counts_only_serving_degradations(monkeypatch):
    from mnemostack.observability import counter

    app, _ = _patched_app(monkeypatch)
    # A real serving-path degradation is counted...
    counter("mnemostack.recall.fallback_triggered", 2)
    # ...but an unrelated ingest failure (which could leak in via a shared
    # recorder) and a routine no-parse are not.
    counter("mnemostack.ingest.failed", 5)
    counter("mnemostack.recall.temporal_no_parse", 3)
    d = TestClient(app).get("/status").json()
    assert d["degraded_events"] == 2


def test_status_counts_trace_only_degradations(monkeypatch):
    # Reranker unavailable / retriever failures live only on the per-call trace;
    # they must still reach /status via the mirrored degradation counter.
    from mnemostack.recall import RecallTrace

    app, _ = _patched_app(monkeypatch)
    tr = RecallTrace()
    tr.mark("reranker:unavailable")
    tr.mark("retriever:bm25:failed")
    tr.mark("temporal:no_parse")  # routine — must NOT count
    tr.mark("reranker:unavailable")  # deduped per trace — must NOT double-count
    d = TestClient(app).get("/status").json()
    assert d["degraded_events"] == 2


def test_status_counts_answer_unavailable_outage(monkeypatch):
    # When the LLM isn't configured, POST /answer 503s before the generator can
    # emit answer.errors — the outage must still surface in degraded_events.
    app, _ = _patched_app(monkeypatch, with_answer=False)
    client = TestClient(app)
    assert client.post("/answer", json={"query": "hi"}).status_code == 503
    assert client.get("/status").json()["degraded_events"] >= 1


def test_build_app_passes_configured_provider_models(monkeypatch):
    import mnemostack.server as srv

    calls = {"embedding": None, "llm": None}

    monkeypatch.setattr(
        srv,
        "VectorStore",
        lambda **_: type("VS", (), {"count": lambda self: 0, "dimension": 3})(),
    )

    class _FakeProvider:
        dimension = 3

        def embed(self, text):
            return [0.0, 0.0, 0.0]

    def _get_provider(name, **kwargs):
        calls["embedding"] = (name, kwargs)
        return _FakeProvider()

    class _FakeLLM:
        pass

    def _get_llm(name, **kwargs):
        calls["llm"] = (name, kwargs)
        return _FakeLLM()

    monkeypatch.setattr(srv, "get_provider", _get_provider)
    monkeypatch.setattr(srv, "get_llm", _get_llm)
    monkeypatch.setattr(srv, "Recaller", lambda **_: _FakeRecaller())
    monkeypatch.setattr(srv, "VectorRetriever", lambda **_: object())
    monkeypatch.setattr(srv, "BM25Retriever", lambda **_: object())
    monkeypatch.setattr(srv, "MemgraphRetriever", lambda **_: object())
    monkeypatch.setattr(srv, "TemporalRetriever", lambda **_: object())
    monkeypatch.setattr(srv, "build_full_pipeline", lambda **_: _FakePipeline())
    monkeypatch.setattr(srv, "FileStateStore", lambda path: object())
    monkeypatch.setattr(srv, "AnswerGenerator", lambda **_: object())
    monkeypatch.setattr(srv, "Reranker", lambda **_: object())

    build_app(
        ServerConfig(
            provider_name="fake",
            embedding_model="embed-custom",
            llm_name="fake-llm",
            llm_model="llm-custom",
        )
    )

    assert calls["embedding"] == ("fake", {"model": "embed-custom"})
    assert calls["llm"] == ("fake-llm", {"model": "llm-custom"})


def test_build_app_passes_configured_rerank_mode(monkeypatch):
    import mnemostack.server as srv

    reranker_kwargs = {}

    class _FakeProvider:
        dimension = 3

        def embed(self, text):
            return [0.0, 0.0, 0.0]

    def _fake_reranker(**kwargs):
        reranker_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(srv, "get_provider", lambda _name, **_kwargs: _FakeProvider())
    monkeypatch.setattr(srv, "get_llm", lambda _name, **_kwargs: object())
    monkeypatch.setattr(srv, "VectorStore", lambda **_: object())
    monkeypatch.setattr(srv, "Recaller", lambda **_: _FakeRecaller())
    monkeypatch.setattr(srv, "VectorRetriever", lambda **_: object())
    monkeypatch.setattr(srv, "BM25Retriever", lambda **_: object())
    monkeypatch.setattr(srv, "MemgraphRetriever", lambda **_: object())
    monkeypatch.setattr(srv, "TemporalRetriever", lambda **_: object())
    monkeypatch.setattr(srv, "build_full_pipeline", lambda **_: _FakePipeline())
    monkeypatch.setattr(srv, "FileStateStore", lambda path: object())
    monkeypatch.setattr(srv, "AnswerGenerator", lambda **_: object())
    monkeypatch.setattr(srv, "Reranker", _fake_reranker)

    build_app(
        ServerConfig(
            provider_name="fake",
            llm_name="fake-llm",
            rerank_mode="full_reorder",
        )
    )

    assert reranker_kwargs["rerank_mode"] == "full_reorder"


def test_recall_endpoint(monkeypatch):
    app, recaller = _patched_app(monkeypatch)
    client = TestClient(app)
    r = client.post("/recall", json={"query": "hello", "limit": 2, "full_pipeline": False})
    assert r.status_code == 200
    data = r.json()
    assert data["query"] == "hello"
    assert len(data["results"]) == 2
    assert data["results"][0]["id"] == "0"
    # Validates the recaller was invoked
    assert recaller.calls


def test_recall_endpoint_applies_vector_floor_after_rerank_and_slice(monkeypatch):
    import mnemostack.server as srv

    class _FloorRecaller:
        def recall(self, query, limit=10, **_):
            return [
                FakeResult(id="lexical-1", text="lexical winner"),
                FakeResult(id="lexical-2", text="lexical runner up"),
                FakeResult(id="vector-strong", text="vector-only match"),
            ]

        def apply_vector_floor_after_rerank(self, results, recalled_results):
            ids = {result.id for result in results}
            output = list(results)
            for result in recalled_results:
                if result.id == "vector-strong" and result.id not in ids:
                    output.append(result)
            return output

    class _DroppingReranker:
        def __init__(self, **_):
            pass

        def rerank(self, q, rs):
            return [result for result in rs if result.id != "vector-strong"]

    monkeypatch.setattr(
        srv, "VectorStore", lambda **_: type("VS", (), {"count": lambda self: 0, "dimension": 3})()
    )

    class _FakeProvider:
        dimension = 3

        def embed(self, text):
            return [0.0, 0.0, 0.0]

    monkeypatch.setattr(srv, "get_provider", lambda _name, **_kwargs: _FakeProvider())
    monkeypatch.setattr(srv, "Recaller", lambda **_: _FloorRecaller())
    monkeypatch.setattr(srv, "VectorRetriever", lambda **_: object())
    monkeypatch.setattr(srv, "BM25Retriever", lambda **_: object())
    monkeypatch.setattr(srv, "MemgraphRetriever", lambda **_: object())
    monkeypatch.setattr(srv, "TemporalRetriever", lambda **_: object())
    monkeypatch.setattr(srv, "build_full_pipeline", lambda **_: _FakePipeline())
    monkeypatch.setattr(srv, "FileStateStore", lambda path: object())
    monkeypatch.setattr(srv, "AnswerGenerator", lambda **_: object())
    monkeypatch.setattr(srv, "Reranker", _DroppingReranker)
    monkeypatch.setattr(srv, "get_llm", lambda _n, **_kwargs: object())

    app = srv.build_app(ServerConfig(provider_name="fake", llm_name="fake"))
    client = TestClient(app)

    r = client.post("/recall", json={"query": "hello", "limit": 2, "full_pipeline": True})

    assert r.status_code == 200
    ids = [result["id"] for result in r.json()["results"]]
    assert ids == ["lexical-1", "lexical-2", "vector-strong"]


def test_recall_endpoint_can_auto_record_ior(monkeypatch):
    class _IorStage:
        def __init__(self):
            self.recorded = []

        def record_recall(self, memory_id, tenant=None):
            self.recorded.append(memory_id)

    ior = _IorStage()
    app, _ = _patched_app(
        monkeypatch,
        pipeline=_FakePipeline([ior]),
        config=ServerConfig(
            provider_name="fake",
            llm_name="fake",
            auto_record_ior=True,
        ),
    )
    client = TestClient(app)
    r = client.post("/recall", json={"query": "hello", "limit": 2, "full_pipeline": False})

    assert r.status_code == 200
    assert ior.recorded == ["0", "1"]


def test_feedback_endpoint_records_qlearning_and_clicked_ior(monkeypatch):
    class _IorStage:
        def __init__(self):
            self.recorded = []

        def record_recall(self, memory_id, tenant=None):
            self.recorded.append(memory_id)

    class _QStage:
        def __init__(self):
            self.feedback = []

        def record_feedback(self, source, query_type, reward, tenant=None):
            self.feedback.append((source, query_type, reward))

    ior = _IorStage()
    qstage = _QStage()
    app, _ = _patched_app(monkeypatch, pipeline=_FakePipeline([ior, qstage]))
    client = TestClient(app)
    r = client.post(
        "/feedback",
        json={
            "hit_id": "mem-1",
            "signal": "clicked",
            "query": "how to configure nginx",
            "sources": ["vector", "bm25", "vector"],
        },
    )

    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["reward"] == 0.7
    assert data["query_type"] == "technical"
    assert data["ior_recorded"] is True
    assert data["q_learning_updates"] == 2
    assert ior.recorded == ["mem-1"]
    assert qstage.feedback == [
        ("vector", "technical", 0.7),
        ("bm25", "technical", 0.7),
    ]


def test_feedback_endpoint_accepts_reward_override(monkeypatch):
    class _QStage:
        def __init__(self):
            self.feedback = []

        def record_feedback(self, source, query_type, reward, tenant=None):
            self.feedback.append((source, query_type, reward))

    qstage = _QStage()
    app, _ = _patched_app(monkeypatch, pipeline=_FakePipeline([qstage]))
    client = TestClient(app)
    r = client.post(
        "/feedback",
        json={
            "hit_id": "mem-2",
            "signal": "irrelevant",
            "query_type": "person",
            "source": "graph",
            "reward": 0.25,
        },
    )

    assert r.status_code == 200
    assert r.json()["reward"] == 0.25
    assert qstage.feedback == [("graph", "person", 0.25)]


def test_recall_endpoint_scrubs_internal_exception(monkeypatch):
    app, recaller = _patched_app(monkeypatch)

    def _raise(*_args, **_kwargs):
        raise RuntimeError("secret backend token leaked")

    recaller.recall = _raise
    client = TestClient(app)
    r = client.post("/recall", json={"query": "hello"})

    assert r.status_code == 500
    assert r.json()["detail"] == "recall failed"
    assert "secret backend token" not in r.text


def test_answer_endpoint(monkeypatch):
    app, _ = _patched_app(monkeypatch)
    client = TestClient(app)
    r = client.post("/answer", json={"query": "what is 42", "limit": 3})
    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "42"
    assert data["confidence"] == 0.9
    assert data["sources"] == ["notes/a.md"]


def test_answer_disabled_when_llm_missing(monkeypatch):
    app, _ = _patched_app(monkeypatch, with_answer=False)
    client = TestClient(app)
    r = client.post("/answer", json={"query": "x"})
    assert r.status_code == 503
    assert "answer generator unavailable" in r.text


def test_metrics_endpoint_emits_prometheus_format(monkeypatch):
    """/metrics must render counters and histograms in Prometheus text format."""
    from mnemostack.observability.recorder import counter, histogram

    app, _ = _patched_app(monkeypatch)
    # build_app installed a fresh InMemoryRecorder. Seed it *after* app build
    # and hit /metrics once so we don't exercise the (mocked) recall path.
    counter("mnemostack.test.ops", 3)
    with histogram("mnemostack.test.latency_ms"):
        pass

    client = TestClient(app)
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "# HELP mnemostack_test_ops_total" in body
    assert "# TYPE mnemostack_test_ops_total counter" in body
    assert "mnemostack_test_ops_total 3" in body
    # Histogram: summary type with sum + count + quantiles
    assert "# TYPE mnemostack_test_latency_ms summary" in body
    assert "mnemostack_test_latency_ms_sum " in body
    assert "mnemostack_test_latency_ms_count 1" in body
    assert 'quantile="0.5"' in body


def test_recall_response_degraded_default_empty(monkeypatch):
    app, _ = _patched_app(monkeypatch)
    client = TestClient(app)
    resp = client.post("/recall", json={"query": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["degraded"] == []
    assert data["notes"] == []  # HTTP surface mirrors the MCP payload shape
    assert data["trace"] is None


def test_recall_no_parse_surfaces_as_note_not_degradation(monkeypatch):
    """HTTP mirror of the MCP regression: a date-less query is routine —
    the response reports a healthy call with a note, never a degradation."""
    app, recaller = _patched_app(monkeypatch)
    real_recall = recaller.recall

    def marking_recall(query, limit=10, *, trace=None, **kw):
        if trace is not None:
            trace.mark("temporal:no_parse")  # temporal retriever's explain_empty
        return real_recall(query, limit, **kw)

    monkeypatch.setattr(recaller, "recall", marking_recall)
    client = TestClient(app)
    resp = client.post("/recall", json={"query": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["degraded"] == []
    assert data["notes"] == ["temporal:no_parse"]


def test_recall_trace_opt_in(monkeypatch):
    app, _ = _patched_app(monkeypatch)
    client = TestClient(app)
    resp = client.post("/recall", json={"query": "hello", "include_trace": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["trace"] is not None
    assert "retrievers" in data["trace"]
    assert "fused" in data["trace"]


def test_recall_degraded_on_reranker_failure(monkeypatch):
    import mnemostack.server as srv

    app, _ = _patched_app(monkeypatch)

    # The app closed over an instance of the fake reranker class that
    # _patched_app installed as srv.Reranker — break its rerank method.
    def _boom(self, q, rs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(srv.Reranker, "rerank", _boom)
    client = TestClient(app)
    resp = client.post("/recall", json={"query": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert "reranker:fallback" in data["degraded"]
    assert data["results"]  # fail-open: results still served


def test_recall_degraded_when_reranker_unavailable(monkeypatch):
    app, _ = _patched_app(monkeypatch, with_answer=False)
    client = TestClient(app)

    resp = client.post("/recall", json={"query": "hello"})

    assert resp.status_code == 200
    data = resp.json()
    assert "reranker:unavailable" in data["degraded"]
    assert data["results"]  # fail-open: results still served


def test_raw_recall_does_not_report_reranker_unavailable(monkeypatch):
    """full_pipeline=false turns the reranker off by request — that is not a
    degradation and must not be tagged as one."""
    app, _ = _patched_app(monkeypatch, with_answer=False)
    client = TestClient(app)

    resp = client.post("/recall", json={"query": "hello", "full_pipeline": False})

    assert resp.status_code == 200
    data = resp.json()
    assert data["degraded"] == []
    assert data["results"]


def test_answer_response_carries_degraded(monkeypatch):
    app, _ = _patched_app(monkeypatch)
    client = TestClient(app)
    resp = client.post("/answer", json={"query": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["degraded"] == []
    assert "trace" in data and data["trace"] is None


def test_recall_endpoint_threads_filters_to_recaller(monkeypatch):
    app, recaller = _patched_app(monkeypatch, with_answer=False)
    client = TestClient(app)

    resp = client.post(
        "/recall", json={"query": "hello", "filters": {"tenant": "a"}}
    )

    assert resp.status_code == 200
    assert recaller.last_filters == {"tenant": "a"}


def test_answer_endpoint_threads_filters_to_recall_and_generator(monkeypatch):
    import mnemostack.server as srv

    app, recaller = _patched_app(monkeypatch)
    client = TestClient(app)

    resp = client.post(
        "/answer", json={"query": "hello", "filters": {"tenant": "b"}}
    )

    assert resp.status_code == 200
    assert recaller.last_filters == {"tenant": "b"}
    # retry sub-recalls must stay inside the same filtered scope
    assert srv.AnswerGenerator.last_generate_kwargs == {
        "recall_filters": {"tenant": "b"},
        "token_budget": None,
        "include_invalidated": False,
        "as_of": None,
        "tenant": None,
    }


def _auth_app(monkeypatch, tmp_path, **kw):
    from mnemostack.auth import FileKeyStore

    ks = FileKeyStore(tmp_path / "keys.json")
    _, read_key = ks.issue("alpha", ["read"])
    _, write_key = ks.issue("alpha", ["write"])
    _, admin_key = ks.issue("alpha", ["admin"])
    cfg = ServerConfig(
        provider_name="fake",
        llm_name="fake",
        graph_uri=None,
        auth_enabled=True,
        keys_file=str(tmp_path / "keys.json"),
    )
    app, recaller = _patched_app(monkeypatch, config=cfg, **kw)
    return app, recaller, read_key, write_key, admin_key


def test_feedback_endpoint_threads_tenant_under_auth(monkeypatch, tmp_path):
    # /feedback records into the key's tenant partition of the learning state.
    import mnemostack.server as srv
    from mnemostack.feedback import FeedbackOutcome

    captured: dict[str, object] = {}

    def _spy(_pipeline, **kw):
        captured.update(kw)
        return FeedbackOutcome(
            hit_id=kw["hit_id"], signal=kw["signal"], reward=1.0,
            query_type="general", ior_recorded=False, q_learning_updates=0,
        )

    monkeypatch.setattr(srv, "apply_feedback", _spy)
    app, _rec, _rk, write_key, _ak = _auth_app(monkeypatch, tmp_path)
    client = TestClient(app)
    r = client.post(
        "/feedback", json={"hit_id": "h", "signal": "useful"},
        headers={"X-API-Key": write_key},
    )
    assert r.status_code == 200
    assert captured["tenant"] == "alpha"  # the key's tenant, threaded into the state


def test_recall_requires_key_when_auth_enabled(monkeypatch, tmp_path):
    app, *_ = _auth_app(monkeypatch, tmp_path)
    client = TestClient(app)
    # default-deny: no key -> 401
    assert client.post("/recall", json={"query": "x"}).status_code == 401
    # invalid key -> 401
    r = client.post("/recall", json={"query": "x"}, headers={"X-API-Key": "msk_bogus"})
    assert r.status_code == 401


def test_recall_valid_key_scopes_tenant(monkeypatch, tmp_path):
    app, recaller, read_key, *_ = _auth_app(monkeypatch, tmp_path)
    client = TestClient(app)
    r = client.post("/recall", json={"query": "x"}, headers={"X-API-Key": read_key})
    assert r.status_code == 200
    assert recaller.last_tenant == "alpha"  # tenant resolved from the key


def test_recall_rate_limited_returns_429(monkeypatch, tmp_path):
    # A tenant with a max_rps quota is throttled: burst of 1, so the second
    # request in the same instant gets 429 with a Retry-After header.
    from mnemostack.auth import FileKeyStore
    from mnemostack.quotas import FileQuotaStore

    ks = FileKeyStore(tmp_path / "keys.json")
    _, read_key = ks.issue("alpha", ["read"])
    FileQuotaStore(tmp_path / "quotas.json").set("alpha", max_rps=1.0, burst=1)
    cfg = ServerConfig(
        provider_name="fake",
        llm_name="fake",
        graph_uri=None,
        auth_enabled=True,
        keys_file=str(tmp_path / "keys.json"),
        quotas_file=str(tmp_path / "quotas.json"),
    )
    app, _rec = _patched_app(monkeypatch, config=cfg)
    client = TestClient(app)
    h = {"X-API-Key": read_key}
    assert client.post("/recall", json={"query": "x"}, headers=h).status_code == 200
    r = client.post("/recall", json={"query": "x"}, headers=h)
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1


def test_recall_tiny_rate_returns_429_not_500(monkeypatch, tmp_path):
    # A vanishingly small but valid max_rps makes 1/rate overflow to inf; the
    # Retry-After header must be capped, not crash header building (500).
    from mnemostack.auth import FileKeyStore
    from mnemostack.quotas import FileQuotaStore

    ks = FileKeyStore(tmp_path / "keys.json")
    _, read_key = ks.issue("alpha", ["read"])
    FileQuotaStore(tmp_path / "quotas.json").set("alpha", max_rps=1e-309)  # burst -> 1
    cfg = ServerConfig(
        provider_name="fake",
        llm_name="fake",
        graph_uri=None,
        auth_enabled=True,
        keys_file=str(tmp_path / "keys.json"),
        quotas_file=str(tmp_path / "quotas.json"),
    )
    app, _rec = _patched_app(monkeypatch, config=cfg)
    client = TestClient(app)
    h = {"X-API-Key": read_key}
    assert client.post("/recall", json={"query": "x"}, headers=h).status_code == 200
    r = client.post("/recall", json={"query": "x"}, headers=h)
    assert r.status_code == 429  # not 500
    assert int(r.headers["Retry-After"]) >= 1  # finite, capped


def test_recall_not_rate_limited_without_quota(monkeypatch, tmp_path):
    # A tenant with no rate quota is never throttled, even under a burst.
    from mnemostack.auth import FileKeyStore
    from mnemostack.quotas import FileQuotaStore

    ks = FileKeyStore(tmp_path / "keys.json")
    _, read_key = ks.issue("alpha", ["read"])
    FileQuotaStore(tmp_path / "quotas.json").set("alpha", max_points=10)  # size only
    cfg = ServerConfig(
        provider_name="fake",
        llm_name="fake",
        graph_uri=None,
        auth_enabled=True,
        keys_file=str(tmp_path / "keys.json"),
        quotas_file=str(tmp_path / "quotas.json"),
    )
    app, _rec = _patched_app(monkeypatch, config=cfg)
    client = TestClient(app)
    h = {"X-API-Key": read_key}
    for _ in range(5):
        assert client.post("/recall", json={"query": "x"}, headers=h).status_code == 200


def test_recall_accepts_bearer_scheme(monkeypatch, tmp_path):
    app, recaller, read_key, *_ = _auth_app(monkeypatch, tmp_path)
    client = TestClient(app)
    r = client.post(
        "/recall", json={"query": "x"}, headers={"Authorization": f"Bearer {read_key}"}
    )
    assert r.status_code == 200
    assert recaller.last_tenant == "alpha"


def test_answer_threads_tenant_from_key(monkeypatch, tmp_path):
    app, recaller, read_key, *_ = _auth_app(monkeypatch, tmp_path)
    client = TestClient(app)
    r = client.post("/answer", json={"query": "x"}, headers={"X-API-Key": read_key})
    assert r.status_code == 200
    assert recaller.last_tenant == "alpha"  # /answer's recall is tenant-scoped too


def test_write_key_rejected_on_read_endpoint(monkeypatch, tmp_path):
    app, _rc, _read, write_key, _admin = _auth_app(monkeypatch, tmp_path)
    client = TestClient(app)
    # a write-only key lacks `read` -> 403 on /recall
    r = client.post("/recall", json={"query": "x"}, headers={"X-API-Key": write_key})
    assert r.status_code == 403


def test_admin_key_satisfies_read_and_write(monkeypatch, tmp_path):
    app, recaller, _read, _write, admin_key = _auth_app(monkeypatch, tmp_path, with_answer=False)
    client = TestClient(app)
    assert (
        client.post("/recall", json={"query": "x"}, headers={"X-API-Key": admin_key}).status_code
        == 200
    )
    fb = client.post(
        "/feedback", json={"hit_id": "h1", "signal": "useful"}, headers={"X-API-Key": admin_key}
    )
    assert fb.status_code == 200


def test_feedback_requires_write_scope(monkeypatch, tmp_path):
    app, _rc, read_key, write_key, _admin = _auth_app(monkeypatch, tmp_path, with_answer=False)
    client = TestClient(app)
    body = {"hit_id": "h1", "signal": "useful"}
    # a read-only key is rejected for a write endpoint (403)
    assert client.post("/feedback", json=body, headers={"X-API-Key": read_key}).status_code == 403
    # a write key is accepted
    assert client.post("/feedback", json=body, headers={"X-API-Key": write_key}).status_code == 200


def test_x_api_key_wins_when_both_headers_present(monkeypatch, tmp_path):
    app, recaller, read_key, *_ = _auth_app(monkeypatch, tmp_path)
    client = TestClient(app)
    r = client.post(
        "/recall",
        json={"query": "x"},
        headers={"X-API-Key": read_key, "Authorization": "Bearer msk_garbage"},
    )
    assert r.status_code == 200  # X-API-Key (valid) is used, not the bad Bearer


def test_probes_stay_public_under_auth(monkeypatch, tmp_path):
    # Liveness/readiness must not require a key (k8s probes can't auth).
    app, *_ = _auth_app(monkeypatch, tmp_path)
    client = TestClient(app)
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code in (200, 503)  # reachable without a key


def test_auth_disabled_is_tenantless_and_open(monkeypatch):
    # Default (auth off): no key needed, tenant stays None (single-tenant).
    app, recaller = _patched_app(monkeypatch)
    r = TestClient(app).post("/recall", json={"query": "x"})
    assert r.status_code == 200
    assert recaller.last_tenant is None


def test_recall_endpoint_reports_tokens_estimate(monkeypatch):
    app, _ = _patched_app(monkeypatch, with_answer=False)
    client = TestClient(app)

    r = client.post("/recall", json={"query": "q", "limit": 3, "full_pipeline": False})

    assert r.status_code == 200
    data = r.json()
    # 3 fake results, each "m{i}" — 2 ASCII chars ≈ 1 estimated token
    assert data["tokens_estimate"] == 3


def test_recall_endpoint_token_budget_trims_results(monkeypatch):
    app, _ = _patched_app(monkeypatch, with_answer=False)
    client = TestClient(app)

    r = client.post(
        "/recall",
        json={"query": "q", "limit": 3, "full_pipeline": False, "token_budget": 2},
    )

    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) == 2
    assert data["tokens_estimate"] == 2


def test_recall_endpoint_rejects_non_positive_token_budget(monkeypatch):
    app, _ = _patched_app(monkeypatch, with_answer=False)
    client = TestClient(app)

    r = client.post("/recall", json={"query": "q", "token_budget": 0})

    assert r.status_code == 422


def test_recall_endpoint_uses_server_default_token_budget(monkeypatch):
    app, _ = _patched_app(
        monkeypatch,
        with_answer=False,
        config=ServerConfig(
            provider_name="fake",
            llm_name="fake",
            graph_uri=None,
            token_budget=1,
        ),
    )
    client = TestClient(app)

    r = client.post("/recall", json={"query": "q", "limit": 3, "full_pipeline": False})

    assert r.status_code == 200
    data = r.json()
    # Server-wide default budget of 1 token keeps only the first 1-token hit.
    assert len(data["results"]) == 1

    # An explicit per-request budget overrides the server default.
    r = client.post(
        "/recall",
        json={"query": "q", "limit": 3, "full_pipeline": False, "token_budget": 3},
    )
    assert len(r.json()["results"]) == 3


def test_answer_endpoint_token_budget_trims_memories_and_reports_tokens(monkeypatch):
    app, _ = _patched_app(monkeypatch)
    client = TestClient(app)

    r = client.post(
        "/answer",
        json={"query": "q", "limit": 3, "full_pipeline": False, "token_budget": 2},
    )

    assert r.status_code == 200
    data = r.json()
    assert len(data["memories"]) == 2
    assert data["tokens_estimate"] == 2
    # The fake answer object carries no provider usage — surfaced as null.
    assert data["tokens_used"] is None
    # The budget must reach the generator too, or its retry-path sub-recalls
    # would prompt unbudgeted.
    import mnemostack.server as srv

    assert srv.AnswerGenerator.last_generate_kwargs["token_budget"] == 2


def test_recall_endpoint_include_invalidated_and_as_of_threaded(monkeypatch):
    app, recaller = _patched_app(monkeypatch, with_answer=False)
    client = TestClient(app)

    captured = {}
    orig = recaller.recall

    def _spy(query, limit=10, filters=None, **kwargs):
        captured.update(kwargs)
        return orig(query, limit=limit, filters=filters)

    recaller.recall = _spy
    resp = client.post(
        "/recall",
        json={"query": "q", "full_pipeline": False,
              "include_invalidated": True, "as_of": "2026-03-01"},
    )
    assert resp.status_code == 200
    assert captured.get("include_invalidated") is True
    assert captured.get("as_of") == "2026-03-01"


def test_auth_works_with_external_verify_only_keystore(monkeypatch, tmp_path):
    # The HTTP auth path must depend only on the KeyStore Protocol (verify), so a
    # verify-only external backend (e.g. OpenBao) is a drop-in via the factory.
    from mnemostack.auth import SCOPES, Principal

    class _VerifyOnly:
        def verify(self, key):
            if key == "msk_ok":
                return Principal(tenant="alpha", scopes=frozenset(SCOPES))
            return None

    monkeypatch.setattr("mnemostack.auth.make_key_store", lambda *_a, **_k: _VerifyOnly())
    cfg = ServerConfig(
        provider_name="fake", llm_name="fake", graph_uri=None, auth_enabled=True,
        quotas_file=str(tmp_path / "quotas.json"),
    )
    app, recaller = _patched_app(monkeypatch, config=cfg)
    client = TestClient(app)
    assert client.post("/recall", json={"query": "x"}).status_code == 401
    r = client.post("/recall", json={"query": "x"}, headers={"X-API-Key": "msk_ok"})
    assert r.status_code == 200
    assert recaller.last_tenant == "alpha"  # tenant resolved via the external store
