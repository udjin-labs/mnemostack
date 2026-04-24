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

    def recall(self, query, limit=10, **_):
        self.calls.append((query, limit))
        return [FakeResult(id=str(i), text=f"m{i}") for i in range(min(limit, 3))]


class _FakePipeline:
    def apply(self, query, results):
        return results


def _patched_app(monkeypatch, with_answer: bool = True):
    """Build the FastAPI app with the heavy retrieval layers mocked out."""
    import mnemostack.server as srv

    monkeypatch.setattr(srv, "VectorStore", lambda **_: type("VS", (), {"count": lambda self: 0, "dimension": 3})())

    class _FakeProvider:
        dimension = 3

        def embed(self, text):  # unused; kept for shape
            return [0.0, 0.0, 0.0]

    monkeypatch.setattr(srv, "get_provider", lambda _name: _FakeProvider())

    fake_recaller = _FakeRecaller()
    monkeypatch.setattr(srv, "Recaller", lambda **_: fake_recaller)
    monkeypatch.setattr(srv, "VectorRetriever", lambda **_: object())
    monkeypatch.setattr(srv, "BM25Retriever", lambda **_: object())
    monkeypatch.setattr(srv, "MemgraphRetriever", lambda **_: object())
    monkeypatch.setattr(srv, "TemporalRetriever", lambda **_: object())
    monkeypatch.setattr(srv, "build_full_pipeline", lambda **_: _FakePipeline())
    monkeypatch.setattr(srv, "FileStateStore", lambda path: object())

    if with_answer:
        class _FakeAns:
            text = "42"
            confidence = 0.9
            sources = ["notes/a.md"]

        class _FakeAnswerGen:
            def __init__(self, llm):
                pass

            def generate(self, q, memories):
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
                return rs

        monkeypatch.setattr(srv, "Reranker", _FakeReranker)
        monkeypatch.setattr(srv, "get_llm", lambda _n: _FakeLLM())
    else:
        def _raise(*_a, **_kw):
            raise RuntimeError("no llm")
        monkeypatch.setattr(srv, "get_llm", _raise)

    app = build_app(ServerConfig(provider_name="fake", llm_name="fake"))
    return app, fake_recaller


def test_health_endpoint(monkeypatch):
    app, _ = _patched_app(monkeypatch)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["provider"] == "fake"
    assert "qdrant" in data and "memgraph" in data


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
