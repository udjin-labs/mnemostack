"""Tests for MCP server — verifies build_server wires up tools correctly."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_fastmcp = pytest.importorskip("fastmcp")

from mnemostack.mcp import build_server  # noqa: E402


def _list_tool_names(mcp) -> set[str]:
    """Extract registered tool names from a FastMCP instance."""
    tools = asyncio.run(mcp.list_tools())
    return {t.name for t in tools}


def test_build_server_returns_fastmcp():
    mcp = build_server(collection="test", embedding_provider="ollama")
    assert mcp is not None
    assert mcp.name == "mnemostack"


def test_build_server_registers_core_tools():
    mcp = build_server(collection="test", embedding_provider="ollama")
    names = _list_tool_names(mcp)
    assert "mnemostack_health" in names
    assert "mnemostack_search" in names
    assert "mnemostack_answer" in names


def test_build_server_without_memgraph_skips_graph_tools():
    mcp = build_server(collection="test", embedding_provider="ollama")
    names = _list_tool_names(mcp)
    assert "mnemostack_graph_query" not in names
    assert "mnemostack_graph_add_triple" not in names


def test_build_server_with_memgraph_adds_graph_tools():
    mcp = build_server(
        collection="test",
        embedding_provider="ollama",
        memgraph_uri="bolt://localhost:7687",
    )
    names = _list_tool_names(mcp)
    assert "mnemostack_graph_query" in names
    assert "mnemostack_graph_add_triple" in names


def test_mcp_search_strips_internal_vector_floor_payload(monkeypatch):
    import mnemostack.mcp.server as srv

    class _FakeEmbedding:
        dimension = 3

    class _FakeVectorStore:
        def __init__(self, **_):
            pass

    class _FakeRecaller:
        def __init__(self, **_):
            pass

        def recall(self, query, limit=10):
            return [
                SimpleNamespace(
                    id="a",
                    text="text",
                    score=0.9,
                    sources=["vector"],
                    payload={
                        "public": "ok",
                        "_vector_floor_candidates": [{"id": "hidden"}],
                    },
                )
            ]

    monkeypatch.setattr(srv, "get_provider", lambda *_args, **_kwargs: _FakeEmbedding())
    monkeypatch.setattr(srv, "VectorStore", _FakeVectorStore)
    monkeypatch.setattr(srv, "VectorRetriever", lambda **_: MagicMock())
    monkeypatch.setattr(srv, "TemporalRetriever", lambda **_: MagicMock())
    monkeypatch.setattr(srv, "build_bm25_docs", lambda _paths: [])
    monkeypatch.setattr(srv, "Recaller", _FakeRecaller)

    mcp = build_server(collection="test", embedding_provider="ollama", vector_floor=1)

    result = asyncio.run(mcp.call_tool("mnemostack_search", {"query": "q", "limit": 1}))

    payload = result.structured_content
    assert payload["results"][0]["payload"] == {"public": "ok"}


def test_mcp_search_passes_configured_rerank_mode(monkeypatch):
    import mnemostack.mcp.server as srv

    reranker_kwargs = {}

    class _FakeEmbedding:
        dimension = 3

    class _FakeVectorStore:
        def __init__(self, **_):
            pass

    class _FakeRecaller:
        def __init__(self, **_):
            pass

        def recall(self, query, limit=10):
            return [
                SimpleNamespace(id="a", text="first", score=0.9, sources=[], payload={}),
                SimpleNamespace(id="b", text="second", score=0.8, sources=[], payload={}),
            ]

    class _FakeReranker:
        def __init__(self, **kwargs):
            reranker_kwargs.update(kwargs)

        def rerank(self, query, results):
            return list(reversed(results))

    monkeypatch.setattr(srv, "get_provider", lambda *_args, **_kwargs: _FakeEmbedding())
    monkeypatch.setattr(srv, "get_llm", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(srv, "VectorStore", _FakeVectorStore)
    monkeypatch.setattr(srv, "VectorRetriever", lambda **_: MagicMock())
    monkeypatch.setattr(srv, "TemporalRetriever", lambda **_: MagicMock())
    monkeypatch.setattr(srv, "build_bm25_docs", lambda _paths: [])
    monkeypatch.setattr(srv, "Recaller", _FakeRecaller)
    monkeypatch.setattr(srv, "Reranker", _FakeReranker)

    mcp = build_server(
        collection="test",
        embedding_provider="ollama",
        rerank_mode="full_reorder",
    )

    result = asyncio.run(mcp.call_tool("mnemostack_search", {"query": "q", "limit": 2}))

    payload = result.structured_content
    assert reranker_kwargs["rerank_mode"] == "full_reorder"
    assert [item["id"] for item in payload["results"]] == ["b", "a"]


def test_mcp_answer_passes_configured_rerank_mode(monkeypatch):
    import mnemostack.mcp.server as srv

    reranker_kwargs = {}
    answer_memories = []

    class _FakeEmbedding:
        dimension = 3

    class _FakeVectorStore:
        def __init__(self, **_):
            pass

    class _FakeRecaller:
        def __init__(self, **_):
            pass

        def recall(self, query, limit=10):
            return [
                SimpleNamespace(id="a", text="first", score=0.9, sources=[], payload={}),
                SimpleNamespace(id="b", text="second", score=0.8, sources=[], payload={}),
            ]

    class _FakeReranker:
        def __init__(self, **kwargs):
            reranker_kwargs.update(kwargs)

        def rerank(self, query, results):
            return list(reversed(results))

    class _FakeAnswerGenerator:
        def __init__(self, **_):
            pass

        def generate(self, query, memories):
            answer_memories.extend(memories)
            return SimpleNamespace(
                ok=True,
                text="answer",
                confidence=0.8,
                sources=[],
                error=None,
            )

        def should_fallback(self, answer):
            return False

    monkeypatch.setattr(srv, "get_provider", lambda *_args, **_kwargs: _FakeEmbedding())
    monkeypatch.setattr(srv, "get_llm", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(srv, "VectorStore", _FakeVectorStore)
    monkeypatch.setattr(srv, "VectorRetriever", lambda **_: MagicMock())
    monkeypatch.setattr(srv, "TemporalRetriever", lambda **_: MagicMock())
    monkeypatch.setattr(srv, "build_bm25_docs", lambda _paths: [])
    monkeypatch.setattr(srv, "Recaller", _FakeRecaller)
    monkeypatch.setattr(srv, "Reranker", _FakeReranker)
    monkeypatch.setattr(srv, "AnswerGenerator", _FakeAnswerGenerator)

    mcp = build_server(
        collection="test",
        embedding_provider="ollama",
        rerank_mode="full_reorder",
    )

    result = asyncio.run(mcp.call_tool("mnemostack_answer", {"query": "q", "limit": 2}))

    assert result.structured_content["ok"] is True
    assert reranker_kwargs["rerank_mode"] == "full_reorder"
    assert [item.id for item in answer_memories] == ["b", "a"]
