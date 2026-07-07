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

        def recall(self, query, limit=10, **kwargs):
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
    result_payload = payload["results"][0]["payload"]
    # pipeline stages may annotate the payload (freshness, q_value, ...);
    # the contract is: public keys present, internal floor metadata stripped
    assert result_payload["public"] == "ok"
    assert "_vector_floor_candidates" not in result_payload


def test_mcp_search_preserves_vector_floor_after_rerank_slice(monkeypatch):
    import mnemostack.mcp.server as srv

    class _FakeEmbedding:
        dimension = 3

    class _FakeVectorStore:
        def __init__(self, **_):
            pass

    class _FakeRecaller:
        def __init__(self, **_):
            pass

        def recall(self, query, limit=10, **kwargs):
            return [
                SimpleNamespace(id="a", text="winner", score=0.9, sources=[], payload={}),
                SimpleNamespace(
                    id="v",
                    text="protected",
                    score=0.7,
                    sources=["vector"],
                    payload={"raw_vector_score": 0.95},
                ),
            ]

        def apply_vector_floor_after_rerank(self, results, recalled_results):
            return results + [r for r in recalled_results if r.id == "v"]

    class _FakeReranker:
        def __init__(self, **_):
            pass

        def rerank(self, query, results):
            return list(results)  # successful no-op rerank returns a new list

    monkeypatch.setattr(srv, "get_provider", lambda *_args, **_kwargs: _FakeEmbedding())
    monkeypatch.setattr(srv, "get_llm", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(srv, "VectorStore", _FakeVectorStore)
    monkeypatch.setattr(srv, "VectorRetriever", lambda **_: MagicMock())
    monkeypatch.setattr(srv, "TemporalRetriever", lambda **_: MagicMock())
    monkeypatch.setattr(srv, "build_bm25_docs", lambda _paths: [])
    monkeypatch.setattr(srv, "Recaller", _FakeRecaller)
    monkeypatch.setattr(srv, "Reranker", _FakeReranker)

    mcp = build_server(collection="test", embedding_provider="ollama", vector_floor=1)

    result = asyncio.run(mcp.call_tool("mnemostack_search", {"query": "q", "limit": 1}))

    payload = result.structured_content
    assert [item["id"] for item in payload["results"]] == ["a", "v"]


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

        def recall(self, query, limit=10, **kwargs):
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

        def recall(self, query, limit=10, **kwargs):
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

        def generate(self, query, memories, **kwargs):
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


def _patch_minimal(monkeypatch, srv, recaller_cls, reranker_cls=None):
    class _FakeEmbedding:
        dimension = 3

    class _FakeVectorStore:
        def __init__(self, **_):
            pass

    class _PassReranker:
        def __init__(self, **_):
            pass

        def rerank(self, query, results):
            return list(results)  # successful no-op rerank returns a new list

    monkeypatch.setattr(srv, "get_provider", lambda *_args, **_kwargs: _FakeEmbedding())
    monkeypatch.setattr(srv, "get_llm", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(srv, "VectorStore", _FakeVectorStore)
    monkeypatch.setattr(srv, "VectorRetriever", lambda **_: MagicMock())
    monkeypatch.setattr(srv, "TemporalRetriever", lambda **_: MagicMock())
    monkeypatch.setattr(srv, "build_bm25_docs", lambda _paths: [])
    monkeypatch.setattr(srv, "Recaller", recaller_cls)
    monkeypatch.setattr(srv, "Reranker", reranker_cls or _PassReranker)


class _OneHitRecaller:
    def __init__(self, **_):
        pass

    def recall(self, query, limit=10, **kwargs):
        return [SimpleNamespace(id="a", text="text", score=0.9, sources=["vector"], payload={})]


def test_mcp_search_returns_degraded_empty_when_healthy(monkeypatch):
    import mnemostack.mcp.server as srv

    _patch_minimal(monkeypatch, srv, _OneHitRecaller)
    mcp = build_server(collection="test", embedding_provider="ollama")

    result = asyncio.run(mcp.call_tool("mnemostack_search", {"query": "q"}))
    payload = result.structured_content

    assert payload["ok"] is True
    assert payload["degraded"] == []
    assert "trace" not in payload


def test_mcp_search_trace_opt_in(monkeypatch):
    import mnemostack.mcp.server as srv

    _patch_minimal(monkeypatch, srv, _OneHitRecaller)
    mcp = build_server(collection="test", embedding_provider="ollama")

    result = asyncio.run(
        mcp.call_tool("mnemostack_search", {"query": "q", "include_trace": True})
    )
    payload = result.structured_content

    assert payload["ok"] is True
    assert "trace" in payload
    assert "fused" in payload["trace"]


def test_mcp_search_degraded_on_reranker_failure(monkeypatch):
    import mnemostack.mcp.server as srv

    class _BoomReranker:
        def __init__(self, **_):
            pass

        def rerank(self, query, results):
            raise RuntimeError("llm down")

    _patch_minimal(monkeypatch, srv, _OneHitRecaller, _BoomReranker)
    mcp = build_server(collection="test", embedding_provider="ollama")

    result = asyncio.run(mcp.call_tool("mnemostack_search", {"query": "q"}))
    payload = result.structured_content

    assert payload["ok"] is True  # fail-open
    assert payload["degraded"] == ["reranker:fallback"]
    assert payload["results"]


def test_mcp_answer_carries_degraded(monkeypatch):
    import mnemostack.mcp.server as srv

    class _FakeAnswerGen:
        def __init__(self, **_):
            pass

        def generate(self, query, memories, **kwargs):
            return SimpleNamespace(
                ok=True, text="42", confidence=0.9, sources=["s"], error=None
            )

        def should_fallback(self, answer):
            return False

    _patch_minimal(monkeypatch, srv, _OneHitRecaller)
    monkeypatch.setattr(srv, "AnswerGenerator", _FakeAnswerGen)
    mcp = build_server(collection="test", embedding_provider="ollama")

    result = asyncio.run(mcp.call_tool("mnemostack_answer", {"query": "q"}))
    payload = result.structured_content

    assert payload["ok"] is True
    assert payload["degraded"] == []


def test_main_state_path_defaults_to_none(monkeypatch):
    """`mnemostack-mcp` must not pin the legacy /tmp path: with the env var
    unset, build_server receives None and resolves default_state_path()."""
    import mnemostack.mcp.server as srv

    captured: dict = {}

    def _fake_build_server(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(run=lambda: None)

    monkeypatch.setattr(srv, "build_server", _fake_build_server)
    monkeypatch.delenv("MNEMOSTACK_STATE_PATH", raising=False)

    srv.main()

    assert captured["state_path"] is None


def test_main_state_path_respects_env(monkeypatch):
    import mnemostack.mcp.server as srv

    captured: dict = {}

    def _fake_build_server(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(run=lambda: None)

    monkeypatch.setattr(srv, "build_server", _fake_build_server)
    monkeypatch.setenv("MNEMOSTACK_STATE_PATH", "/custom/state.json")

    srv.main()

    assert captured["state_path"] == "/custom/state.json"


def test_mcp_search_threads_filters_to_recaller(monkeypatch):
    import mnemostack.mcp.server as srv

    captured = {}

    class _FilterCapturingRecaller:
        def __init__(self, **_):
            pass

        def recall(self, query, limit=10, filters=None, **kwargs):
            captured["filters"] = filters
            return [
                SimpleNamespace(id="a", text="text", score=0.9, sources=["vector"], payload={})
            ]

    _patch_minimal(monkeypatch, srv, _FilterCapturingRecaller)
    mcp = build_server(collection="test", embedding_provider="ollama")

    result = asyncio.run(
        mcp.call_tool("mnemostack_search", {"query": "q", "filters": {"tenant": "a"}})
    )

    assert result.structured_content["ok"] is True
    assert captured["filters"] == {"tenant": "a"}


class _ThreeHitRecaller:
    def __init__(self, **_):
        pass

    def recall(self, query, limit=10, **kwargs):
        # each text is 4 ASCII chars ≈ 1 estimated token
        return [
            SimpleNamespace(
                id=str(i), text="tttt", score=0.9 - i * 0.1, sources=["vector"], payload={}
            )
            for i in range(min(limit, 3))
        ]


def test_mcp_search_reports_tokens_estimate(monkeypatch):
    import mnemostack.mcp.server as srv

    _patch_minimal(monkeypatch, srv, _ThreeHitRecaller)
    mcp = build_server(collection="test", embedding_provider="ollama")

    result = asyncio.run(mcp.call_tool("mnemostack_search", {"query": "q", "limit": 3}))
    payload = result.structured_content

    assert payload["ok"] is True
    assert payload["tokens_estimate"] == 3


def test_mcp_search_token_budget_trims_results(monkeypatch):
    import mnemostack.mcp.server as srv

    _patch_minimal(monkeypatch, srv, _ThreeHitRecaller)
    mcp = build_server(collection="test", embedding_provider="ollama")

    result = asyncio.run(
        mcp.call_tool("mnemostack_search", {"query": "q", "limit": 3, "token_budget": 2})
    )
    payload = result.structured_content

    assert payload["ok"] is True
    assert payload["count"] == 2
    assert payload["tokens_estimate"] == 2


def test_mcp_server_default_token_budget_applies(monkeypatch):
    import mnemostack.mcp.server as srv

    _patch_minimal(monkeypatch, srv, _ThreeHitRecaller)
    mcp = build_server(collection="test", embedding_provider="ollama", token_budget=1)

    result = asyncio.run(mcp.call_tool("mnemostack_search", {"query": "q", "limit": 3}))
    assert result.structured_content["count"] == 1

    # per-call budget overrides the server-wide default
    result = asyncio.run(
        mcp.call_tool("mnemostack_search", {"query": "q", "limit": 3, "token_budget": 3})
    )
    assert result.structured_content["count"] == 3


def test_mcp_answer_reports_tokens(monkeypatch):
    import mnemostack.mcp.server as srv

    class _FakeAnswerGen:
        def __init__(self, **_):
            pass

        def generate(self, query, memories, **kwargs):
            return SimpleNamespace(
                ok=True, text="42", confidence=0.9, sources=["s"], error=None, tokens_used=123
            )

        def should_fallback(self, answer):
            return False

    _patch_minimal(monkeypatch, srv, _ThreeHitRecaller)
    monkeypatch.setattr(srv, "AnswerGenerator", _FakeAnswerGen)
    mcp = build_server(collection="test", embedding_provider="ollama")

    result = asyncio.run(
        mcp.call_tool("mnemostack_answer", {"query": "q", "limit": 3, "token_budget": 2})
    )
    payload = result.structured_content

    assert payload["ok"] is True
    assert payload["tokens_estimate"] == 2
    assert payload["tokens_used"] == 123


def test_mcp_answer_prefers_generator_context_estimate(monkeypatch):
    import mnemostack.mcp.server as srv

    class _FakeAnswerGen:
        def __init__(self, **_):
            pass

        def generate(self, query, memories, **kwargs):
            return SimpleNamespace(
                ok=True,
                text="42",
                confidence=0.9,
                sources=["s"],
                error=None,
                tokens_used=None,
                # the retry paths can answer from a different pool than the
                # primary recall — the surface must report this value
                context_tokens_estimate=77,
            )

        def should_fallback(self, answer):
            return False

    _patch_minimal(monkeypatch, srv, _ThreeHitRecaller)
    monkeypatch.setattr(srv, "AnswerGenerator", _FakeAnswerGen)
    mcp = build_server(collection="test", embedding_provider="ollama")

    result = asyncio.run(mcp.call_tool("mnemostack_answer", {"query": "q", "limit": 3}))
    assert result.structured_content["tokens_estimate"] == 77


def test_mcp_invalidate_tool_marks_ids(monkeypatch):
    import mnemostack.mcp.server as srv

    class _RecordingVector:
        def __init__(self, **_):
            self.calls = []

        def invalidate(self, ids, *, invalidated_at=None, valid_until=None, index_root=None):
            self.calls.append((list(ids), invalidated_at, valid_until))
            return len(ids)

    vec = _RecordingVector()

    class _FakeEmbedding:
        dimension = 3

    monkeypatch.setattr(srv, "get_provider", lambda *_a, **_k: _FakeEmbedding())
    monkeypatch.setattr(srv, "VectorStore", lambda **_: vec)
    mcp = build_server(collection="test", embedding_provider="ollama")

    result = asyncio.run(
        mcp.call_tool(
            "mnemostack_invalidate",
            {"ids": ["a", "b"], "valid_until": "2026-06-01"},
        )
    )
    payload = result.structured_content
    assert payload["ok"] is True
    assert payload["requested"] == 2
    assert payload["invalidated"] == 2
    assert vec.calls == [(["a", "b"], None, "2026-06-01")]


def test_mcp_invalidate_tool_registered():
    mcp = build_server(collection="test", embedding_provider="ollama")
    names = _list_tool_names(mcp)
    assert "mnemostack_invalidate" in names


def test_mcp_search_threads_validity_params(monkeypatch):
    import mnemostack.mcp.server as srv

    captured = {}

    class _CapturingRecaller:
        def __init__(self, **_):
            pass

        def recall(self, query, limit=10, filters=None, include_invalidated=False,
                   as_of=None, **_):
            captured["include_invalidated"] = include_invalidated
            captured["as_of"] = as_of
            return [SimpleNamespace(id="a", text="t", score=0.9, sources=["v"], payload={})]

        def apply_vector_floor_after_rerank(self, results, recalled):
            return results

    _patch_minimal(monkeypatch, srv, _CapturingRecaller)
    mcp = build_server(collection="test", embedding_provider="ollama")

    asyncio.run(
        mcp.call_tool(
            "mnemostack_search",
            {"query": "q", "include_invalidated": True, "as_of": "2026-03-01"},
        )
    )
    assert captured == {"include_invalidated": True, "as_of": "2026-03-01"}


def test_mcp_invalidate_works_without_embedding_provider(monkeypatch):
    import mnemostack.mcp.server as srv

    class _RecordingVector:
        def __init__(self, **_):
            self.called = False

        def invalidate(self, ids, **_):
            self.called = True
            return len(ids)

    vec = _RecordingVector()

    def _no_provider(*_a, **_k):
        raise RuntimeError("GEMINI_API_KEY not set")

    # Embedding provider is unavailable — invalidation is payload-only and must
    # still work (it builds a dimension-only store, no embeddings).
    monkeypatch.setattr(srv, "get_provider", _no_provider)
    monkeypatch.setattr(srv, "VectorStore", lambda **_: vec)
    mcp = build_server(collection="test", embedding_provider="ollama")

    result = asyncio.run(mcp.call_tool("mnemostack_invalidate", {"ids": ["7"]}))
    payload = result.structured_content
    assert payload["ok"] is True
    assert payload["invalidated"] == 1
    assert vec.called is True


def test_mcp_invalidate_accepts_numeric_ids(monkeypatch):
    import mnemostack.mcp.server as srv

    class _RecordingVector:
        def __init__(self, **_):
            self.ids = None

        def invalidate(self, ids, **_):
            self.ids = list(ids)
            return len(ids)

    vec = _RecordingVector()

    class _FakeEmbedding:
        dimension = 3

    monkeypatch.setattr(srv, "get_provider", lambda *_a, **_k: _FakeEmbedding())
    monkeypatch.setattr(srv, "VectorStore", lambda **_: vec)
    mcp = build_server(collection="test", embedding_provider="ollama")

    # JSON integer ids must not be rejected by the tool schema, and reach
    # invalidate as ints (matching numeric-id Qdrant collections).
    result = asyncio.run(mcp.call_tool("mnemostack_invalidate", {"ids": [1, 2]}))
    assert result.structured_content["ok"] is True
    assert vec.ids == [1, 2]


def test_mcp_invalidate_passes_index_root(monkeypatch):
    import mnemostack.mcp.server as srv

    class _RecordingVector:
        def __init__(self, **_):
            self.kwargs = None

        def invalidate(self, ids, **kwargs):
            self.kwargs = kwargs
            return len(ids)

    vec = _RecordingVector()

    class _FakeEmbedding:
        dimension = 3

    monkeypatch.setattr(srv, "get_provider", lambda *_a, **_k: _FakeEmbedding())
    monkeypatch.setattr(srv, "VectorStore", lambda **_: vec)
    mcp = build_server(collection="test", embedding_provider="ollama")

    asyncio.run(mcp.call_tool(
        "mnemostack_invalidate", {"ids": ["a"], "index_root": "/root/A"}
    ))
    assert vec.kwargs["index_root"] == "/root/A"


# --- service-key auth (multi-tenant) ---


def _auth_mcp(tmp_path, monkeypatch, *, scopes="read", tenant="alpha", rec=None, memgraph=None):
    import mnemostack.mcp.server as srv
    from mnemostack.auth import FileKeyStore

    monkeypatch.setattr(srv, "get_provider", lambda *a, **k: SimpleNamespace(dimension=3))
    monkeypatch.setattr(srv, "VectorStore", lambda **_: MagicMock())
    monkeypatch.setattr(srv, "VectorRetriever", lambda **_: MagicMock())
    monkeypatch.setattr(srv, "TemporalRetriever", lambda **_: MagicMock())
    monkeypatch.setattr(srv, "build_bm25_docs", lambda _paths: [])

    class _Rec:
        def __init__(self, **_):
            pass

        def recall(self, query, limit=10, **kwargs):
            if rec is not None:
                rec["tenant"] = kwargs.get("tenant")
            return []

    monkeypatch.setattr(srv, "Recaller", _Rec)
    ks = tmp_path / "keys.json"
    _kid, key = FileKeyStore(ks).issue(tenant, scopes)
    return build_server(
        collection="test",
        embedding_provider="ollama",
        memgraph_uri=memgraph,
        auth_enabled=True,
        api_key=key,
        keys_file=str(ks),
    )


def test_mcp_auth_boot_validates_key(tmp_path):
    from mnemostack.auth import FileKeyStore

    with pytest.raises(ValueError):  # auth on but no key
        build_server(collection="t", embedding_provider="ollama", auth_enabled=True, api_key=None)
    ks = tmp_path / "keys.json"
    FileKeyStore(ks).issue("alpha", "read")
    with pytest.raises(ValueError):  # auth on but wrong key
        build_server(
            collection="t",
            embedding_provider="ollama",
            auth_enabled=True,
            api_key="msk_wrong",
            keys_file=str(ks),
        )


def test_mcp_auth_scope_enforced(tmp_path, monkeypatch):
    mcp = _auth_mcp(tmp_path, monkeypatch, scopes="read")  # read only
    s = asyncio.run(mcp.call_tool("mnemostack_search", {"query": "q", "limit": 1}))
    assert s.structured_content["ok"] is True  # read scope: allowed
    f = asyncio.run(mcp.call_tool("mnemostack_feedback", {"hit_id": "x", "signal": "useful"}))
    assert f.structured_content["ok"] is False and "scope" in f.structured_content["error"]


def test_mcp_auth_threads_tenant_into_feedback(tmp_path, monkeypatch):
    # mnemostack_feedback records into the key's tenant partition of the learning
    # state, so one tenant's feedback can't move another's ranking.
    import mnemostack.mcp.server as srv

    captured: dict[str, object] = {}

    def _spy(_pipeline, **kw):
        captured.update(kw)
        return SimpleNamespace(to_dict=lambda: {"ok": True})

    monkeypatch.setattr(srv, "apply_feedback", _spy)
    mcp = _auth_mcp(tmp_path, monkeypatch, tenant="acme", scopes="write")
    asyncio.run(mcp.call_tool("mnemostack_feedback", {"hit_id": "h", "signal": "useful"}))
    assert captured["tenant"] == "acme"


def test_mcp_auth_threads_tenant_into_recall(tmp_path, monkeypatch):
    rec = {}
    mcp = _auth_mcp(tmp_path, monkeypatch, tenant="acme", rec=rec)
    asyncio.run(mcp.call_tool("mnemostack_search", {"query": "q", "limit": 1}))
    assert rec["tenant"] == "acme"  # recall scoped to the key's tenant


def test_mcp_auth_revoked_key_denies_mid_session(tmp_path, monkeypatch):
    import mnemostack.mcp.server as srv
    from mnemostack.auth import FileKeyStore

    monkeypatch.setattr(srv, "get_provider", lambda *a, **k: SimpleNamespace(dimension=3))
    monkeypatch.setattr(srv, "VectorStore", lambda **_: MagicMock())
    monkeypatch.setattr(srv, "VectorRetriever", lambda **_: MagicMock())
    monkeypatch.setattr(srv, "TemporalRetriever", lambda **_: MagicMock())
    monkeypatch.setattr(srv, "build_bm25_docs", lambda _paths: [])
    monkeypatch.setattr(srv, "Recaller", lambda **_: SimpleNamespace(recall=lambda *a, **k: []))
    ks = tmp_path / "keys.json"
    store = FileKeyStore(ks)
    kid, key = store.issue("alpha", "read")
    mcp = build_server(
        collection="t", embedding_provider="ollama", auth_enabled=True, api_key=key, keys_file=str(ks)
    )
    store.revoke(kid)  # revoked after boot — per-call re-verify must catch it
    r = asyncio.run(mcp.call_tool("mnemostack_search", {"query": "q"}))
    assert r.structured_content["ok"] is False and "revoked" in r.structured_content["error"]


def test_mcp_graph_tools_scoped_under_tenant(tmp_path, monkeypatch):
    # Graph tools are tenant-scoped now (not fail-closed): the caller's tenant is
    # threaded into the structured query/write so it only ever touches its own
    # tenant's subgraph.
    import mnemostack.graph.factory as gf

    rec: dict[str, str | None] = {}

    class _FakeGS:
        def query_triples(self, **kw):
            rec["q_tenant"] = kw.get("tenant")
            return []

        def add_triple(self, **kw):
            rec["w_tenant"] = kw.get("tenant")

        def close(self):
            pass

    monkeypatch.setattr(gf, "make_graph_store", lambda *a, **k: _FakeGS())
    # admin implies read + write, so one key exercises both tools.
    mcp = _auth_mcp(tmp_path, monkeypatch, tenant="acme", scopes="admin", memgraph="bolt://x")

    q = asyncio.run(mcp.call_tool("mnemostack_graph_query", {}))
    assert q.structured_content["ok"] is True
    assert rec["q_tenant"] == "acme"  # read query confined to the key's tenant

    w = asyncio.run(
        mcp.call_tool(
            "mnemostack_graph_add_triple", {"subject": "a", "predicate": "KNOWS", "obj": "b"}
        )
    )
    assert w.structured_content["ok"] is True
    assert rec["w_tenant"] == "acme"  # write stamped with the key's tenant


def test_mcp_graph_add_triple_needs_write_scope(tmp_path, monkeypatch):
    # A read-only key can't write graph facts even though the tool is un-gated.
    import mnemostack.graph.factory as gf

    monkeypatch.setattr(gf, "make_graph_store", lambda *a, **k: MagicMock())
    mcp = _auth_mcp(tmp_path, monkeypatch, tenant="acme", scopes="read", memgraph="bolt://x")
    w = asyncio.run(
        mcp.call_tool(
            "mnemostack_graph_add_triple", {"subject": "a", "predicate": "KNOWS", "obj": "b"}
        )
    )
    assert w.structured_content["ok"] is False and "scope" in w.structured_content["error"]


def test_mcp_auth_recall_pipeline_keeps_scoped_graph(tmp_path, monkeypatch):
    # The graph is tenant-scoped now, so under auth the recall pipeline KEEPS the
    # graph: GraphResurrection confines its walk to the caller's tenant (via the
    # pipeline context) rather than being dropped entirely.
    import mnemostack.mcp.server as srv

    real = srv.build_full_pipeline
    captured = {}

    def _spy(**kw):
        captured.setdefault("graph_uri", kw.get("graph_uri"))
        return real(**kw)

    monkeypatch.setattr(srv, "build_full_pipeline", _spy)
    mcp = _auth_mcp(tmp_path, monkeypatch, memgraph="bolt://x")
    asyncio.run(mcp.call_tool("mnemostack_search", {"query": "q", "limit": 1}))
    assert captured["graph_uri"] == "bolt://x"
