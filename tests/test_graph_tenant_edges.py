"""Tenantless graph reads outside recall (#207): the MCP graph query tool and
the unauthenticated serve startup check fail closed over several tenants,
like the recall guard."""

from __future__ import annotations

import asyncio

import pytest

from mnemostack.graph.store import Triple
from mnemostack.recall import CrossTenantRecallError


class _FakeGraph:
    def __init__(self, sample):
        self.sample = sample
        self.probes = 0
        self.queries = 0

    def tenant_sample(self):
        self.probes += 1
        return self.sample

    def query_triples(self, **kw):
        self.queries += 1
        return [Triple(subject="a", predicate="uses", obj="b")]

    def close(self):
        pass


def _graph_query(monkeypatch, graph, **server_kw):
    pytest.importorskip("fastmcp")
    import mnemostack.graph.factory as factory
    from mnemostack.mcp import build_server

    monkeypatch.setattr(factory, "make_graph_store", lambda *a, **k: graph)
    mcp = build_server(
        collection="c", embedding_provider="ollama", memgraph_uri="bolt://x", **server_kw
    )
    result = asyncio.run(mcp.call_tool("mnemostack_graph_query", {"subject": "a"}))
    return result.structured_content


def test_mcp_graph_query_refuses_a_tenantless_query_over_several_tenants(monkeypatch):
    graph = _FakeGraph(["t1", "t2"])
    out = _graph_query(monkeypatch, graph)
    assert out["ok"] is False and "several tenants" in out["error"]
    assert graph.queries == 0


@pytest.mark.parametrize(
    "sample, server_kw",
    [
        (["t1"], {}),
        ([], {}),
        (None, {}),  # the graph could not answer: not a refusal on its own
        (["t1", "t2"], {"default_tenant": "t1"}),
        (["t1", "t2"], {"allow_cross_tenant": True}),
    ],
)
def test_mcp_graph_query_runs_when_safe(monkeypatch, sample, server_kw):
    graph = _FakeGraph(sample)
    out = _graph_query(monkeypatch, graph, **server_kw)
    assert out["ok"] is True and graph.queries == 1
    if server_kw:
        assert graph.probes == 0  # scoped or opted out: no probe


def test_mcp_graph_query_caches_a_conclusive_sample(monkeypatch):
    pytest.importorskip("fastmcp")
    import mnemostack.graph.factory as factory
    from mnemostack.mcp import build_server

    graph = _FakeGraph(["t1"])
    monkeypatch.setattr(factory, "make_graph_store", lambda *a, **k: graph)
    mcp = build_server(collection="c", embedding_provider="ollama", memgraph_uri="bolt://x")
    for _ in range(3):
        asyncio.run(mcp.call_tool("mnemostack_graph_query", {"subject": "a"}))
    assert graph.probes == 1


def _build_app_with_guard(monkeypatch, guard, **cfg_kw):
    pytest.importorskip("fastapi")
    import mnemostack.server as server_mod

    monkeypatch.setattr(
        server_mod, "get_provider", lambda *a, **k: type("P", (), {"dimension": 8})()
    )
    monkeypatch.setattr(server_mod.VectorStore, "__init__", lambda self, **kw: None, raising=False)
    monkeypatch.setattr(
        server_mod.VectorStore, "distinct_tenant_count", lambda self, limit=2: 1, raising=False
    )
    real = server_mod.Recaller

    class _Recaller(real):
        def _guard_tenantless_recall(self, sources=None):
            guard()

    monkeypatch.setattr(server_mod, "Recaller", _Recaller)
    return server_mod.build_app(server_mod.ServerConfig(provider_name="gemini", **cfg_kw))


def test_unauth_serve_refuses_when_the_stores_together_hold_several_tenants(monkeypatch):
    def guard():
        raise CrossTenantRecallError("graph t1 + collection t2")

    with pytest.raises(CrossTenantRecallError, match="collection, graph"):
        _build_app_with_guard(monkeypatch, guard)


def test_unauth_serve_starts_when_the_guard_passes(monkeypatch):
    calls = []
    try:
        _build_app_with_guard(monkeypatch, lambda: calls.append(1))
    except CrossTenantRecallError:
        pytest.fail("a single-tenant setup must not hit the tenant gate")
    assert calls == [1]


def test_scoped_serve_does_not_probe_at_startup(monkeypatch):
    calls = []
    try:
        _build_app_with_guard(monkeypatch, lambda: calls.append(1), default_tenant="t1")
    except CrossTenantRecallError:
        pytest.fail("a scoped server must not hit the tenant gate")
    assert calls == []


def test_graph_store_tenant_sample():
    from mnemostack.graph.store import GraphStore

    class _Result:
        def __init__(self, rows):
            self.rows = rows

        def data(self):
            return self.rows

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def run(self, cypher, **params):
            tenants = [t for t in ("t1", "t2") if t != params.get("t1")]
            return _Result([{"t": tenants[0]}])

    class _Driver:
        def session(self, **_):
            return _Session()

    class _Down:
        def session(self, **_):
            raise OSError("connection refused")

    store = GraphStore.__new__(GraphStore)
    store.database = None
    store.driver = _Driver()
    assert store.tenant_sample() == ["t1", "t2"]
    store.driver = _Down()
    assert store.tenant_sample() is None
