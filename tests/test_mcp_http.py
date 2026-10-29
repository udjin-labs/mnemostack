"""Authenticated stateless HTTP calls through the real MCP transport."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

pytest.importorskip("fastmcp")
from mcp import ClientSession  # noqa: E402
from mcp.client.streamable_http import streamable_http_client  # noqa: E402

from mnemostack.auth import FileKeyStore  # noqa: E402
from mnemostack.mcp import build_server  # noqa: E402


@asynccontextmanager
async def _client(app, key, *, modern=False):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), headers={"Authorization": f"Bearer {key}"},
    ) as http:
        async with streamable_http_client("http://fixture.local/mcp", http_client=http) as (read, write, *_):
            async with ClientSession(read, write) as client:
                if modern:
                    await client.discover()
                else:
                    await client.initialize()
                yield client


@pytest.fixture
def configured(tmp_path, monkeypatch):
    import mnemostack.mcp.server as srv

    store = FileKeyStore(tmp_path / "keys.json")
    alpha_id, alpha = store.issue("alpha", "read,write")
    _, beta = store.issue("beta", "read,write")
    _, reader = store.issue("alpha", "read")
    recorded = []

    def feedback(_pipeline, **kwargs):
        recorded.append((kwargs["tenant"], kwargs["hit_id"]))
        return SimpleNamespace(to_dict=lambda: {"ok": True})

    monkeypatch.setattr(srv, "apply_feedback", feedback)
    monkeypatch.setattr(srv, "build_full_pipeline", lambda **_: SimpleNamespace(
        apply=lambda query, results, **kwargs: results,
    ))
    monkeypatch.setattr(srv, "get_llm", lambda *a, **kw: object())
    monkeypatch.setattr(srv, "Reranker", lambda **_: SimpleNamespace(rerank=lambda query, results: results))
    monkeypatch.setattr(srv, "get_provider", lambda *a, **kw: SimpleNamespace(dimension=3))
    monkeypatch.setattr(srv, "VectorStore", lambda **_: MagicMock())
    monkeypatch.setattr(srv, "VectorRetriever", lambda **_: MagicMock())
    monkeypatch.setattr(srv, "TemporalRetriever", lambda **_: MagicMock())
    monkeypatch.setattr(srv, "build_bm25_docs", lambda _: [])

    class Recaller:
        def __init__(self, **kwargs):
            pass

        def recall(self, query, limit=10, **kwargs):
            from mnemostack.recall import RecallResult

            tenant = kwargs["tenant"]
            return [RecallResult(id=tenant, text=tenant, score=1, payload={"tenant_id": tenant})]

    monkeypatch.setattr(srv, "Recaller", Recaller)
    server = build_server(
        transport="http", auth_enabled=True, keys_file=str(store.path),
        state_path=str(tmp_path / "state.json"),
    )
    return SimpleNamespace(
        server=server, app=server.http_app(stateless_http=True, json_response=True),
        store=store, alpha=alpha, alpha_id=alpha_id, beta=beta, reader=reader, recorded=recorded,
    )


def test_http_requires_request_authentication():
    with pytest.raises(ValueError, match="requires service-key"):
        build_server(transport="http")
    with pytest.raises(ValueError, match="not a process API key"):
        build_server(transport="http", auth_enabled=True, api_key="fixture-process")


@pytest.mark.asyncio
async def test_http_modern_discovery_without_initialize(configured):
    if not hasattr(ClientSession, "discover"):
        pytest.skip("MCP SDK does not support the 2026-07-28 protocol")
    cfg = configured
    async with cfg.app.router.lifespan_context(cfg.app):
        async with _client(cfg.app, cfg.beta, modern=True) as client:
            assert client.protocol_version == "2026-07-28"
            catalog = await client.list_tools()
            assert "mnemostack_search" in {tool.name for tool in catalog.tools}
            result = await client.call_tool("mnemostack_feedback", {"hit_id": "modern", "signal": "useful"})
            assert result.structuredContent["ok"] is True
    assert cfg.recorded == [("beta", "modern")]


@pytest.mark.asyncio
async def test_http_missing_invalid_revoked_keys_fail_closed(configured):
    cfg = configured
    async with cfg.app.router.lifespan_context(cfg.app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=cfg.app)) as http:
            request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
            response = await http.post("http://fixture.local/mcp", json=request, headers={
                "Accept": "application/json, text/event-stream",
                "Authorization": f"Bearer {cfg.alpha}",
            })
            assert response.status_code == 200
            cfg.store.revoke(cfg.alpha_id)
            for key in (None, "fixture-invalid", cfg.alpha):
                headers = {"Accept": "application/json, text/event-stream"}
                if key is not None:
                    headers["Authorization"] = f"Bearer {key}"
                response = await http.post("http://fixture.local/mcp", json=request, headers=headers)
                assert response.status_code == 401
                assert "mcp-session-id" not in response.headers
    assert cfg.recorded == []


@pytest.mark.asyncio
async def test_http_parallel_clients_keep_tenants_and_scopes(configured):
    import asyncio

    cfg = configured

    async def call(key, tenant):
        async with _client(cfg.app, key) as client:
            catalog = await client.list_tools()
            assert "mnemostack_search" in {tool.name for tool in catalog.tools}
            result = await client.call_tool("mnemostack_search", {"query": "fixture"})
            assert result.structuredContent["ok"] is True, result.structuredContent
            assert [hit["id"] for hit in result.structuredContent["results"]] == [tenant]
            result = await client.call_tool("mnemostack_feedback", {"hit_id": tenant, "signal": "useful"})
            assert result.structuredContent["ok"] is True

    async with cfg.app.router.lifespan_context(cfg.app):
        await asyncio.gather(*(call(key, tenant) for _ in range(5) for key, tenant in (
            (cfg.alpha, "alpha"), (cfg.beta, "beta"),
        )))
        async with _client(cfg.app, cfg.reader) as client:
            result = await client.call_tool("mnemostack_feedback", {"hit_id": "forbidden", "signal": "useful"})
            assert result.structuredContent["ok"] is False
            assert "scope" in result.structuredContent["error"]
    assert sorted(cfg.recorded) == [("alpha", "alpha")] * 5 + [("beta", "beta")] * 5
