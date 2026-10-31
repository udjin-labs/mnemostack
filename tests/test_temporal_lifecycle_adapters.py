"""Lifecycle visibility through real CLI, HTTP and MCP recall paths."""

import asyncio
import json
import os
import socket
from datetime import datetime, timezone

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance

from mnemostack.embeddings.base import EmbeddingProvider
from mnemostack.ingest import IngestItem, Ingestor, stable_chunk_id
from mnemostack.recall import InMemoryStateStore
from mnemostack.vector import VectorStore

START = "2026-01-01T00:00:00Z"
TRANSITION = "2026-02-01T00:00:00Z"
RETRACTED = "2026-02-03T12:00:00Z"
NOW = datetime(2026, 3, 1, 12, tzinfo=timezone.utc)
QUERY = "Who owns project P?"
TENANT = "tenant-a"
LIMIT = 20


class _Embedding(EmbeddingProvider):
    name = "lifecycle-fixture"
    dimension = 4

    def embed(self, text):
        return [1.0, 0.0, 0.0, 0.0]

    def embed_batch(self, texts):
        return [self.embed(text) for text in texts]


class _NoGeneration:
    def generate(self, *args, **kwargs):
        pytest.fail("Lifecycle regression tests must not generate answers")


class _SuccessfulReranker:
    def __init__(self):
        self.calls = 0

    def rerank(self, query, results):
        self.calls += 1
        return list(results)


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    import dotenv

    import mnemostack.ingest as ingest
    from mnemostack.recall.pipeline import stages

    for key in tuple(os.environ):
        if key.startswith(("MNEMOSTACK_", "OLLAMA_", "GEMINI_", "GOOGLE_", "OPENAI_", "ANTHROPIC_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("MNEMOSTACK_CONFIG", str(tmp_path / "absent-config.yaml"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    # An existing state file prevents migration from the shared legacy path.
    state_dir = tmp_path / "state" / "mnemostack"
    state_dir.mkdir(parents=True)
    (state_dir / "server-state.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    monkeypatch.setenv("FASTMCP_CHECK_FOR_UPDATES", "off")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: False)
    monkeypatch.setattr(dotenv, "find_dotenv", lambda *a, **kw: "")

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)

    monkeypatch.setattr(ingest, "datetime", Clock)
    monkeypatch.setattr(stages, "datetime", Clock)
    attempts = []

    def no_network(*args, **kwargs):
        attempts.append(True)
        pytest.fail("Lifecycle regression tests must not connect to services")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    yield
    assert not attempts, "A component attempted an undeclared network fallback"


@pytest.fixture
def corpus(isolated):
    store = VectorStore.__new__(VectorStore)
    store.client = QdrantClient(":memory:")
    store.collection = "lifecycle"
    store.dimension = 4
    store.distance = Distance.COSINE
    store.sparse_text = False
    store.text_key = "text"
    store.search_params = None
    store._sparse_encoder = None
    store.ensure_collection()
    items = {
        "old": IngestItem(
            text="Project P owner is Alice", source="p-owner-v1.txt", timestamp=START,
            metadata={"project": "P", "valid_from": START, "valid_until": TRANSITION},
        ),
        "current": IngestItem(
            text="Project P owner is Bob", source="p-owner-v2.txt", timestamp=TRANSITION,
            metadata={"project": "P", "valid_from": TRANSITION},
        ),
        "other_project": IngestItem(
            text="Project Q owner is Carol", source="q-owner.txt", timestamp="2026-02-28T00:00:00Z",
            metadata={"project": "Q"},
        ),
    }
    ids = {}
    try:
        for tenant in (TENANT, "tenant-b"):
            writer = Ingestor(embedding=_Embedding(), vector_store=store, tenant=tenant)
            for name, item in items.items():
                # The writer, not caller metadata, owns the tenant partition.
                supplied = IngestItem(
                    text=item.text, source=item.source, timestamp=item.timestamp,
                    metadata={**item.metadata, "tenant_id": TENANT},
                )
                stats = writer.ingest([supplied])
                pid = stable_chunk_id(item.source, 0, item.text, tenant=tenant)
                assert stats.failed == 0 and stats.ids == [pid]
                assert store.retrieve_payload(pid, tenant=tenant)["tenant_id"] == tenant
                ids[tenant, name] = pid
        assert ids[TENANT, "old"] != ids["tenant-b", "old"]
        yield store, items, ids
    finally:
        store.client.close()


@pytest.fixture(params=["cli", "http", "mcp"])
def search(request, corpus, monkeypatch, tmp_path, capsys):
    store, _, _ = corpus
    reranker = _SuccessfulReranker()
    close = None

    if request.param == "cli":
        import mnemostack.cli as cli

        monkeypatch.setattr(cli, "get_provider", lambda *a, **kw: _Embedding())
        monkeypatch.setattr(cli, "VectorStore", lambda **kw: store)
        monkeypatch.setattr(cli, "get_llm", lambda *a, **kw: _NoGeneration())
        monkeypatch.setattr(cli, "Reranker", lambda **kw: reranker)
        monkeypatch.setattr(cli, "FileStateStore", lambda *a: InMemoryStateStore())

        def call(filters=None, as_of=None):
            argv = [
                "search", QUERY, "--limit", str(LIMIT), "--json", "--tenant", TENANT,
                "--vector-floor", "0", "--require-rerank",
            ]
            if filters is not None:
                argv += ["--filters", json.dumps(filters)]
            if as_of is not None:
                argv += ["--as-of", as_of]
            capsys.readouterr()
            assert cli.main(argv) == 0
            return json.loads(capsys.readouterr().out)

    elif request.param == "http":
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        import mnemostack.server as server

        monkeypatch.setattr(server, "get_provider", lambda *a, **kw: _Embedding())
        monkeypatch.setattr(server, "VectorStore", lambda **kw: store)
        monkeypatch.setattr(server, "_make_probe_client", lambda *a: store.client)
        monkeypatch.setattr(server, "get_llm", lambda *a, **kw: _NoGeneration())
        monkeypatch.setattr(server, "Reranker", lambda **kw: reranker)
        monkeypatch.setattr(server, "FileStateStore", lambda *a: InMemoryStateStore())
        app = server.build_app(server.ServerConfig(
            provider_name="ollama", graph_uri=None, text_search="off", default_tenant=TENANT,
            vector_floor=0, state_path=str(tmp_path / "http-state.json"),
            record_access=False, retry_on_weak=False,
        ))
        client = TestClient(app)

        def call(filters=None, as_of=None):
            response = client.post("/recall", json={
                "query": QUERY, "limit": LIMIT, "filters": filters, "as_of": as_of,
                "full_pipeline": True, "retry_on_weak": False,
            })
            assert response.status_code == 200, response.text
            data = response.json()
            assert not set(data["degraded"]) - set(data["notes"])
            return [{**row, "payload": row["metadata"]} for row in data["results"]]

        close = client.close
    else:
        pytest.importorskip("fastmcp")
        import mnemostack.mcp.server as server

        monkeypatch.setattr(server, "get_provider", lambda *a, **kw: _Embedding())
        monkeypatch.setattr(server, "VectorStore", lambda **kw: store)
        monkeypatch.setattr(server, "get_llm", lambda *a, **kw: _NoGeneration())
        monkeypatch.setattr(server, "FileStateStore", lambda *a: InMemoryStateStore())
        mcp = server.build_server(
            embedding_provider="ollama", memgraph_uri=None, text_search="off",
            default_tenant=TENANT, vector_floor=0, reranker=reranker,
            state_path=str(tmp_path / "mcp-state.json"),
        )

        def call(filters=None, as_of=None):
            result = asyncio.run(mcp.call_tool("mnemostack_search", {
                "query": QUERY, "limit": LIMIT, "filters": filters, "as_of": as_of,
            }))
            data = result.structured_content
            assert data["ok"], data
            assert not set(data["degraded"]) - set(data["notes"])
            return data["results"]

    def checked_call(**kwargs):
        before = reranker.calls
        rows = call(**kwargs)
        assert reranker.calls == before + 1, "Search silently skipped the declared reranker"
        return rows

    try:
        yield checked_call
    finally:
        if close is not None:
            close()


def _assert_records(rows, names, corpus):
    _, items, ids = corpus
    expected = {ids[TENANT, name]: items[name] for name in names}
    assert {row["id"] for row in rows} == set(expected)
    assert len(rows) == len(expected)
    for row in rows:
        item = expected[row["id"]]
        assert row["text"] == item.text
        assert row["payload"]["source"] == item.source
        assert row["payload"]["offset"] == 0
        assert row["payload"]["tenant_id"] == TENANT
        assert row["payload"]["project"] == item.metadata["project"]


def test_owner_replacement_preserves_current_and_historical_sources(corpus, search):
    store, items, ids = corpus
    foreign = {name: store.retrieve_payload(ids["tenant-b", name], tenant="tenant-b") for name in items}
    assert store.invalidate(
        ids[TENANT, "old"], invalidated_at=RETRACTED, valid_until=TRANSITION, tenant=TENANT,
    ) == 1

    _assert_records(search(filters={"project": "P"}), ["current"], corpus)
    for instant in (START, "2026-01-15T00:00:00Z"):
        _assert_records(search(filters={"project": "P"}, as_of=instant), ["old"], corpus)
    _assert_records(search(filters={"project": "P"}, as_of=TRANSITION), ["current"], corpus)
    assert foreign == {
        name: store.retrieve_payload(ids["tenant-b", name], tenant="tenant-b") for name in items
    }


def test_project_filter_excludes_newer_other_project_and_foreign_tenant(corpus, search):
    store, _, ids = corpus
    store.invalidate(ids[TENANT, "old"], invalidated_at=RETRACTED, tenant=TENANT)

    _assert_records(search(filters={"project": "P"}), ["current"], corpus)
    _assert_records(search(filters={"project": "Q"}), ["other_project"], corpus)
    # Without a project filter, eligibility is not semantic project inference.
    # Do not require a particular order for equal synthetic embedding vectors.
    _assert_records(search(), ["current", "other_project"], corpus)
