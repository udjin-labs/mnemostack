"""Creation-only indexes preserve payload filtering and recall results."""
from unittest.mock import Mock

import pytest
from qdrant_client import QdrantClient, models

from mnemostack.config import Config
from mnemostack.recall import filter_by_validity
from mnemostack.recall.retrievers import VectorRetriever
from mnemostack.vector.qdrant import VectorStore


class Embedding:
    dimension = 2

    def embed(self, text):
        return [1.0, 0.0]


def make_store(config=None):
    store = VectorStore.__new__(VectorStore)
    store.collection = "indexes"
    store.dimension = 2
    store.distance = models.Distance.COSINE
    store.timestamp_index = config
    store.client = QdrantClient(":memory:")
    return store


@pytest.mark.parametrize("config,expected", [
    (None, {}),
    (("captured", "iso"), {"captured": "datetime"}),
    (("captured", "epoch"), {"captured": "float"}),
    (("captured", "epoch_ms"), {"captured": "float"}),
    (("captured", "unknown"), {}),
])
def test_new_collection_indexes(config, expected):
    store = make_store(config)
    try:
        create = Mock(wraps=store.client.create_payload_index)
        store.client.create_payload_index = create
        assert store.ensure_collection()
        assert {c.kwargs["field_name"]: c.kwargs["field_schema"].value
                for c in create.call_args_list} == {
            "tenant_id": "keyword", "source": "keyword", "invalidated_at": "datetime", **expected,
        }
    finally:
        store.client.close()


def test_existing_collection_never_adds_indexes(monkeypatch):
    store = make_store(("captured", "iso"))
    try:
        store.client.create_collection(
            store.collection, vectors_config=models.VectorParams(size=2, distance=store.distance),
        )
        client = store.client
        monkeypatch.setattr("mnemostack.vector.qdrant.QdrantClient", lambda **kwargs: client)
        store = VectorStore("indexes", 2, timestamp_index=("captured", "iso"))
        before = store.payload_indexes()
        create = Mock(wraps=store.client.create_payload_index)
        store.client.create_payload_index = create
        assert not store.ensure_collection()
        store.upsert(1, [1.0, 0.0], {"text": "one"})
        assert VectorRetriever(Embedding(), store).search("query")[0].id == 1
        assert store.payload_indexes() == before
        create.assert_not_called()
    finally:
        store.client.close()


def test_new_index_failures_match_keyword_index_failures():
    store = make_store(("captured", "iso"))
    try:
        store.client.create_payload_index = Mock(side_effect=RuntimeError("unavailable"))
        assert store.ensure_collection()
        assert store.client.create_payload_index.call_count == 4
        assert store.collection_exists()
    finally:
        store.client.close()


@pytest.mark.parametrize("fmt", ["iso", "epoch", "epoch_ms"])
def test_indexed_recall_matches_unindexed(fmt):
    indexed = make_store(("captured", fmt))
    baseline = make_store()
    baseline.index_payload_field = Mock()
    try:
        indexed.ensure_collection()
        baseline.ensure_collection()
        stamp = {"iso": "2026-01-02T00:00:00Z", "epoch": 1767312000,
                 "epoch_ms": 1767312000000}[fmt]
        markers = [{}, {"invalidated_at": None}, {"invalidated_at": []},
                   {"invalidated_at": ""}, {"invalidated_at": "2026-02-01T00:00:00Z"}]
        for store in (indexed, baseline):
            for i, marker in enumerate(markers, 1):
                store.upsert(i, [1.0, i / 10], {
                    "text": str(i), "captured": stamp, "tenant_id": "a",
                    "valid_from": "2026-01-01", **marker,
                }, tenant="a")
            store.upsert(6, [1.0, 0.0], {"text": "other", "captured": stamp}, tenant="b")
        embedding = Embedding()
        retrievers = [VectorRetriever(embedding, s, timestamp_key="captured", timestamp_format=fmt)
                      for s in (indexed, baseline)]
        for kwargs in ({}, {"include_invalidated": True}, {"as_of": "2026-01-15"},
                       {"tenant": "a"}, {"tenant": "b"},
                       {"filters": {"captured": {"gte": "2026-01-01", "lte": "2026-01-03"}}}):
            results = []
            for retriever in retrievers:
                hits = retriever.search("query", **kwargs)
                results.append(filter_by_validity(
                    hits, as_of=kwargs.get("as_of"),
                    include_invalidated=kwargs.get("include_invalidated", False),
                ))
            assert results[0] == results[1]
            assert results[0], kwargs
    finally:
        indexed.client.close()
        baseline.client.close()


@pytest.mark.parametrize("file_settings,env,expected", [
    ("", {}, None),
    ("  timestamp_key: captured\n", {}, None),
    ("  timestamp_format: epoch\n", {}, None),
    ("  timestamp_key: captured\n  timestamp_format: epoch_ms\n", {}, ("captured", "epoch_ms")),
    ("  timestamp_key: captured\n", {"MNEMOSTACK_TIMESTAMP_FORMAT": "iso"}, ("captured", "iso")),
    ("", {"MNEMOSTACK_TIMESTAMP_KEY": "captured", "MNEMOSTACK_TIMESTAMP_FORMAT": "epoch"},
     ("captured", "epoch")),
])
def test_timestamp_index_requires_explicit_key_and_format(tmp_path, monkeypatch, file_settings, env, expected):
    for key in ("MNEMOSTACK_TIMESTAMP_KEY", "MNEMOSTACK_TIMESTAMP_FORMAT"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    path = tmp_path / "config.yaml"
    path.write_text("recall:\n" + (file_settings or "  text_key: text\n"))
    cfg = Config.load(path)
    assert cfg.timestamp_index == expected
    assert "timestamp_index" not in cfg.to_dict()


def test_explicit_timestamp_reaches_configured_store_factories(monkeypatch, tmp_path):
    import argparse

    import mnemostack.cli as cli
    from mnemostack.server import ServerConfig

    path = tmp_path / "config.yaml"
    path.write_text("recall:\n  timestamp_key: captured\n  timestamp_format: epoch_ms\n")
    monkeypatch.setenv("MNEMOSTACK_CONFIG", str(path))
    monkeypatch.delenv("MNEMOSTACK_TIMESTAMP_KEY", raising=False)
    monkeypatch.delenv("MNEMOSTACK_TIMESTAMP_FORMAT", raising=False)
    cfg = Config.load()
    monkeypatch.setattr(cli.Config, "load", lambda: cfg)
    constructor = Mock()
    monkeypatch.setattr(cli, "VectorStore", constructor)
    cli._payload_schema.cache_clear()
    try:
        cli._indexing_store(argparse.Namespace(collection="test", qdrant="http://qdrant"), Mock(dimension=2))
        assert constructor.call_args.kwargs["timestamp_index"] == ("captured", "epoch_ms")
        assert ServerConfig.from_env().timestamp_index == ("captured", "epoch_ms")
    finally:
        cli._payload_schema.cache_clear()


@pytest.mark.parametrize("configured,expected", [
    (("captured", "epoch_ms"), ("captured", "epoch_ms")),
    (("other", "iso"), None),
    (None, None),
])
def test_mcp_factory_only_indexes_matching_explicit_schema(monkeypatch, tmp_path, configured, expected):
    import mnemostack.mcp.server as server

    class FakeMCP:
        def __init__(self, *args, **kwargs):
            self.tools = {}

        def tool(self):
            def register(function):
                self.tools[function.__name__] = function
                return function
            return register

    cfg = Config()
    cfg.timestamp_index = configured
    monkeypatch.setattr(server.Config, "load", lambda: cfg)
    monkeypatch.setattr(server, "FastMCP", FakeMCP)
    monkeypatch.setattr(server, "_FASTMCP_AVAILABLE", True)
    embedding = Mock(dimension=2, name="embedding")
    embedding.health_check.return_value = (True, "ok")
    monkeypatch.setattr(server, "get_provider", lambda *args, **kwargs: embedding)
    constructor = Mock()
    constructor.return_value.collection_exists.return_value = False
    monkeypatch.setattr(server, "VectorStore", constructor)
    mcp = server.build_server(
        collection="test", timestamp_key="captured", timestamp_format="epoch_ms",
        state_path=str(tmp_path / "state.json"),
    )
    assert mcp.tools["mnemostack_health"]()["components"]["vector"]["ok"]
    assert constructor.call_args.kwargs.get("timestamp_index") == expected


def test_unreadable_config_does_not_guess_a_timestamp_index(monkeypatch):
    import mnemostack.cli as cli

    monkeypatch.setattr(cli.Config, "load", Mock(side_effect=ValueError("invalid config")))
    assert cli._timestamp_index_kwargs() == {}


@pytest.mark.parametrize("key", [123, ["a"], None, "", "tenant_id", "source", "invalidated_at"])
def test_no_timestamp_index_for_a_malformed_or_already_indexed_key(key):
    from mnemostack.vector.qdrant import timestamp_index_schema

    assert timestamp_index_schema((key, "iso")) is None
    assert timestamp_index_schema((key, "epoch")) is None


def test_non_mapping_recall_section_is_not_an_explicit_timestamp_setting(tmp_path, monkeypatch):
    from mnemostack.config import Config

    for name in ("MNEMOSTACK_TIMESTAMP_KEY", "MNEMOSTACK_TIMESTAMP_FORMAT"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "config.yaml"
    path.write_text("recall: [timestamp_key, timestamp_format]\n")
    try:
        loaded = Config.load(str(path))
    except Exception:  # noqa: BLE001
        return  # a malformed section may be refused outright
    assert loaded.timestamp_index is None
