"""Graph auth (user/password/database) threading through the factory + entry points.

`GraphStore` / `MemgraphRetriever` always supported auth, but CLI/MCP/HTTP used
to construct them with only uri+timeout, so an auth'd or non-default-database
Memgraph/Neo4j was unusable from those surfaces. These tests pin the wiring
without needing a live server (the neo4j driver is stubbed / not connected).
"""

from __future__ import annotations

import argparse
from typing import Any


class _FakeGraphStore:
    """Records the kwargs it was constructed with (stands in for GraphStore)."""

    calls: list[dict[str, Any]] = []

    def __init__(self, uri="", user="", password="", database=None, timeout=5.0):
        self.kwargs = {
            "uri": uri,
            "user": user,
            "password": password,
            "database": database,
            "timeout": timeout,
        }
        _FakeGraphStore.calls.append(self.kwargs)


# ---------- factory ----------


def test_make_graph_store_threads_auth(monkeypatch):
    import mnemostack.graph.store as store_mod
    from mnemostack.graph.factory import make_graph_store

    _FakeGraphStore.calls.clear()
    monkeypatch.setattr(store_mod, "GraphStore", _FakeGraphStore)

    make_graph_store(
        "bolt://db:7687",
        timeout=9.0,
        user="neo4j",
        password="secret",
        database="memories",
    )

    assert _FakeGraphStore.calls == [
        {
            "uri": "bolt://db:7687",
            "user": "neo4j",
            "password": "secret",
            "database": "memories",
            "timeout": 9.0,
        }
    ]


def test_make_graph_store_defaults_are_anonymous(monkeypatch):
    import mnemostack.graph.store as store_mod
    from mnemostack.graph.factory import make_graph_store

    _FakeGraphStore.calls.clear()
    monkeypatch.setattr(store_mod, "GraphStore", _FakeGraphStore)

    make_graph_store("bolt://localhost:7687")

    call = _FakeGraphStore.calls[0]
    assert call["user"] == "" and call["password"] == "" and call["database"] is None


# ---------- CLI namespace seeding ----------


def test_cli_seeds_graph_auth_from_config(monkeypatch):
    """build_parser stamps graph.user/password/database onto every command's args."""
    from mnemostack import cli
    from mnemostack.config import Config, GraphConfig

    cfg = Config()
    cfg.graph = GraphConfig(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="secret",
        database="memories",
    )
    monkeypatch.setattr(Config, "load", classmethod(lambda cls, *a, **k: cfg))

    parser = cli.build_parser()
    args = parser.parse_args(["health"])
    assert args.graph_user == "neo4j"
    assert args.graph_password == "secret"
    assert args.graph_database == "memories"


def test_graph_auth_helper_reads_namespace():
    from mnemostack.cli import _graph_auth

    args = argparse.Namespace(
        graph_user="neo4j", graph_password="secret", graph_database="memories"
    )
    assert _graph_auth(args) == {
        "user": "neo4j",
        "password": "secret",
        "database": "memories",
    }


def test_graph_auth_helper_missing_attrs_default_anonymous():
    # A namespace built by tests / older callers without the seeded defaults
    # must still yield an anonymous (back-compat) connection, not crash.
    from mnemostack.cli import _graph_auth

    assert _graph_auth(argparse.Namespace()) == {
        "user": "",
        "password": "",
        "database": None,
    }


# ---------- MemgraphRetriever carries database into the session ----------


def test_memgraph_retriever_passes_database_to_session():
    from mnemostack.recall.retrievers import MemgraphRetriever

    seen: dict[str, Any] = {}

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def run(self, *a, **k):
            class _R:
                def __iter__(self_inner):
                    return iter(())

                def single(self_inner):
                    return None

            return _R()

    class _Driver:
        def session(self, database=None):
            seen["database"] = database
            return _Session()

    r = MemgraphRetriever(uri="bolt://x", database="memories", driver=_Driver())
    r.search("alice")
    assert seen["database"] == "memories"


def test_memgraph_retriever_database_defaults_none():
    from mnemostack.recall.retrievers import MemgraphRetriever

    r = MemgraphRetriever(uri="bolt://x")
    assert r.database is None


# ---------- MCP build_server threads auth into constructed components ----------


def test_mcp_build_server_accepts_graph_auth():
    # The build_server signature must expose the auth params so cmd_mcp_serve can
    # pass cfg.graph.user/password/database. (Smoke: signature only — no server run.)
    import inspect

    from mnemostack.mcp.server import build_server

    params = inspect.signature(build_server).parameters
    assert {"graph_user", "graph_password", "graph_database"} <= set(params)


def test_server_config_carries_graph_auth():
    from mnemostack.server import ServerConfig

    cfg = ServerConfig(graph_user="neo4j", graph_password="secret", graph_database="m")
    assert cfg.graph_user == "neo4j"
    assert cfg.graph_password == "secret"
    assert cfg.graph_database == "m"


def test_build_full_pipeline_accepts_graph_database():
    import inspect

    from mnemostack.recall.pipeline import build_full_pipeline

    assert "graph_database" in inspect.signature(build_full_pipeline).parameters
