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


def test_memgraph_retriever_tolerates_no_arg_session_driver():
    # An injected driver whose session() predates the database= kwarg must still
    # work when no non-default database is configured — otherwise the broad
    # except swallows the TypeError and graph recall silently goes empty.
    from mnemostack.recall.retrievers import MemgraphRetriever

    calls = {"n": 0}

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def run(self, *a, **k):
            calls["n"] += 1

            class _R:
                def __iter__(self_inner):
                    return iter(())

                def single(self_inner):
                    return None

            return _R()

    class _NoArgDriver:
        def session(self):  # no database= parameter
            return _Session()

    r = MemgraphRetriever(uri="bolt://x", driver=_NoArgDriver())  # database defaults None
    r.search("alice")
    assert calls["n"] > 0  # queries ran; session() was called without database=


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


# ---------- positional back-compat: new params appended at the tail ----------


def test_new_graph_params_are_appended_not_inserted():
    """A mid-signature insert would shift existing positional args for library
    callers; the new params must be the last ones on each widened signature.
    """
    import inspect

    from mnemostack.mcp.server import build_server
    from mnemostack.recall.pipeline import build_full_pipeline
    from mnemostack.recall.pipeline.resurrection import GraphResurrection
    from mnemostack.recall.retrievers import MemgraphRetriever

    def _last(fn):
        # The guard protects POSITIONAL compatibility — keyword-only params
        # (which can never shift a positional caller) don't count as the tail.
        params = [
            name
            for name, p in inspect.signature(fn).parameters.items()
            if p.kind is not inspect.Parameter.KEYWORD_ONLY
        ]
        return params[-1]

    assert _last(build_full_pipeline) == "graph_database"
    assert _last(GraphResurrection.__init__) == "database"
    assert _last(MemgraphRetriever.__init__) == "database"
    # build_server: the auth trio must sit after the pre-existing token_budget
    bs = list(inspect.signature(build_server).parameters)
    assert bs.index("graph_user") > bs.index("token_budget")


def test_memgraph_retriever_positional_call_unshifted():
    # Old-style positional call (uri, user, password, min_word) must still bind
    # min_word — not the newly added database.
    from mnemostack.recall.retrievers import MemgraphRetriever

    r = MemgraphRetriever("bolt://x", "u", "p", 7)
    assert r.min_word == 7
    assert r.database is None


# ---------- env overrides feed graph auth (not just YAML) ----------


def test_env_overrides_graph_auth(monkeypatch):
    from mnemostack.config import Config

    monkeypatch.setenv("MNEMOSTACK_GRAPH_USER", "neo4j")
    monkeypatch.setenv("MNEMOSTACK_GRAPH_PASSWORD", "secret")
    monkeypatch.setenv("MNEMOSTACK_GRAPH_DATABASE", "memories")

    cfg = Config.load()
    # env wins over any local config file, so this holds regardless of the host.
    assert cfg.graph.user == "neo4j"
    assert cfg.graph.password == "secret"
    assert cfg.graph.database == "memories"


# ---------- CLI service paths (serve / mcp-serve / non-raw search) ----------


def _parsed_args(monkeypatch, argv):
    """A CLI namespace with graph auth seeded from a stubbed config."""
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
    return cli.build_parser().parse_args(argv)


def test_cmd_serve_threads_graph_auth(monkeypatch):
    import sys
    import types

    import mnemostack.server as server_mod
    from mnemostack import cli

    args = _parsed_args(monkeypatch, ["serve"])
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        server_mod, "build_app", lambda cfg: captured.update(cfg=cfg) or "app"
    )
    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda *a, **k: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    assert cli.cmd_serve(args) == 0
    cfg = captured["cfg"]
    assert cfg.graph_user == "neo4j"
    assert cfg.graph_password == "secret"
    assert cfg.graph_database == "memories"


def test_cmd_mcp_serve_threads_graph_auth(monkeypatch):
    import mnemostack.mcp as mcp_mod
    from mnemostack import cli

    args = _parsed_args(monkeypatch, ["mcp-serve"])
    captured: dict[str, Any] = {}

    class _Srv:
        def run(self):
            return None

    monkeypatch.setattr(mcp_mod, "build_server", lambda **kw: captured.update(kw) or _Srv())

    assert cli.cmd_mcp_serve(args) == 0
    assert captured["graph_user"] == "neo4j"
    assert captured["graph_password"] == "secret"
    assert captured["graph_database"] == "memories"


def test_recall_for_cli_threads_graph_auth(monkeypatch):
    import pytest

    from mnemostack import cli

    args = _parsed_args(monkeypatch, ["search", "q"])
    captured: dict[str, Any] = {}

    class _Stop(Exception):
        pass

    def fake_build_full_pipeline(**kw):
        captured.update(kw)
        raise _Stop

    monkeypatch.setattr(cli, "build_full_pipeline", fake_build_full_pipeline)

    with pytest.raises(_Stop):
        cli._recall_for_cli(args, recaller=None, query="q", limit=10)
    assert captured["graph_user"] == "neo4j"
    assert captured["graph_password"] == "secret"
    assert captured["graph_database"] == "memories"
