"""Quantization search parameters on the dense query path (#196).

Contract under test: unset knobs send no ``search_params`` at all (requests are
unchanged); set knobs ride on every DENSE query and never on the sparse one;
bad values fail the config load; and every dense-searching surface (HTTP
server, inspector, MCP, CLI search / answer / synthesize / serve / mcp-serve /
inspect) receives the configured values.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, QuantizationSearchParams, SearchParams

from mnemostack.config import Config, quantization_kwargs
from mnemostack.vector import VectorStore
from mnemostack.vector.async_qdrant import AsyncVectorStore
from mnemostack.vector.qdrant import quantization_search_params
from mnemostack.vector.sparse import SparseTextEncoder

VEC = [1.0, 0.0, 0.0, 0.0]
EXPECTED = SearchParams(quantization=QuantizationSearchParams(rescore=True, oversampling=2.0))


class _Stop(Exception):
    """Raised by a capturing fake to end a build right after the store."""


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MNEMOSTACK_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    return monkeypatch


def _store(sparse: bool = False, **quant) -> VectorStore:
    s = VectorStore.__new__(VectorStore)
    s.collection = "quant"
    s.dimension = 4
    s.distance = Distance.COSINE
    s.client = QdrantClient(":memory:")
    s.sparse_text = sparse
    s._sparse_encoder = SparseTextEncoder() if sparse else None
    s.search_params = quantization_search_params(**quant)
    s.ensure_collection()
    s.upsert(1, VEC, {"text": "postgres backup notes"})
    s.client = MagicMock(wraps=s.client)
    return s


# ---------- store ----------


def test_unset_knobs_build_no_search_params():
    assert quantization_search_params() is None
    assert quantization_kwargs() == {}
    assert VectorStore(collection="c", dimension=4).search_params is None


def test_set_knobs_build_search_params():
    assert quantization_search_params(rescore=True, oversampling=2.0) == EXPECTED
    store = VectorStore(
        collection="c", dimension=4, quantization_rescore=True, quantization_oversampling=2.0
    )
    assert store.search_params == EXPECTED


def test_unset_dense_query_carries_no_search_params():
    s = _store()
    assert [h.id for h in s.search(VEC, limit=5)] == [1]
    assert "search_params" not in s.client.query_points.call_args.kwargs


def test_set_dense_query_carries_search_params():
    s = _store(rescore=True, oversampling=2.0)
    assert [h.id for h in s.search(VEC, limit=5)] == [1]
    assert s.client.query_points.call_args.kwargs["search_params"] == EXPECTED


def test_sparse_query_never_carries_search_params():
    s = _store(sparse=True, rescore=True, oversampling=2.0)
    assert [h.id for h in s.sparse_search("postgres", limit=5)] == [1]
    assert "search_params" not in s.client.query_points.call_args.kwargs


@pytest.mark.parametrize("quant, expected", [({}, None), ({"rescore": False}, "set")])
def test_async_dense_query(quant, expected):
    s = AsyncVectorStore.__new__(AsyncVectorStore)
    s.collection = "quant"
    s.search_params = quantization_search_params(**quant)
    s.client = MagicMock()
    s.client.query_points = AsyncMock(return_value=MagicMock(points=[]))
    asyncio.run(s.search(VEC, limit=5))
    kwargs = s.client.query_points.call_args.kwargs
    if expected is None:
        assert "search_params" not in kwargs
    else:
        assert kwargs["search_params"] == SearchParams(
            quantization=QuantizationSearchParams(rescore=False)
        )


# ---------- config ----------


def test_env_knobs(isolated_env):
    isolated_env.setenv("MNEMOSTACK_QUANTIZATION_RESCORE", "true")
    isolated_env.setenv("MNEMOSTACK_QUANTIZATION_OVERSAMPLING", "2")
    cfg = Config.load()
    assert cfg.vector.quantization_rescore is True
    assert cfg.vector.quantization_oversampling == 2.0
    isolated_env.setenv("MNEMOSTACK_QUANTIZATION_RESCORE", "off")
    assert Config.load().vector.quantization_rescore is False


def test_defaults_are_unset(isolated_env):
    cfg = Config.load()
    assert cfg.vector.quantization_rescore is None
    assert cfg.vector.quantization_oversampling is None


def test_env_rescore_rejects_non_boolean(isolated_env):
    isolated_env.setenv("MNEMOSTACK_QUANTIZATION_RESCORE", "maybe")
    with pytest.raises(ValueError, match="MNEMOSTACK_QUANTIZATION_RESCORE"):
        Config.load()


def test_env_oversampling_names_the_variable(isolated_env):
    isolated_env.setenv("MNEMOSTACK_QUANTIZATION_OVERSAMPLING", "abc")
    with pytest.raises(ValueError, match="MNEMOSTACK_QUANTIZATION_OVERSAMPLING"):
        Config.load()


@pytest.mark.parametrize(
    "quant",
    [
        {"rescore": "yes"},
        {"oversampling": 0.5},
        {"oversampling": float("nan")},
        {"oversampling": float("inf")},
        {"oversampling": True},
        {"oversampling": "2"},
    ],
)
def test_sdk_values_are_validated(quant):
    with pytest.raises(ValueError, match="quantization_"):
        quantization_search_params(**quant)


def test_rejected_values_create_no_client(monkeypatch):
    # Validation precedes the client: nothing is left behind to close.
    import mnemostack.vector.async_qdrant as aq
    import mnemostack.vector.qdrant as q

    made: list = []
    monkeypatch.setattr(q, "QdrantClient", lambda **kw: made.append(kw))
    monkeypatch.setattr(aq, "AsyncQdrantClient", lambda **kw: made.append(kw))
    with pytest.raises(ValueError):
        VectorStore(collection="c", dimension=4, quantization_oversampling=0.5)
    with pytest.raises(ValueError):
        AsyncVectorStore(collection="c", dimension=4, quantization_rescore="yes")
    assert made == []


def test_config_show_explicit_bad_file_is_a_clean_error(isolated_env, tmp_path, capsys):
    import mnemostack.cli as cli

    path = tmp_path / "bad.yaml"
    path.write_text("vector: [\n")
    assert cli.main(["config", "--config", str(path)]) == 2
    assert "error: invalid configuration:" in capsys.readouterr().err


def test_cli_bad_config_is_a_clean_error(isolated_env, capsys):
    import mnemostack.cli as cli

    isolated_env.setenv("MNEMOSTACK_QUANTIZATION_OVERSAMPLING", "0.5")
    assert cli.main(["health"]) == 2
    assert "error: invalid configuration:" in capsys.readouterr().err


@pytest.mark.parametrize(
    "body, reason",
    [
        ("vector: [\n", "not valid YAML"),
        ("- a\n- b\n", "must be a mapping"),
        ("[]\n", "must be a mapping"),
        ("false\n", "must be a mapping"),
        ("recall:\n  token_budget: []\n", ""),
    ],
)
def test_cli_malformed_config_file_is_a_clean_error(isolated_env, tmp_path, capsys, body, reason):
    import mnemostack.cli as cli

    path = tmp_path / "config.yaml"
    path.write_text(body)
    isolated_env.setenv("MNEMOSTACK_CONFIG", str(path))
    assert cli.main(["health"]) == 2
    err = capsys.readouterr().err
    assert "error: invalid configuration:" in err and reason in err


@pytest.mark.parametrize("body", ["", "# only a comment\n", "null\n"])
def test_empty_config_file_is_still_accepted(isolated_env, tmp_path, body):
    path = tmp_path / "config.yaml"
    path.write_text(body)
    assert Config.load(path).vector.collection == "mnemostack"


@pytest.mark.parametrize(
    "line",
    [
        "quantization_rescore: 'yes'",
        "quantization_oversampling: 0.5",
        "quantization_oversampling: .nan",
        "quantization_oversampling: .inf",
        "quantization_oversampling: true",
        "quantization_oversampling: '2'",
    ],
)
def test_file_values_are_validated_at_load(isolated_env, tmp_path, line):
    path = tmp_path / "config.yaml"
    path.write_text(f"vector:\n  {line}\n")
    with pytest.raises(ValueError, match="vector.quantization_"):
        Config.load(path)


def test_file_oversampling_int_becomes_float(isolated_env, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("vector:\n  quantization_rescore: true\n  quantization_oversampling: 2\n")
    cfg = Config.load(path)
    assert cfg.vector.quantization_rescore is True
    assert cfg.vector.quantization_oversampling == 2.0
    assert isinstance(cfg.vector.quantization_oversampling, float)


# ---------- surfaces ----------


def _capture(sink):
    def _fake(**kwargs):
        sink.update(kwargs)
        raise _Stop

    return _fake


QUANT = {"quantization_rescore": True, "quantization_oversampling": 2.0}


def test_server_forwards_from_env(isolated_env):
    pytest.importorskip("fastapi")
    import mnemostack.server as srv

    isolated_env.setenv("MNEMOSTACK_QUANTIZATION_RESCORE", "1")
    isolated_env.setenv("MNEMOSTACK_QUANTIZATION_OVERSAMPLING", "2.0")
    cfg = srv.ServerConfig.from_env()
    assert (cfg.quantization_rescore, cfg.quantization_oversampling) == (True, 2.0)

    seen: dict = {}
    isolated_env.setattr(srv, "get_provider", lambda *_a, **_k: MagicMock(dimension=4))
    isolated_env.setattr(srv, "VectorStore", _capture(seen))
    with pytest.raises(_Stop):
        srv.build_app(cfg)
    assert {k: seen[k] for k in QUANT} == QUANT


def test_inspector_forwards(isolated_env):
    pytest.importorskip("fastapi")
    import mnemostack.inspector as ins
    from mnemostack.server import ServerConfig

    seen: dict = {}
    isolated_env.setattr(ins, "VectorStore", _capture(seen))
    with pytest.raises(_Stop):
        ins.build_inspector_app(ServerConfig(provider_name="fake", **QUANT))
    assert {k: seen[k] for k in QUANT} == QUANT


def test_inspector_unscoped_search_carries_search_params(isolated_env):
    # The unscoped (legacy-only) search queries Qdrant directly, not through
    # VectorStore.search — it must carry the store's params too.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import mnemostack.inspector as ins
    from mnemostack.server import ServerConfig

    s = _store(rescore=True, oversampling=2.0)

    class _Provider:
        dimension = 4
        name = "fake:quant"

        def embed(self, _text):
            return VEC

    isolated_env.setattr(ins, "get_provider", lambda *_a, **_k: _Provider())
    isolated_env.setattr(ins, "VectorStore", lambda **_: s)
    isolated_env.setattr(ins, "_make_probe_client", lambda *_a, **_k: s.client)
    app = ins.build_inspector_app(
        ServerConfig(provider_name="fake", collection="quant", graph_uri=None)
    )
    rows = TestClient(app).get("/api/records?q=x").json()["records"]
    assert [r["id"] for r in rows] == ["1"]
    assert s.client.query_points.call_args.kwargs["search_params"] == EXPECTED


def test_mcp_forwards(isolated_env):
    pytest.importorskip("fastmcp")
    import mnemostack.mcp.server as msrv

    seen: dict = {}

    def _fake_store(**kwargs):
        seen.update(kwargs)
        return MagicMock()

    isolated_env.setattr(msrv, "get_provider", lambda *_a, **_k: MagicMock(dimension=4))
    isolated_env.setattr(msrv, "VectorStore", _fake_store)
    isolated_env.setattr(msrv, "VectorRetriever", lambda **_: MagicMock())
    isolated_env.setattr(msrv, "TemporalRetriever", lambda **_: MagicMock())
    isolated_env.setattr(msrv, "build_bm25_docs", lambda _paths: [])
    isolated_env.setattr(msrv, "Recaller", lambda **_: MagicMock(recall=MagicMock(return_value=[])))
    mcp = msrv.build_server(collection="c", embedding_provider="ollama", **QUANT)
    asyncio.run(mcp.call_tool("mnemostack_search", {"query": "q", "limit": 1}))
    assert {k: seen[k] for k in QUANT} == QUANT


@pytest.mark.parametrize("argv", [["search", "q"], ["answer", "q"], ["synthesize", "q"]])
def test_cli_dense_commands_forward(isolated_env, argv):
    import mnemostack.cli as cli

    isolated_env.setenv("MNEMOSTACK_QUANTIZATION_RESCORE", "true")
    isolated_env.setenv("MNEMOSTACK_QUANTIZATION_OVERSAMPLING", "2")
    seen: dict = {}
    isolated_env.setattr(cli, "get_provider", lambda *_a, **_k: MagicMock(dimension=4))
    isolated_env.setattr(cli, "VectorStore", _capture(seen))
    args = cli.build_parser().parse_args(argv)
    with pytest.raises(_Stop):
        args.func(args)
    assert {k: seen[k] for k in QUANT} == QUANT


def test_cli_search_unset_passes_nothing(isolated_env):
    import mnemostack.cli as cli

    seen: dict = {}
    isolated_env.setattr(cli, "get_provider", lambda *_a, **_k: MagicMock(dimension=4))
    isolated_env.setattr(cli, "VectorStore", _capture(seen))
    args = cli.build_parser().parse_args(["search", "q"])
    with pytest.raises(_Stop):
        args.func(args)
    assert not set(QUANT) & set(seen)


def test_cli_serve_inspect_and_mcp_serve_forward(isolated_env):
    pytest.importorskip("fastapi")
    pytest.importorskip("fastmcp")
    import mnemostack.cli as cli
    import mnemostack.inspector as ins
    import mnemostack.mcp as mcp_pkg
    import mnemostack.server as srv

    isolated_env.setenv("MNEMOSTACK_QUANTIZATION_RESCORE", "true")
    isolated_env.setenv("MNEMOSTACK_QUANTIZATION_OVERSAMPLING", "2")
    got: dict = {}

    def _app(cfg):
        got["serve"] = (cfg.quantization_rescore, cfg.quantization_oversampling)
        raise _Stop

    def _inspector(cfg):
        got["inspect"] = (cfg.quantization_rescore, cfg.quantization_oversampling)
        raise _Stop

    def _mcp(**kwargs):
        got["mcp-serve"] = (kwargs["quantization_rescore"], kwargs["quantization_oversampling"])
        raise _Stop

    isolated_env.setattr(srv, "build_app", _app)
    isolated_env.setattr(ins, "build_inspector_app", _inspector)
    isolated_env.setattr(mcp_pkg, "build_server", _mcp)
    for cmd in ("serve", "inspect", "mcp-serve"):
        args = cli.build_parser().parse_args([cmd])
        with pytest.raises(_Stop):
            args.func(args)
    assert got == {name: (True, 2.0) for name in ("serve", "inspect", "mcp-serve")}
