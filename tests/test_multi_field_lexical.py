"""Multi-field lexical arms — per-field MatchText gates with fusion weights.

Contract under test: `recall.text_search_fields` turns each configured
payload field into its OWN lexically-gated arm (gating on that field,
returning chunk text from `text_key`), with a distinct retriever name — so
fusion weights, the adaptive profile, and telemetry stay per-arm — and a
fusion-level weight (title/heading fields are typically far more precise
lexical signals than chunk bodies).
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance

from mnemostack.config import (
    Config,
    ensure_text_fields_mode,
    parse_text_search_fields,
)
from mnemostack.recall.retrievers import (
    BM25Retriever,
    QdrantSparseRetriever,
    QdrantTextRetriever,
    build_qdrant_text_arms,
)
from mnemostack.vector import VectorStore

_V1 = [1.0, 0.0, 0.0, 0.0]
_V2 = [0.0, 1.0, 0.0, 0.0]


class _FakeEmbedder:
    dimension = 4
    name = "fake:text"

    def embed(self, text):
        return _V1

    def embed_batch(self, texts):
        return [_V1 for _ in texts]


def _store(sparse: bool = False) -> VectorStore:
    s = VectorStore.__new__(VectorStore)
    s.collection = "mf"
    s.dimension = 4
    s.distance = Distance.COSINE
    s.client = QdrantClient(":memory:")
    s.sparse_text = sparse
    s.text_key = "text"
    if sparse:
        from mnemostack.vector.sparse import SparseTextEncoder

        s._sparse_encoder = SparseTextEncoder()
    else:
        s._sparse_encoder = None
    s.ensure_collection()
    return s


def _seed_titled(s: VectorStore) -> None:
    # Title mentions "postgres" but the body does NOT (and vice versa on 2):
    # the difference between gating on title vs body is observable.
    s.upsert(
        1,
        _V1,
        {"title": "postgres servers", "text": "backup restore runbook"},
        tenant="acme",
    )
    s.upsert(
        2,
        _V2,
        {"title": "network map", "text": "postgres appears in body only"},
        tenant="acme",
    )
    s.upsert(
        3,
        _V1,
        {"title": "postgres tuning", "text": "shared buffers notes"},
        tenant="globex",
    )


# ---------- config parsing ----------


def test_parse_fields_env_string_forms():
    assert parse_text_search_fields("title:2.0,text") == {"title": 2.0, "text": 1.0}
    assert parse_text_search_fields(" title : 2 , text : 0.5 ") == {
        "title": 2.0,
        "text": 0.5,
    }
    assert parse_text_search_fields("") == {}
    assert parse_text_search_fields(None) == {}


def test_parse_fields_mapping_forms():
    assert parse_text_search_fields({"title": 2, "text": "1.5"}) == {
        "title": 2.0,
        "text": 1.5,
    }


@pytest.mark.parametrize(
    "bad",
    [
        "title:0",
        "title:-1",
        "title:inf",
        "title:nan",
        "title:abc",
        ":2.0",
        "title:2.0,title:3.0",  # duplicate key must not silently last-win
        {"title": None},
        {"title": True},  # bool is a float subclass; YAML `yes` is a typo
        {"title": 2.0, " title ": 3.0},  # collides after trimming
        {"": 1.0},
        ["title"],
        42,
    ],
)
def test_parse_fields_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_text_search_fields(bad)


def test_fields_require_lexical_mode():
    ensure_text_fields_mode("lexical", {"title": 2.0})  # fine
    ensure_text_fields_mode("sparse", {})  # empty fields never conflict
    for mode in ("off", "bm25", "qdrant_bm25", "sparse"):
        with pytest.raises(ValueError, match="text_search=lexical"):
            ensure_text_fields_mode(mode, {"title": 2.0})


def test_config_load_normalizes_env_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("MNEMOSTACK_TEXT_SEARCH_FIELDS", "title:2.0,text")
    cfg = Config.load(tmp_path / "absent.yaml")
    assert cfg.recall.text_search_fields == {"title": 2.0, "text": 1.0}


def test_config_load_normalizes_yaml_fields(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "recall:\n  text_search: lexical\n  text_search_fields:\n    title: 2.0\n    text: 1\n"
    )
    cfg = Config.load(p)
    assert cfg.recall.text_search_fields == {"title": 2.0, "text": 1.0}


# ---------- retriever name override ----------


def test_name_override_on_all_lexical_arms():
    s = _store()
    title_arm = QdrantTextRetriever(
        embedding=_FakeEmbedder(), vector_store=s, name="qdrant_text:title"
    )
    assert title_arm.name == "qdrant_text:title"
    # The class attribute (and other instances) stay untouched.
    assert QdrantTextRetriever.name == "qdrant_text"
    assert QdrantTextRetriever(embedding=_FakeEmbedder(), vector_store=s).name == "qdrant_text"

    sp = _store(sparse=True)
    assert QdrantSparseRetriever(vector_store=sp, name="sparse2").name == "sparse2"
    assert BM25Retriever(docs=[], name="bm25:aux").name == "bm25:aux"


def test_name_override_rejects_empty():
    s = _store()
    with pytest.raises(ValueError, match="non-empty"):
        QdrantTextRetriever(embedding=_FakeEmbedder(), vector_store=s, name="  ")


# ---------- arm factory ----------


def test_factory_default_is_the_single_historical_arm():
    s = _store()
    arms, weights = build_qdrant_text_arms(embedding=_FakeEmbedder(), vector_store=s)
    assert [a.name for a in arms] == ["qdrant_text"]
    assert arms[0].gate_key == "text" and arms[0].text_key == "text"
    assert weights == {}


def test_factory_multi_field_names_gates_and_weights():
    s = _store()
    arms, weights = build_qdrant_text_arms(
        embedding=_FakeEmbedder(),
        vector_store=s,
        fields={"title": 2.0, "text": 1.0},
    )
    assert [a.name for a in arms] == ["qdrant_text:title", "qdrant_text"]
    assert [a.gate_key for a in arms] == ["title", "text"]
    # Every arm RETURNS the body text regardless of its gate field.
    assert all(a.text_key == "text" for a in arms)
    # Weight 1.0 stays adaptive (no static override) — only the boost lands.
    assert weights == {"qdrant_text:title": 2.0}


# ---------- gate semantics ----------


def test_title_arm_gates_on_title_but_returns_body():
    s = _store()
    _seed_titled(s)
    arm = QdrantTextRetriever(
        embedding=_FakeEmbedder(),
        vector_store=s,
        text_key="text",
        gate_key="title",
        name="qdrant_text:title",
    )
    hits = arm.search("postgres", limit=10)
    # Gate matched TITLES: 1 and 3, not the body-only 2.
    assert {h.id for h in hits} == {1, 3}
    # ...but the result text is the chunk body, not the title.
    by_id = {h.id: h for h in hits}
    assert by_id[1].text == "backup restore runbook"
    assert all(h.sources == ["qdrant_text:title"] for h in hits)


def test_body_arm_unchanged_by_default():
    s = _store()
    _seed_titled(s)
    arm = QdrantTextRetriever(embedding=_FakeEmbedder(), vector_store=s)
    # Only point 2 carries "postgres" in its BODY (1 and 3 only in the title).
    assert {h.id for h in arm.search("postgres", limit=10)} == {2}


def test_title_gate_composes_with_tenant():
    s = _store()
    _seed_titled(s)
    arm = QdrantTextRetriever(
        embedding=_FakeEmbedder(), vector_store=s, gate_key="title"
    )
    assert [h.id for h in arm.search("postgres", limit=10, tenant="acme")] == [1]


def test_explain_empty_reports_the_instance_name():
    s = _store()
    arm = QdrantTextRetriever(
        embedding=_FakeEmbedder(), vector_store=s, gate_key="title", name="qdrant_text:title"
    )
    # A query with no gate-able tokens must carry THIS arm's name in the tag.
    assert arm.explain_empty("a b") == "qdrant_text:title:no_tokens"


def test_metric_names_sanitize_every_invalid_character():
    from mnemostack.recall.recaller import _metric_name

    assert _metric_name("qdrant_text:title") == "qdrant_text_title"
    # Field names are operator-configured — a slash (or anything else outside
    # the Prometheus grammar) must not reach the metric identifier either.
    assert _metric_name("qdrant_text:metadata/title") == "qdrant_text_metadata_title"
    assert _metric_name("vector") == "vector"


def test_sparse_and_bm25_sources_carry_the_instance_name():
    sp = _store(sparse=True)
    sp.upsert(1, _V1, {"text": "postgres backup"})
    arm = QdrantSparseRetriever(vector_store=sp, name="sparse:aux")
    hits = arm.search("postgres", limit=5)
    assert hits and all(h.sources == ["sparse:aux"] for h in hits)

    from mnemostack.recall.bm25 import BM25Doc

    bm = BM25Retriever(
        docs=[BM25Doc(id=1, text="postgres backup", payload={})], name="bm25:aux"
    )
    bhits = bm.search("postgres", limit=5)
    assert bhits and all(h.sources == ["bm25:aux"] for h in bhits)


def test_direct_retriever_synthesis_matches_the_lexical_family():
    from mnemostack.synthesis import _query_retrievers

    s = _store()
    _seed_titled(s)
    arm = QdrantTextRetriever(
        embedding=_FakeEmbedder(), vector_store=s, gate_key="title", name="qdrant_text:title"
    )
    # The "bm25" umbrella must select the suffixed arm on the DIRECT
    # retrievers path too, not only through a supplied Recaller.
    hits = _query_retrievers([arm], "postgres", 10, {"bm25"}, None)
    assert {h.id for h in hits} == {1, 3}
    assert _query_retrievers([arm], "postgres", 10, {"memgraph"}, None) == []


def test_multi_field_arms_share_one_query_embedding():
    calls = []

    class _CountingEmbedder(_FakeEmbedder):
        def embed(self, text):
            calls.append(text)
            return _V1

    s = _store()
    _seed_titled(s)
    arms, _ = build_qdrant_text_arms(
        embedding=_CountingEmbedder(),
        vector_store=s,
        fields={"title": 2.0, "text": 1.0},
    )
    for arm in arms:
        arm.search("postgres", limit=5)
    # Two arms, one query — one billable embedding call, not two.
    assert calls == ["postgres"]


def test_mcp_build_server_validates_fields_mode_eagerly():
    pytest.importorskip("fastmcp")
    from mnemostack.mcp.server import build_server

    with pytest.raises(ValueError, match="text_search=lexical"):
        build_server(
            text_search="sparse", text_search_fields={"title": 2.0}
        )


def test_cli_fields_reader_propagates_malformed_config(monkeypatch):
    cli = _cli_env(monkeypatch, MNEMOSTACK_TEXT_SEARCH_FIELDS="title:notanumber")
    # A malformed fields value must NOT silently degrade to the body-only
    # arm — the reader propagates, matching the servers' refusal to boot.
    with pytest.raises(ValueError, match="must be a number"):
        cli._text_search_fields()
    cli._text_search_fields.cache_clear()


# ---------- fusion weights + telemetry keying ----------


def test_recaller_weights_key_per_arm():
    from mnemostack.recall.recaller import Recaller

    s = _store()
    _seed_titled(s)
    arms, weights = build_qdrant_text_arms(
        embedding=_FakeEmbedder(),
        vector_store=s,
        fields={"title": 2.0, "text": 1.0},
    )
    rec = Recaller(retrievers=arms, retriever_weights=weights)
    assert rec._weight_for("qdrant_text:title", "postgres") == 2.0
    assert rec._weight_for("qdrant_text", "postgres") == 1.0
    results = rec.recall("postgres", limit=10)
    # Union of both gates: titles {1,3} + bodies {2,3}.
    assert {r.id for r in results} == {1, 2, 3}


# ---------- --source umbrella keeps suffixed arms ----------


def test_source_filter_umbrella_accepts_suffixed_arm_names():
    from mnemostack.synthesis import _result_source_enabled, _source_enabled

    assert _source_enabled("qdrant_text:title", {"bm25"})
    assert _source_enabled("qdrant_text", {"bm25"})
    assert not _source_enabled("memgraph", {"bm25"})

    class _R:
        sources = ["qdrant_text:title"]

    assert _result_source_enabled(_R(), {"bm25"})
    assert not _result_source_enabled(_R(), {"memgraph"})


# ---------- CLI wiring ----------


def _cli_env(monkeypatch, **env):
    import mnemostack.cli as cli

    for k, v in env.items():
        monkeypatch.setenv(k, v)
    cli._payload_schema.cache_clear()
    cli._text_search_mode.cache_clear()
    cli._text_search_fields.cache_clear()
    return cli


def test_cli_builds_multi_field_arms_with_weights(monkeypatch):
    import argparse

    cli = _cli_env(
        monkeypatch,
        MNEMOSTACK_TEXT_SEARCH="lexical",
        MNEMOSTACK_TEXT_SEARCH_FIELDS="title:2.0,text:1.0",
    )
    s = _store()
    _seed_titled(s)
    rec = cli._build_recaller(
        argparse.Namespace(collection="mf", vector_floor=0),
        _FakeEmbedder(),
        s,
    )
    names = [r.name for r in rec.retrievers]
    assert "qdrant_text:title" in names and "qdrant_text" in names
    assert rec.retriever_weights == {"qdrant_text:title": 2.0}
    cli._payload_schema.cache_clear()
    cli._text_search_mode.cache_clear()
    cli._text_search_fields.cache_clear()


def test_cli_rejects_fields_without_lexical_mode(monkeypatch):
    import argparse

    cli = _cli_env(
        monkeypatch,
        MNEMOSTACK_TEXT_SEARCH="sparse",
        MNEMOSTACK_TEXT_SEARCH_FIELDS="title:2.0",
    )
    with pytest.raises(ValueError, match="text_search=lexical"):
        cli._build_recaller(
            argparse.Namespace(collection="mf", vector_floor=0),
            _FakeEmbedder(),
            _store(),
        )
    cli._payload_schema.cache_clear()
    cli._text_search_mode.cache_clear()
    cli._text_search_fields.cache_clear()


def test_cmd_serve_forwards_fields_into_server_config(monkeypatch):
    # cmd_serve builds ServerConfig EXPLICITLY (never from_env) — a field
    # missing there silently degrades the HTTP surface to the body-only arm
    # while MCP honors the config (the pre-PR review's P1).
    import argparse

    pytest.importorskip("uvicorn")
    import mnemostack.server as srv

    cli = _cli_env(
        monkeypatch,
        MNEMOSTACK_TEXT_SEARCH="lexical",
        MNEMOSTACK_TEXT_SEARCH_FIELDS="title:2.0,text:1.0",
    )
    captured: dict = {}

    def _fake_build_app(cfg):
        captured["cfg"] = cfg
        raise RuntimeError("stop before uvicorn")

    monkeypatch.setattr(srv, "build_app", _fake_build_app)
    with pytest.raises(RuntimeError, match="stop before uvicorn"):
        cli.cmd_serve(
            argparse.Namespace(
                provider="gemini",
                model=None,
                embedding_model=None,
                llm="gemini",
                llm_model=None,
                collection="mf",
                qdrant="http://x",
                memgraph_uri="",
                graph_user="",
                graph_password="",
                graph_database=None,
                graph_timeout=5.0,
                qdrant_health_timeout=2,
                bm25_path=[],
                vector_floor=0,
                rerank_mode="relevant_only",
                token_budget=None,
                state_path=None,
                auto_record_ior=False,
                auth=False,
                keys_file=None,
                quotas_file=None,
                host="127.0.0.1",
                port=8000,
                reload=False,
            )
        )
    assert captured["cfg"].text_search_fields == {"title": 2.0, "text": 1.0}
    cli._payload_schema.cache_clear()
    cli._text_search_mode.cache_clear()
    cli._text_search_fields.cache_clear()


def test_cli_text_index_covers_every_gate_field(monkeypatch, capsys):
    import argparse

    cli = _cli_env(
        monkeypatch,
        MNEMOSTACK_TEXT_SEARCH="lexical",
        MNEMOSTACK_TEXT_SEARCH_FIELDS="title:2.0,text:1.0",
    )
    store = _store()
    indexed: list[str] = []
    monkeypatch.setattr(
        store, "ensure_text_index", lambda field=None: indexed.append(field)
    )
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    rc = cli.cmd_text_index(argparse.Namespace(collection="mf", qdrant="http://x"))
    assert rc == 0
    assert indexed == ["title", "text"]
    out = capsys.readouterr().out
    assert "'title'" in out and "'text'" in out
    cli._payload_schema.cache_clear()
    cli._text_search_mode.cache_clear()
    cli._text_search_fields.cache_clear()
