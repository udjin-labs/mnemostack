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
        {True: 1.0},  # YAML `on:`/`yes:` key parses as a boolean
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


def test_config_load_rejects_exact_duplicate_yaml_keys(tmp_path):
    # PyYAML silently keeps the LAST duplicate key, collapsing the mapping
    # before any parse-level duplicate check can see it — the strict loader
    # rejects it at load, for text_search_fields and every other config key.
    p = tmp_path / "dup.yaml"
    p.write_text(
        "recall:\n  text_search: lexical\n  text_search_fields:\n"
        "    title: 2.0\n    title: 3.0\n"
    )
    with pytest.raises(ValueError, match="duplicate key 'title'"):
        Config.load(p)
    p2 = tmp_path / "dup2.yaml"
    p2.write_text("vector:\n  collection: a\n  collection: b\n")
    with pytest.raises(ValueError, match="duplicate key 'collection'"):
        Config.load(p2)


def test_empty_env_value_clears_yaml_fields(monkeypatch, tmp_path):
    # Env has higher precedence than the file, and "" parses to an empty
    # mapping — the env-var way to clear YAML-configured fields (e.g. when
    # the env also switches the mode away from lexical, where inherited
    # fields would refuse startup).
    p = tmp_path / "c.yaml"
    p.write_text(
        "recall:\n  text_search: lexical\n  text_search_fields:\n    title: 2.0\n"
    )
    monkeypatch.setenv("MNEMOSTACK_TEXT_SEARCH_FIELDS", "")
    monkeypatch.setenv("MNEMOSTACK_TEXT_SEARCH", "sparse")
    cfg = Config.load(p)
    assert cfg.recall.text_search_fields == {}
    # The cleared mapping passes the fields/mode check under the new mode.
    ensure_text_fields_mode("sparse", cfg.recall.text_search_fields)


def test_strict_loader_accepts_yaml_merge_key_overrides(tmp_path):
    # An anchor-based config overriding one inherited key is VALID YAML —
    # the explicit override wins over the merged default and must not be
    # rejected as a duplicate.
    p = tmp_path / "merge.yaml"
    p.write_text(
        "defaults: &defaults\n  host: http://default:6333\n  collection: base\n"
        "vector:\n  <<: *defaults\n  host: http://override:6333\n"
    )
    cfg = Config.load(p)
    assert cfg.vector.host == "http://override:6333"
    assert cfg.vector.collection == "base"


def test_prometheus_labels_escape_configured_names():
    pytest.importorskip("fastapi")
    from mnemostack.server import _prometheus_dump

    class _FakeRecorder:
        def snapshot_counters(self):
            return {("mnemostack.degraded", ("reason", 'arm"with\nnewline\\x')): 3}

        def snapshot_histograms(self):
            return {}

    out = _prometheus_dump(_FakeRecorder())
    # One sample line, metacharacters escaped per the Prometheus text format —
    # an unescaped newline would inject an extra exposition line.
    sample = [line for line in out.splitlines() if not line.startswith("#")]
    assert sample == ['mnemostack_degraded_total{reason="arm\\"with\\nnewline\\\\x"} 3']


def test_bm25_from_qdrant_threads_the_name_override():
    s = _store()
    _seed_titled(s)
    bm = BM25Retriever.from_qdrant(s.client, "mf", name="bm25:aux")
    assert bm.name == "bm25:aux"
    assert BM25Retriever.from_qdrant(s.client, "mf").name == "bm25"


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


def test_name_override_rejects_cross_family_collisions():
    # The recaller keys semantics on exact identities ("vector" drives the
    # vector floor and the no-vector fallback) — a BM25 arm named "vector"
    # (or "vector:x", which the base-name filters would family-match) would
    # masquerade through them.
    for bad in ("vector", "vector:x", "sparse", "memgraph"):
        with pytest.raises(ValueError, match="reserved"):
            BM25Retriever(docs=[], name=bad)
    # A family's OWN name, suffixed or not, stays legal.
    assert BM25Retriever(docs=[], name="bm25:aux").name == "bm25:aux"


@pytest.mark.parametrize("bad", [",", ",,,", "title:2.0,", ",title", "title,,text"])
def test_parse_fields_rejects_empty_segments(bad):
    # Only a genuinely blank string clears the mapping — separator-only or
    # dangling-comma values are malformed template expansions.
    with pytest.raises(ValueError, match="empty comma-separated segment"):
        parse_text_search_fields(bad)


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


def test_metric_names_are_injectively_encoded():
    import re as _re

    from mnemostack.recall.recaller import _metric_name

    # Built-in names pass through VERBATIM — existing dashboards keep
    # their series identifiers.
    assert _metric_name("vector") == "vector"
    assert _metric_name("qdrant_text") == "qdrant_text"
    # Everything else lands in the disjoint arm_ namespace, inside the
    # Prometheus identifier grammar.
    encoded = _metric_name("qdrant_text:metadata/title")
    assert encoded.startswith("arm_") and _re.fullmatch(r"[A-Za-z0-9_]+", encoded)
    # Injective by construction: distinct names never merge into one series —
    # including a name vs the literal escape-text of another (the collision
    # verbatim passthrough allowed), and punctuation-only differences.
    pairs = [
        ("qdrant_text:metadata/title", "qdrant_text:metadata-title"),
        ("qdrant_text:@a*a^", "qdrant_text:#a.a:"),  # 16-bit-CRC collision pair
        ("a/", "a_2f"),  # name vs the literal escape encoding of that name
        ("qdrant_text:/", "qdrant__text_3a_2f"),
    ]
    for a, b in pairs:
        assert _metric_name(a) != _metric_name(b)
    # ...and the mapping is deterministic (dashboards need stable ids).
    assert _metric_name("qdrant_text:title") == _metric_name("qdrant_text:title")


def test_factory_validates_weights_at_the_programmatic_boundary():
    # ServerConfig/build_server/library callers bypass Config.load — an
    # invalid weight must fail HERE, not silently zero the arm in RRF.
    s = _store()
    for bad in ({"title": 0.0}, {"title": -1.0}, {"title": float("nan")}):
        with pytest.raises(ValueError):
            build_qdrant_text_arms(
                embedding=_FakeEmbedder(), vector_store=s, fields=bad
            )


def test_shared_embedding_is_single_flight_and_skips_failures():
    import threading

    calls = []
    release = threading.Event()

    class _SlowEmbedder(_FakeEmbedder):
        def embed(self, text):
            calls.append(text)
            release.wait(5)
            return _V1

    from mnemostack.recall.retrievers import _SharedQueryEmbedding

    shared = _SharedQueryEmbedding(_SlowEmbedder())
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(shared.embed("q")))
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    # All four are in flight before the provider returns — the race the
    # sequential test masked.
    release.set()
    for t in threads:
        t.join()
    assert calls == ["q"]  # one leader, three waiters
    assert results == [_V1] * 4

    # Failures (empty vector) are returned but NOT memoized — the next call
    # retries the recovered provider instead of pinning the arms dead.
    flaky_calls = []

    class _FlakyEmbedder(_FakeEmbedder):
        def embed(self, text):
            flaky_calls.append(text)
            return [] if len(flaky_calls) == 1 else _V1

    shared2 = _SharedQueryEmbedding(_FlakyEmbedder())
    assert shared2.embed("q") == []
    assert shared2.embed("q") == _V1
    assert len(flaky_calls) == 2
    # ...while successes ARE memoized.
    assert shared2.embed("q") == _V1
    assert len(flaky_calls) == 2


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
    # Invalid WEIGHTS must also fail at build time, before the server
    # advertises tools whose every recall would then error.
    with pytest.raises(ValueError, match="positive finite"):
        build_server(
            text_search="lexical", text_search_fields={"title": 0.0}
        )


def test_cli_text_index_refuses_fields_mode_mismatch(monkeypatch, capsys):
    import argparse

    cli = _cli_env(
        monkeypatch,
        MNEMOSTACK_TEXT_SEARCH="sparse",
        MNEMOSTACK_TEXT_SEARCH_FIELDS="title:2.0",
    )
    store = _store()
    indexed: list[str] = []
    monkeypatch.setattr(
        store, "ensure_text_index", lambda field=None: indexed.append(field)
    )
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    rc = cli.cmd_text_index(argparse.Namespace(collection="mf", qdrant="http://x"))
    # Refused BEFORE mutating Qdrant — no index created, no false success.
    assert rc == 2
    assert indexed == []
    assert "text_search=lexical" in capsys.readouterr().err
    cli._payload_schema.cache_clear()
    cli._text_search_mode.cache_clear()
    cli._text_search_fields.cache_clear()


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
