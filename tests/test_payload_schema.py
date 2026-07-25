"""Configurable payload schema — foreign collections with their own field
names (text/timestamp keys) and numeric epoch timestamps.

The contract under test: recall mounts a pre-existing Qdrant collection
without rewriting its payloads — keys are configurable end-to-end, timestamp
VALUES parse tolerantly everywhere (epoch int/ms, datetime, ISO), and the
temporal window filter is emitted in the collection's own domain (numeric
Range for epoch fields — a DatetimeRange never matches them).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance

from mnemostack.recall.filters import payload_matches
from mnemostack.recall.recaller import Recaller
from mnemostack.recall.retrievers import TemporalRetriever, VectorRetriever
from mnemostack.recall.validity import parse_payload_instant
from mnemostack.vector import VectorStore

_VEC = [1.0, 0.0, 0.0, 0.0]

#: 2026-04-15 12:00:00 UTC
_EPOCH = 1776254400


class _FakeEmbedder:
    dimension = 4
    name = "fake:schema"

    def embed(self, text):
        return _VEC

    def embed_batch(self, texts):
        return [_VEC for _ in texts]


def _foreign_store() -> VectorStore:
    """In-memory collection with a FOREIGN schema: content/updated_at(epoch)."""
    s = VectorStore.__new__(VectorStore)
    s.collection = "foreign"
    s.dimension = 4
    s.distance = Distance.COSINE
    s.client = QdrantClient(":memory:")
    s.ensure_collection()
    s.upsert(1, _VEC, {"content": "april note", "updated_at": _EPOCH})
    s.upsert(2, _VEC, {"content": "march note", "updated_at": _EPOCH - 40 * 86400})
    return s


# ---------- parse_payload_instant ----------


@pytest.mark.parametrize(
    ("value", "expected_date"),
    [
        ("2026-04-15T12:00:00+00:00", "2026-04-15"),
        ("2026-04-15T12:00:00Z", "2026-04-15"),
        ("2026-04-15", "2026-04-15"),
        (_EPOCH, "2026-04-15"),  # epoch seconds (int)
        (float(_EPOCH), "2026-04-15"),  # epoch seconds (float)
        (_EPOCH * 1000, "2026-04-15"),  # epoch MILLISECONDS, by magnitude
        (str(_EPOCH), "2026-04-15"),  # numeric string
        (datetime(2026, 4, 15, 12, tzinfo=timezone.utc), "2026-04-15"),
        (datetime(2026, 4, 15, 12), "2026-04-15"),  # naive → assumed UTC
    ],
)
def test_parse_payload_instant_accepts_every_common_shape(value, expected_date):
    dt = parse_payload_instant(value)
    assert dt is not None and dt.tzinfo is not None
    assert dt.astimezone(timezone.utc).strftime("%Y-%m-%d") == expected_date


@pytest.mark.parametrize(
    "value",
    [None, True, False, "not a time", "", float("nan"), float("inf"), [], {}],
)
def test_parse_payload_instant_rejects_non_instants(value):
    assert parse_payload_instant(value) is None


# ---------- crash regressions (epoch int in a standard-key payload) ----------


def test_freshness_blend_survives_epoch_timestamps():
    # An epoch int has no .tzinfo — this used to AttributeError out of
    # pipeline.apply and fail the whole recall request.
    from mnemostack.recall.pipeline.stages import FreshnessBlend
    from mnemostack.recall.recaller import RecallResult

    class _Ctx:
        extras: dict = {}

    stage = FreshnessBlend()
    results = [RecallResult(id=1, text="x", score=1.0, payload={"timestamp": _EPOCH})]
    out = stage.apply(_Ctx(), results)
    assert out and "freshness" in out[0].payload  # decayed normally, no crash


def test_answer_display_ts_renders_epoch():
    from mnemostack.recall.answer import _display_ts

    assert _display_ts(_EPOCH) == "2026-04-15 12:00"
    assert _display_ts("2026-04-15") == "2026-04-15"
    assert _display_ts("garbage-string") == "garbage-st"  # historical fallback


def test_specificity_format_survives_epoch():
    from mnemostack.recall.recaller import RecallResult
    from mnemostack.recall.specificity import _format_memories

    out = _format_memories(
        [RecallResult(id=1, text="note", score=1.0, payload={"timestamp": _EPOCH})]
    )
    assert "[2026-04-15]" in out  # used to TypeError on ts[:10]


def test_curiosity_boost_reads_epoch_age():
    from mnemostack.recall.pipeline.stages import CuriosityBoost
    from mnemostack.recall.recaller import RecallResult

    class _Store:
        def get(self, key):
            return []

    class _Ctx:
        extras: dict = {}

    old_epoch = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp())
    stage = CuriosityBoost(state_store=_Store())
    results = [RecallResult(id=1, text="x", score=1.0, payload={"timestamp": old_epoch})]
    out = stage.apply(_Ctx(), results)
    assert out[0].payload.get("curiosity_boosted") is True  # full bonus, not the half


# ---------- in-memory filter mirror (BM25 path) ----------


def test_payload_matches_intersects_mixed_timestamp_domains():
    payload = {"updated_at": _EPOCH}
    kw = {"timestamp_key": "updated_at"}
    # ISO bounds vs epoch value — used to TypeError → drop every candidate.
    assert payload_matches(
        payload, {"updated_at": {"gte": "2026-04-01", "lte": "2026-04-30"}}, **kw
    )
    assert not payload_matches(payload, {"updated_at": {"gte": "2026-05-01"}}, **kw)
    # And the mirror image: epoch bounds vs ISO value.
    assert payload_matches(
        {"updated_at": "2026-04-15T12:00:00Z"}, {"updated_at": {"gte": _EPOCH - 60}}, **kw
    )
    # Genuinely incomparable stays excluded.
    assert not payload_matches({"updated_at": "junk"}, {"updated_at": {"gte": _EPOCH}}, **kw)


def test_payload_matches_never_coerces_non_timestamp_fields():
    # The instant fallback is scoped to THE timestamp key only: a numeric-
    # looking string in an unrelated field must not sneak past a numeric range
    # as an accidental "epoch" — strict Qdrant semantics (incomparable →
    # excluded) hold everywhere else.
    assert not payload_matches({"priority": "42"}, {"priority": {"gte": 0, "lte": 100}})
    assert not payload_matches(
        {"priority": "42"},
        {"priority": {"gte": 0, "lte": 100}},
        timestamp_key="updated_at",  # unrelated schema key changes nothing
    )
    # Same-domain comparisons on arbitrary fields still work as before.
    assert payload_matches({"priority": 42}, {"priority": {"gte": 0, "lte": 100}})


# ---------- retriever text_key ----------


def test_vector_retriever_reads_configured_text_key():
    store = _foreign_store()
    r = VectorRetriever(embedding=_FakeEmbedder(), vector_store=store, text_key="content")
    hits = r.search("anything", limit=5)
    assert {h.text for h in hits} == {"april note", "march note"}


def test_recaller_legacy_path_reads_configured_text_key():
    store = _foreign_store()
    rec = Recaller(
        embedding_provider=_FakeEmbedder(), vector_store=store, text_key="content"
    )
    results = rec.recall("anything", limit=5)
    assert results and all(r.text for r in results)
    assert "april note" in {r.text for r in results}


# ---------- temporal retriever over a foreign epoch schema ----------


def _april_window(_query):
    return (
        "2026-04-01T00:00:00+00:00",
        "2026-04-30T23:59:59+00:00",
    )


def test_temporal_retriever_epoch_format_matches_numeric_field():
    store = _foreign_store()
    r = TemporalRetriever(
        embedding=_FakeEmbedder(),
        vector_store=store,
        extractor=_april_window,
        text_key="content",
        timestamp_key="updated_at",
        timestamp_format="epoch",
    )
    hits = r.search("what happened in april", limit=5)
    # Numeric Range filter over the epoch field: April point in, March out.
    assert [h.text for h in hits] == ["april note"]
    assert all(h.payload.get("temporal_match") for h in hits)


def test_temporal_retriever_iso_format_would_miss_the_numeric_field():
    # The reason timestamp_format exists: ISO (DatetimeRange) bounds cannot
    # match a numeric payload field — configured wrong, temporal goes empty.
    store = _foreign_store()
    r = TemporalRetriever(
        embedding=_FakeEmbedder(),
        vector_store=store,
        extractor=_april_window,
        text_key="content",
        timestamp_key="updated_at",  # right key, wrong format:
        timestamp_format="iso",
    )
    assert r.search("what happened in april", limit=5) == []


def test_temporal_retriever_intersects_epoch_caller_bounds_with_iso_window():
    # A caller filter in epoch domain against the extractor's ISO window used
    # to TypeError → silently zero temporal contribution.
    store = _foreign_store()
    r = TemporalRetriever(
        embedding=_FakeEmbedder(),
        vector_store=store,
        extractor=_april_window,
        text_key="content",
        timestamp_key="updated_at",
        timestamp_format="epoch",
    )
    hits = r.search(
        "april", limit=5, filters={"updated_at": {"gte": _EPOCH - 86400}}
    )
    assert [h.text for h in hits] == ["april note"]
    # An out-of-window caller scope stays empty (intersection is honored).
    assert r.search("april", limit=5, filters={"updated_at": {"gte": _EPOCH + 86400}}) == []


def test_temporal_retriever_rejects_unknown_timestamp_format():
    with pytest.raises(ValueError, match="timestamp_format"):
        TemporalRetriever(
            embedding=_FakeEmbedder(),
            vector_store=_foreign_store(),
            timestamp_format="unixtime",
        )


# ---------- config threading ----------


def test_recall_config_env_overrides(monkeypatch):
    from mnemostack.config import Config

    monkeypatch.setenv("MNEMOSTACK_TEXT_KEY", "content")
    monkeypatch.setenv("MNEMOSTACK_TIMESTAMP_KEY", "updated_at")
    monkeypatch.setenv("MNEMOSTACK_TIMESTAMP_FORMAT", "epoch")
    cfg = Config.load()
    assert cfg.recall.text_key == "content"
    assert cfg.recall.timestamp_key == "updated_at"
    assert cfg.recall.timestamp_format == "epoch"


def test_server_config_mirrors_payload_schema(monkeypatch):
    pytest.importorskip("fastapi")
    from mnemostack.server import ServerConfig

    monkeypatch.setenv("MNEMOSTACK_TEXT_KEY", "content")
    monkeypatch.setenv("MNEMOSTACK_TIMESTAMP_KEY", "updated_at")
    monkeypatch.setenv("MNEMOSTACK_TIMESTAMP_FORMAT", "epoch_ms")
    cfg = ServerConfig.from_env()
    assert (cfg.text_key, cfg.timestamp_key, cfg.timestamp_format) == (
        "content",
        "updated_at",
        "epoch_ms",
    )


def test_example_config_documents_the_schema_keys():
    from mnemostack.config import generate_example_config

    yaml_text = generate_example_config()
    assert "text_key" in yaml_text and "timestamp_format" in yaml_text


# ---------- review-driven regressions ----------


def test_parse_payload_instant_pre_2001_milliseconds():
    # 946684800000 = 2000-01-01 in ms — BELOW the 1e12 magnitude threshold,
    # and out of datetime range as seconds (year 31969) → must retry as ms.
    dt = parse_payload_instant(946684800000)
    assert dt is not None and dt.strftime("%Y-%m-%d") == "2000-01-01"
    # 1e11 = 1973 in ms but a VALID (implausible) year-5138 datetime in
    # seconds — the implausible-future guard prefers the ms reading.
    dt = parse_payload_instant(100000000000)
    assert dt is not None and dt.year == 1973
    # Epoch 0 is a real instant, not "missing".
    dt = parse_payload_instant(0)
    assert dt is not None and dt.strftime("%Y-%m-%d") == "1970-01-01"


def test_epoch_zero_keeps_its_date_prefix():
    from mnemostack.recall.answer import AnswerGenerator
    from mnemostack.recall.recaller import RecallResult

    out = AnswerGenerator._format_context(
        [RecallResult(id=1, text="genesis", score=1.0, payload={"timestamp": 0})]
    )
    assert "[1970-01-01]" in out  # epoch 0 is falsy but real


def test_curiosity_boost_full_bonus_for_epoch_zero():
    from mnemostack.recall.pipeline.stages import CuriosityBoost
    from mnemostack.recall.recaller import RecallResult

    class _Store:
        def get(self, key):
            return []

    class _Ctx:
        extras: dict = {}

    stage = CuriosityBoost(state_store=_Store())
    results = [RecallResult(id=1, text="x", score=1.0, payload={"timestamp": 0})]
    out = stage.apply(_Ctx(), results)
    assert out[0].payload.get("curiosity_boosted") is True  # not the half-bonus path


def test_synthesis_facts_and_timeline_respect_the_schema():
    from mnemostack.recall.recaller import RecallResult
    from mnemostack.synthesis import _facts_from_results, _timeline

    results = [
        RecallResult(
            id=1,
            text="newer fact",
            score=0.9,
            payload={"content": "newer fact", "updated_at": _EPOCH},
        ),
        RecallResult(
            id=2,
            text="older fact",
            score=0.8,
            payload={"content": "older fact", "updated_at": _EPOCH - 86400 * 30},
        ),
    ]
    facts = _facts_from_results(results, text_key="content", timestamp_key="updated_at")
    assert all(f.timestamp is not None for f in facts)  # configured key was read
    assert all("content" not in f.metadata for f in facts)  # text not duplicated
    timeline = _timeline(facts)
    assert [f.text for f in timeline] == ["older fact", "newer fact"]  # epoch-sorted


def test_cmd_serve_threads_the_schema_into_server_config(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")  # cmd_serve imports it before building the config
    import mnemostack.cli as cli

    cli._payload_schema.cache_clear()
    monkeypatch.setenv("MNEMOSTACK_TEXT_KEY", "content")
    monkeypatch.setenv("MNEMOSTACK_TIMESTAMP_KEY", "updated_at")
    monkeypatch.setenv("MNEMOSTACK_TIMESTAMP_FORMAT", "epoch")
    captured = {}

    def _fake_build_app(cfg):
        captured["cfg"] = cfg
        raise SystemExit(0)  # stop before uvicorn

    monkeypatch.setattr("mnemostack.server.build_app", _fake_build_app)
    ns = argparse.Namespace(
        provider="gemini", embedding_model=None, llm="gemini", llm_model=None,
        collection="mt", qdrant="http://localhost:6333", memgraph_uri=None,
        graph_timeout=5.0, graph_user=None, graph_password=None,
        graph_database=None, bm25_path=[], vector_floor=0,
        rerank_mode="relevant_only", token_budget=None, state_path=None,
        auto_record_ior=False, host="127.0.0.1", port=8000, auth=False,
        keys_file=None, quotas_file=None, qdrant_health_timeout=2,
    )
    with pytest.raises(SystemExit):
        cli.cmd_serve(ns)
    cfg = captured["cfg"]
    assert (cfg.text_key, cfg.timestamp_key, cfg.timestamp_format) == (
        "content", "updated_at", "epoch",
    )
    cli._payload_schema.cache_clear()


def test_parse_payload_instant_configured_unit_beats_heuristic():
    # 86400000 = 1970-01-02 in ms, but a plausible 1972 as seconds — only the
    # CONFIGURED unit can disambiguate early millisecond epochs.
    dt = parse_payload_instant(86400000, numeric_unit="ms")
    assert dt is not None and dt.strftime("%Y-%m-%d") == "1970-01-02"
    dt = parse_payload_instant(86400000, numeric_unit="s")
    assert dt is not None and dt.year == 1972  # explicit seconds honored too
    from mnemostack.recall.validity import numeric_unit_for

    assert numeric_unit_for("epoch_ms") == "ms"
    assert numeric_unit_for("epoch") == "s"
    assert numeric_unit_for("iso") == "auto"


def test_temporal_bound_rounds_inward_on_fractional_edges():
    # int() truncation of a fractional bound WIDENS the window: gte .5 must
    # round UP, lte .5 must round DOWN — never admit the excluded half-second.
    r = TemporalRetriever(
        embedding=_FakeEmbedder(),
        vector_store=_foreign_store(),
        timestamp_format="epoch",
    )
    half = datetime(2026, 4, 15, 12, 0, 0, 500000, tzinfo=timezone.utc)
    assert r._bound(half, "gte") == int(half.timestamp()) + 1  # rounds up
    assert r._bound(half, "lte") == int(half.timestamp())  # rounds down
    whole = datetime(2026, 4, 15, 12, tzinfo=timezone.utc)
    assert r._bound(whole, "gte") == r._bound(whole, "lte")  # exact unchanged


def test_filters_cross_domain_strings_compare_as_instants():
    # Two STRINGS from different domains compare lexicographically without a
    # TypeError — the timestamp key must compare on the time line instead:
    # an ISO 2025 payload is NOT >= a numeric-string bound meaning Apr 2026.
    assert not payload_matches(
        {"timestamp": "2025-01-01T00:00:00Z"},
        {"timestamp": {"gte": str(_EPOCH)}},
    )
    assert payload_matches(
        {"timestamp": "2026-05-01T00:00:00Z"},
        {"timestamp": {"gte": str(_EPOCH)}},
    )


def test_mcp_standalone_main_threads_the_schema(monkeypatch):
    pytest.importorskip("mcp")
    import mnemostack.mcp.server as mcp_mod

    monkeypatch.setenv("MNEMOSTACK_TEXT_KEY", "content")
    monkeypatch.setenv("MNEMOSTACK_TIMESTAMP_KEY", "updated_at")
    monkeypatch.setenv("MNEMOSTACK_TIMESTAMP_FORMAT", "epoch")
    captured = {}

    class _FakeServer:
        def run(self):
            pass

    def _fake_build_server(**kwargs):
        captured.update(kwargs)
        return _FakeServer()

    monkeypatch.setattr(mcp_mod, "build_server", _fake_build_server)
    mcp_mod.main()
    assert captured["text_key"] == "content"
    assert captured["timestamp_key"] == "updated_at"
    assert captured["timestamp_format"] == "epoch"


def test_synthesis_derives_schema_from_a_supplied_recaller():
    # A caller passing a Recaller already configured for a foreign collection
    # (as the CLI does) must not have to repeat the keys — the facts/timeline
    # read the recaller's schema.
    from mnemostack.recall.recaller import RecallResult
    from mnemostack.synthesis import synthesize

    class _SchemaRecaller:
        text_key = "content"
        timestamp_key = "updated_at"
        timestamp_format = "epoch"
        retrievers: list = []

        def recall(self, query, limit=10, filters=None):
            return [
                RecallResult(
                    id=1,
                    text="acme launched the beta",
                    score=0.9,
                    payload={"content": "acme launched the beta", "updated_at": _EPOCH},
                )
            ]

    result = synthesize("acme", recaller=_SchemaRecaller())
    assert result.facts and result.facts[0].timestamp == str(_EPOCH)  # key was read
    assert all("content" not in f.metadata for f in result.facts)
    assert result.timeline  # epoch timestamp sorted, not dropped


def test_cli_payload_schema_warns_on_broken_config(monkeypatch, tmp_path, capsys):
    import mnemostack.cli as cli
    from mnemostack.config import Config

    cli._payload_schema.cache_clear()

    def _boom(cls):
        raise ValueError("broken yaml")

    monkeypatch.setattr(Config, "load", classmethod(_boom))
    assert cli._payload_schema() == ("text", "timestamp", "iso")
    assert "NOT in effect" in capsys.readouterr().err  # loud, not silent
    cli._payload_schema.cache_clear()
