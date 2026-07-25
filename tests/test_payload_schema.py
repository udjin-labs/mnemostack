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
import sys
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


def test_temporal_bound_preserves_fractional_precision():
    # Bounds are emitted as EXACT floats: integral rounding in either
    # direction is lossy — truncation widens the scope (admits the excluded
    # half-second), inward rounding narrows it (a time.time() payload at
    # 23:59:59.5 would fall outside an lte cut to 23:59:59).
    r = TemporalRetriever(
        embedding=_FakeEmbedder(),
        vector_store=_foreign_store(),
        timestamp_format="epoch",
    )
    half = datetime(2026, 4, 15, 12, 0, 0, 500000, tzinfo=timezone.utc)
    assert r._bound(half) == half.timestamp()  # .5 kept exactly
    r_ms = TemporalRetriever(
        embedding=_FakeEmbedder(),
        vector_store=_foreign_store(),
        timestamp_format="epoch_ms",
    )
    assert r_ms._bound(half) == half.timestamp() * 1000


def test_vector_retriever_converts_iso_bounds_for_epoch_field():
    # The NORMAL vector arm (not just temporal/BM25): an ISO caller constraint
    # over a numeric field must be converted to the collection domain, or
    # Qdrant builds a DatetimeRange that matches nothing.
    store = _foreign_store()
    r = VectorRetriever(
        embedding=_FakeEmbedder(),
        vector_store=store,
        text_key="content",
        timestamp_key="updated_at",
        timestamp_format="epoch",
    )
    hits = r.search(
        "anything", limit=5,
        filters={"updated_at": {"gte": "2026-04-01", "lte": "2026-04-30"}},
    )
    assert [h.text for h in hits] == ["april note"]


def test_string_epoch_beats_iso_lookalike_under_explicit_unit():
    # "20240101" is valid basic-ISO AND a number: an explicit unit means the
    # field is numeric, so the numeric reading wins; auto keeps ISO-first.
    dt = parse_payload_instant("20240101", numeric_unit="ms")
    assert dt is not None and dt.year == 1970
    # auto mode: ISO-first — but basic-format ISO ("20240101") is only parsed
    # by fromisoformat on 3.11+; on 3.10 auto legitimately reads it numerically.
    dt = parse_payload_instant("20240101")
    assert dt is not None
    if sys.version_info >= (3, 11):
        assert dt.strftime("%Y-%m-%d") == "2024-01-01"


def test_convert_filter_leaves_same_domain_values_verbatim():
    from mnemostack.recall.retrievers import convert_timestamp_filter

    # Exact ISO MatchValue is string equality: a same-domain value must pass
    # through untouched ("...Z" != a rewritten "...+00:00").
    f = {"timestamp": "2026-04-01T00:00:00Z"}
    assert convert_timestamp_filter(
        f, timestamp_key="timestamp", timestamp_format="iso"
    ) == f
    # Same-domain numerics on an epoch collection stay verbatim (int stays int).
    f2 = {"updated_at": _EPOCH}
    assert convert_timestamp_filter(
        f2, timestamp_key="updated_at", timestamp_format="epoch"
    ) == {"updated_at": _EPOCH}
    # Cross-domain still converts — and an integral instant emits as int.
    out = convert_timestamp_filter(
        {"updated_at": {"gte": "2026-04-15T12:00:00Z"}},
        timestamp_key="updated_at",
        timestamp_format="epoch",
    )
    assert out == {"updated_at": {"gte": _EPOCH}} and isinstance(out["updated_at"]["gte"], int)


def test_bm25_from_qdrant_converts_iso_window_for_epoch_collection():
    from mnemostack.recall.retrievers import bm25_docs_from_qdrant

    store = _foreign_store()
    docs = bm25_docs_from_qdrant(
        store.client, "foreign",
        text_key="content",
        timestamp_key="updated_at",
        timestamp_format="epoch",
        newer_than="2026-04-01T00:00:00Z",  # documented ISO window over epoch field
    )
    assert [d.text for d in docs] == ["april note"]  # corpus not silently empty


def test_inspector_records_convert_timestamp_filters(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import mnemostack.inspector as insp
    from mnemostack.server import ServerConfig

    store = _foreign_store()
    monkeypatch.setattr(insp, "VectorStore", lambda **_: store)
    monkeypatch.setattr(insp, "_make_probe_client", lambda *a, **k: store.client)
    app = insp.build_inspector_app(
        ServerConfig(
            provider_name="fake", collection="foreign", graph_uri=None,
            text_key="content", timestamp_key="updated_at", timestamp_format="epoch",
        )
    )
    c = TestClient(app)
    d = c.get(
        "/api/records",
        params={"filters": '{"updated_at": {"gte": "2026-04-01", "lte": "2026-04-30"}}'},
    ).json()
    assert [r["text"] for r in d["records"]] == ["april note"]


def test_payload_matches_exact_timestamp_across_domains():
    # An exact ISO condition and an epoch payload naming the same moment must
    # match on the timestamp key (BM25/post-pipeline mirror of the vector
    # arm's converted filter); other keys keep plain equality.
    assert payload_matches(
        {"updated_at": _EPOCH},
        {"updated_at": "2026-04-15T12:00:00Z"},
        timestamp_key="updated_at",
    )
    assert not payload_matches(
        {"updated_at": _EPOCH},
        {"updated_at": "2026-04-16T12:00:00Z"},
        timestamp_key="updated_at",
    )
    assert not payload_matches({"kind": "42"}, {"kind": 42})  # non-timestamp: strict


def test_convert_filter_exact_cross_domain_becomes_degenerate_range():
    from mnemostack.recall.retrievers import convert_timestamp_filter

    # ISO collection + numeric exact value: scalar MatchValue would be
    # representation-sensitive — a degenerate DatetimeRange compares instants.
    out = convert_timestamp_filter(
        {"timestamp": _EPOCH}, timestamp_key="timestamp", timestamp_format="iso"
    )
    cond = out["timestamp"]
    assert isinstance(cond, dict) and cond["gte"] == cond["lte"]
    # Epoch collection + ISO exact value: degenerate numeric Range.
    out = convert_timestamp_filter(
        {"updated_at": "2026-04-15T12:00:00Z"},
        timestamp_key="updated_at",
        timestamp_format="epoch",
    )
    assert out["updated_at"] == {"gte": _EPOCH, "lte": _EPOCH}
    # Vector e2e: exact ISO over the epoch field finds the point.
    store = _foreign_store()
    r = VectorRetriever(
        embedding=_FakeEmbedder(), vector_store=store, text_key="content",
        timestamp_key="updated_at", timestamp_format="epoch",
    )
    hits = r.search("x", limit=5, filters={"updated_at": "2026-04-15T12:00:00Z"})
    assert [h.text for h in hits] == ["april note"]


def test_convert_filter_numeric_string_on_iso_collection():
    from mnemostack.recall.retrievers import convert_timestamp_filter

    # A JSON-string epoch on an ISO collection is cross-domain: it becomes a
    # degenerate DatetimeRange (a string MatchValue could never equal RFC3339).
    out = convert_timestamp_filter(
        {"timestamp": str(_EPOCH)}, timestamp_key="timestamp", timestamp_format="iso"
    )
    cond = out["timestamp"]
    assert isinstance(cond, dict) and cond["gte"] == cond["lte"]
    assert cond["gte"].startswith("2026-04-15")
    # A native ISO string still passes verbatim.
    f = {"timestamp": "2026-04-15T12:00:00Z"}
    assert convert_timestamp_filter(
        f, timestamp_key="timestamp", timestamp_format="iso"
    ) == f


def test_synthesis_derives_schema_from_direct_retrievers():
    # The documented retrievers=[...] construction (no recaller) must also
    # carry the schema into facts/timeline.
    from mnemostack.synthesis import synthesize

    store = _foreign_store()
    retr = VectorRetriever(
        embedding=_FakeEmbedder(), vector_store=store, text_key="content",
        timestamp_key="updated_at", timestamp_format="epoch",
    )
    result = synthesize("april", retrievers=[retr])
    assert result.facts
    assert all(f.timestamp is not None for f in result.facts)
    assert all("content" not in f.metadata for f in result.facts)
    assert result.timeline


def test_recaller_derives_schema_from_its_retrievers():
    # Retriever-mode: a schema configured on the INNER retriever must reach
    # the outer Recaller (filter mirrors, post-pipeline backstop) without the
    # caller repeating it.
    store = _foreign_store()
    inner = VectorRetriever(
        embedding=_FakeEmbedder(), vector_store=store, text_key="content",
        timestamp_key="updated_at", timestamp_format="epoch",
    )
    rec = Recaller(retrievers=[inner])
    assert (rec.text_key, rec.timestamp_key, rec.timestamp_format) == (
        "content", "updated_at", "epoch",
    )
    # explicit outer args still win
    rec2 = Recaller(retrievers=[inner], timestamp_key="ts")
    assert rec2.timestamp_key == "ts"
    # post-pipeline flow backstop keeps the epoch hit under an ISO range
    from mnemostack.recall.flow import recall_flow

    results = recall_flow(
        rec, "april", limit=5,
        filters={"updated_at": {"gte": "2026-04-01", "lte": "2026-04-30"}},
    )
    assert [r.text for r in results] == ["april note"]


def test_exact_fractional_epoch_becomes_degenerate_range():
    from mnemostack.recall.retrievers import convert_timestamp_filter

    # Qdrant MatchValue rejects floats — a same-domain fractional exact value
    # must ride as a degenerate numeric range instead.
    out = convert_timestamp_filter(
        {"updated_at": 1776254400.5},
        timestamp_key="updated_at",
        timestamp_format="epoch",
    )
    assert out["updated_at"] == {"gte": 1776254400.5, "lte": 1776254400.5}
    # Same-domain INT exacts stay scalar (MatchValue admits ints).
    out = convert_timestamp_filter(
        {"updated_at": _EPOCH}, timestamp_key="updated_at", timestamp_format="epoch"
    )
    assert out["updated_at"] == _EPOCH


def test_answer_generator_derives_schema_from_recaller():
    from mnemostack.recall.answer import AnswerGenerator

    class _FakeLLM:
        def generate(self, *a, **k):
            raise NotImplementedError

    class _SchemaRecaller:
        timestamp_key = "updated_at"
        timestamp_format = "epoch"

    gen = AnswerGenerator(llm=_FakeLLM(), recaller=_SchemaRecaller())
    assert gen.timestamp_key == "updated_at" and gen.timestamp_format == "epoch"
    # explicit kwargs still win
    gen2 = AnswerGenerator(
        llm=_FakeLLM(), recaller=_SchemaRecaller(), timestamp_key="ts"
    )
    assert gen2.timestamp_key == "ts"
    # no recaller: standard defaults
    gen3 = AnswerGenerator(llm=_FakeLLM())
    assert gen3.timestamp_key == "timestamp" and gen3.timestamp_format == "iso"


def test_synthesis_bm25_docs_get_the_schema():
    from mnemostack.recall.bm25 import BM25Doc
    from mnemostack.synthesis import _build_recaller_from_kwargs

    rec = _build_recaller_from_kwargs(
        None,
        {
            "bm25_docs": [BM25Doc(id="1", text="acme note", payload={"updated_at": _EPOCH})],
            "timestamp_key": "updated_at",
            "timestamp_format": "epoch",
        },
    )
    assert rec is not None
    bm25_retr = rec.retrievers[0]
    assert bm25_retr.timestamp_key == "updated_at"
    assert bm25_retr.timestamp_format == "epoch"
    # mixed-domain filter now matches through the BM25 predicate
    hits = bm25_retr.search("acme", filters={"updated_at": {"gte": "2026-04-01"}})
    assert [h.text for h in hits] == ["acme note"]


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


def test_synthesis_source_filter_preserves_the_schema():
    # sources= rebuilds the recaller — the rebuilt one must carry the
    # original's schema, or the timeline/metadata silently reset to defaults.
    from mnemostack.recall.retrievers import VectorRetriever as _VR
    from mnemostack.synthesis import _filter_recaller

    rec = Recaller(
        retrievers=[_VR(embedding=_FakeEmbedder(), vector_store=_foreign_store())],
        text_key="content",
        timestamp_key="updated_at",
        timestamp_format="epoch",
    )
    rebuilt = _filter_recaller(rec, {"vector"})
    assert rebuilt is not rec
    assert (rebuilt.text_key, rebuilt.timestamp_key, rebuilt.timestamp_format) == (
        "content", "updated_at", "epoch",
    )


def test_bm25_from_qdrant_forwards_timestamp_format():
    from mnemostack.recall.retrievers import BM25Retriever

    store = _foreign_store()
    r = BM25Retriever.from_qdrant(
        store.client, "foreign", text_key="content",
        timestamp_key="updated_at", timestamp_format="epoch_ms",
    )
    assert r.timestamp_key == "updated_at" and r.timestamp_format == "epoch_ms"


def test_doctor_flags_invalid_timestamp_format(monkeypatch, capsys):
    import mnemostack.cli as cli

    monkeypatch.setenv("MNEMOSTACK_TIMESTAMP_FORMAT", "unixtime")
    rc = cli.cmd_doctor(
        argparse.Namespace(
            json=True, provider="gemini", embedding_model=None,
            qdrant="http://localhost:1", collection="mt",
            memgraph_uri=None, graph_timeout=1.0, timeout=1,
        )
    )
    out = capsys.readouterr().out
    assert '"config.timestamp_format"' in out.replace("'", '"') or "timestamp_format" in out
    assert "unixtime" in out
    assert rc == 2  # misconfig exit, not a false green


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
