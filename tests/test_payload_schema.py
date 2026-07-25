"""Configurable payload schema — foreign collections with their own field
names (text/timestamp keys) and numeric epoch timestamps.

The contract under test: recall mounts a pre-existing Qdrant collection
without rewriting its payloads — keys are configurable end-to-end, timestamp
VALUES parse tolerantly everywhere (epoch int/ms, datetime, ISO), and the
temporal window filter is emitted in the collection's own domain (numeric
Range for epoch fields — a DatetimeRange never matches them).
"""

from __future__ import annotations

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
    # ISO bounds vs epoch value — used to TypeError → drop every candidate.
    assert payload_matches(
        payload, {"updated_at": {"gte": "2026-04-01", "lte": "2026-04-30"}}
    )
    assert not payload_matches(payload, {"updated_at": {"gte": "2026-05-01"}})
    # And the mirror image: epoch bounds vs ISO value.
    assert payload_matches(
        {"updated_at": "2026-04-15T12:00:00Z"}, {"updated_at": {"gte": _EPOCH - 60}}
    )
    # Genuinely incomparable stays excluded.
    assert not payload_matches({"updated_at": "junk"}, {"updated_at": {"gte": _EPOCH}})


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
