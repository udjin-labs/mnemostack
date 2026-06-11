"""Tests for the streaming ingest API.

We don't talk to a real Qdrant — a fake VectorStore records upserts and we
assert on the observed calls. The point is to verify:

- deterministic ids (re-run = no duplicates)
- batching (correct batch_size handling)
- stats accounting (seen/embedded/upserted/skipped/failed)
- skip_seen LRU behaviour in a single process
- graceful handling of embedding failures
"""

from __future__ import annotations

from mnemostack.ingest import (
    IngestItem,
    Ingestor,
    stable_chunk_id,
)


class _FakeEmbedding:
    dimension = 3

    def embed(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0] if text else []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class _FakeStore:
    def __init__(self):
        self.upserts: list[tuple[str, list[float], dict]] = []

    def upsert(self, id, vector, payload):
        self.upserts.append((id, vector, payload))

    def upsert_batch(self, points):
        for p in points:
            self.upsert(p[0], p[1], p[2])
        return len(points)


def test_stable_chunk_id_deterministic():
    a = stable_chunk_id("src", 0, "hello")
    b = stable_chunk_id("src", 0, "hello")
    assert a == b
    import uuid

    assert uuid.UUID(a)


def test_stable_chunk_id_distinguishes_inputs():
    assert stable_chunk_id("a", 0, "hi") != stable_chunk_id("b", 0, "hi")
    assert stable_chunk_id("a", 0, "hi") != stable_chunk_id("a", 1, "hi")
    assert stable_chunk_id("a", 0, "hi") != stable_chunk_id("a", 0, "hello")


def test_ingest_basic_happy_path():
    emb = _FakeEmbedding()
    store = _FakeStore()
    ing = Ingestor(embedding=emb, vector_store=store, batch_size=2)
    items = [
        IngestItem(text="first", source="notes/a.md"),
        IngestItem(text="second", source="notes/a.md", offset=100),
        IngestItem(text="third", source="notes/a.md", offset=200),
    ]
    stats = ing.ingest(items)
    assert stats.seen == 3
    assert stats.upserted == 3
    assert stats.skipped == 0
    assert stats.failed == 0
    assert len(store.upserts) == 3
    # Each point got a unique UUID id
    ids = {u[0] for u in store.upserts}
    assert len(ids) == 3


def test_ingest_is_idempotent_across_calls():
    emb = _FakeEmbedding()
    store = _FakeStore()
    ing = Ingestor(embedding=emb, vector_store=store, batch_size=5)
    items = [IngestItem(text="hello", source="a.md")]
    first = ing.ingest(items)
    second = ing.ingest(items)
    # Same id produced both times — Qdrant upserts onto itself, but LRU cache
    # short-circuits the second call (no extra work)
    assert first.upserted == 1
    assert second.upserted == 0
    assert second.skipped == 1


def test_ingest_does_not_dup_in_single_call():
    emb = _FakeEmbedding()
    store = _FakeStore()
    ing = Ingestor(embedding=emb, vector_store=store, batch_size=10)
    items = [
        IngestItem(text="dup", source="a.md"),
        IngestItem(text="dup", source="a.md"),  # duplicate on purpose
    ]
    stats = ing.ingest(items)
    # Both items present initially (seen=2) but one gets skipped after the
    # first embed flushes its id into the cache mid-call.
    assert stats.seen == 2
    # Both produce identical id, so only one is actually stored.
    assert len({u[0] for u in store.upserts}) == 1


def test_ingest_records_failed_embeddings():
    class _EmbedEmpty:
        dimension = 3

        def embed(self, t):
            return []

        def embed_batch(self, ts):
            return [[] for _ in ts]

    store = _FakeStore()
    ing = Ingestor(embedding=_EmbedEmpty(), vector_store=store, batch_size=2)
    stats = ing.ingest([IngestItem(text="a"), IngestItem(text="b")])
    assert stats.failed == 2
    assert stats.upserted == 0
    assert store.upserts == []


def test_stream_yields_per_batch():
    emb = _FakeEmbedding()
    store = _FakeStore()
    ing = Ingestor(embedding=emb, vector_store=store, batch_size=2, skip_seen=False)
    items = [IngestItem(text=f"t{i}", source="s.md", offset=i) for i in range(5)]
    batches = list(ing.stream(iter(items)))
    assert len(batches) == 3  # 2 + 2 + 1
    total_upserts = sum(b.upserted for b in batches)
    assert total_upserts == 5


def test_stream_does_not_materialize_full_iterator():
    emb = _FakeEmbedding()
    store = _FakeStore()
    ing = Ingestor(
        embedding=emb,
        vector_store=store,
        batch_size=1,
        skip_seen=False,
        window_size=2,
    )
    consumed: list[int] = []

    def items():
        for i in range(3):
            consumed.append(i)
            yield IngestItem(text=f"t{i}", source="s.md", offset=i)

    batches = ing.stream(items())
    first = next(batches)

    assert first.upserted == 1
    assert consumed == [0]


def test_ingest_does_not_materialize_full_iterator():
    emb = _FakeEmbedding()
    store = _FakeStore()
    ing = Ingestor(
        embedding=emb,
        vector_store=store,
        batch_size=1,
        skip_seen=False,
        window_size=2,
    )
    consumed: list[int] = []

    def items():
        for i in range(3):
            consumed.append(i)
            yield IngestItem(text=f"t{i}", source="s.md", offset=i)
            if i == 0:
                assert len(store.upserts) == 1

    stats = ing.ingest(items())

    assert stats.upserted == 5
    assert consumed == [0, 1, 2]


def test_ingest_one_shortcut():
    emb = _FakeEmbedding()
    store = _FakeStore()
    ing = Ingestor(embedding=emb, vector_store=store)
    stats = ing.ingest_one(IngestItem(text="solo", source="s.md"))
    assert stats.upserted == 1


def test_ingest_window_size_1_matches_current_behavior():
    emb = _FakeEmbedding()
    baseline_store = _FakeStore()
    window_store = _FakeStore()
    items = [
        IngestItem(text="a", source="chat", offset=0),
        IngestItem(text="b", source="chat", offset=1),
        IngestItem(text="c", source="chat", offset=2),
    ]

    baseline = Ingestor(embedding=emb, vector_store=baseline_store, batch_size=10)
    windowed = Ingestor(embedding=emb, vector_store=window_store, batch_size=10, window_size=1)

    assert baseline.ingest(items).upserted == 3
    assert windowed.ingest(items).upserted == 3

    def _normalized(upserts):
        # indexed_at is wall-clock — equal up to microseconds, strip it
        return [(pid, vec, {k: v for k, v in p.items() if k != "indexed_at"}) for pid, vec, p in upserts]

    assert _normalized(baseline_store.upserts) == _normalized(window_store.upserts)


def test_ingest_sliding_window_adds_context_chunks_with_middle_metadata():
    emb = _FakeEmbedding()
    store = _FakeStore()
    ing = Ingestor(embedding=emb, vector_store=store, batch_size=10, window_size=3)
    items = [
        IngestItem(text="A", source="chat", offset=0, metadata={"speaker": "s0"}),
        IngestItem(text="B", source="chat", offset=1, metadata={"speaker": "s1"}),
        IngestItem(text="C", source="chat", offset=2, metadata={"speaker": "s2"}),
        IngestItem(text="D", source="chat", offset=3, metadata={"speaker": "s3"}),
        IngestItem(text="E", source="chat", offset=4, metadata={"speaker": "s4"}),
    ]

    stats = ing.ingest(items)

    assert stats.upserted == 8  # 5 solos + 3 windows
    payloads = [payload for _id, _vec, payload in store.upserts]
    window_payloads = [p for p in payloads if p.get("chunk_window") == 3]
    assert [p["text"] for p in window_payloads] == ["A\nB\nC", "B\nC\nD", "C\nD\nE"]
    assert [p["speaker"] for p in window_payloads] == ["s1", "s2", "s3"]
    assert [p["offset"] for p in window_payloads] == [1, 2, 3]


# ---------- timestamp handling ----------


def _flush_payloads(items, **ingestor_kwargs):
    store = _FakeStore()
    ing = Ingestor(_FakeEmbedding(), store, **ingestor_kwargs)
    ing.ingest(items)
    return [payload for _id, _vec, payload in store.upserts]


def test_timestamp_field_lands_in_payload():
    payloads = _flush_payloads([IngestItem(text="hi", timestamp="2026-06-01T13:41:00")])
    assert payloads[0]["timestamp"] == "2026-06-01T13:41:00"


def test_metadata_timestamp_back_compat():
    payloads = _flush_payloads(
        [IngestItem(text="hi", metadata={"timestamp": "2026-06-01T13:41:00"})]
    )
    assert payloads[0]["timestamp"] == "2026-06-01T13:41:00"


def test_timestamp_field_wins_over_metadata():
    payloads = _flush_payloads(
        [
            IngestItem(
                text="hi",
                timestamp="2026-06-02T00:00:00",
                metadata={"timestamp": "2020-01-01T00:00:00"},
            )
        ]
    )
    assert payloads[0]["timestamp"] == "2026-06-02T00:00:00"


def test_indexed_at_auto_utc_and_not_overwritten():
    from datetime import datetime

    payloads = _flush_payloads(
        [
            IngestItem(text="a", source="s", offset=0),
            IngestItem(text="b", source="s", offset=1, metadata={"indexed_at": "pinned"}),
        ]
    )
    auto = payloads[0]["indexed_at"]
    parsed = datetime.fromisoformat(auto)
    assert parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0
    assert payloads[1]["indexed_at"] == "pinned"


def test_window_chunks_carry_ts_range():
    items = [
        IngestItem(text=f"m{i}", source="conv", offset=i, timestamp=f"2026-06-0{i + 1}T10:00:00")
        for i in range(3)
    ]
    payloads = _flush_payloads(items, window_size=3)
    windows = [p for p in payloads if p.get("chunk_kind") == "sliding_window"]
    assert len(windows) == 1
    w = windows[0]
    assert w["window_start_ts"] == "2026-06-01T10:00:00"
    assert w["window_end_ts"] == "2026-06-03T10:00:00"
    assert w["timestamp"] == "2026-06-02T10:00:00"  # middle item


def test_window_chunks_without_ts_omit_range_keys():
    items = [IngestItem(text=f"m{i}", source="conv", offset=i) for i in range(3)]
    payloads = _flush_payloads(items, window_size=3)
    windows = [p for p in payloads if p.get("chunk_kind") == "sliding_window"]
    assert len(windows) == 1
    assert "window_start_ts" not in windows[0]
    assert "window_end_ts" not in windows[0]
    assert "timestamp" not in windows[0]


def test_enrich_merges_into_payload():
    emb = _FakeEmbedding()
    store = _FakeStore()

    def enricher(item):
        return {"char_count": len(item.text), "lang_hint": "neutral"}

    ing = Ingestor(embedding=emb, vector_store=store, enrich=enricher)
    ing.ingest([IngestItem(text="participant A sent a payment", source="notes/a.md")])

    payload = store.upserts[0][2]
    assert payload["char_count"] == len("participant A sent a payment")
    assert payload["lang_hint"] == "neutral"


def test_enrich_failure_is_fail_open():
    emb = _FakeEmbedding()
    store = _FakeStore()

    def broken(item):
        raise RuntimeError("user extractor exploded")

    ing = Ingestor(embedding=emb, vector_store=store, enrich=broken)
    stats = ing.ingest([IngestItem(text="still indexed", source="notes/a.md")])

    assert stats.upserted == 1
    payload = store.upserts[0][2]
    assert payload["text"] == "still indexed"


def test_enrich_non_dict_return_ignored():
    emb = _FakeEmbedding()
    store = _FakeStore()

    ing = Ingestor(embedding=emb, vector_store=store, enrich=lambda item: ["not", "a", "dict"])
    stats = ing.ingest([IngestItem(text="still indexed", source="notes/a.md")])

    assert stats.upserted == 1


def test_enrich_cannot_override_protected_keys():
    emb = _FakeEmbedding()
    store = _FakeStore()

    def hostile(item):
        return {
            "text": "REPLACED",
            "source": "elsewhere.md",
            "offset": 999,
            "index_root": "/evil",
            "timestamp": "1970-01-01T00:00:00Z",
            "extra": "kept",
        }

    ing = Ingestor(embedding=emb, vector_store=store, enrich=hostile)
    ing.ingest(
        [
            IngestItem(
                text="original",
                source="notes/a.md",
                timestamp="2026-06-01T10:00:00Z",
            )
        ]
    )

    payload = store.upserts[0][2]
    assert payload["text"] == "original"
    assert payload["source"] == "notes/a.md"
    assert payload["offset"] == 0
    assert "index_root" not in payload
    assert payload["timestamp"] == "2026-06-01T10:00:00Z"  # explicit item ts wins
    assert payload["extra"] == "kept"


def test_enrich_without_explicit_timestamp_may_set_it():
    emb = _FakeEmbedding()
    store = _FakeStore()

    ing = Ingestor(
        embedding=emb,
        vector_store=store,
        enrich=lambda item: {"timestamp": "2026-05-01T00:00:00Z"},
    )
    ing.ingest([IngestItem(text="undated note", source="notes/a.md")])

    assert store.upserts[0][2]["timestamp"] == "2026-05-01T00:00:00Z"


def test_enrich_sees_assembled_window_text():
    emb = _FakeEmbedding()
    store = _FakeStore()
    seen_texts: list[str] = []

    def spy(item):
        seen_texts.append(item.text)
        return {}

    ing = Ingestor(embedding=emb, vector_store=store, window_size=2, enrich=spy)
    ing.ingest(
        [
            IngestItem(text="first", source="notes/a.md", offset=0),
            IngestItem(text="second", source="notes/a.md", offset=100),
        ]
    )

    assert any("first" in t and "second" in t for t in seen_texts)
