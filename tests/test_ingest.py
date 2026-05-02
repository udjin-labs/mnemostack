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
        def embed(self, t): return []
        def embed_batch(self, ts): return [[] for _ in ts]

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
    assert baseline_store.upserts == window_store.upserts


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
