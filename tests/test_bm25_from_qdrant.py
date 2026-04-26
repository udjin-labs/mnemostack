from datetime import datetime, timezone
from types import SimpleNamespace

from mnemostack.recall import BM25Retriever, Recaller, RecallResult, bm25_docs_from_qdrant


class FakeQdrantClient:
    def __init__(self, points):
        self.points = points
        self.calls = []

    def scroll(self, **kwargs):
        self.calls.append(kwargs)
        points = self._apply_filter(kwargs.get("scroll_filter"))
        offset = kwargs.get("offset") or 0
        limit = kwargs.get("limit") or 10
        end = offset + limit
        batch = points[offset:end]
        next_offset = end if end < len(points) else None
        return batch, next_offset

    def _apply_filter(self, scroll_filter):
        if scroll_filter is None:
            return self.points
        must = getattr(scroll_filter, "must", None) or []
        points = self.points
        for condition in must:
            if getattr(condition, "key", None) != "timestamp":
                continue
            time_range = getattr(condition, "range", None)
            gte = _iso_key(getattr(time_range, "gte", None))
            lte = _iso_key(getattr(time_range, "lte", None))
            filtered = []
            for item in points:
                timestamp = _iso_key((item.payload or {}).get("timestamp"))
                if timestamp is None:
                    continue
                if gte is not None and timestamp < gte:
                    continue
                if lte is not None and timestamp > lte:
                    continue
                filtered.append(item)
            points = filtered
        return points


def _iso_key(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, str):
        return value.replace("+00:00", "Z")
    return str(value)


def point(pid, payload):
    return SimpleNamespace(id=pid, payload=payload)


def test_bm25_docs_from_qdrant_reads_payload_text_and_metadata():
    client = FakeQdrantClient([
        point("a", {"text": "MERGED pull request 71706", "source_file": "transcript:one.jsonl"}),
        point("b", {"text": "ordinary note", "source": "notes.md"}),
        point("c", {"text": "   "}),
        point("d", {"other": "missing text"}),
    ])

    docs = bm25_docs_from_qdrant(client, "memory", batch_size=2, limit=10)

    assert [d.id for d in docs] == ["a", "b"]
    assert docs[0].text == "MERGED pull request 71706"
    assert docs[0].payload["qdrant_id"] == "a"
    assert docs[0].payload["source"] == "transcript:one.jsonl"
    assert docs[1].payload["source"] == "notes.md"
    assert len(client.calls) == 2
    assert client.calls[0]["collection_name"] == "memory"
    assert client.calls[0]["with_payload"] is True
    assert client.calls[0]["with_vectors"] is False


def test_bm25_docs_from_qdrant_supports_filter_and_limit():
    client = FakeQdrantClient([
        point("a", {"text": "keep exact-token", "chunk_type": "transcript"}),
        point("b", {"text": "skip exact-token", "chunk_type": "other"}),
        point("c", {"text": "keep another exact-token", "chunk_type": "transcript"}),
    ])

    docs = bm25_docs_from_qdrant(
        client,
        "memory",
        limit=2,
        batch_size=1,
        id_prefix="transcript",
        payload_filter=lambda p: p.get("chunk_type") == "transcript",
    )

    assert [d.id for d in docs] == [
        "bm25:transcript:a",
        "bm25:transcript:c",
    ]


def test_bm25_retriever_from_qdrant_finds_exact_token():
    client = FakeQdrantClient([
        point("msg-1", {"text": "[09:20] user said MERGED! PR 71706", "source_file": "transcript:s.jsonl"}),
        point("msg-2", {"text": "unrelated conversation about weather"}),
    ])

    retriever = BM25Retriever.from_qdrant(client, "memory")
    results = retriever.search("MERGED 71706", limit=1)

    assert len(results) == 1
    assert results[0].id == "msg-1"
    assert "MERGED" in results[0].text
    assert results[0].sources == ["bm25"]


def test_bm25_from_qdrant_fuses_with_vector_result_by_qdrant_id():
    class FakeVectorRetriever:
        name = "vector"

        def search(self, query, limit=20, filters=None):
            return [
                RecallResult(
                    id="msg-1",
                    text="[09:20] user said MERGED! PR 71706",
                    score=0.9,
                    payload={"text": "[09:20] user said MERGED! PR 71706"},
                    sources=["vector"],
                )
            ]

    client = FakeQdrantClient([
        point("msg-1", {"text": "[09:20] user said MERGED! PR 71706"}),
    ])

    recaller = Recaller(
        retrievers=[
            FakeVectorRetriever(),
            BM25Retriever.from_qdrant(client, "memory"),
        ]
    )
    results = recaller.recall("MERGED 71706", limit=5)

    assert len(results) == 1
    assert results[0].id == "msg-1"
    assert set(results[0].sources) == {"vector", "bm25"}


def test_bm25_from_qdrant_unbounded_by_default():
    client = FakeQdrantClient([
        point(f"p-{idx}", {"text": f"chunk {idx}"}) for idx in range(5_000)
    ])

    docs = bm25_docs_from_qdrant(client, "memory", batch_size=777)

    assert len(docs) == 5_000
    assert docs[0].id == "p-0"
    assert docs[-1].id == "p-4999"
    assert len(client.calls) == 7


def test_bm25_from_qdrant_explicit_limit_respects_cap():
    client = FakeQdrantClient([
        point(f"p-{idx}", {"text": f"chunk {idx}"}) for idx in range(5_000)
    ])

    docs = bm25_docs_from_qdrant(client, "memory", limit=100, batch_size=33)

    assert len(docs) == 100
    assert docs[-1].id == "p-99"
    assert sum(call["limit"] for call in client.calls) == 100


def test_bm25_from_qdrant_newer_than_filter():
    client = FakeQdrantClient([
        point("old", {"text": "old chunk", "timestamp": "2026-04-01T00:00:00Z"}),
        point("new", {"text": "new chunk", "timestamp": "2026-04-20T00:00:00Z"}),
        point("missing", {"text": "missing timestamp"}),
    ])

    docs = bm25_docs_from_qdrant(
        client,
        "memory",
        newer_than="2026-04-10T00:00:00Z",
    )

    assert [doc.id for doc in docs] == ["new"]


def test_bm25_from_qdrant_older_than_filter():
    client = FakeQdrantClient([
        point("old", {"text": "old chunk", "timestamp": "2026-04-01T00:00:00Z"}),
        point("new", {"text": "new chunk", "timestamp": "2026-04-20T00:00:00Z"}),
        point("missing", {"text": "missing timestamp"}),
    ])

    docs = bm25_docs_from_qdrant(
        client,
        "memory",
        older_than="2026-04-10T00:00:00Z",
    )

    assert [doc.id for doc in docs] == ["old"]


def test_bm25_from_qdrant_combined_time_filter():
    client = FakeQdrantClient([
        point("too-old", {"text": "too old", "timestamp": "2026-04-01T00:00:00Z"}),
        point("inside", {"text": "inside", "timestamp": "2026-04-15T00:00:00Z"}),
        point("too-new", {"text": "too new", "timestamp": "2026-04-30T00:00:00Z"}),
    ])

    docs = bm25_docs_from_qdrant(
        client,
        "memory",
        newer_than="2026-04-10T00:00:00Z",
        older_than="2026-04-20T00:00:00Z",
    )

    assert [doc.id for doc in docs] == ["inside"]
    condition = client.calls[0]["scroll_filter"].must[0]
    assert condition.key == "timestamp"
