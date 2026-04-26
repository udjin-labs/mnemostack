from types import SimpleNamespace

from mnemostack.recall import BM25Retriever, bm25_docs_from_qdrant


class FakeQdrantClient:
    def __init__(self, points):
        self.points = points
        self.calls = []

    def scroll(self, **kwargs):
        self.calls.append(kwargs)
        offset = kwargs.get("offset") or 0
        limit = kwargs.get("limit") or 10
        end = offset + limit
        batch = self.points[offset:end]
        next_offset = end if end < len(self.points) else None
        return batch, next_offset


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

    assert [d.id for d in docs] == ["bm25:qdrant:a", "bm25:qdrant:b"]
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
    assert results[0].id == "bm25:qdrant:msg-1"
    assert "MERGED" in results[0].text
    assert results[0].sources == ["bm25"]
