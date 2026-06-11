"""Adversarial isolation tests for recall filters.

The contract under test is isolation, not ranking: a recall scoped with
`filters=` must never return a point outside the filtered scope through ANY
retriever. Before this suite, BM25 silently ignored filters — in a
multi-tenant deployment the fused output mixed foreign tenants' chunks in.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance

from mnemostack.embeddings.base import EmbeddingProvider
from mnemostack.recall import BM25Doc, Recaller, payload_matches, recall_flow
from mnemostack.recall.retrievers import (
    BM25Retriever,
    MemgraphRetriever,
    VectorRetriever,
)
from mnemostack.vector import VectorStore


class FakeEmbedder(EmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 8

    @property
    def name(self) -> str:
        return "fake"

    def embed(self, text: str) -> list[float]:
        buckets = [0.0] * 8
        for c in text.lower():
            if c.isalpha():
                buckets[ord(c) % 8] += 1
        s = sum(buckets) or 1.0
        return [b / s for b in buckets]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


# ---------- payload_matches ----------


def test_payload_matches_exact_and_missing_key():
    assert payload_matches({"tenant": "a"}, {"tenant": "a"})
    assert not payload_matches({"tenant": "b"}, {"tenant": "a"})
    # a point that cannot be attributed to the scope must not pass it
    assert not payload_matches({}, {"tenant": "a"})
    assert not payload_matches(None, {"tenant": "a"})


def test_payload_matches_ranges():
    p = {"timestamp": "2026-06-05T10:00:00Z", "count": 7}
    assert payload_matches(p, {"timestamp": {"gte": "2026-06-01", "lte": "2026-06-30"}})
    assert not payload_matches(p, {"timestamp": {"gte": "2026-06-10"}})
    assert payload_matches(p, {"count": {"gte": 5, "lte": 10}})
    assert not payload_matches(p, {"count": {"lte": 5}})
    # incomparable types cannot be proven inside the range — excluded
    assert not payload_matches({"count": "seven"}, {"count": {"gte": 5}})


def test_payload_matches_empty_filters_passes_everything():
    assert payload_matches({}, None)
    assert payload_matches({"x": 1}, {})


# ---------- BM25 ----------


def _bm25_two_tenants() -> BM25Retriever:
    docs = [
        BM25Doc(id="a1", text="quarterly report numbers", payload={"tenant": "a"}),
        BM25Doc(id="a2", text="meeting notes for the report", payload={"tenant": "a"}),
        BM25Doc(id="b1", text="report report report report", payload={"tenant": "b"}),
        BM25Doc(id="b2", text="the secret report of tenant b", payload={"tenant": "b"}),
        BM25Doc(id="b3", text="report draft report final", payload={"tenant": "b"}),
    ]
    return BM25Retriever(docs)


def test_bm25_filters_exclude_foreign_tenant():
    retr = _bm25_two_tenants()

    results = retr.search("report", limit=10, filters={"tenant": "a"})

    assert results
    assert all(r.payload["tenant"] == "a" for r in results)


def test_bm25_filters_before_topk_cut():
    """Foreign docs out-score tenant A's on this query; a post-cut filter
    would return fewer than the available matching docs."""
    retr = _bm25_two_tenants()

    results = retr.search("report", limit=2, filters={"tenant": "a"})

    assert len(results) == 2  # both A docs found despite B's higher scores
    assert {r.id for r in results} == {"a1", "a2"}


def test_bm25_no_filters_unchanged():
    retr = _bm25_two_tenants()
    results = retr.search("report", limit=10)
    assert {r.payload["tenant"] for r in results} == {"a", "b"}


# ---------- Memgraph ----------


def test_memgraph_excluded_under_filters():
    """Graph nodes carry no chunk payload — they cannot be attributed to the
    filtered scope, so under filters the retriever must contribute nothing."""
    driver = MagicMock()
    retr = MemgraphRetriever(driver=driver)

    assert retr.search("some entity name", filters={"tenant": "a"}) == []
    driver.session.assert_not_called()


# ---------- fused isolation ----------


@pytest.fixture
def fused_recaller():
    embedder = FakeEmbedder()
    store = VectorStore.__new__(VectorStore)
    store.collection = "test"
    store.dimension = embedder.dimension
    store.distance = Distance.COSINE
    store.client = QdrantClient(":memory:")
    store.ensure_collection()

    corpus = [
        ("11111111-0000-0000-0000-000000000001", "tenant a quarterly report", "a"),
        ("11111111-0000-0000-0000-000000000002", "tenant a planning notes", "a"),
        ("22222222-0000-0000-0000-000000000001", "tenant b quarterly report", "b"),
        ("22222222-0000-0000-0000-000000000002", "tenant b secret roadmap", "b"),
    ]
    bm25_docs = []
    for pid, text, tenant in corpus:
        payload = {"text": text, "tenant": tenant}
        store.upsert(pid, embedder.embed(text), payload)
        bm25_docs.append(BM25Doc(id=pid, text=text, payload=payload))

    recaller = Recaller(
        retrievers=[
            VectorRetriever(embedding=embedder, vector_store=store),
            BM25Retriever(bm25_docs),
        ]
    )
    return recaller


def test_fused_recall_never_leaks_foreign_tenant(fused_recaller):
    results = fused_recaller.recall("quarterly report", limit=10, filters={"tenant": "a"})

    assert results
    assert all(r.payload.get("tenant") == "a" for r in results)


def test_fused_recall_without_filters_sees_everything(fused_recaller):
    results = fused_recaller.recall("quarterly report", limit=10)
    assert {r.payload.get("tenant") for r in results} == {"a", "b"}


def test_recall_flow_threads_filters(fused_recaller):
    results = recall_flow(
        fused_recaller, "quarterly report", limit=10, filters={"tenant": "b"}
    )

    assert results
    assert all(r.payload.get("tenant") == "b" for r in results)
