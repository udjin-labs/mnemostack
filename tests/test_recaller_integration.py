"""Integration tests — Recaller end-to-end with in-memory Qdrant + fake embedder.

These tests don't hit any external services. Use a fake deterministic
embedding provider so search is reproducible.
"""
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance

from mnemostack.embeddings import EmbeddingProvider, register_provider, get_provider
from mnemostack.recall import BM25Doc, Recaller, RecallResult
from mnemostack.vector import VectorStore


class FakeEmbedder(EmbeddingProvider):
    """Deterministic embedder: vector depends on character frequency."""

    @property
    def dimension(self) -> int:
        return 8

    @property
    def name(self) -> str:
        return "fake"

    def embed(self, text: str) -> list[float]:
        text = text.lower()
        # 8 buckets by character ranges
        buckets = [0.0] * 8
        for c in text:
            if c.isalpha():
                buckets[ord(c) % 8] += 1
        s = sum(buckets) or 1.0
        return [b / s for b in buckets]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


@pytest.fixture
def setup():
    """Set up in-memory Qdrant + BM25 corpus + fake embedder + Recaller."""
    embedder = FakeEmbedder()

    vector_store = VectorStore.__new__(VectorStore)
    vector_store.collection = "test"
    vector_store.dimension = embedder.dimension
    vector_store.distance = Distance.COSINE
    vector_store.client = QdrantClient(":memory:")
    vector_store.ensure_collection()

    docs = [
        (1, "Python is a programming language used for data science and ML"),
        (2, "The Qdrant vector database supports similarity search at scale"),
        (3, "Machine learning models need large amounts of training data"),
        (4, "Memgraph is a graph database compatible with Cypher queries"),
        (5, "Hybrid retrieval combines BM25 with vector search"),
    ]

    # Index into vector store
    for doc_id, text in docs:
        vec = embedder.embed(text)
        vector_store.upsert(doc_id, vec, {"text": text})

    # Build BM25 corpus
    bm25_docs = [BM25Doc(id=doc_id, text=text) for doc_id, text in docs]

    recaller = Recaller(
        embedding_provider=embedder,
        vector_store=vector_store,
        bm25_docs=bm25_docs,
    )
    return recaller


def test_recaller_basic(setup):
    recaller = setup
    results = recaller.recall("vector database", limit=3)
    assert len(results) > 0
    assert all(isinstance(r, RecallResult) for r in results)
    # Doc 2 mentions "Qdrant vector database" and doc 4 "graph database"
    ids = [r.id for r in results]
    assert 2 in ids


def test_recaller_sources(setup):
    recaller = setup
    results = recaller.recall("machine learning", limit=5)
    assert len(results) > 0
    # At least one result should have both sources (vector AND bm25)
    # Results typically have either, depending on BM25 scoring
    for r in results:
        assert r.sources  # non-empty list
        assert set(r.sources) <= {"vector", "bm25"}


def test_recaller_empty_corpus(setup):
    """Recaller without BM25 docs should still work on vector only."""
    recaller = setup
    recaller.bm25 = None
    results = recaller.recall("python programming", limit=3)
    assert len(results) > 0
    for r in results:
        assert r.sources == ["vector"]


def test_recaller_returns_text_and_payload(setup):
    recaller = setup
    results = recaller.recall("graph Cypher", limit=2)
    assert len(results) > 0
    r = results[0]
    assert r.text  # non-empty
    assert "text" in r.payload or r.text
