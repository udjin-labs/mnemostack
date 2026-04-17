"""Tests for VectorStore — uses in-memory Qdrant (no external server needed)."""
import pytest
from qdrant_client import QdrantClient

from memvault.vector import Hit, VectorStore


@pytest.fixture
def store(monkeypatch):
    """VectorStore pointing at in-memory Qdrant."""
    store = VectorStore.__new__(VectorStore)  # bypass __init__
    store.collection = "test_collection"
    store.dimension = 4
    from qdrant_client.models import Distance

    store.distance = Distance.COSINE
    store.client = QdrantClient(":memory:")
    return store


def test_ensure_collection_creates(store):
    assert not store.collection_exists()
    created = store.ensure_collection()
    assert created
    assert store.collection_exists()


def test_ensure_collection_idempotent(store):
    store.ensure_collection()
    # second call — should NOT recreate
    created = store.ensure_collection()
    assert not created


def test_ensure_collection_recreate(store):
    store.ensure_collection()
    store.upsert(1, [1.0, 0.0, 0.0, 0.0], {"text": "hello"})
    assert store.count() == 1
    store.ensure_collection(recreate=True)
    assert store.count() == 0


def test_upsert_and_search(store):
    store.ensure_collection()
    store.upsert(1, [1.0, 0.0, 0.0, 0.0], {"text": "dog", "class": "animal"})
    store.upsert(2, [0.0, 1.0, 0.0, 0.0], {"text": "car", "class": "vehicle"})
    store.upsert(3, [0.9, 0.1, 0.0, 0.0], {"text": "cat", "class": "animal"})

    hits = store.search([1.0, 0.0, 0.0, 0.0], limit=2)
    assert len(hits) == 2
    assert hits[0].id == 1  # closest match
    assert isinstance(hits[0], Hit)
    assert hits[0].payload["text"] == "dog"


def test_search_with_filter(store):
    store.ensure_collection()
    store.upsert(1, [1.0, 0.0, 0.0, 0.0], {"text": "dog", "class": "animal"})
    store.upsert(2, [0.9, 0.1, 0.0, 0.0], {"text": "car", "class": "vehicle"})

    hits = store.search([1.0, 0.0, 0.0, 0.0], limit=5, filters={"class": "vehicle"})
    assert len(hits) == 1
    assert hits[0].id == 2


def test_upsert_batch(store):
    store.ensure_collection()
    points = [
        (i, [float(i) / 10, 0.0, 0.0, 0.0], {"idx": i}) for i in range(1, 26)
    ]
    n = store.upsert_batch(points, batch_size=10)
    assert n == 25
    assert store.count() == 25


def test_search_min_score_filter(store):
    store.ensure_collection()
    store.upsert(1, [1.0, 0.0, 0.0, 0.0], {"text": "exact"})
    store.upsert(2, [-1.0, 0.0, 0.0, 0.0], {"text": "opposite"})  # score ~ -1 (cosine)

    hits = store.search([1.0, 0.0, 0.0, 0.0], limit=10, min_score=0.5)
    assert len(hits) == 1
    assert hits[0].id == 1
