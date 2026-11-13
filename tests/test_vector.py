"""Tests for VectorStore — uses in-memory Qdrant (no external server needed)."""

import pytest
from qdrant_client import QdrantClient

from mnemostack.vector import Hit, VectorStore


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


def test_collection_exists_reraises_non_not_found_errors():
    store = VectorStore.__new__(VectorStore)
    store.collection = "test_collection"

    class FailingClient:
        def get_collection(self, _collection):
            raise RuntimeError("qdrant auth failed")

    store.client = FailingClient()

    with pytest.raises(RuntimeError, match="qdrant auth failed"):
        store.collection_exists()


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
    points = [(i, [float(i) / 10, 0.0, 0.0, 0.0], {"idx": i}) for i in range(1, 26)]
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


def test_scroll_iterates_all_points(store):
    store.ensure_collection()
    total = 50
    points = [(i, [float(i % 4) / 10, 0.0, 0.0, 0.0], {"idx": i}) for i in range(1, total + 1)]
    store.upsert_batch(points, batch_size=10)

    collected = list(store.scroll(batch_size=15))
    assert len(collected) == total
    ids = {hit.id for hit in collected}
    assert ids == set(range(1, total + 1))


def test_scroll_with_filter(store):
    store.ensure_collection()
    store.upsert(1, [1.0, 0.0, 0.0, 0.0], {"class": "A"})
    store.upsert(2, [0.0, 1.0, 0.0, 0.0], {"class": "B"})
    store.upsert(3, [0.0, 0.0, 1.0, 0.0], {"class": "A"})

    a_points = list(store.scroll(batch_size=10, filters={"class": "A"}))
    assert len(a_points) == 2
    assert {p.id for p in a_points} == {1, 3}


def test_iter_ids_lightweight(store):
    store.ensure_collection()
    for i in range(1, 11):
        store.upsert(i, [0.1, 0.1, 0.1, 0.1], {"i": i})

    ids = list(store.iter_ids(batch_size=4))
    assert set(ids) == set(range(1, 11))


def test_scroll_empty_collection(store):
    store.ensure_collection()
    assert list(store.scroll()) == []
    assert list(store.iter_ids()) == []


# ---------- dimension validation ----------


def _store_with_client(client, dimension):
    from qdrant_client.models import Distance

    from mnemostack.vector import VectorStore as VS

    s = VS.__new__(VS)
    s.collection = "test_collection"
    s.dimension = dimension
    s.distance = Distance.COSINE
    s.client = client
    return s


def test_ensure_collection_dimension_mismatch_raises(store):
    from mnemostack.vector import DimensionMismatchError

    store.ensure_collection()  # creates with dimension 4
    other = _store_with_client(store.client, dimension=8)
    with pytest.raises(DimensionMismatchError, match="4-dim.*8-dim"):
        other.ensure_collection()


def test_ensure_collection_matching_dimension_ok(store):
    store.ensure_collection()
    other = _store_with_client(store.client, dimension=4)
    assert other.ensure_collection() is False  # exists, validated, kept


def test_ensure_collection_recreate_skips_validation(store):
    store.ensure_collection()
    other = _store_with_client(store.client, dimension=8)
    assert other.ensure_collection(recreate=True) is True
    # collection is now 8-dim; original store would mismatch
    from mnemostack.vector import DimensionMismatchError

    with pytest.raises(DimensionMismatchError):
        store.ensure_collection()


def test_scroll_resumes_when_the_first_batch_is_only_the_cursor(store):
    """R6 (codex P2): Qdrant's offset is INCLUSIVE, so a small batch can
    contain nothing but the cursor we then drop. Treating that empty
    filtered batch as exhaustion silently omitted every successor."""
    store.ensure_collection()
    for pid in range(1, 6):
        store.upsert(pid, [0.1, 0.2, 0.3, 0.4], {"source": "s.md"})
    ids = [hit.id for hit in store.scroll(batch_size=1, start_after=1)]
    assert ids == [2, 3, 4, 5]
    # And a cursor that IS the last point still terminates.
    assert [hit.id for hit in store.scroll(batch_size=1, start_after=5)] == []


def test_lifecycle_reads_project_guards(store, monkeypatch):
    store.ensure_collection()
    payloads = [{}, {"tenant_id": "a"}, {"tenant_id": "b", "index_root": "other"}]
    for pid, payload in enumerate(payloads, 1):
        store.upsert(pid, [1.0, 0.0, 0.0, 0.0], {**payload, "text": "large" * 1000}, tenant=payload.get("tenant_id"))
    calls = []
    retrieve = store.client.retrieve

    def tracked(*args, **kwargs):
        calls.append(kwargs)
        return retrieve(*args, **kwargs)

    monkeypatch.setattr(store.client, "retrieve", tracked)
    for pid in (1, 2, 3, 4):
        store.set_payload(pid, {"changed": True}, tenant="a")
    assert all(c["with_payload"] == ["tenant_id"] and c["with_vectors"] is False for c in calls)
    calls.clear()
    assert store.invalidate([1, 2, 3, 4], tenant="a", index_root="root") == 1
    assert calls[0]["with_payload"] == ["index_root", "tenant_id"]
    assert calls[0]["with_vectors"] is False
    rows = {p.id: p.payload for p in retrieve(store.collection, ids=[1, 2, 3], with_payload=True)}
    assert "changed" not in rows[1] and "invalidated_at" not in rows[1]
    assert rows[2]["changed"] and rows[2]["invalidated_at"]
    assert "changed" not in rows[3] and "invalidated_at" not in rows[3]


def test_scroll_payload_projection_keeps_missing_keys_and_order(store):
    store.ensure_collection()
    store.upsert(1, [1.0, 0.0, 0.0, 0.0], {"text": "large"})
    store.upsert(2, [1.0, 0.0, 0.0, 0.0], {"source": "note", "text": "large"})
    full = list(store.scroll())
    narrow = list(store.scroll(payload_keys=["source", "index_root"]))
    assert [h.id for h in narrow] == [h.id for h in full]
    assert [h.payload for h in narrow] == [{}, {"source": "note"}]
    assert all(h.vector is None for h in narrow)


@pytest.mark.parametrize("key", ["meta.author", "tags[]", "tags[0]", "trailing]", '"author"', "display name", ""])
def test_projection_preserves_flat_path_keys(store, monkeypatch, key):
    store.ensure_collection()
    store.upsert(1, [1.0, 0.0, 0.0, 0.0], {key: "value", "text": "body"}, tenant="a")
    calls = []
    retrieve = store.client.retrieve

    def tracked(*args, **kwargs):
        calls.append(kwargs["with_payload"])
        return retrieve(*args, **kwargs)

    monkeypatch.setattr(store.client, "retrieve", tracked)
    assert store.retrieve_payload_fields([1], [key], tenant="a") == {"1": {key: "value"}}
    assert store.retrieve_payload_fields([1], [key], tenant="b") == {}
    assert calls == [True, True]
    assert list(store.scroll(payload_keys=[key], tenant="a"))[0].payload[key] == "value"
