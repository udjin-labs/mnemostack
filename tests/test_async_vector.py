"""Tests for AsyncVectorStore — uses in-memory async Qdrant."""
import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance

from mnemostack.vector import AsyncVectorStore, Hit


@pytest_asyncio.fixture
async def store():
    s = AsyncVectorStore.__new__(AsyncVectorStore)
    s.collection = "test_async"
    s.dimension = 4
    s.distance = Distance.COSINE
    s.client = AsyncQdrantClient(":memory:")
    await s.ensure_collection()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_async_ensure_collection_idempotent(store):
    created = await store.ensure_collection()
    assert not created  # already created in fixture


@pytest.mark.asyncio
async def test_async_upsert_and_search(store):
    await store.upsert(1, [1.0, 0.0, 0.0, 0.0], {"text": "dog"})
    await store.upsert(2, [0.0, 1.0, 0.0, 0.0], {"text": "car"})
    hits = await store.search([1.0, 0.0, 0.0, 0.0], limit=1)
    assert len(hits) == 1
    assert isinstance(hits[0], Hit)
    assert hits[0].id == 1
    assert hits[0].payload["text"] == "dog"


@pytest.mark.asyncio
async def test_async_upsert_batch(store):
    points = [(i, [float(i) / 10, 0.0, 0.0, 0.0], {"idx": i}) for i in range(1, 6)]
    n = await store.upsert_batch(points)
    assert n == 5
    count = await store.count()
    assert count == 5


@pytest.mark.asyncio
async def test_async_search_with_filter(store):
    await store.upsert(1, [1.0, 0.0, 0.0, 0.0], {"class": "A"})
    await store.upsert(2, [1.0, 0.0, 0.0, 0.0], {"class": "B"})
    hits = await store.search([1.0, 0.0, 0.0, 0.0], limit=10, filters={"class": "B"})
    assert len(hits) == 1
    assert hits[0].id == 2


@pytest.mark.asyncio
async def test_async_context_manager():
    async with AsyncVectorStore(collection="test_ctx", dimension=4) as s:
        # Just ensure we can call async methods
        # Skip actual Qdrant call if no real server — in-memory works
        pass
