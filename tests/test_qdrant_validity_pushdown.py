"""Server-side validity push-down: VectorStore.search(hide_invalidated=...) (#90).

The default validity view (hide facts carrying an ``invalidated_at`` marker) is
pushed into Qdrant as an ``IsEmpty`` filter so stale points are never fetched,
instead of fetched-then-dropped client-side. ``as_of`` point-in-time recall is
deliberately NOT pushed (bare-date bounds make a server-side range unsafe), and
the client-side ``filter_by_validity`` stays the backstop.
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance

from mnemostack.recall.retrievers import VectorRetriever
from mnemostack.vector import VectorStore

VEC = [1.0, 0.0, 0.0, 0.0]


def _store():
    s = VectorStore.__new__(VectorStore)
    s.collection = "pushdown_test"
    s.dimension = 4
    s.distance = Distance.COSINE
    s.client = QdrantClient(":memory:")
    s.ensure_collection()
    return s


def _seed(store):
    store.upsert(1, VEC, {"text": "current", "source": "a"})
    store.upsert(2, VEC, {"text": "stale", "source": "a", "invalidated_at": "2026-07-04T00:00:00Z"})
    store.upsert(3, VEC, {"text": "current-b", "source": "b"})


# ---------- VectorStore.search push-down ----------


def test_hide_invalidated_excludes_stale_points():
    store = _store()
    _seed(store)
    got = {h.id for h in store.search(VEC, limit=10, hide_invalidated=True)}
    assert got == {1, 3}  # the stale point (2) is filtered server-side


def test_default_returns_everything():
    store = _store()
    _seed(store)
    got = {h.id for h in store.search(VEC, limit=10)}
    assert got == {1, 2, 3}  # hide_invalidated defaults False -> unchanged behavior


def test_hide_invalidated_combines_with_payload_filter():
    store = _store()
    _seed(store)
    # payload filter (source=a) AND validity: only the current 'a' point.
    got = {h.id for h in store.search(VEC, limit=10, filters={"source": "a"}, hide_invalidated=True)}
    assert got == {1}


# ---------- VectorRetriever decides when to push ----------


class _SpyStore:
    def __init__(self):
        self.calls: list[bool] = []

    def search(self, vec, limit, filters=None, *, hide_invalidated=False):
        self.calls.append(hide_invalidated)
        return []


class _Embed:
    def embed(self, text):
        return VEC


def test_retriever_pushes_default_view():
    spy = _SpyStore()
    VectorRetriever(embedding=_Embed(), vector_store=spy).search("q")
    assert spy.calls == [True]  # default: hide invalidated is pushed down


def test_retriever_does_not_push_for_include_invalidated():
    spy = _SpyStore()
    VectorRetriever(embedding=_Embed(), vector_store=spy).search("q", include_invalidated=True)
    assert spy.calls == [False]


def test_retriever_does_not_push_for_as_of():
    spy = _SpyStore()
    # as_of stays client-side (bare-date bounds unsafe for a server-side range).
    VectorRetriever(embedding=_Embed(), vector_store=spy).search("q", as_of="2026-03-01")
    assert spy.calls == [False]


def test_retriever_advertises_validity_awareness():
    r = VectorRetriever(embedding=_Embed(), vector_store=_SpyStore())
    assert r.accepts_as_of is True
    assert r.accepts_include_invalidated is True


# ---------- async mirror ----------


@pytest.mark.asyncio
async def test_async_hide_invalidated_excludes_stale():
    from qdrant_client import AsyncQdrantClient

    from mnemostack.vector import AsyncVectorStore

    s = AsyncVectorStore.__new__(AsyncVectorStore)
    s.collection = "pushdown_async"
    s.dimension = 4
    s.distance = Distance.COSINE
    s.client = AsyncQdrantClient(":memory:")
    await s.ensure_collection()
    await s.upsert(1, VEC, {"text": "current"})
    await s.upsert(2, VEC, {"text": "stale", "invalidated_at": "2026-07-04T00:00:00Z"})

    got = {h.id for h in await s.search(VEC, limit=10, hide_invalidated=True)}
    assert got == {1}
    allids = {h.id for h in await s.search(VEC, limit=10)}
    assert allids == {1, 2}
