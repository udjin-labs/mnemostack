"""Smoke tests against a REAL Qdrant server (not the in-memory local client).

The regular suite runs on ``QdrantClient(":memory:")``, whose semantics
diverge from real servers exactly where the text-search arms live: payload
full-text indexes (local mode matches without one, servers REQUIRE it for
MatchText), ``update_collection`` (servers refuse adding a sparse space
post-hoc), and version-dependent ``update_vectors`` behavior. These tests
replay the release-verification scenarios against an actual server.

Gated by ``MNEMOSTACK_TEST_QDRANT_URL`` (e.g. ``http://localhost:6333``):
unset → the whole module skips, so local runs are unaffected. CI provides a
`qdrant/qdrant` service container pinned to the same version the deployment
docs' compose example pins — the documented deployment is the tested one.
"""

from __future__ import annotations

import os
import uuid

import pytest

from mnemostack.vector import VectorStore

SERVER_URL = os.environ.get("MNEMOSTACK_TEST_QDRANT_URL", "").strip()

pytestmark = pytest.mark.skipif(
    not SERVER_URL,
    reason="set MNEMOSTACK_TEST_QDRANT_URL to run real-server smoke tests",
)

_V1 = [1.0, 0.0, 0.0, 0.0]
_V2 = [0.0, 1.0, 0.0, 0.0]


@pytest.fixture(scope="module", autouse=True)
def _server_reachable():
    if not SERVER_URL:  # pragma: no cover - module already skipped
        return
    import httpx

    try:
        httpx.get(SERVER_URL, timeout=5).raise_for_status()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Qdrant server at {SERVER_URL} unreachable: {e}")


@pytest.fixture
def collection_name():
    """A unique collection per test, dropped afterwards even on failure."""
    name = f"smoke_{uuid.uuid4().hex[:12]}"
    yield name
    try:
        VectorStore(collection=name, dimension=4, host=SERVER_URL).client.delete_collection(
            name
        )
    except Exception:  # noqa: BLE001 - never fail a test on cleanup
        pass


def _sparse_store(name: str) -> VectorStore:
    s = VectorStore(collection=name, dimension=4, host=SERVER_URL, sparse_text=True)
    s.ensure_collection(recreate=True)
    return s


def _seed(s: VectorStore) -> None:
    s.upsert(1, _V1, {"text": "postgres backup restore procedure"}, tenant="acme")
    s.upsert(2, _V2, {"text": "kubernetes ingress setup"}, tenant="acme")
    s.upsert(3, _V1, {"text": "postgres tuning notes"}, tenant="globex")


def test_sparse_search_with_server_side_idf_ordering(collection_name):
    s = _sparse_store(collection_name)
    _seed(s)
    hits = s.sparse_search("postgres backup", limit=5)
    # Both postgres chunks match; the one also matching "backup" ranks first.
    assert [h.id for h in hits][:2] == [1, 3]
    assert hits[0].score > hits[1].score


def test_sparse_tenant_boundary(collection_name):
    s = _sparse_store(collection_name)
    _seed(s)
    assert [h.id for h in s.sparse_search("postgres", limit=5, tenant="acme")] == [1]


def test_matchtext_gate_against_a_real_text_index(collection_name):
    # The scenario local mode CANNOT represent: a real server requires the
    # full-text payload index before it accepts MatchText filters.
    s = _sparse_store(collection_name)
    _seed(s)
    s.ensure_text_index()
    assert {h.id for h in s.search(_V1, limit=5, text_any=["postgres"])} == {1, 3}
    # any-of semantics across tokens
    assert {h.id for h in s.search(_V1, limit=5, text_any=["kubernetes", "backup"])} == {1, 2}


def test_gate_composes_with_tenant(collection_name):
    s = _sparse_store(collection_name)
    _seed(s)
    s.ensure_text_index()
    hits = s.search(_V1, limit=5, text_any=["postgres"], tenant="acme")
    assert [h.id for h in hits] == [1]


def test_scroll_preserves_dense_vector_in_named_layout(collection_name):
    s = _sparse_store(collection_name)
    _seed(s)
    vecs = [h.vector for h in s.scroll(with_vectors=True, tenant="acme")]
    assert vecs and all(v is not None and len(v) == 4 for v in vecs)


def test_server_refuses_post_hoc_sparse_space(collection_name):
    # The documented migration reality: an existing dense collection cannot
    # gain the sparse space via update_collection — the refusal must surface
    # loudly with recreate/re-index guidance, never silently.
    dense = VectorStore(collection=collection_name, dimension=4, host=SERVER_URL)
    dense.ensure_collection(recreate=True)
    dense.upsert(10, _V1, {"text": "alpha zk-9931"})
    upgraded = VectorStore(
        collection=collection_name, dimension=4, host=SERVER_URL, sparse_text=True
    )
    with pytest.raises(RuntimeError, match="refused to add"):
        upgraded.ensure_collection()


def test_coverage_gap_and_backfill_flow(collection_name):
    # Space exists, but points were written dense-only (the async-store
    # shape): the gap is detected server-side (has_vector), ensure refuses
    # until backfilled, and the backfill makes the points recallable.
    sparse = _sparse_store(collection_name)
    dense_writer = VectorStore(collection=collection_name, dimension=4, host=SERVER_URL)
    dense_writer.upsert(20, _V1, {"text": "gamma delta zk-777"})
    dense_writer.upsert(21, _V2, {"text": "epsilon"})
    assert sparse.sparse_coverage_gap() == 2
    with pytest.raises(RuntimeError, match="sparse-backfill"):
        sparse.ensure_collection()
    assert sparse.backfill_sparse_text() == 2
    assert sparse.sparse_coverage_gap() == 0
    sparse.ensure_collection()  # now clean
    assert [h.id for h in sparse.sparse_search("zk-777", limit=5)] == [20]
