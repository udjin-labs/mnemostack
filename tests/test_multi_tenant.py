"""Multi-tenant data-isolation boundary (vector layer).

The security-critical property: with a `tenant` set, a read can NEVER see
another tenant's points, and a write can NEVER land in another tenant's
namespace — via any parameter combination. These run against an in-memory
Qdrant (`:memory:`), no external services.
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance

from mnemostack.vector import TENANT_ID_KEY, VectorStore


def _store() -> VectorStore:
    s = VectorStore.__new__(VectorStore)
    s.collection = "mt"
    s.dimension = 4
    s.distance = Distance.COSINE
    s.client = QdrantClient(":memory:")
    s.ensure_collection()
    return s


_VEC = [1.0, 0.0, 0.0, 0.0]


def _seed_two_tenants(store: VectorStore) -> None:
    # Same vector for both tenants so similarity can't be what separates them —
    # only the tenant filter can.
    store.upsert(1, _VEC, {"text": "alpha-1"}, tenant="alpha")
    store.upsert(2, _VEC, {"text": "alpha-2"}, tenant="alpha")
    store.upsert(3, _VEC, {"text": "beta-1"}, tenant="beta")


def test_write_stamps_tenant_id():
    store = _store()
    store.upsert(1, _VEC, {"text": "x"}, tenant="alpha")
    hit = store.search(_VEC, limit=1)[0]
    assert hit.payload[TENANT_ID_KEY] == "alpha"


def test_read_is_isolated_by_tenant():
    store = _store()
    _seed_two_tenants(store)

    alpha = {h.id for h in store.search(_VEC, limit=10, tenant="alpha")}
    beta = {h.id for h in store.search(_VEC, limit=10, tenant="beta")}
    assert alpha == {1, 2}
    assert beta == {3}
    # No leakage in either direction.
    assert alpha.isdisjoint(beta)


def test_no_tenant_sees_everything_backward_compat():
    # Single-tenant / legacy: without a tenant, search is unfiltered.
    store = _store()
    _seed_two_tenants(store)
    all_ids = {h.id for h in store.search(_VEC, limit=10)}
    assert all_ids == {1, 2, 3}


def test_server_tenant_overrides_caller_supplied_tenant_id():
    # A client can't smuggle a different tenant_id into the payload — the
    # server-supplied tenant wins.
    store = _store()
    store.upsert(1, _VEC, {"text": "x", TENANT_ID_KEY: "evil"}, tenant="alpha")
    hit = store.search(_VEC, limit=1, tenant="alpha")[0]
    assert hit.payload[TENANT_ID_KEY] == "alpha"
    # And it's genuinely not visible as tenant "evil".
    assert store.search(_VEC, limit=10, tenant="evil") == []


def test_count_is_tenant_scoped():
    store = _store()
    _seed_two_tenants(store)
    assert store.count() == 3
    assert store.count(tenant="alpha") == 2
    assert store.count(tenant="beta") == 1


def test_scroll_is_tenant_scoped():
    store = _store()
    _seed_two_tenants(store)
    assert {h.id for h in store.scroll(tenant="alpha")} == {1, 2}
    assert {h.id for h in store.scroll(tenant="beta")} == {3}


def test_upsert_batch_stamps_tenant():
    store = _store()
    store.upsert_batch(
        [(1, _VEC, {"text": "a"}), (2, _VEC, {"text": "b"})], tenant="alpha"
    )
    assert store.count(tenant="alpha") == 2
    assert store.search(_VEC, limit=10, tenant="beta") == []


def test_invalidate_will_not_touch_another_tenant():
    store = _store()
    _seed_two_tenants(store)
    # Ask to invalidate beta's point (id 3) but scoped to tenant alpha -> skipped.
    updated = store.invalidate([3], tenant="alpha")
    assert updated == 0
    # id 3 is still current (not hidden) for beta.
    beta = {h.id for h in store.search(_VEC, limit=10, tenant="beta", hide_invalidated=True)}
    assert 3 in beta
    # Scoped to its own tenant, it invalidates.
    assert store.invalidate([3], tenant="beta") == 1
    beta_after = {
        h.id for h in store.search(_VEC, limit=10, tenant="beta", hide_invalidated=True)
    }
    assert 3 not in beta_after


@pytest.mark.parametrize("filters", [None, {"text": "alpha-1"}])
def test_tenant_and_caller_filters_combine(filters):
    # Caller filters AND with the tenant filter — they never widen scope.
    store = _store()
    _seed_two_tenants(store)
    hits = store.search(_VEC, limit=10, filters=filters, tenant="alpha")
    ids = {h.id for h in hits}
    assert ids <= {1, 2}  # never beta's id 3
    if filters:
        assert ids == {1}


def test_stamp_tenant_migration_is_idempotent():
    # A legacy corpus with no tenant_id is migrated into one named tenant.
    store = _store()
    store.upsert(1, _VEC, {"text": "legacy-1"})
    store.upsert(2, _VEC, {"text": "legacy-2"})
    assert store.count(tenant="alpha") == 0  # not yet stamped

    stamped = store.stamp_tenant("alpha", only_missing=True)
    assert stamped == 2
    assert store.count(tenant="alpha") == 2
    # Re-run: nothing left to stamp (idempotent).
    assert store.stamp_tenant("alpha", only_missing=True) == 0
    # And a different tenant's later write isn't swept up by a re-run.
    store.upsert(3, _VEC, {"text": "beta-1"}, tenant="beta")
    assert store.stamp_tenant("alpha", only_missing=True) == 0
    assert store.count(tenant="beta") == 1


class _FakeEmbedder:
    dimension = 4

    @property
    def name(self):
        return "fake:mt"

    def embed(self, text):
        return _VEC

    def embed_batch(self, texts):
        return [_VEC for _ in texts]


def test_ingestor_stamps_tenant():
    from mnemostack import IngestItem, Ingestor

    store = _store()
    Ingestor(embedding=_FakeEmbedder(), vector_store=store, tenant="alpha").ingest(
        [IngestItem(text="a note", source="n1"), IngestItem(text="another", source="n2")]
    )
    assert store.count(tenant="alpha") >= 1
    assert store.search(_VEC, limit=10, tenant="beta") == []
    for hit in store.search(_VEC, limit=10, tenant="alpha"):
        assert hit.payload[TENANT_ID_KEY] == "alpha"


def test_same_content_across_tenants_does_not_collide():
    # Two tenants ingesting the IDENTICAL (source, offset, text) must NOT share a
    # point id — otherwise the second write clobbers the first (data loss).
    from mnemostack import IngestItem, Ingestor

    store = _store()
    item = IngestItem(text="shared content", source="doc.md")
    Ingestor(embedding=_FakeEmbedder(), vector_store=store, tenant="alpha").ingest([item])
    Ingestor(embedding=_FakeEmbedder(), vector_store=store, tenant="beta").ingest([item])

    assert store.count(tenant="alpha") == 1
    assert store.count(tenant="beta") == 1
    assert store.count() == 2  # both survive — no clobber


def test_metadata_cannot_set_tenant_id():
    from mnemostack import IngestItem, Ingestor

    # Under a tenant-scoped Ingestor, a planted metadata tenant_id is overridden.
    store = _store()
    Ingestor(embedding=_FakeEmbedder(), vector_store=store, tenant="alpha").ingest(
        [IngestItem(text="x", source="a", metadata={TENANT_ID_KEY: "evil"})]
    )
    assert store.search(_VEC, limit=10, tenant="evil") == []
    assert store.count(tenant="alpha") == 1

    # Under a single-tenant Ingestor, a planted metadata tenant_id is stripped
    # (the Ingestor's tenant is the only authority), so it can't lie in wait.
    store2 = _store()
    Ingestor(embedding=_FakeEmbedder(), vector_store=store2).ingest(
        [IngestItem(text="y", source="b", metadata={TENANT_ID_KEY: "ghost"})]
    )
    assert store2.count(tenant="ghost") == 0
    hit = store2.search(_VEC, limit=1)[0]
    assert TENANT_ID_KEY not in hit.payload


def test_set_payload_will_not_relabel_another_tenant():
    store = _store()
    _seed_two_tenants(store)  # ids 1,2 = alpha; 3 = beta
    # Beta tries to grab alpha's point id 1 -> skipped, no change.
    store.set_payload(1, {"note": "stolen", TENANT_ID_KEY: "beta"}, tenant="beta")
    assert store.count(tenant="beta") == 1  # still just id 3
    assert {h.id for h in store.search(_VEC, limit=10, tenant="alpha")} == {1, 2}
    # Owner can set payload, but can't change its own tenant_id.
    store.set_payload(1, {"note": "ok", TENANT_ID_KEY: "beta"}, tenant="alpha")
    alpha_hit = next(h for h in store.search(_VEC, limit=10, tenant="alpha") if h.id == 1)
    assert alpha_hit.payload["note"] == "ok"
    assert alpha_hit.payload[TENANT_ID_KEY] == "alpha"


def test_prune_is_tenant_scoped():
    from mnemostack import stable_chunk_id
    from mnemostack.ingest import prune_stale_chunks

    store = _store()
    # Same source in two tenants; alpha's fresh set only lists alpha's id.
    a_id = stable_chunk_id("doc.md", 0, "a-text", tenant="alpha")
    b_id = stable_chunk_id("doc.md", 0, "b-text", tenant="beta")
    store.upsert(a_id, _VEC, {"text": "a-text", "source": "doc.md"}, tenant="alpha")
    store.upsert(b_id, _VEC, {"text": "b-text", "source": "doc.md"}, tenant="beta")

    # Prune alpha with a fresh set that keeps a_id — must NOT delete beta's b_id.
    removed = prune_stale_chunks(store, {"doc.md": {a_id}}, tenant="alpha")
    assert removed == 0
    assert store.count(tenant="beta") == 1
    assert store.count(tenant="alpha") == 1


async def test_async_store_isolates_by_tenant():
    from qdrant_client import AsyncQdrantClient

    from mnemostack.vector import AsyncVectorStore

    store = AsyncVectorStore.__new__(AsyncVectorStore)
    store.collection = "mt_async"
    store.dimension = 4
    store.distance = Distance.COSINE
    store.client = AsyncQdrantClient(":memory:")
    await store.ensure_collection()

    await store.upsert(1, _VEC, {"text": "a"}, tenant="alpha")
    await store.upsert(2, _VEC, {"text": "b"}, tenant="beta")

    alpha = {h.id for h in await store.search(_VEC, limit=10, tenant="alpha")}
    assert alpha == {1}
    assert await store.count(tenant="beta") == 1
    await store.close()
