"""Batched payload patching: the store hook and its scalar-equivalent contract."""

from __future__ import annotations

from mnemostack.vector.patch import PayloadPatch, apply_patches_via


def _store(dim: int = 2):
    from qdrant_client import QdrantClient

    from mnemostack.vector.qdrant import VectorStore

    s = VectorStore.__new__(VectorStore)
    s.collection = "patch-test"
    s.dimension = dim
    s.client = QdrantClient(":memory:")
    s.sparse_text = False
    s.client.create_collection(
        collection_name=s.collection,
        vectors_config={"size": dim, "distance": "Cosine"},
    )
    return s


def _seed(s, cid: str, payload: dict) -> None:
    from qdrant_client import models

    s.client.upsert(
        collection_name=s.collection,
        points=[models.PointStruct(id=cid, vector=[0.1, 0.2], payload=payload)],
    )


def _payload_of(s, cid: str) -> dict:
    pt = s.client.retrieve(collection_name=s.collection, ids=[cid], with_payload=True)
    return dict(pt[0].payload or {})


CID_A = "11111111-1111-1111-1111-111111111111"
CID_B = "22222222-2222-2222-2222-222222222222"
CID_C = "33333333-3333-3333-3333-333333333333"


def test_batch_patches_match_scalar_semantics():
    s = _store()
    _seed(s, CID_A, {"text": "old", "title": "Old", "stale": "x", "foreign": "keep"})
    applied = s.apply_payload_patches(
        [
            PayloadPatch(
                id=CID_A,
                set_values={"text": "new", "title": "New"},
                delete_keys=("stale",),
            )
        ]
    )
    assert applied == 1
    payload = _payload_of(s, CID_A)
    assert payload["text"] == "new" and payload["title"] == "New"
    assert "stale" not in payload
    assert payload["foreign"] == "keep"  # merge semantics: untouched


def test_batch_size_bounds_round_trips(monkeypatch):
    s = _store()
    for cid in (CID_A, CID_B, CID_C):
        _seed(s, cid, {"v": "old"})
    calls: list[int] = []
    real = s.client.batch_update_points

    def counting(collection_name, update_operations, **kw):
        calls.append(len(update_operations))
        return real(
            collection_name=collection_name, update_operations=update_operations, **kw
        )

    monkeypatch.setattr(s.client, "batch_update_points", counting)
    patches = [
        PayloadPatch(id=cid, set_values={"v": "new"}) for cid in (CID_A, CID_B, CID_C)
    ]
    assert s.apply_payload_patches(patches, batch_size=2) == 3
    assert len(calls) == 2  # groups of 2 + 1
    assert all(_payload_of(s, cid)["v"] == "new" for cid in (CID_A, CID_B, CID_C))


def test_tenant_ownership_validated_for_whole_group_before_mutation():
    from mnemostack.vector.qdrant import TENANT_ID_KEY

    s = _store()
    _seed(s, CID_A, {"v": "old", TENANT_ID_KEY: "acme"})
    _seed(s, CID_B, {"v": "old", TENANT_ID_KEY: "other"})
    applied = s.apply_payload_patches(
        [
            PayloadPatch(id=CID_A, set_values={"v": "new"}),
            PayloadPatch(id=CID_B, set_values={"v": "new"}),
        ],
        tenant="acme",
    )
    assert applied == 1  # foreign point silently skipped, like the scalars
    assert _payload_of(s, CID_A)["v"] == "new"
    assert _payload_of(s, CID_B)["v"] == "old"
    # …and the owner label survives the scoped write.
    assert _payload_of(s, CID_A)[TENANT_ID_KEY] == "acme"


def test_unscoped_patches_cannot_touch_tenant_id():
    from mnemostack.vector.qdrant import TENANT_ID_KEY

    s = _store()
    _seed(s, CID_A, {"v": "old", TENANT_ID_KEY: "acme"})
    s.apply_payload_patches(
        [
            PayloadPatch(
                id=CID_A,
                set_values={"v": "new", TENANT_ID_KEY: "mallory"},
                delete_keys=(TENANT_ID_KEY,),
            )
        ]
    )
    payload = _payload_of(s, CID_A)
    assert payload["v"] == "new"
    assert payload[TENANT_ID_KEY] == "acme"  # neither reassigned nor deleted


def test_facade_falls_back_to_scalars_for_duck_stores():
    class _Duck:
        def __init__(self):
            self.deletes: list = []
            self.sets: list = []

        def delete_payload_keys(self, cid, keys):
            self.deletes.append((cid, keys))

        def set_payload(self, cid, payload):
            self.sets.append((cid, payload))

    duck = _Duck()
    n = apply_patches_via(
        duck,
        [PayloadPatch(id="p1", set_values={"a": 1}, delete_keys=("b",))],
    )
    assert n == 1
    assert duck.deletes == [("p1", ["b"])]
    assert duck.sets == [("p1", {"a": 1})]


def test_facade_routes_to_the_store_hook_when_present():
    class _Hooked:
        def __init__(self):
            self.batches: list = []

        def apply_payload_patches(self, patches, *, tenant=None, batch_size=100):
            self.batches.append((len(patches), tenant, batch_size))
            return len(patches)

    hooked = _Hooked()
    patches = [PayloadPatch(id=f"p{i}", set_values={"a": i}) for i in range(3)]
    assert apply_patches_via(hooked, patches, tenant="t1", batch_size=2) == 3
    assert hooked.batches == [(3, "t1", 2)]


def test_unscoped_batch_skips_missing_points_instead_of_404ing():
    """A real Qdrant server 404s (and aborts) a batch naming a since-deleted
    point — the in-memory client silently no-ops, which is exactly why this
    guard exists: existence is pre-checked per group, so a vanished point is
    skipped and NOT counted."""
    s = _store()
    _seed(s, CID_A, {"v": "old"})
    applied = s.apply_payload_patches(
        [
            PayloadPatch(id=CID_A, set_values={"v": "new"}),
            PayloadPatch(id=CID_B, set_values={"v": "new"}),  # never existed
        ]
    )
    assert applied == 1
    assert _payload_of(s, CID_A)["v"] == "new"


def test_batch_retries_after_a_mid_flight_404_and_skips_the_vanished_point(monkeypatch):
    """The residual race: a point passes the existence check, then vanishes
    before the batch lands and the server 404s (having possibly applied the
    earlier operations). The group is re-filtered and re-applied — idempotent
    — and the vanished point is not counted."""
    from qdrant_client.http.exceptions import UnexpectedResponse

    s = _store()
    _seed(s, CID_A, {"v": "old"})
    _seed(s, CID_B, {"v": "old"})
    real = s.client.batch_update_points
    calls: list[int] = []

    def racing(collection_name, update_operations, **kw):
        calls.append(len(update_operations))
        if len(calls) == 1:
            # B vanishes AFTER the existence check, and the server rejects
            # the whole batch the way a real one does.
            s.client.delete(collection_name, points_selector=[CID_B])
            raise UnexpectedResponse(404, "Not Found", b"", None)
        return real(
            collection_name=collection_name, update_operations=update_operations, **kw
        )

    monkeypatch.setattr(s.client, "batch_update_points", racing)

    applied = s.apply_payload_patches(
        [
            PayloadPatch(id=CID_A, set_values={"v": "new"}),
            PayloadPatch(id=CID_B, set_values={"v": "new"}),
        ]
    )

    assert applied == 1  # only the surviving point
    assert len(calls) == 2  # first attempt + one retry with the survivor
    assert _payload_of(s, CID_A)["v"] == "new"


def test_batch_404_without_a_vanished_point_propagates(monkeypatch):
    """A 404 the existence re-check cannot explain (nothing vanished) must
    raise, not spin or get swallowed."""
    import pytest
    from qdrant_client.http.exceptions import UnexpectedResponse

    s = _store()
    _seed(s, CID_A, {"v": "old"})

    def always_404(collection_name, update_operations, **kw):
        raise UnexpectedResponse(404, "Not Found", b"", None)

    monkeypatch.setattr(s.client, "batch_update_points", always_404)

    with pytest.raises(UnexpectedResponse):
        s.apply_payload_patches([PayloadPatch(id=CID_A, set_values={"v": "new"})])


def test_sync_flushes_bounded_groups_beyond_one_batch():
    """More changed points than PAYLOAD_PATCH_BATCH flush in bounded groups."""
    from mnemostack.markdown.sync import PAYLOAD_PATCH_BATCH, upsert_markdown_chunks

    class _Prov:
        def embed(self, text):
            return [0.1]

        def embed_batch(self, texts):
            return [[0.1] for _ in texts]

    class _HookedStore:
        def __init__(self):
            self.batches: list[int] = []

        def apply_payload_patches(self, patches, *, tenant=None, batch_size=100):
            self.batches.append(len(patches))
            return len(patches)

        def upsert(self, cid, vec, payload, **kw):
            pass

        def upsert_batch(self, points, **kw):
            pass

    n = PAYLOAD_PATCH_BATCH * 2 + 50
    chunks = [
        (f"id{i}", "body", {"text": "body", "source": "a.md", "title": f"T{i}"})
        for i in range(n)
    ]
    existing = {
        f"id{i}": {"text": "body", "source": "a.md", "title": "OLD"} for i in range(n)
    }
    store = _HookedStore()
    res = upsert_markdown_chunks(store, _Prov(), chunks, existing_payloads=existing)
    assert res.refreshed == n
    assert store.batches == [PAYLOAD_PATCH_BATCH, PAYLOAD_PATCH_BATCH, 50]


def test_refreshed_counts_only_patches_the_store_confirmed():
    """A tenant-skipped / vanished point must not inflate `refreshed`: the
    counter takes the store's own applied count, not the attempt count."""
    from mnemostack.markdown.sync import upsert_markdown_chunks

    class _Prov:
        def embed(self, text):
            return [0.1]

        def embed_batch(self, texts):
            return [[0.1] for _ in texts]

    class _SkippingStore:
        """Applies all but one patch per batch (e.g. a foreign-tenant skip)."""

        def apply_payload_patches(self, patches, *, tenant=None, batch_size=100):
            return max(0, len(patches) - 1)

        def upsert(self, cid, vec, payload, **kw):
            pass

        def upsert_batch(self, points, **kw):
            pass

    chunks = [
        (f"id{i}", "body", {"text": "body", "source": "a.md", "title": f"T{i}"})
        for i in range(3)
    ]
    existing = {
        f"id{i}": {"text": "body", "source": "a.md", "title": "OLD"} for i in range(3)
    }
    res = upsert_markdown_chunks(_SkippingStore(), _Prov(), chunks, existing_payloads=existing)
    assert res.compared == 3
    assert res.refreshed == 2  # 3 attempted, 2 confirmed by the store


def test_markdown_sync_patches_through_the_hook_in_bounded_groups():
    from mnemostack.markdown.sync import upsert_markdown_chunks

    class _Prov:
        def embed(self, text):
            return [0.1]

        def embed_batch(self, texts):
            return [[0.1] for _ in texts]

    class _HookedStore:
        def __init__(self):
            self.batches: list[int] = []
            self.scalar_sets = 0

        def apply_payload_patches(self, patches, *, tenant=None, batch_size=100):
            self.batches.append(len(patches))
            return len(patches)

        def set_payload(self, cid, payload, **kw):
            self.scalar_sets += 1

        def delete_payload_keys(self, cid, keys, **kw):
            pass

        def upsert(self, cid, vec, payload, **kw):
            pass

        def upsert_batch(self, points, **kw):
            pass

    store = _HookedStore()
    chunks = [
        (f"id{i}", "body", {"text": "body", "source": "a.md", "title": f"T{i}"})
        for i in range(5)
    ]
    existing = {f"id{i}": {"text": "body", "source": "a.md", "title": "OLD"} for i in range(5)}
    res = upsert_markdown_chunks(store, _Prov(), chunks, existing_payloads=existing)
    assert res.refreshed == 5
    assert store.scalar_sets == 0  # everything flowed through the hook
    assert sum(store.batches) == 5
