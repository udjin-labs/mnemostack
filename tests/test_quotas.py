"""Per-tenant storage quotas — store, enforcement helper, and ingest wiring."""

from __future__ import annotations

import json

import pytest

from mnemostack.ingest import IngestItem, Ingestor
from mnemostack.quotas import (
    FileQuotaStore,
    QuotaExceededError,
    QuotaStoreError,
    TenantQuota,
    enforce_points_quota,
)

# ---------- FileQuotaStore ----------


def test_quota_store_roundtrip(tmp_path):
    store = FileQuotaStore(tmp_path / "quotas.json")
    assert store.get("acme") is None  # unset
    store.set("acme", max_points=1000)
    assert store.get("acme") == TenantQuota(max_points=1000)
    # list + json shape
    assert store.list_quotas() == [{"tenant": "acme", "max_points": 1000}]
    # replace
    store.set("acme", max_points=500)
    assert store.get("acme").max_points == 500
    # remove
    assert store.remove("acme") is True
    assert store.get("acme") is None
    assert store.remove("acme") is False


def test_quota_store_set_none_is_unlimited(tmp_path):
    store = FileQuotaStore(tmp_path / "q.json")
    store.set("t", max_points=None)
    assert store.get("t") == TenantQuota(max_points=None)


def test_quota_store_rejects_negative(tmp_path):
    store = FileQuotaStore(tmp_path / "q.json")
    with pytest.raises(ValueError):
        store.set("t", max_points=-1)


def test_quota_store_rejects_bool(tmp_path):
    # bool is an int subclass — True must not be stored as max_points=1.
    store = FileQuotaStore(tmp_path / "q.json")
    with pytest.raises(ValueError):
        store.set("t", max_points=True)


def test_quota_store_corrupt_file_fails_open(tmp_path, caplog):
    p = tmp_path / "q.json"
    p.write_text("{ not json")
    store = FileQuotaStore(p)
    # get() fails OPEN (no limit) — a broken quota file must not block ingest.
    assert store.get("acme") is None
    # management ops surface the corruption loudly.
    with pytest.raises(QuotaStoreError):
        store.set("acme", max_points=1)


def test_quota_store_malformed_max_points_ignored(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps({"quotas": {"acme": {"max_points": "lots"}}}))
    # a bad max_points is dropped to None (no limit), not raised.
    assert FileQuotaStore(p).get("acme") == TenantQuota(max_points=None)


# ---------- enforce_points_quota ----------


def test_enforce_points_quota():
    # no-op when unscoped or no limit
    enforce_points_quota(None, 100, 50, 10)  # unscoped
    enforce_points_quota("t", 100, 50, None)  # no limit
    # within limit ok
    enforce_points_quota("t", 90, 10, 100)
    # exceeding raises
    with pytest.raises(QuotaExceededError) as ei:
        enforce_points_quota("t", 95, 10, 100)
    assert ei.value.tenant == "t" and ei.value.limit == 100 and ei.value.attempted == 105


# ---------- Ingestor enforcement ----------


class _Emb:
    dimension = 3

    def embed(self, text):
        return [0.1, 0.2, 0.3] if text else []

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


class _CountingStore:
    """Fake store modeling deterministic-id idempotency: re-upserting an existing
    id doesn't grow the count. ``existing`` is a base of pre-stored points with
    ids the batch won't collide with."""

    def __init__(self, existing=0):
        self._base = existing
        self._ids: set[str] = set()
        self.upserts: list = []

    def count(self, tenant=None):
        return self._base + len(self._ids)

    def retrieve_existing_ids(self, ids):
        return {str(i) for i in ids if str(i) in self._ids}

    def upsert_batch(self, points, tenant=None):
        for p in points:
            self.upsert(p[0], p[1], p[2])
        return len(points)

    def upsert(self, id, vec, payload, tenant=None):
        self.upserts.append((id, vec, payload))
        self._ids.add(str(id))  # a set — re-upsert of the same id doesn't grow

    def set_payload(self, cid, payload, tenant=None):
        pass

    def delete_payload_keys(self, cid, keys, tenant=None):
        pass


def _items(n):
    return [IngestItem(source=f"s{i}", text=f"text {i}", offset=0) for i in range(n)]


def test_ingest_refuses_over_quota():
    store = _CountingStore(existing=8)  # tenant already has 8 points
    ing = Ingestor(embedding=_Emb(), vector_store=store, batch_size=5,
                   tenant="acme", max_points=10)
    # First flush of 5 would make 8+5=13 > 10 → refused before any upsert.
    with pytest.raises(QuotaExceededError):
        ing.ingest(_items(5))
    assert store.upserts == []  # nothing written


def test_ingest_within_quota_succeeds():
    store = _CountingStore(existing=0)
    ing = Ingestor(embedding=_Emb(), vector_store=store, batch_size=5,
                   tenant="acme", max_points=10)
    stats = ing.ingest(_items(4))
    assert stats.upserted == 4 and len(store.upserts) == 4


def test_reingest_at_limit_is_not_falsely_rejected():
    # Fill a tenant to its limit, then re-ingest the SAME data with a fresh
    # Ingestor (bypassing the in-process skip_seen cache): the deterministic ids
    # already exist, so no NEW points — it must NOT be rejected.
    store = _CountingStore(existing=0)
    items = _items(5)
    Ingestor(embedding=_Emb(), vector_store=store, batch_size=5,
             tenant="acme", max_points=5).ingest(items)
    assert store.count(tenant="acme") == 5
    # fresh Ingestor, same store + same items, at the exact limit
    stats = Ingestor(embedding=_Emb(), vector_store=store, batch_size=5,
                     tenant="acme", max_points=5).ingest(items)
    assert stats.upserted == 5  # re-upserted (idempotent), not rejected
    assert store.count(tenant="acme") == 5  # count didn't grow


def test_ingest_no_quota_when_unscoped_or_unlimited():
    # unscoped (tenant=None) never checks the quota even if max_points set
    store = _CountingStore(existing=100)
    ing = Ingestor(embedding=_Emb(), vector_store=store, batch_size=5,
                   tenant=None, max_points=10)
    assert ing.ingest(_items(3)).upserted == 3
    # scoped but no limit
    store2 = _CountingStore(existing=100)
    ing2 = Ingestor(embedding=_Emb(), vector_store=store2, batch_size=5,
                    tenant="acme", max_points=None)
    assert ing2.ingest(_items(3)).upserted == 3


def test_ingest_quota_end_to_end_inmemory():
    # Real VectorStore (in-memory Qdrant): exercises retrieve_existing_ids + the
    # tenant-filtered count, not just the fake.
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance

    from mnemostack.vector import VectorStore

    vs = VectorStore.__new__(VectorStore)
    vs.collection = "q"
    vs.dimension = 3
    vs.distance = Distance.COSINE
    vs.client = QdrantClient(":memory:")
    vs.ensure_collection()

    def _ing():
        return Ingestor(embedding=_Emb(), vector_store=vs, batch_size=3,
                        tenant="acme", max_points=4)

    _ing().ingest(_items(4))  # exactly at the limit
    assert vs.count(tenant="acme") == 4
    _ing().ingest(_items(4))  # re-ingest same data — no new points → allowed
    assert vs.count(tenant="acme") == 4
    with pytest.raises(QuotaExceededError):  # one genuinely-new item → exceeds
        _ing().ingest(_items(5))


# ---------- markdown enforcement ----------


def test_markdown_upsert_refuses_over_quota():
    from mnemostack.markdown.sync import upsert_markdown_chunks

    store = _CountingStore(existing=9)
    chunks = [(f"id{i}", f"t{i}", {"source": "a.md"}) for i in range(5)]
    with pytest.raises(QuotaExceededError):
        upsert_markdown_chunks(store, _Emb(), chunks, {}, tenant="acme", max_points=10)
    assert store.upserts == []


def test_markdown_edit_at_cap_counts_net_not_gross():
    from mnemostack.markdown.sync import upsert_markdown_chunks

    _md = ["text", "source", "offset"]  # a real markdown chunk always has these
    store = _CountingStore(existing=0)
    v1 = [("id1", "t", {"source": "a.md", "_md_keys": _md}),
          ("id2", "t", {"source": "a.md", "_md_keys": _md})]
    upsert_markdown_chunks(store, _Emb(), v1, {}, tenant="acme", max_points=2)
    assert store.count(tenant="acme") == 2  # at cap
    # edit a.md → 2 chunks with NEW ids; the old 2 will be pruned (net 0).
    existing = {"id1": {"source": "a.md", "_md_keys": _md},
                "id2": {"source": "a.md", "_md_keys": _md}}
    v2 = [("id3", "t", {"source": "a.md", "_md_keys": _md}),
          ("id4", "t", {"source": "a.md", "_md_keys": _md})]
    res = upsert_markdown_chunks(store, _Emb(), v2, existing, tenant="acme", max_points=2)
    assert res.inserted == 2  # allowed — net change is 0, not +2


def test_ingest_dedup_within_flush_counts_once():
    # A duplicated item (same source/offset/text → same id) in one flush stores
    # one point, so it must count once — a tenant with room for 1 accepts it.
    store = _CountingStore(existing=0)
    ing = Ingestor(embedding=_Emb(), vector_store=store, batch_size=10,
                   tenant="acme", max_points=1)
    item = IngestItem(source="s", text="dup", offset=0)
    ing.ingest([item, item])  # must not raise
    assert store.count(tenant="acme") == 1


def test_watch_skips_over_quota_file_without_retry():
    from mnemostack.markdown.watch import MarkdownWatcher

    class _Syncer:
        index_root = "/root"

        def index_file(self, path):
            raise QuotaExceededError("acme", 10, 11)

        def remove_file(self, path):
            pass

    errors: list = []
    w = MarkdownWatcher(_Syncer(), "/root", on_error=lambda p, k, e: errors.append((p, e)))
    w._apply("/root/big.md", "modify")
    assert "/root/big.md" not in w._failed  # not queued for retry
    assert errors and isinstance(errors[0][1], QuotaExceededError)  # reported once


def test_markdown_upsert_refresh_does_not_count():
    from mnemostack.markdown.sync import upsert_markdown_chunks

    store = _CountingStore(existing=10)  # already at limit
    # all chunks already indexed → refreshes only, no NEW points → allowed
    chunks = [("id0", "t0", {"source": "a.md"})]
    existing = {"id0": {"_md_keys": []}}
    res = upsert_markdown_chunks(store, _Emb(), chunks, existing, tenant="acme", max_points=10)
    assert res.refreshed == 1 and res.inserted == 0


# ---------- CLI ----------


def test_cli_quota_set_list_rm_roundtrip(tmp_path, capsys):
    import argparse

    from mnemostack import cli

    qf = str(tmp_path / "quotas.json")
    assert cli.cmd_quota_set(
        argparse.Namespace(tenant="acme", max_points=5000, quotas_file=qf)
    ) == 0
    capsys.readouterr()  # discard the set output
    assert cli.cmd_quota_list(argparse.Namespace(quotas_file=qf, json=True)) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed == [{"tenant": "acme", "max_points": 5000}]
    assert cli.cmd_quota_rm(argparse.Namespace(tenant="acme", quotas_file=qf)) == 0
    assert cli.cmd_quota_rm(argparse.Namespace(tenant="acme", quotas_file=qf)) == 1  # gone


def test_cli_resolve_max_points(tmp_path):
    import argparse

    from mnemostack import cli

    qf = str(tmp_path / "quotas.json")
    FileQuotaStore(qf).set("acme", max_points=42)
    args = argparse.Namespace(quotas_file=qf)
    assert cli._resolve_max_points(args, "acme") == 42  # resolved from the store
    assert cli._resolve_max_points(args, "other") is None  # no quota set
    assert cli._resolve_max_points(args, None) is None  # unscoped ingest
