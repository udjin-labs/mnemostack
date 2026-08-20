"""`mnemostack index --tenant` (closes #160).

The remote write surface scopes ids by tenant and `index-markdown`
already had `--tenant`, but the fixed-window `index` command wrote
unscoped ids — so an operator indexing the same document a client had
POSTed to /memories created a SECOND, unscoped copy that tenant-filtered
recall never saw and `--prune` could not manage.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from mnemostack import cli
from mnemostack.ingest import IngestItem, ingest_remote_items, stable_chunk_id
from mnemostack.vector import VectorStore
from mnemostack.vector.qdrant import TENANT_ID_KEY


class _FakeEmbedding:
    dimension = 3

    def __init__(self) -> None:
        self.embedded = 0

    def embed(self, text: str) -> list[float]:
        self.embedded += 1
        h = abs(hash(text))
        return [(h % 97) / 97.0, (h % 89) / 89.0, 1.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def health_check(self):
        return True, "ok"


@pytest.fixture()
def indexer(monkeypatch, tmp_path):
    """cli.cmd_index wired to an in-memory store and a fake embedder."""
    store = VectorStore(collection="idx", dimension=3)
    store.client = QdrantClient(":memory:")
    store.ensure_collection()
    emb = _FakeEmbedding()
    monkeypatch.setattr(cli, "get_provider", lambda *_a, **_k: emb)
    monkeypatch.setattr(cli, "_indexing_store", lambda *_a, **_k: store)

    def _run(path: Path, **overrides):
        args = _index_args(path, **overrides)
        rc = cli.cmd_index(args)
        assert rc == 0, rc
        return rc

    return _run, store, emb


def _index_args(path: Path, **overrides):
    """Real parser defaults — a hand-built Namespace silently omits the
    flags cmd_index reads and turns a contract test into a crash test."""
    argv = ["index", str(path), "--yes"]
    if overrides.pop("tenant", None) is not None:
        argv += ["--tenant", overrides.pop("_tenant_value")]
    if overrides.pop("prune", False):
        argv.append("--prune")
    if overrides.pop("refresh_payloads", False):
        argv.append("--refresh-payloads")
    args = cli.build_parser().parse_args(argv)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _doc(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _payloads(store) -> list[dict]:
    return [hit.payload for hit in store.scroll()]


def test_tenant_indexing_stamps_and_scopes_ids(indexer, tmp_path):
    run, store, _emb = indexer
    doc = _doc(tmp_path, "notes.txt", "the deploy window is Friday at 15:00")
    run(doc, tenant=True, _tenant_value="acme")
    payloads = _payloads(store)
    assert payloads and all(p[TENANT_ID_KEY] == "acme" for p in payloads)
    # The id is the library's tenant-scoped id, not the unscoped one.
    text = payloads[0]["text"]
    source = payloads[0]["source"]
    offset = payloads[0].get("offset", 0)
    scoped = stable_chunk_id(source, offset, text, tenant="acme")
    unscoped = stable_chunk_id(source, offset, text)
    ids = {str(pid) for pid in store.iter_ids()}
    assert scoped in ids and unscoped not in ids


def test_index_and_remote_write_share_ids_for_a_tenant(indexer, tmp_path):
    """The point of the issue: an operator re-indexing what a client wrote
    must land on the SAME points, not a parallel unscoped copy."""
    run, store, emb = indexer
    text = "the staging cluster lives in eu-central-1"
    doc = _doc(tmp_path, "ops.txt", text)
    run(doc, tenant=True, _tenant_value="acme")
    before = {str(pid) for pid in store.iter_ids()}
    payload = _payloads(store)[0]

    # The same content through the remote surface, same tenant, same source.
    results = ingest_remote_items(
        emb,
        store,
        [IngestItem(text=payload["text"], source=payload["source"], offset=0)],
        tenant="acme",
    )
    assert [r.status for r in results] == ["duplicate"]  # deduplicated, not doubled
    assert {str(pid) for pid in store.iter_ids()} == before


def test_two_tenants_do_not_collide_on_one_document(indexer, tmp_path):
    run, store, _emb = indexer
    doc = _doc(tmp_path, "shared.txt", "one document indexed by two tenants")
    run(doc, tenant=True, _tenant_value="acme")
    run(doc, tenant=True, _tenant_value="globex")
    tenants = {p[TENANT_ID_KEY] for p in _payloads(store)}
    assert tenants == {"acme", "globex"}
    assert len({str(pid) for pid in store.iter_ids()}) == 2  # no overwrite


def test_prune_is_tenant_scoped(indexer, tmp_path):
    """Re-indexing an edited document prunes only that tenant's stale
    chunks — another tenant's copy of the same source survives."""
    run, store, _emb = indexer
    doc = _doc(tmp_path, "edited.txt", "first version of the document")
    run(doc, tenant=True, _tenant_value="acme")
    run(doc, tenant=True, _tenant_value="globex")
    doc.write_text("second version of the document entirely", encoding="utf-8")
    run(doc, tenant=True, _tenant_value="acme", prune=True)
    by_tenant: dict[str, list[str]] = {}
    for p in _payloads(store):
        by_tenant.setdefault(p[TENANT_ID_KEY], []).append(p["text"])
    assert by_tenant["acme"] == ["second version of the document entirely"]
    assert by_tenant["globex"] == ["first version of the document"]


def test_unscoped_indexing_is_unchanged(indexer, tmp_path):
    """tenant=None must reproduce the historical id and carry no stamp —
    single-tenant deployments are untouched by this feature."""
    run, store, _emb = indexer
    doc = _doc(tmp_path, "legacy.txt", "a single-tenant corpus entry")
    run(doc)
    payloads = _payloads(store)
    assert payloads and all(TENANT_ID_KEY not in p for p in payloads)
    expected = stable_chunk_id(
        payloads[0]["source"], payloads[0].get("offset", 0), payloads[0]["text"]
    )
    assert expected in {str(pid) for pid in store.iter_ids()}


def test_empty_tenant_is_rejected(indexer, tmp_path):
    _run, _store, _emb = indexer
    doc = _doc(tmp_path, "x.txt", "content")
    assert cli.cmd_index(_index_args(doc, tenant=True, _tenant_value="   ")) == 2


def test_recreate_is_refused_under_a_tenant(indexer, tmp_path):
    """R1 (codex P1): --recreate drops the WHOLE collection, so under a
    tenant it would delete every other tenant's points. Refused, as
    index-markdown has refused since the graph-tenancy work."""
    _run, store, _emb = indexer
    doc = _doc(tmp_path, "keep.txt", "another tenant's content lives here")
    args = _index_args(doc, tenant=True, _tenant_value="globex")
    assert cli.cmd_index(args) == 0
    before = {str(pid) for pid in store.iter_ids()}
    args = _index_args(doc, tenant=True, _tenant_value="acme")
    args.recreate = True
    assert cli.cmd_index(args) == 2  # refused, not executed
    assert {str(pid) for pid in store.iter_ids()} == before  # nothing dropped


def _quota_file(tmp_path: Path, tenant: str, max_points: int) -> str:
    from mnemostack.quotas import FileQuotaStore

    path = tmp_path / "quotas.json"
    FileQuotaStore(str(path)).set(tenant, max_points=max_points)
    return str(path)


def test_quota_refuses_the_run_and_writes_nothing(indexer, tmp_path):
    """R1 (review agent): `index --tenant` was the ONE write path that
    ignored `mnemostack quota set` — 158 points landed against a cap of 2.
    The refusal must also cost no embedding: the check runs before the
    provider is called, so a rejected run is free."""
    _run, store, emb = indexer
    doc = _doc(tmp_path, "big.txt", "quota" * 200)
    args = _index_args(doc, tenant=True, _tenant_value="acme")
    args.quotas_file = _quota_file(tmp_path, "acme", 2)
    args.chunk_size = 40
    embedded_before = emb.embedded
    assert cli.cmd_index(args) == 2
    assert list(store.iter_ids()) == []  # nothing written
    assert emb.embedded == embedded_before  # nothing embedded


def test_quota_admits_a_run_that_fits(indexer, tmp_path):
    """The guard rejects growth past the cap, not indexing itself."""
    _run, store, _emb = indexer
    doc = _doc(tmp_path, "small.txt", "one chunk of content")
    args = _index_args(doc, tenant=True, _tenant_value="acme")
    args.quotas_file = _quota_file(tmp_path, "acme", 50)
    assert cli.cmd_index(args) == 0
    assert len(list(store.iter_ids())) == 1


def test_another_tenants_quota_does_not_bind(indexer, tmp_path):
    """The cap is per tenant: globex's own limit is what applies to globex,
    and a tenant with no quota row is unlimited."""
    _run, store, _emb = indexer
    doc = _doc(tmp_path, "big.txt", "quota" * 200)
    args = _index_args(doc, tenant=True, _tenant_value="globex")
    args.quotas_file = _quota_file(tmp_path, "acme", 2)  # acme's cap, not globex's
    args.chunk_size = 40
    assert cli.cmd_index(args) == 0
    assert len(list(store.iter_ids())) > 2


def test_refresh_payload_writes_are_tenant_scoped(indexer, tmp_path):
    """R1 (codex P1): the refresh path scoped its snapshot READ but wrote
    patches unscoped — a point recreated under another owner mid-run could
    be patched, and tenant-aware stores got an unscoped operation."""
    run, store, _emb = indexer
    doc = _doc(tmp_path, "refresh.txt", "content whose payload gets refreshed")
    run(doc, tenant=True, _tenant_value="acme")
    seen: list[dict] = []
    orig = store.apply_payload_patches

    def _recording(patches, **kwargs):
        seen.append(kwargs)
        return orig(patches, **kwargs)

    store.apply_payload_patches = _recording  # type: ignore[method-assign]
    # A stale ownership marker in the STORED payload guarantees the refresh
    # diff produces a patch (the indexer must clear keys it used to own).
    pid = next(iter(store.iter_ids()))
    store.client.set_payload(
        collection_name=store.collection,
        payload={"_enrich_keys": ["ghost_field"], "ghost_field": "left over"},
        points=[pid],
    )
    args = _index_args(doc, tenant=True, _tenant_value="acme")
    args.refresh_payloads = True
    assert cli.cmd_index(args) == 0
    assert seen, "no payload patch was issued"
    assert all(kw.get("tenant") == "acme" for kw in seen), seen
