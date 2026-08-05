"""Single-scan pruning: stale discovery from a snapshot, never per-source scans."""

from __future__ import annotations

import argparse
import types
from typing import Any

import pytest
from qdrant_client import QdrantClient

import mnemostack.cli as cli
from mnemostack.ingest import (
    prune_stale_chunks,
    prune_stale_chunks_from_snapshot,
    stable_chunk_id,
)
from mnemostack.vector import VectorStore

VEC = [1.0, 0.0, 0.0, 0.0]


def _make_store(name: str = "test_collection"):
    s = VectorStore.__new__(VectorStore)  # bypass __init__
    s.collection = name
    s.dimension = 4
    from qdrant_client.models import Distance

    s.distance = Distance.COSINE
    s.client = QdrantClient(":memory:")
    s.ensure_collection()
    return s


@pytest.fixture
def store():
    """VectorStore pointing at in-memory Qdrant."""
    return _make_store()


def _put(store, source: str, offset: int, text: str, index_root: str | None = None) -> str:
    pid = stable_chunk_id(source, offset, text)
    payload = {"text": text, "source": source, "offset": offset}
    if index_root is not None:
        payload["index_root"] = index_root
    store.upsert(pid, VEC, payload)
    return pid


def _snapshot(store, filters=None) -> list[tuple[str, dict]]:
    return [(str(hit.id), hit.payload or {}) for hit in store.scroll(filters=filters)]


def _remaining(store) -> set[str]:
    return {str(pid) for pid in store.iter_ids()}


def _prune_with_reads_forbidden(store, *args: Any, **kwargs: Any) -> int:
    """Run the snapshot prune while ANY store read is an instant failure."""

    def _no_reads(*_a, **_kw):  # pragma: no cover - failure path
        raise AssertionError("snapshot mode must not read from the store")

    store.scroll = _no_reads
    store.iter_ids = _no_reads
    try:
        return prune_stale_chunks_from_snapshot(store, *args, **kwargs)
    finally:
        del store.scroll, store.iter_ids  # uncover the bound methods


# ------------------------------------------------- snapshot-mode semantics


def test_snapshot_prune_matches_per_source_prune():
    """The snapshot path removes exactly what the per-source scans removed."""

    def _corpus(s) -> dict[str, set[str]]:
        fresh = _put(s, "a.md", 0, "current text")
        _put(s, "a.md", 100, "old tail that no longer exists")
        _put(s, "b.md", 0, "untouched source")
        return {"a.md": {fresh}}

    reference = _make_store()
    ref_map = _corpus(reference)
    assert prune_stale_chunks(reference, ref_map) == 1

    snapshotted = _make_store()
    snap_map = _corpus(snapshotted)
    snapshot = _snapshot(snapshotted)
    removed = _prune_with_reads_forbidden(snapshotted, snap_map, snapshot)

    assert removed == 1
    assert _remaining(snapshotted) == _remaining(reference)


def test_snapshot_mode_issues_zero_store_reads(store):
    fresh = _put(store, "a.md", 0, "keep")
    stale = _put(store, "a.md", 100, "drop")
    snapshot = _snapshot(store)

    removed = _prune_with_reads_forbidden(store, {"a.md": {fresh}}, snapshot)

    assert removed == 1
    remaining = _remaining(store)
    assert fresh in remaining and stale not in remaining


def test_snapshot_prune_respects_index_root_and_unattributed(store):
    """Foreign-root and rootless points are skipped even when the snapshot is
    broader than the prune scope (the cmd_index snapshot is unfiltered)."""
    ours_stale = _put(store, "note.md", 100, "our stale", index_root="/data/a")
    foreign = _put(store, "note.md", 0, "their content", index_root="/data/b")
    legacy = _put(store, "note.md", 200, "indexed before root tracking")
    snapshot = _snapshot(store)  # unfiltered: includes all three

    removed = _prune_with_reads_forbidden(
        store, {"note.md": set()}, snapshot, index_root="/data/a"
    )

    assert removed == 1
    remaining = _remaining(store)
    assert ours_stale not in remaining
    assert foreign in remaining and legacy in remaining


def test_snapshot_prune_only_touches_listed_sources(store):
    a = _put(store, "a.md", 0, "a-chunk")
    b = _put(store, "b.md", 0, "b-chunk")
    snapshot = _snapshot(store)

    removed = _prune_with_reads_forbidden(store, {"a.md": {a}}, snapshot)

    assert removed == 0
    assert _remaining(store) == {a, b}


def test_non_string_source_payloads_never_crash_or_match(store):
    keep = _put(store, "a.md", 0, "fresh")
    snapshot = _snapshot(store) + [
        ("weird-1", {"source": 5}),
        ("weird-2", {"source": ["a.md"]}),  # unhashable — must not crash `in`
        ("weird-3", {}),
    ]

    removed = _prune_with_reads_forbidden(store, {"a.md": {keep}}, snapshot)

    assert removed == 0
    assert keep in _remaining(store)


def test_tenant_ids_revalidated_at_delete(store):
    """Even a wrongly-scoped snapshot cannot delete a foreign tenant's point:
    the tenant-aware delete re-checks ownership server-side."""
    ours = stable_chunk_id("doc.md", 0, "ours")
    theirs = stable_chunk_id("doc.md", 100, "theirs")
    store.upsert(ours, VEC, {"source": "doc.md"}, tenant="acme")
    store.upsert(theirs, VEC, {"source": "doc.md"}, tenant="other")
    # A (buggy) snapshot claiming both points for acme's scope.
    snapshot = [(ours, {"source": "doc.md"}), (theirs, {"source": "doc.md"})]

    removed = _prune_with_reads_forbidden(
        store, {"doc.md": set()}, snapshot, tenant="acme"
    )

    assert removed == 1
    remaining = _remaining(store)
    assert theirs in remaining and ours not in remaining


def test_deletes_are_bounded_by_delete_batch_size(store, monkeypatch):
    ids = [_put(store, "a.md", i * 100, f"chunk {i}") for i in range(5)]
    snapshot = _snapshot(store)
    calls: list[int] = []
    real = store.delete_points

    def counting(pids, **kw):
        calls.append(len(pids))
        return real(pids, **kw)

    monkeypatch.setattr(store, "delete_points", counting)

    removed = prune_stale_chunks_from_snapshot(
        store, {"a.md": set()}, snapshot, delete_batch_size=2
    )

    assert removed == 5
    assert calls == [2, 2, 1]
    assert not any(i in _remaining(store) for i in ids)


def test_quota_estimate_skips_non_string_sources_like_the_prune():
    """markdown_prune_count mirrors the real prune — including the guard for
    a non-string/unhashable stored source, which must not crash the quota
    hook while the prune itself would have skipped the point."""
    from mnemostack.markdown.sync import markdown_prune_count

    existing = {
        "stale-1": {"source": "gone.md"},
        "weird-1": {"source": 5},
        "weird-2": {"source": ["gone.md"]},  # unhashable
        "weird-3": {},
    }
    count = markdown_prune_count(
        existing,
        [],
        set(),
        prune=True,
        full_root=True,
        failed_sources=set(),
        md_owned_only=False,
    )
    assert count == 1  # only the real string-sourced stale point


# ------------------------------------------------- fallback (no snapshot)


def test_fallback_keeps_working_for_stores_without_scroll():
    """A custom store exposing only iter_ids/delete_points (the old
    prune_stale_chunks contract) still prunes via the per-source path."""

    class _LegacyStore:
        def __init__(self):
            self.points = {
                "stale": {"source": "a.md"},
                "fresh": {"source": "a.md"},
            }
            self.deleted: list = []

        def iter_ids(self, filters=None, **_kw):
            for pid, pl in self.points.items():
                if not filters or all(pl.get(k) == v for k, v in filters.items()):
                    yield pid

        def delete_points(self, ids, **_kw):
            self.deleted.extend(ids)
            return len(ids)

    store = _LegacyStore()
    removed = prune_stale_chunks_from_snapshot(store, {"a.md": {"fresh"}})

    assert removed == 1
    assert store.deleted == ["stale"]


def test_fallback_uses_one_scroll_and_never_per_source_scans(store, monkeypatch):
    root = "/data/a"
    fresh_map: dict[str, set[str]] = {}
    stale_ids = []
    for i in range(10):
        src = f"doc{i}.md"
        keep = _put(store, src, 0, f"keep {i}", index_root=root)
        stale_ids.append(_put(store, src, 100, f"stale {i}", index_root=root))
        fresh_map[src] = {keep}
    scroll_calls: list[dict | None] = []
    real_scroll = store.scroll

    def counting_scroll(*a, filters=None, **kw):
        scroll_calls.append(filters)
        return real_scroll(*a, filters=filters, **kw)

    monkeypatch.setattr(store, "scroll", counting_scroll)
    monkeypatch.setattr(
        store,
        "iter_ids",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("per-source iter_ids must never run")
        ),
    )

    removed = prune_stale_chunks_from_snapshot(store, fresh_map, index_root=root)

    assert removed == 10
    assert scroll_calls == [{"index_root": root}]  # ONE root-scoped scan
    monkeypatch.undo()
    assert not any(i in _remaining(store) for i in stale_ids)


def test_fallback_without_root_scans_once_unfiltered(store, monkeypatch):
    fresh = _put(store, "a.md", 0, "keep")
    stale = _put(store, "a.md", 100, "drop")
    scroll_calls: list[dict | None] = []
    real_scroll = store.scroll

    def counting_scroll(*a, filters=None, **kw):
        scroll_calls.append(filters)
        return real_scroll(*a, filters=filters, **kw)

    monkeypatch.setattr(store, "scroll", counting_scroll)

    removed = prune_stale_chunks_from_snapshot(store, {"a.md": {fresh}})

    assert removed == 1
    assert scroll_calls == [None]
    assert stale not in _remaining(store)


def test_empty_fresh_map_short_circuits_without_any_scan(store):
    keep = _put(store, "a.md", 0, "keep")

    assert _prune_with_reads_forbidden(store, {}) == 0
    assert _remaining(store) == {keep}


# ------------------------------------------------- CLI wiring


class _Provider:
    dimension = 4

    def embed(self, text):
        return VEC

    def embed_batch(self, texts):
        return [VEC for _ in texts]


def _instrument(monkeypatch, store):
    """Record every read the CLI issues as (method, filters) pairs."""
    reads: list[tuple[str, dict | None]] = []
    real_scroll, real_iter = store.scroll, store.iter_ids

    def scroll(*a, filters=None, **kw):
        reads.append(("scroll", filters))
        return real_scroll(*a, filters=filters, **kw)

    def iter_ids(*a, filters=None, **kw):
        reads.append(("iter_ids", filters))
        return real_iter(*a, filters=filters, **kw)

    monkeypatch.setattr(store, "scroll", scroll)
    monkeypatch.setattr(store, "iter_ids", iter_ids)
    return reads


def _patch_stack(monkeypatch, store):
    monkeypatch.setattr(cli, "get_provider", lambda *_a, **_k: _Provider())
    monkeypatch.setattr(cli, "VectorStore", lambda **_kw: store)
    monkeypatch.setattr(cli, "_embedding_model", lambda _a: None, raising=False)
    monkeypatch.setattr(cli, "model_kwargs", lambda _m: {})
    monkeypatch.setattr(cli.sys, "stdin", types.SimpleNamespace(isatty=lambda: False))


def _index_args(tmp_path, **overrides) -> argparse.Namespace:
    defaults = dict(
        path=str(tmp_path), provider="fake", collection="test_collection",
        qdrant="http://localhost:6333", recreate=False, yes=False, prune=True,
        enrich=None, refresh_payloads=False, chunk_size=800, window_size=1,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_cmd_index_prune_never_scans_per_source(monkeypatch, tmp_path, store, capsys):
    (tmp_path / "note.md").write_text("hello world", encoding="utf-8")
    (tmp_path / "other.md").write_text("second doc", encoding="utf-8")
    root = str(tmp_path.resolve())
    stale = _put(store, "note.md", 800, "removed second page", index_root=root)
    reads = _instrument(monkeypatch, store)
    _patch_stack(monkeypatch, store)

    rc = cli.cmd_index(_index_args(tmp_path))

    assert rc == 0
    assert stale not in _remaining(store)
    assert "pruned 1 stale" in capsys.readouterr().out
    # No read anywhere carries a per-source filter, and prune discovery is
    # exactly one root-scoped scroll (the id preload stays unfiltered).
    assert all(f is None or "source" not in f for _m, f in reads)
    assert [f for m, f in reads if m == "scroll" and f] == [{"index_root": root}]


def test_cmd_index_refresh_prune_reuses_snapshot(monkeypatch, tmp_path, store, capsys):
    (tmp_path / "note.md").write_text("hello world", encoding="utf-8")
    root = str(tmp_path.resolve())
    stale = _put(store, "note.md", 800, "removed second page", index_root=root)
    reads = _instrument(monkeypatch, store)
    _patch_stack(monkeypatch, store)

    rc = cli.cmd_index(_index_args(tmp_path, refresh_payloads=True))

    assert rc == 0
    assert stale not in _remaining(store)
    assert "pruned 1 stale" in capsys.readouterr().out
    # The --refresh-payloads snapshot doubles as the prune snapshot: no
    # filtered read of any kind happens after it.
    assert all(f is None for _m, f in reads)


def test_cmd_index_markdown_full_root_walk_scans_root_exactly_once(
    monkeypatch, tmp_path, store
):
    (tmp_path / "a.md").write_text("alpha body", encoding="utf-8")
    root = str(tmp_path.resolve())
    # A point for a file deleted from disk — the full-root reconcile must
    # still find and prune it, now from the snapshot instead of a 2nd scroll.
    gone = stable_chunk_id("gone.md", 0, "vanished")
    store.upsert(
        gone, VEC, {"source": "gone.md", "index_root": root, "_md_keys": ["title"]}
    )
    reads = _instrument(monkeypatch, store)
    _patch_stack(monkeypatch, store)
    args = argparse.Namespace(
        path=str(tmp_path), provider="fake", embedding_model=None,
        collection="test_collection", qdrant="http://localhost:6333",
        chunk_size=1200, memgraph_uri=None, graph_timeout=5.0,
        recreate=False, prune=True, yes=True,
    )

    rc = cli.cmd_index_markdown(args)

    assert rc == 0
    assert gone not in _remaining(store)
    # ONE root-scoped scroll total (the snapshot load) — the reconcile pass
    # and the prune both reuse it; nothing reads per source.
    root_scoped = [f for _m, f in reads if f and "index_root" in f]
    assert root_scoped == [{"index_root": root}]
    assert all(f is None or "source" not in f for _m, f in reads)
