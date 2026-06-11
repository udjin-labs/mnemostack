"""Tests for stale-chunk pruning — VectorStore.delete_points, prune_stale_chunks,
and the `index --prune` CLI wiring."""

from __future__ import annotations

import argparse
import types

import pytest
from qdrant_client import QdrantClient

import mnemostack.cli as cli
from mnemostack.ingest import prune_stale_chunks, stable_chunk_id
from mnemostack.vector import VectorStore

VEC = [1.0, 0.0, 0.0, 0.0]


@pytest.fixture
def store():
    """VectorStore pointing at in-memory Qdrant."""
    s = VectorStore.__new__(VectorStore)  # bypass __init__
    s.collection = "test_collection"
    s.dimension = 4
    from qdrant_client.models import Distance

    s.distance = Distance.COSINE
    s.client = QdrantClient(":memory:")
    s.ensure_collection()
    return s


def _put(store, source: str, offset: int, text: str) -> str:
    pid = stable_chunk_id(source, offset, text)
    store.upsert(pid, VEC, {"text": text, "source": source, "offset": offset})
    return pid


# ---------- VectorStore.delete_points ----------


def test_delete_points_removes_only_listed(store):
    keep = _put(store, "a.md", 0, "keep")
    drop1 = _put(store, "a.md", 100, "drop1")
    drop2 = _put(store, "b.md", 0, "drop2")

    deleted = store.delete_points([drop1, drop2])

    assert deleted == 2
    remaining = {str(pid) for pid in store.iter_ids()}
    assert remaining == {keep}


def test_delete_points_empty_list_is_noop(store):
    _put(store, "a.md", 0, "keep")
    assert store.delete_points([]) == 0
    assert store.count() == 1


# ---------- prune_stale_chunks ----------


def test_prune_removes_stale_ids_of_reindexed_source(store):
    fresh = _put(store, "a.md", 0, "current text")
    stale = _put(store, "a.md", 100, "old tail that no longer exists")
    other = _put(store, "b.md", 0, "untouched source")

    removed = prune_stale_chunks(store, {"a.md": {fresh}})

    assert removed == 1
    remaining = {str(pid) for pid in store.iter_ids()}
    assert remaining == {fresh, other}
    assert stale not in remaining


def test_prune_noop_when_fresh_covers_everything(store):
    a = _put(store, "a.md", 0, "one")
    b = _put(store, "a.md", 100, "two")

    removed = prune_stale_chunks(store, {"a.md": {a, b}})

    assert removed == 0
    assert store.count() == 2


def test_prune_only_touches_listed_sources(store):
    _put(store, "a.md", 0, "a-chunk")
    b = _put(store, "b.md", 0, "b-chunk")

    # b.md is not in the fresh map — its points must survive even though
    # they would be "stale" relative to a.md's fresh set.
    removed = prune_stale_chunks(store, {"a.md": {stable_chunk_id("a.md", 0, "a-chunk")}})

    assert removed == 0
    assert str(b) in {str(pid) for pid in store.iter_ids()}


# ---------- CLI wiring ----------


class _FakeProvider:
    dimension = 4

    def embed(self, text: str) -> list[float]:
        return VEC


def _index_args(tmp_path, **overrides) -> argparse.Namespace:
    defaults = dict(
        path=str(tmp_path),
        provider="fake",
        collection="test_collection",
        qdrant="http://localhost:6333",
        recreate=False,
        yes=False,
        prune=True,
        chunk_size=800,
        window_size=1,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _patch_stack(monkeypatch, store) -> None:
    monkeypatch.setattr(cli, "get_provider", lambda _name, **_kw: _FakeProvider())
    monkeypatch.setattr(cli, "VectorStore", lambda **_kw: store)
    monkeypatch.setattr(cli, "_embedding_model", lambda _args: None, raising=False)
    monkeypatch.setattr(cli, "model_kwargs", lambda _model: {})
    monkeypatch.setattr(cli.sys, "stdin", types.SimpleNamespace(isatty=lambda: False))


def test_cli_index_prune_deletes_stale_chunks(monkeypatch, tmp_path, store, capsys):
    doc = tmp_path / "note.md"
    doc.write_text("hello world", encoding="utf-8")
    # A chunk this file used to produce (different offset/text) — now stale.
    stale = _put(store, "note.md", 800, "removed second page")
    # A chunk from another source — must survive.
    other = _put(store, "other.md", 0, "foreign chunk")
    _patch_stack(monkeypatch, store)

    rc = cli.cmd_index(_index_args(tmp_path))

    assert rc == 0
    remaining = {str(pid) for pid in store.iter_ids()}
    assert stale not in remaining
    assert other in remaining
    assert stable_chunk_id("note.md", 0, "hello world") in remaining
    assert "pruned 1 stale" in capsys.readouterr().out


def test_cli_index_prune_spares_sources_with_failed_embeddings(monkeypatch, tmp_path, store, capsys):
    """If the fresh chunk failed to embed, the old chunk is the only copy of
    that source's data — pruning must leave it alone."""
    doc = tmp_path / "note.md"
    doc.write_text("hello world", encoding="utf-8")
    old = _put(store, "note.md", 800, "previous revision chunk")

    class _FailingProvider:
        dimension = 4

        def embed(self, text: str) -> list[float]:
            return []

    monkeypatch.setattr(cli, "get_provider", lambda _name, **_kw: _FailingProvider())
    monkeypatch.setattr(cli, "VectorStore", lambda **_kw: store)
    monkeypatch.setattr(cli, "_embedding_model", lambda _args: None, raising=False)
    monkeypatch.setattr(cli, "model_kwargs", lambda _model: {})
    monkeypatch.setattr(cli.sys, "stdin", types.SimpleNamespace(isatty=lambda: False))

    rc = cli.cmd_index(_index_args(tmp_path))

    assert rc == 0
    assert old in {str(pid) for pid in store.iter_ids()}
    captured = capsys.readouterr()
    assert "pruned 0 stale" in captured.out
    assert "prune skipped" in captured.err


def test_cli_index_without_prune_keeps_stale_chunks(monkeypatch, tmp_path, store):
    doc = tmp_path / "note.md"
    doc.write_text("hello world", encoding="utf-8")
    stale = _put(store, "note.md", 800, "removed second page")
    _patch_stack(monkeypatch, store)

    rc = cli.cmd_index(_index_args(tmp_path, prune=False))

    assert rc == 0
    assert stale in {str(pid) for pid in store.iter_ids()}


def test_index_parser_accepts_prune_flag():
    parser = cli.build_parser()
    args = parser.parse_args(["index", "some/path", "--prune"])
    assert args.prune is True
    args = parser.parse_args(["index", "some/path"])
    assert args.prune is False
