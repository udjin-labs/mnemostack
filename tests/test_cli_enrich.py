"""Tests for `index --enrich` and `index --refresh-payloads`."""

from __future__ import annotations

import argparse
import types

import pytest
from qdrant_client import QdrantClient

from mnemostack import cli
from mnemostack.ingest import stable_chunk_id
from mnemostack.vector import VectorStore

VEC = [1.0, 0.0, 0.0, 0.0]


@pytest.fixture
def store():
    s = VectorStore.__new__(VectorStore)  # bypass __init__
    s.collection = "test_collection"
    s.dimension = 4
    from qdrant_client.models import Distance

    s.distance = Distance.COSINE
    s.client = QdrantClient(":memory:")
    s.ensure_collection()
    return s


@pytest.fixture
def enricher_module(tmp_path, monkeypatch):
    """A real importable module so --enrich exercises the dotted-path load."""
    mod = tmp_path / "my_enrichers.py"
    mod.write_text(
        "def char_count(item):\n"
        "    return {'char_count': len(item.text)}\n"
        "\n"
        "not_callable = 42\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    return "my_enrichers"


class _CountingProvider:
    dimension = 4

    def __init__(self):
        self.embed_calls = 0

    def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        return VEC


def _args(doc_dir, **overrides) -> argparse.Namespace:
    defaults = dict(
        path=str(doc_dir),
        provider="fake",
        collection="test_collection",
        qdrant="http://localhost:6333",
        recreate=False,
        yes=False,
        prune=False,
        enrich=None,
        refresh_payloads=False,
        chunk_size=800,
        window_size=1,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _patch_stack(monkeypatch, store, provider=None):
    provider = provider or _CountingProvider()
    monkeypatch.setattr(cli, "get_provider", lambda _name, **_kw: provider)
    monkeypatch.setattr(cli, "VectorStore", lambda **_kw: store)
    monkeypatch.setattr(cli, "_embedding_model", lambda _args: None, raising=False)
    monkeypatch.setattr(cli, "model_kwargs", lambda _model: {})
    monkeypatch.setattr(cli.sys, "stdin", types.SimpleNamespace(isatty=lambda: False))
    return provider


def test_set_payload_merges_without_touching_vector(store):
    pid = stable_chunk_id("a.md", 0, "hello")
    store.upsert(pid, VEC, {"text": "hello", "source": "a.md", "offset": 0})

    store.set_payload(pid, {"char_count": 5, "source": "a.md"})

    hit = next(iter(store.scroll()))
    assert hit.payload["char_count"] == 5
    assert hit.payload["text"] == "hello"  # merge keeps existing keys


def test_cli_index_enrich_lands_in_payload(monkeypatch, tmp_path, store, enricher_module):
    doc = tmp_path / "note.md"
    doc.write_text("hello world", encoding="utf-8")
    _patch_stack(monkeypatch, store)

    rc = cli.cmd_index(_args(tmp_path, enrich=f"{enricher_module}:char_count"))

    assert rc == 0
    hit = next(iter(store.scroll()))
    assert hit.payload["char_count"] == len("hello world")


def test_cli_refresh_payloads_updates_without_reembedding(
    monkeypatch, tmp_path, store, enricher_module
):
    doc = tmp_path / "note.md"
    doc.write_text("hello world", encoding="utf-8")
    # Chunk is already indexed, without enrichment.
    pid = stable_chunk_id("note.md", 0, "hello world")
    store.upsert(pid, VEC, {"text": "hello world", "source": "note.md", "offset": 0})
    provider = _patch_stack(monkeypatch, store)

    rc = cli.cmd_index(
        _args(
            tmp_path,
            enrich=f"{enricher_module}:char_count",
            refresh_payloads=True,
        )
    )

    assert rc == 0
    assert provider.embed_calls == 0  # payload-only refresh, no re-embedding
    hit = next(iter(store.scroll()))
    assert hit.payload["char_count"] == len("hello world")
    assert hit.payload["index_root"] == str(tmp_path.resolve())  # new fields applied too


def test_cli_refresh_payloads_noop_without_flag(monkeypatch, tmp_path, store, enricher_module):
    doc = tmp_path / "note.md"
    doc.write_text("hello world", encoding="utf-8")
    pid = stable_chunk_id("note.md", 0, "hello world")
    store.upsert(pid, VEC, {"text": "hello world", "source": "note.md", "offset": 0})
    _patch_stack(monkeypatch, store)

    rc = cli.cmd_index(_args(tmp_path, enrich=f"{enricher_module}:char_count"))

    assert rc == 0
    hit = next(iter(store.scroll()))
    assert "char_count" not in hit.payload  # existing chunk untouched by default


@pytest.mark.parametrize(
    "spec",
    ["no-colon", "my_enrichers:missing_func", "my_enrichers:not_callable", "ghost.module:f"],
)
def test_cli_enrich_bad_specs_exit_2(monkeypatch, tmp_path, store, enricher_module, spec, capsys):
    doc = tmp_path / "note.md"
    doc.write_text("hello world", encoding="utf-8")
    _patch_stack(monkeypatch, store)

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_index(_args(tmp_path, enrich=spec))

    assert exc_info.value.code == 2
    assert "--enrich" in capsys.readouterr().err


def test_index_parser_accepts_new_flags():
    parser = cli.build_parser()
    args = parser.parse_args(
        ["index", "some/path", "--enrich", "pkg.mod:func", "--refresh-payloads"]
    )
    assert args.enrich == "pkg.mod:func"
    assert args.refresh_payloads is True
    args = parser.parse_args(["index", "some/path"])
    assert args.enrich is None
    assert args.refresh_payloads is False


def test_refresh_payloads_does_not_hijack_foreign_roots(
    monkeypatch, tmp_path, store, enricher_module, capsys
):
    """Identical (source, offset, text) under two roots shares one chunk id —
    refresh must not rewrite a point another root recorded as its own, or
    that root's prune isolation breaks."""
    doc = tmp_path / "note.md"
    doc.write_text("hello world", encoding="utf-8")
    pid = stable_chunk_id("note.md", 0, "hello world")
    store.upsert(
        pid,
        VEC,
        {"text": "hello world", "source": "note.md", "offset": 0, "index_root": "/elsewhere"},
    )
    _patch_stack(monkeypatch, store)

    rc = cli.cmd_index(
        _args(tmp_path, enrich=f"{enricher_module}:char_count", refresh_payloads=True)
    )

    assert rc == 0
    hit = next(iter(store.scroll()))
    assert hit.payload["index_root"] == "/elsewhere"  # ownership untouched
    assert "char_count" not in hit.payload
    assert "owned by another index root" in capsys.readouterr().err


def test_refresh_payloads_adopts_unattributed_legacy_points(
    monkeypatch, tmp_path, store, enricher_module
):
    """Points indexed before root tracking carry no index_root — adopting
    them is the documented migration path."""
    doc = tmp_path / "note.md"
    doc.write_text("hello world", encoding="utf-8")
    pid = stable_chunk_id("note.md", 0, "hello world")
    store.upsert(pid, VEC, {"text": "hello world", "source": "note.md", "offset": 0})
    _patch_stack(monkeypatch, store)

    rc = cli.cmd_index(
        _args(tmp_path, enrich=f"{enricher_module}:char_count", refresh_payloads=True)
    )

    assert rc == 0
    hit = next(iter(store.scroll()))
    assert hit.payload["index_root"] == str(tmp_path.resolve())
    assert hit.payload["char_count"] == len("hello world")


def test_refresh_removes_stale_enrichment_keys(monkeypatch, tmp_path, store, enricher_module):
    """An enrichment key the new enricher no longer produces must disappear
    on refresh — otherwise filters/context_fields keep seeing a stale fact."""
    doc = tmp_path / "note.md"
    doc.write_text("hello world", encoding="utf-8")
    pid = stable_chunk_id("note.md", 0, "hello world")
    store.upsert(
        pid,
        VEC,
        {
            "text": "hello world",
            "source": "note.md",
            "offset": 0,
            "index_root": str(tmp_path.resolve()),
            "amount": 100,  # written by a previous enricher...
            "_enrich_keys": ["amount"],  # ...which recorded ownership
            "user_meta": "kept",  # foreign field, NOT enrichment-owned
        },
    )
    _patch_stack(monkeypatch, store)

    # the new enricher returns char_count only — amount is stale
    rc = cli.cmd_index(
        _args(tmp_path, enrich=f"{enricher_module}:char_count", refresh_payloads=True)
    )

    assert rc == 0
    hit = next(iter(store.scroll()))
    assert "amount" not in hit.payload
    assert hit.payload["char_count"] == len("hello world")
    assert hit.payload["_enrich_keys"] == ["char_count"]
    assert hit.payload["user_meta"] == "kept"  # foreign fields untouched


def test_refresh_without_enricher_clears_all_owned_keys(monkeypatch, tmp_path, store):
    """Refreshing with no enricher at all removes everything the previous
    enrichment owned, plus the ownership record itself."""
    doc = tmp_path / "note.md"
    doc.write_text("hello world", encoding="utf-8")
    pid = stable_chunk_id("note.md", 0, "hello world")
    store.upsert(
        pid,
        VEC,
        {
            "text": "hello world",
            "source": "note.md",
            "offset": 0,
            "index_root": str(tmp_path.resolve()),
            "amount": 100,
            "_enrich_keys": ["amount"],
        },
    )
    _patch_stack(monkeypatch, store)

    rc = cli.cmd_index(_args(tmp_path, refresh_payloads=True))

    assert rc == 0
    hit = next(iter(store.scroll()))
    assert "amount" not in hit.payload
    assert "_enrich_keys" not in hit.payload


def test_delete_payload_keys(store):
    pid = stable_chunk_id("a.md", 0, "hello")
    store.upsert(pid, VEC, {"text": "hello", "source": "a.md", "amount": 7})

    store.delete_payload_keys(pid, ["amount"])
    store.delete_payload_keys(pid, [])  # no-op

    hit = next(iter(store.scroll()))
    assert "amount" not in hit.payload
    assert hit.payload["text"] == "hello"
