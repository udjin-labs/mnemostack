"""Tests for idempotent `mnemostack index` behaviour.

The important property: running `index` twice over the same content must
not create duplicate points. We cover that by exercising `_stable_chunk_id`
directly, since the full CLI path needs a live Qdrant.
"""
from __future__ import annotations

import argparse

import pytest

import mnemostack.cli as cli
from mnemostack.cli import _stable_chunk_id


def test_stable_chunk_id_is_deterministic():
    a = _stable_chunk_id("notes/a.md", 0, "hello world")
    b = _stable_chunk_id("notes/a.md", 0, "hello world")
    assert a == b


def test_stable_chunk_id_differs_on_content_change():
    a = _stable_chunk_id("notes/a.md", 0, "hello world")
    b = _stable_chunk_id("notes/a.md", 0, "hello world!")
    assert a != b


def test_stable_chunk_id_differs_on_offset_change():
    a = _stable_chunk_id("notes/a.md", 0, "hello world")
    b = _stable_chunk_id("notes/a.md", 800, "hello world")
    assert a != b


def test_stable_chunk_id_differs_on_source_change():
    a = _stable_chunk_id("notes/a.md", 0, "hello world")
    b = _stable_chunk_id("notes/b.md", 0, "hello world")
    assert a != b


def test_stable_chunk_id_is_valid_uuid_string():
    import uuid
    v = _stable_chunk_id("s", 0, "x")
    assert uuid.UUID(v)


def test_index_validates_path_before_recreate(monkeypatch, tmp_path):
    """A typo with --recreate must not drop/recreate a collection first."""

    def fail_get_provider(_name):
        pytest.fail("embedding provider should not be initialized for a missing path")

    class FailingVectorStore:
        def __init__(self, **_kwargs):
            pytest.fail("vector store should not be initialized for a missing path")

    monkeypatch.setattr(cli, "get_provider", fail_get_provider)
    monkeypatch.setattr(cli, "VectorStore", FailingVectorStore)

    args = argparse.Namespace(
        provider="ollama",
        collection="mnemostack",
        qdrant="http://localhost:6333",
        recreate=True,
        path=str(tmp_path / "missing"),
        chunk_size=800,
    )

    assert cli.cmd_index(args) == 2


def test_build_recaller_keeps_vector_handles_for_retry():
    provider = object()
    store = object()
    args = argparse.Namespace(
        query_expansion=False,
        bm25_path=[],
        memgraph_uri=None,
    )

    recaller = cli._build_recaller(args, provider, store)

    assert recaller.embedding is provider
    assert recaller.vector is store
