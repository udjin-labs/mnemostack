"""Tests for idempotent `mnemostack index` behaviour.

The important property: running `index` twice over the same content must
not create duplicate points. We cover that by exercising `_stable_chunk_id`
directly, since the full CLI path needs a live Qdrant.
"""
from __future__ import annotations

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
