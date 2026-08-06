"""Syntax-aware code chunking: partition contract, boundaries, identifiers."""

from __future__ import annotations

import argparse
import types

import pytest
from qdrant_client import QdrantClient

import mnemostack.cli as cli
from mnemostack.code import (
    CODE_EXTENSIONS,
    chunk_code,
    identifier_tokens,
    language_for,
)
from mnemostack.vector import VectorStore

# ------------------------------------------------------------ partition


PY_SOURCE = (
    '"""Module doc."""\n'
    "import os\n\n\n"
    "def first_function(x):\n"
    + "".join(f"    a{i} = x + {i}\n" for i in range(20))
    + "    return x\n\n\n"
    "class SecondThing:\n"
    + "".join(f"    attr{i} = {i}\n" for i in range(20))
    + "\n\n"
    "def third_helper():\n"
    + "".join(f"    b{i} = {i}\n" for i in range(20))
    + "    return None\n"
)


def _assert_partition(text: str, chunks) -> None:
    """Offsets are exact, chunks are in order without overlap, and every
    gap between chunks (and around them) is whitespace-only — i.e. no real
    content is ever silently dropped."""
    pos = 0
    for c in chunks:
        assert c.offset >= pos
        assert text[c.offset : c.offset + len(c.text)] == c.text
        assert text[pos : c.offset].strip() == ""  # gap holds no content
        pos = c.offset + len(c.text)
    assert text[pos:].strip() == ""  # nothing real after the last chunk


def test_python_chunks_split_at_top_level_defs():
    chunks = chunk_code(PY_SOURCE, "python", max_chars=2000)
    _assert_partition(PY_SOURCE, chunks)
    symbols = [c.symbol for c in chunks]
    assert "second_thing" not in symbols  # names come from source, verbatim
    assert "first_function" in symbols
    assert "SecondThing" in symbols
    assert "third_helper" in symbols
    # Each definition starts its own chunk (segments are large enough here).
    for c in chunks:
        if c.symbol == "third_helper":
            assert c.text.startswith("def third_helper")


def test_offsets_survive_the_real_resolve_contract():
    """Chunk offsets are exact character offsets — text[offset:offset+len]
    round-trips, which is what `mnemostack resolve` verifies by position."""
    chunks = chunk_code(PY_SOURCE, "python", max_chars=2000)
    for c in chunks:
        assert PY_SOURCE[c.offset : c.offset + len(c.text)] == c.text


def test_small_definitions_merge_into_one_chunk():
    src = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
    chunks = chunk_code(src, "python", max_chars=2000)
    assert len(chunks) == 1  # both under MIN_SEGMENT_CHARS — merged
    assert chunks[0].symbol == "a"  # named after the opener


def test_oversized_segment_falls_back_to_char_splitting():
    body = "".join(f"    line{i} = {i}\n" for i in range(400))
    src = f"def huge():\n{body}"
    chunks = chunk_code(src, "python", max_chars=500)
    _assert_partition(src, chunks)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)
    # Only the first piece carries the symbol — continuations are unnamed.
    assert chunks[0].symbol == "huge"
    assert all(c.symbol is None for c in chunks[1:])


def test_brace_family_boundaries_go_and_js():
    go_src = (
        "package main\n\nimport \"fmt\"\n\n"
        "func FirstThing(x int) int {\n"
        + "".join(f"\ty{i} := x + {i}\n" for i in range(20))
        + "\treturn x\n}\n\n"
        "func (s *Server) SecondMethod() error {\n"
        + "".join(f"\tz{i} := {i}\n" for i in range(20))
        + "\treturn nil\n}\n"
    )
    chunks = chunk_code(go_src, "go", max_chars=2000)
    _assert_partition(go_src, chunks)
    assert "FirstThing" in [c.symbol for c in chunks]

    js_src = (
        "import x from 'y';\n\n"
        "export function handleRequest(req) {\n"
        + "".join(f"  const a{i} = {i};\n" for i in range(20))
        + "  return req;\n}\n\n"
        "export class ApiClient {\n"
        + "".join(f"  method{i}() {{ return {i}; }}\n" for i in range(20))
        + "}\n"
    )
    js_chunks = chunk_code(js_src, "javascript", max_chars=2000)
    _assert_partition(js_src, js_chunks)
    assert "handleRequest" in [c.symbol for c in js_chunks]
    assert "ApiClient" in [c.symbol for c in js_chunks]


def test_no_boundaries_degrades_to_plain_char_chunking():
    """A file the heuristics don't understand behaves exactly like the
    classic chunker — the documented worst case."""
    src = "x = 1\n" * 500  # no top-level defs at all
    chunks = chunk_code(src, "python", max_chars=300)
    _assert_partition(src, chunks)
    assert all(len(c.text) <= 300 for c in chunks)


def test_whitespace_only_pieces_are_dropped():
    src = "\n\n\n\n"
    assert chunk_code(src, "python", max_chars=100) == []


def test_partition_survives_crlf_and_missing_trailing_newline():
    crlf = PY_SOURCE.replace("\n", "\r\n")
    chunks = chunk_code(crlf, "python", max_chars=2000)
    _assert_partition(crlf, chunks)
    assert "first_function" in [c.symbol for c in chunks]

    no_trailing = PY_SOURCE.rstrip("\n")
    chunks2 = chunk_code(no_trailing, "python", max_chars=2000)
    _assert_partition(no_trailing, chunks2)


def test_c_type_prefixed_functions_are_boundaries():
    c_src = (
        "#include <stdio.h>\n\n"
        "static int parse_request(const char *buf, size_t len) {\n"
        + "".join(f"    int a{i} = {i};\n" for i in range(20))
        + "    return 0;\n}\n\n"
        "void handle_signal(int sig) {\n"
        + "".join(f"    int b{i} = {i};\n" for i in range(20))
        + "}\n\n"
        "int declared_only(int x);\n"
    )
    chunks = chunk_code(c_src, "c", max_chars=2000)
    _assert_partition(c_src, chunks)
    symbols = [c.symbol for c in chunks]
    assert "parse_request" in symbols
    assert "handle_signal" in symbols
    # A prototype (`;` on the line) is a declaration, not a boundary.
    assert "declared_only" not in symbols


def test_leading_doc_comment_travels_with_its_definition():
    src = (
        "def first():\n"
        + "".join(f"    a{i} = {i}\n" for i in range(20))
        + "    return 1\n\n"
        "# Documents the second function.\n"
        "# Spanning two comment lines.\n"
        "def second():\n"
        + "".join(f"    b{i} = {i}\n" for i in range(20))
        + "    return 2\n"
    )
    chunks = chunk_code(src, "python", max_chars=2000)
    _assert_partition(src, chunks)
    second = next(c for c in chunks if c.symbol == "second")
    assert second.text.startswith("# Documents the second function.")
    first = next(c for c in chunks if c.symbol == "first")
    assert "Documents" not in first.text


def test_leading_block_comment_travels_in_brace_family():
    src = (
        "export function first(x) {\n"
        + "".join(f"  const a{i} = {i};\n" for i in range(20))
        + "  return x;\n}\n\n"
        "/**\n"
        " * Documents the second function.\n"
        " */\n"
        "export function second(y) {\n"
        + "".join(f"  const b{i} = {i};\n" for i in range(20))
        + "  return y;\n}\n"
    )
    chunks = chunk_code(src, "javascript", max_chars=2000)
    _assert_partition(src, chunks)
    second = next(c for c in chunks if c.symbol == "second")
    assert second.text.startswith("/**")


def test_small_chunk_size_still_bounds_and_partitions():
    # --chunk-size below the merge minimum must still be honored: pieces
    # never exceed max_chars and the partition holds.
    chunks = chunk_code(PY_SOURCE, "python", max_chars=100)
    _assert_partition(PY_SOURCE, chunks)
    assert all(len(c.text) <= 100 for c in chunks)


# ------------------------------------------------------------ identifiers


def test_identifier_tokens_split_camel_and_snake():
    tokens = identifier_tokens("def parseHttpRequest(user_id): return XMLHttpRequest")
    assert "parse" in tokens and "http" in tokens and "request" in tokens
    assert "parsehttprequest" in tokens  # full identifier kept for exact-name gates
    assert "user" in tokens and "id" in tokens
    assert "xml" in tokens  # acronym boundary XMLHttp -> XML + Http


def test_identifier_tokens_dedupe_preserve_order_and_bound():
    tokens = identifier_tokens("aa aa aa bb", limit=1)
    assert tokens == ["aa"]
    many = identifier_tokens(" ".join(f"name{i}" for i in range(1000)))
    assert len(many) <= 256


def test_language_for_extensions():
    assert language_for("app.py") == "python"
    assert language_for("Component.TSX") == "typescript"
    assert language_for("notes.md") is None
    assert language_for("Makefile") is None
    assert all(ext.startswith(".") for ext in CODE_EXTENSIONS)


# ------------------------------------------------------------ CLI wiring


class _Provider:
    dimension = 4

    def embed(self, text):
        return [1.0, 0.0, 0.0, 0.0]

    def embed_batch(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
def store():
    s = VectorStore.__new__(VectorStore)
    s.collection = "test_collection"
    s.dimension = 4
    from qdrant_client.models import Distance

    s.distance = Distance.COSINE
    s.client = QdrantClient(":memory:")
    s.ensure_collection()
    return s


def _patch_stack(monkeypatch, store):
    monkeypatch.setattr(cli, "get_provider", lambda *_a, **_k: _Provider())
    monkeypatch.setattr(cli, "VectorStore", lambda **_kw: store)
    monkeypatch.setattr(cli, "_embedding_model", lambda _a: None, raising=False)
    monkeypatch.setattr(cli, "model_kwargs", lambda _m: {})
    monkeypatch.setattr(cli.sys, "stdin", types.SimpleNamespace(isatty=lambda: False))


def _index_args(tmp_path, **overrides) -> argparse.Namespace:
    defaults = dict(
        path=str(tmp_path), provider="fake", collection="test_collection",
        qdrant="http://localhost:6333", recreate=False, yes=False, prune=False,
        enrich=None, refresh_payloads=False, chunk_size=2000, window_size=1,
        code=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_cmd_index_without_code_flag_ignores_source_files(monkeypatch, tmp_path, store):
    (tmp_path / "app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("hello world", encoding="utf-8")
    _patch_stack(monkeypatch, store)

    assert cli.cmd_index(_index_args(tmp_path)) == 0

    sources = {(h.payload or {}).get("source") for h in store.scroll()}
    assert sources == {"notes.md"}  # .py untouched without --code


def test_cmd_index_code_flag_indexes_code_with_metadata(monkeypatch, tmp_path, store):
    (tmp_path / "app.py").write_text(
        "def handle_request(user_id):\n"
        + "".join(f"    v{i} = {i}\n" for i in range(30))
        + "    return user_id\n",
        encoding="utf-8",
    )
    (tmp_path / "notes.md").write_text("hello world", encoding="utf-8")
    _patch_stack(monkeypatch, store)

    assert cli.cmd_index(_index_args(tmp_path, code=True)) == 0

    by_source = {}
    for h in store.scroll():
        by_source.setdefault((h.payload or {})["source"], []).append(h.payload)
    assert set(by_source) == {"app.py", "notes.md"}
    code_payload = by_source["app.py"][0]
    assert code_payload["language"] == "python"
    assert code_payload["chunk_kind"] == "code"
    assert code_payload["symbol"] == "handle_request"
    assert "handle" in code_payload["code_tokens"].split()
    assert "user" in code_payload["code_tokens"].split()
    # Prose files keep the classic chunker: no code metadata.
    md_payload = by_source["notes.md"][0]
    assert "language" not in md_payload and "code_tokens" not in md_payload


def test_cmd_index_code_skips_vendored_trees(monkeypatch, tmp_path, store):
    (tmp_path / "app.py").write_text("def real():\n    return 1\n", encoding="utf-8")
    vendored = tmp_path / "node_modules" / "lib"
    vendored.mkdir(parents=True)
    (vendored / "dep.js").write_text("function dep() { return 1; }\n", encoding="utf-8")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "junk.py").write_text("x = 1\n", encoding="utf-8")
    _patch_stack(monkeypatch, store)

    assert cli.cmd_index(_index_args(tmp_path, code=True)) == 0

    sources = {(h.payload or {}).get("source") for h in store.scroll()}
    assert sources == {"app.py"}


def test_cmd_index_code_finds_uppercase_extensions(monkeypatch, tmp_path, store):
    """Per-extension globs are case-sensitive on POSIX — the extension-
    agnostic walk must still find MAIN.CPP / Component.TSX."""
    (tmp_path / "Component.TSX").write_text(
        "export function widget() { return 1; }\n", encoding="utf-8"
    )
    _patch_stack(monkeypatch, store)

    assert cli.cmd_index(_index_args(tmp_path, code=True)) == 0

    payloads = [h.payload or {} for h in store.scroll()]
    assert any(p.get("language") == "typescript" for p in payloads)


def test_enricher_cannot_fabricate_a_symbol_on_an_unnamed_chunk(
    monkeypatch, tmp_path, store
):
    (tmp_path / "data.py").write_text("x = 1\n" * 40, encoding="utf-8")  # no defs
    mod = types.ModuleType("fake_enricher_mod")

    def enrich(item):
        return {"symbol": "INJECTED", "language": "klingon"}

    mod.enrich = enrich
    monkeypatch.setitem(__import__("sys").modules, "fake_enricher_mod", mod)
    _patch_stack(monkeypatch, store)

    rc = cli.cmd_index(_index_args(tmp_path, code=True, enrich="fake_enricher_mod:enrich"))

    assert rc == 0
    payloads = [h.payload or {} for h in store.scroll()]
    assert payloads
    for p in payloads:
        assert p.get("language") == "python"  # authoritative, not "klingon"
        assert "symbol" not in p  # unnamed chunk stays unnamed


def test_cmd_index_code_warm_second_run_is_zero_write(monkeypatch, tmp_path, store, capsys):
    """Code chunk ids and payloads are deterministic — the diff-based
    refresh must see a warm re-index as fully unchanged."""
    (tmp_path / "app.py").write_text(
        "def stable_function(x):\n"
        + "".join(f"    y{i} = x + {i}\n" for i in range(30))
        + "    return x\n",
        encoding="utf-8",
    )
    _patch_stack(monkeypatch, store)
    args = _index_args(tmp_path, code=True, refresh_payloads=True)

    assert cli.cmd_index(args) == 0
    first = {str(h.id): dict(h.payload or {}) for h in store.scroll()}
    capsys.readouterr()

    assert cli.cmd_index(args) == 0
    out = capsys.readouterr().out
    second = {str(h.id): dict(h.payload or {}) for h in store.scroll()}
    assert second == first
    assert "0 patched" in out


def test_cmd_index_code_prune_removes_stale_chunks(monkeypatch, tmp_path, store):
    src = tmp_path / "app.py"
    src.write_text(
        "def old_one():\n"
        + "".join(f"    a{i} = {i}\n" for i in range(30))
        + "\n\ndef keeper():\n"
        + "".join(f"    b{i} = {i}\n" for i in range(30)),
        encoding="utf-8",
    )
    _patch_stack(monkeypatch, store)
    args = _index_args(tmp_path, code=True, prune=True)
    assert cli.cmd_index(args) == 0
    before = {str(pid) for pid in store.iter_ids()}
    assert len(before) >= 2

    # The first function disappears — its chunks must be pruned.
    src.write_text(
        "def keeper():\n" + "".join(f"    b{i} = {i}\n" for i in range(30)),
        encoding="utf-8",
    )
    assert cli.cmd_index(args) == 0
    after_payloads = [h.payload or {} for h in store.scroll()]
    assert all("old_one" != p.get("symbol") for p in after_payloads)
