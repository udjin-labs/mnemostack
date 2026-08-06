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


def test_kotlin_fun_definitions_are_boundaries():
    kt_src = (
        "package demo\n\n"
        "fun firstThing(x: Int): Int {\n"
        + "".join(f"    val a{i} = x + {i}\n" for i in range(20))
        + "    return x\n}\n\n"
        "fun secondThing(y: Int): Int {\n"
        + "".join(f"    val b{i} = {i}\n" for i in range(20))
        + "    return y\n}\n"
    )
    chunks = chunk_code(kt_src, "kotlin", max_chars=2000)
    _assert_partition(kt_src, chunks)
    symbols = [c.symbol for c in chunks]
    assert "firstThing" in symbols and "secondThing" in symbols


def test_annotations_and_templates_travel_with_their_definition():
    java_src = (
        "public class First {\n"
        + "".join(f"    int a{i} = {i};\n" for i in range(20))
        + "}\n\n"
        "@Deprecated\n"
        "public class Second {\n"
        + "".join(f"    int b{i} = {i};\n" for i in range(20))
        + "}\n"
    )
    chunks = chunk_code(java_src, "java", max_chars=2000)
    _assert_partition(java_src, chunks)
    second = next(c for c in chunks if c.symbol == "Second")
    assert second.text.startswith("@Deprecated")

    cpp_src = (
        "static int first(void) {\n"
        + "".join(f"    int a{i} = {i};\n" for i in range(20))
        + "    return 0;\n}\n\n"
        "template <typename T>\n"
        "T second(T value) {\n"
        + "".join(f"    T b{i} = value;\n" for i in range(20))
        + "    return value;\n}\n"
    )
    cpp_chunks = chunk_code(cpp_src, "cpp", max_chars=2000)
    _assert_partition(cpp_src, cpp_chunks)
    templated = next(c for c in cpp_chunks if c.symbol == "second")
    assert templated.text.startswith("template <typename T>")


def test_typed_typescript_const_definitions_are_boundaries():
    ts_src = (
        "import { Handler } from 'lib';\n\n"
        "export const firstHandler: Handler = async (req) => {\n"
        + "".join(f"  const a{i} = {i};\n" for i in range(20))
        + "  return req;\n};\n\n"
        "export const secondHandler: Handler<Thing> = (req) => {\n"
        + "".join(f"  const b{i} = {i};\n" for i in range(20))
        + "  return req;\n};\n"
    )
    chunks = chunk_code(ts_src, "typescript", max_chars=2000)
    _assert_partition(ts_src, chunks)
    symbols = [c.symbol for c in chunks]
    assert "firstHandler" in symbols and "secondHandler" in symbols


def test_ruby_modules_and_methods_are_boundaries():
    rb_src = (
        "module Outer\n"
        + "".join(f"  CONST_{i} = {i}\n" for i in range(20))
        + "end\n\n"
        "def helper_method?\n"
        + "".join(f"  x{i} = {i}\n" for i in range(20))
        + "end\n"
    )
    chunks = chunk_code(rb_src, "ruby", max_chars=2000)
    _assert_partition(rb_src, chunks)
    symbols = [c.symbol for c in chunks]
    assert "Outer" in symbols
    assert "helper_method?" in symbols


def test_scoped_enum_captures_the_declared_name():
    src = (
        "enum class Color {\n"
        + "".join(f"    VALUE_{i},\n" for i in range(20))
        + "};\n\n"
        "enum Plain {\n"
        + "".join(f"    P_{i},\n" for i in range(20))
        + "};\n"
    )
    chunks = chunk_code(src, "cpp", max_chars=2000)
    symbols = [c.symbol for c in chunks]
    assert "Color" in symbols
    assert "class" not in symbols
    assert "Plain" in symbols


def test_c_preprocessor_lines_are_not_carried_as_comments():
    src = (
        "#ifdef FEATURE\n"
        "static int guarded(void) {\n"
        + "".join(f"    int a{i} = {i};\n" for i in range(20))
        + "    return 0;\n}\n"
        "#endif\n"
        "static int after(void) {\n"
        + "".join(f"    int b{i} = {i};\n" for i in range(20))
        + "    return 1;\n}\n"
    )
    chunks = chunk_code(src, "c", max_chars=2000)
    _assert_partition(src, chunks)
    guarded = next(c for c in chunks if c.symbol == "guarded")
    after = next(c for c in chunks if c.symbol == "after")
    # The #endif terminator stays with its conditional block.
    assert "#endif" in guarded.text
    assert not after.text.startswith("#endif")


def test_one_line_c_function_definitions_are_boundaries():
    src = (
        "struct buf { int n; };\n"
        + "".join(f"static int pad{i}(void) {{ return {i}; }}\n" for i in range(10))
        + "\ninline int size(void) { return 4; }\n\n"
        "int declared_only(int x);\n"
        "static int big_one(void) {\n"
        + "".join(f"    int c{i} = {i};\n" for i in range(20))
        + "    return 0;\n}\n"
    )
    chunks = chunk_code(src, "c", max_chars=400)
    _assert_partition(src, chunks)
    symbols = [c.symbol for c in chunks]
    assert "big_one" in symbols
    assert "declared_only" not in symbols  # prototype: ends with ;
    # One-line definitions were recognized as boundaries (any of them naming
    # a chunk proves the `;`-in-body case no longer rejects the line).
    assert any(s and s.startswith("pad") for s in symbols) or "size" in symbols


def test_c_prototypes_with_trailing_comments_are_not_boundaries():
    src = (
        "int commented_proto(int x); // API declaration\n"
        "int block_proto(int y); /* legacy */\n\n"
        + "".join(
            f"static int real_def{i}(void) {{ return {i}; }} // one-liner\n"
            for i in range(6)
        )
        + "\nstatic int big_def(void) {\n"
        + "".join(f"    int a{i} = {i};\n" for i in range(20))
        + "    return 0;\n}\n"
    )
    chunks = chunk_code(src, "c", max_chars=2000)
    _assert_partition(src, chunks)
    symbols = [c.symbol for c in chunks]
    # Commented prototypes never name a chunk; real definitions do.
    assert "commented_proto" not in symbols
    assert "block_proto" not in symbols
    assert any(s and s.startswith("real_def") for s in symbols)


def test_oversized_merge_resplits_at_the_internal_definition_boundary():
    """A small helper merged with a large following function must split at
    the function's start, not at an arbitrary character offset."""
    helper = "def helper():\n" + "".join(f"    h{i} = {i}\n" for i in range(12))
    big = "def big_function():\n" + "".join(
        f"    value_{i} = {i} * 2\n" for i in range(90)
    )
    src = helper + big
    assert len(helper) < 200  # below the merge minimum — gets merged
    assert len(helper) + len(big) > 2000 > len(big)  # combined would char-split
    chunks = chunk_code(src, "python", max_chars=2000)
    _assert_partition(src, chunks)
    by_symbol = {c.symbol: c for c in chunks}
    assert by_symbol["helper"].text == helper
    assert by_symbol["big_function"].text == big  # clean cut at the def line


def test_rust_attributes_travel_with_their_definition():
    src = (
        "fn first() -> u32 {\n"
        + "".join(f"    let a{i} = {i};\n" for i in range(20))
        + "    0\n}\n\n"
        "#[derive(Debug, Clone)]\n"
        "#[cfg(feature = \"extra\")]\n"
        "struct Config {\n"
        + "".join(f"    field{i}: u32,\n" for i in range(20))
        + "}\n"
    )
    chunks = chunk_code(src, "rust", max_chars=2000)
    _assert_partition(src, chunks)
    config = next(c for c in chunks if c.symbol == "Config")
    assert config.text.startswith("#[derive")
    first = next(c for c in chunks if c.symbol == "first")
    assert "#[derive" not in first.text


def test_lua_local_functions_are_boundaries():
    src = (
        "local M = {}\n\n"
        "local function helper_one(x)\n"
        + "".join(f"    local a{i} = x + {i}\n" for i in range(20))
        + "    return x\nend\n\n"
        "local function helper_two(y)\n"
        + "".join(f"    local b{i} = {i}\n" for i in range(20))
        + "    return y\nend\n"
    )
    chunks = chunk_code(src, "lua", max_chars=2000)
    _assert_partition(src, chunks)
    symbols = [c.symbol for c in chunks]
    assert "helper_one" in symbols and "helper_two" in symbols


def test_cpp_qualified_member_definitions_are_boundaries():
    src = (
        "#include \"widget.h\"\n\n"
        "int Widget::render(const Frame &f) {\n"
        + "".join(f"    int a{i} = {i};\n" for i in range(20))
        + "    return 0;\n}\n\n"
        "Widget::~Widget() {\n"
        + "".join(f"    int b{i} = {i};\n" for i in range(20))
        + "}\n"
    )
    chunks = chunk_code(src, "cpp", max_chars=2000)
    _assert_partition(src, chunks)
    symbols = [c.symbol for c in chunks]
    # Qualified names record the member, not the qualification.
    assert "render" in symbols
    assert not any(s and "::" in s for s in symbols)


def test_decorator_stays_with_its_definition_when_resplitting():
    """A tiny helper + decorated large function: the re-split must cut at
    the DECORATOR line, never between the decorator and its def."""
    helper = "def helper():\n" + "".join(f"    h{i} = {i}\n" for i in range(12))
    big = (
        "@app.route('/endpoint')\n"
        "@cached\n"
        "def big_function():\n"
        + "".join(f"    value_{i} = {i} * 2\n" for i in range(85))
    )
    src = helper + big
    assert len(helper) < 200 and len(helper) + len(big) > 2000 > len(big)
    chunks = chunk_code(src, "python", max_chars=2000)
    _assert_partition(src, chunks)
    by_symbol = {c.symbol: c for c in chunks}
    assert by_symbol["helper"].text == helper
    assert by_symbol["big_function"].text == big  # decorators included
    assert not any(c.text.strip() == "@app.route('/endpoint')\n@cached".strip()
                   for c in chunks)


def test_sql_with_cte_keeps_its_final_select():
    filler = "".join(f"    , col_{i} AS (SELECT {i})\n" for i in range(20))
    src = (
        "CREATE TABLE t (id INT);\n\n"
        "WITH base AS (\n    SELECT 1 AS x\n)\n"
        + filler
        + "SELECT * FROM base;\n\n"
        "INSERT INTO t VALUES (1);\n"
    )
    chunks = chunk_code(src, "sql", max_chars=4000)
    _assert_partition(src, chunks)
    # The CTE's final SELECT never starts a chunk of its own — it stays in
    # the same chunk as the WITH clause it belongs to.
    assert not any(c.text.startswith("SELECT") for c in chunks)
    with_chunk = next(c for c in chunks if "WITH base AS" in c.text)
    assert "SELECT * FROM base;" in with_chunk.text


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
    assert "user" in tokens and "user_id" in tokens
    assert "xml" in tokens  # acronym boundary XMLHttp -> XML + Http
    # Sub-3-char parts are dropped BY DESIGN: the lexical retriever's default
    # min_token_len=3 strips them from every query gate, so emitting them
    # would create tokens that can never match.
    assert "id" not in tokens


def test_identifier_tokens_dedupe_preserve_order_and_bound():
    tokens = identifier_tokens("aaa aaa aaa bbb", limit=1)
    assert tokens == ["aaa"]
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


def test_refresh_without_code_flag_removes_stale_code_metadata(
    monkeypatch, tmp_path, store, capsys
):
    """Opting back out of --code must not leave points falsely code-marked:
    a small file keeps the same chunk id either way (same source/offset/
    text), so the refresh must DELETE the code-owned keys, not merge over
    them."""
    src = tmp_path / "app.py"
    src.write_text("def tiny():\n    return 1\n", encoding="utf-8")
    _patch_stack(monkeypatch, store)

    assert cli.cmd_index(_index_args(tmp_path, path=str(src), code=True)) == 0
    code_payloads = [h.payload or {} for h in store.scroll()]
    assert any(p.get("chunk_kind") == "code" for p in code_payloads)
    code_ids = {str(h.id) for h in store.scroll()}

    # Same single-file target, code mode off, payload refresh on: the chunk
    # id is identical (same source/offset/text), payload must go prose.
    assert cli.cmd_index(_index_args(tmp_path, path=str(src), refresh_payloads=True)) == 0
    prose_payloads = {str(h.id): dict(h.payload or {}) for h in store.scroll()}
    assert set(prose_payloads) == code_ids  # ids unchanged — refreshed in place
    for p in prose_payloads.values():
        for key in ("language", "chunk_kind", "code_tokens", "symbol", "_code_keys"):
            assert key not in p


def test_code_keys_record_is_reserved_and_validated(monkeypatch, tmp_path, store):
    """An enricher cannot plant _code_keys (reserved key), and a malformed
    stored record neither crashes the refresh nor deletes arbitrary fields
    — it is simply cleared."""
    src = tmp_path / "note.md"
    src.write_text("hello world", encoding="utf-8")
    mod = types.ModuleType("planting_enricher_mod")
    mod.enrich = lambda item: {"_code_keys": ["text"], "extra": "kept"}
    monkeypatch.setitem(__import__("sys").modules, "planting_enricher_mod", mod)
    _patch_stack(monkeypatch, store)

    rc = cli.cmd_index(_index_args(tmp_path, enrich="planting_enricher_mod:enrich"))
    assert rc == 0
    payload = next(iter(h.payload or {} for h in store.scroll()))
    assert "_code_keys" not in payload  # reserved from enrichers

    # Simulate a legacy/tampered malformed record already in the store: the
    # refresh must survive it, keep real fields, and clear the record.
    cid = next(str(h.id) for h in store.scroll())
    store.set_payload(cid, {"_code_keys": 1})
    rc = cli.cmd_index(
        _index_args(tmp_path, enrich="planting_enricher_mod:enrich", refresh_payloads=True)
    )
    assert rc == 0
    payload = next(iter(h.payload or {} for h in store.scroll()))
    assert payload.get("text") == "hello world"
    assert payload.get("extra") == "kept"
    assert "_code_keys" not in payload

    # A valid-LOOKING planted record naming a foreign field must not delete
    # it either: membership in the closed code-owned set is enforced.
    store.set_payload(cid, {"_code_keys": ["extra", "tags"], "tags": ["keep-me"]})
    rc = cli.cmd_index(
        _index_args(tmp_path, enrich="planting_enricher_mod:enrich", refresh_payloads=True)
    )
    assert rc == 0
    payload = next(iter(h.payload or {} for h in store.scroll()))
    assert payload.get("extra") == "kept"
    assert payload.get("tags") == ["keep-me"]  # foreign field survived
    assert "_code_keys" not in payload


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
