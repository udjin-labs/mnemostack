"""Tests for the generic markdown indexer: parse + collect + graph sync + CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from mnemostack.markdown import collect_markdown, extract_links, parse_frontmatter

# ---------- frontmatter ----------


def test_frontmatter_parsed_and_body_stripped():
    meta, body = parse_frontmatter("---\ntitle: Foo\ntags: [a, b]\n---\n# Head\ntext")
    assert meta == {"title": "Foo", "tags": ["a", "b"]}
    assert body == "# Head\ntext"


def test_no_frontmatter_returns_full_text():
    meta, body = parse_frontmatter("# Just a heading\nbody")
    assert meta == {}
    assert body == "# Just a heading\nbody"


def test_malformed_frontmatter_is_fail_open():
    meta, body = parse_frontmatter("---\ntitle: : bad: yaml:\n- x\n---\nbody")
    assert meta == {}
    assert "body" in body


def test_non_mapping_frontmatter_ignored():
    meta, body = parse_frontmatter("---\n- just\n- a\n- list\n---\nbody")
    assert meta == {}


def test_frontmatter_closing_fence_at_eof():
    # A frontmatter-only note ending right after the closing fence (no trailing
    # newline) must still parse its metadata, not embed the raw YAML fences.
    meta, body = parse_frontmatter("---\ntitle: Only Meta\ntags: [x]\n---")
    assert meta == {"title": "Only Meta", "tags": ["x"]}
    assert body == ""


def test_frontmatter_with_utf8_bom():
    # A file saved with a UTF-8 BOM starts with U+FEFF; frontmatter must still
    # be recognized rather than embedded as body.
    meta, body = parse_frontmatter("\ufeff---\ntitle: Bom\n---\n# Note\nbody")
    assert meta == {"title": "Bom"}
    assert body == "# Note\nbody"


def test_extract_links_skips_unterminated_code_fence():
    # An opening fence with no closing fence runs to EOF (CommonMark); links
    # after it are still code samples, not references.
    text = "real [[Live]]\n\n```\n[[Fenced]] and [x](f.md)\nno closing fence"
    targets = {link.target for link in extract_links(text)}
    assert "Live" in targets
    assert "Fenced" not in targets and "f" not in targets


def test_empty_frontmatter_block_strips_fences():
    # An empty frontmatter block (YAML loads to None) must still strip the
    # recognized fence, not embed the --- lines as body text.
    meta, body = parse_frontmatter("---\n\n---\n# Note\nbody")
    assert meta == {}
    assert body == "# Note\nbody"
    assert "---" not in body


def test_empty_frontmatter_immediately_closed():
    # No blank content line at all (---\n---): still a recognized (empty) block.
    meta, body = parse_frontmatter("---\n---\n# Note\nbody")
    assert meta == {}
    assert body == "# Note\nbody"


def test_extract_links_closing_fence_needs_bare_line():
    # A line with the fence plus trailing text is code, not a closer, so the
    # block stays open and its sample links are excluded.
    text = "real [[Live]]\n\n```\n``` not a close [[Fenced]]\n[x](f.md)\n```\n"
    targets = {link.target for link in extract_links(text)}
    assert "Live" in targets
    assert "Fenced" not in targets and "f" not in targets


def test_extract_links_skips_indented_code_blocks():
    # A 4-space indented code block after a blank line is code; its sample links
    # must not become edges.
    text = "real [[Live]]\n\n    [[Indented]] and [x](ind.md)\n\nplain [r](real.md)"
    targets = {link.target for link in extract_links(text)}
    assert "Live" in targets and "real" in targets
    assert "Indented" not in targets and "ind" not in targets


def test_extract_links_accepts_single_quote_and_paren_titles():
    # CommonMark allows '...' and (...) link titles, not just "...".
    links = extract_links("[a](x.md 'see also') [b](y.md (note)) [c](z.md \"t\")")
    targets = {link.target for link in links}
    assert targets == {"x", "y", "z"}


# ---------- links ----------


def test_extract_wikilinks_with_alias_and_anchor():
    links = extract_links("see [[Target]], [[Other|alias]], [[Third#section]]")
    assert [(link.target, link.is_wikilink) for link in links] == [
        ("Target", True),
        ("Other", True),
        ("Third", True),
    ]


def test_extract_markdown_links_notes_only():
    # .md and extensionless targets are notes; external, image (.png), embeds,
    # and pure anchors are not edges.
    text = "[a](notes/foo.md) [b](bare) [ext](https://x.com) [img](y.png) ![pic](z.png) [t](#top)"
    links = extract_links(text)
    assert [(link.target, link.is_wikilink) for link in links] == [
        ("notes/foo", False),
        ("bare", False),
    ]


def test_links_deduped_per_style():
    # Same style + target dedupes ([[A]] [[A]] -> one), but a wikilink and an
    # inline link to the same name both survive: they resolve differently
    # (corpus-wide vs. relative to the source file).
    links = extract_links("[[A]] [[A]] [b](A.md)")
    assert [(link.target, link.is_wikilink) for link in links] == [("A", True), ("A", False)]


def test_extract_links_skips_embeds_and_non_note_wikilinks():
    # Obsidian embeds (![[...]]) and non-note wikilink targets (.png/.pdf) are
    # not notes, so they must not become File -[LINKS_TO]-> File edges.
    text = "note [[Real Note]], embed ![[diagram.png]], asset [[paper.pdf]]"
    links = extract_links(text)
    assert [(link.target, link.is_wikilink) for link in links] == [("Real Note", True)]


def test_extract_links_skips_fenced_code_blocks():
    # Link syntax shown inside a fenced code block is a sample, not a real
    # reference — it must not become a LINKS_TO edge.
    text = (
        "real [[Live Note]] and [r](real.md)\n\n"
        "```\nsample [[Fenced Note]] and [x](fenced.md)\n```\n"
    )
    links = extract_links(text)
    targets = {link.target for link in links}
    assert "Live Note" in targets and "real" in targets
    assert "Fenced Note" not in targets and "fenced" not in targets


def test_extract_links_mixed_fence_delimiters_stay_excluded():
    # A ~~~ block containing an inner ``` line must stay one block: links after
    # the inner mismatched fence are still code samples, not references.
    text = (
        "real [[Live]]\n\n"
        "~~~\ncode ``` inner\n[[Fenced]] and [x](fenced.md)\n~~~\n"
    )
    targets = {link.target for link in extract_links(text)}
    assert "Live" in targets
    assert "Fenced" not in targets and "fenced" not in targets


def test_extract_links_skips_indented_code_fences():
    # A fence indented up to 3 spaces is still a valid code block.
    text = "real [[Live]]\n\n   ```\n   [[Indented]] and [x](ind.md)\n   ```\n"
    targets = {link.target for link in extract_links(text)}
    assert "Live" in targets
    assert "Indented" not in targets and "ind" not in targets


def test_extract_links_skips_inline_code_spans():
    # Link syntax shown in an inline code span is documentation, not a link.
    text = "Use `[[Example]]` or `[x](sample.md)` in a note; but [[Real]] links."
    targets = {link.target for link in extract_links(text)}
    assert "Real" in targets
    assert "Example" not in targets and "sample" not in targets


def test_extract_links_keeps_dotted_note_names():
    # A note basename that merely contains dots (a daily note, a version note)
    # is a real note; only known asset extensions are rejected.
    links = extract_links("daily [[2026.07.04]], asset [[paper.pdf]], ver [v](v1.2.0.md)")
    targets = [(link.target, link.is_wikilink) for link in links]
    assert ("2026.07.04", True) in targets     # dotted note kept
    assert ("v1.2.0", False) in targets        # dotted .md note kept
    assert all(t != "paper.pdf" for t, _ in targets)  # .pdf asset dropped


def test_extract_links_spaced_and_escaped_inline_targets():
    # Angle-bracketed (<My Note.md>) and %20-escaped inline destinations are
    # valid intra-corpus note references; both normalize to the same key.
    links = extract_links("[a](<My Note.md>) and [b](My%20Note.md) and [c](Plain.md)")
    targets = [(link.target, link.is_wikilink) for link in links]
    assert ("My Note", False) in targets
    assert ("Plain", False) in targets
    # the two spellings of the same note dedupe to one edge
    assert sum(1 for t, _ in targets if t == "My Note") == 1


# ---------- collect_markdown ----------


def _vault(tmp: Path) -> Path:
    (tmp / "a.md").write_text("---\nauthor: X\ntopic: t1\n---\n# A\nlinks [[B]] and [c](sub/c.md)\n")
    (tmp / "sub").mkdir()
    (tmp / "sub" / "c.md").write_text("# C\ncontent here")
    (tmp / "b.md").write_text("# B\nrefers [[Missing Note]]")
    return tmp


def test_collect_frontmatter_into_payload(tmp_path):
    col = collect_markdown(_vault(tmp_path), index_root="/root")
    a_chunks = [c for c in col.chunks if c.payload["source"] == "a.md"]
    assert a_chunks
    p = a_chunks[0].payload
    assert p["author"] == "X" and p["topic"] == "t1"
    assert p["source"] == "a.md" and p["index_root"] == "/root"
    assert "heading_path" in p  # markdown chunker carries the heading


def test_collect_protected_keys_win_over_frontmatter(tmp_path):
    (tmp_path / "x.md").write_text("---\nsource: HACKED\noffset: 999\n---\n# X\nbody")
    col = collect_markdown(tmp_path)
    p = col.chunks[0].payload
    assert p["source"] == "x.md"   # not "HACKED"
    assert p["offset"] == 0        # not 999


def test_collect_link_resolution_case_insensitive_and_dangling(tmp_path):
    col = collect_markdown(_vault(tmp_path))
    by_source = {(e.source, e.target): e.resolved for e in col.edges}
    # [[B]] resolves to b.md despite case difference in the note name
    assert by_source[("a.md", "b.md")] is True
    # [c](sub/c.md) resolves by relative path
    assert by_source[("a.md", "sub/c.md")] is True
    # [[Missing Note]] has no file -> dangling edge kept
    assert by_source[("b.md", "Missing Note")] is False


def test_collect_stable_ids_are_idempotent(tmp_path):
    v = _vault(tmp_path)
    ids1 = sorted(c.id for c in collect_markdown(v).chunks)
    ids2 = sorted(c.id for c in collect_markdown(v).chunks)
    assert ids1 == ids2


def test_collect_relative_link_resolves_up_a_directory(tmp_path):
    # An inline link ../index.md from sub/page.md must resolve to index.md,
    # not stay dangling — relative resolution walks up from the source dir.
    (tmp_path / "index.md").write_text("# Index\ntop-level note")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "page.md").write_text("# Page\nback to [home](../index.md)")
    col = collect_markdown(tmp_path)
    edge = next(e for e in col.edges if e.source == "sub/page.md")
    assert edge.target == "index.md"
    assert edge.resolved is True


def test_collect_relative_link_prefers_own_directory(tmp_path):
    # Two notes share the basename note.md in different folders. A same-dir
    # inline link must resolve to the sibling, not the other folder's note.
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    (tmp_path / "one" / "note.md").write_text("# One note")
    (tmp_path / "one" / "src.md").write_text("see [n](note.md)")
    (tmp_path / "two" / "note.md").write_text("# Two note")
    col = collect_markdown(tmp_path)
    edge = next(e for e in col.edges if e.source == "one/src.md")
    assert edge.target == "one/note.md"


def test_collect_relative_link_case_insensitive_sibling(tmp_path):
    # [x](note.md) from b/src.md must resolve to the same-dir b/Note.md
    # (case-insensitive) rather than falling through to a/note.md elsewhere.
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "note.md").write_text("# A note")
    (tmp_path / "b" / "Note.md").write_text("# B note")
    (tmp_path / "b" / "src.md").write_text("see [x](note.md)")
    col = collect_markdown(tmp_path)
    edge = next(e for e in col.edges if e.source == "b/src.md")
    assert edge.target == "b/Note.md"
    assert edge.resolved is True


def test_collect_single_file_resolves_sibling_links(tmp_path):
    # index-markdown on ONE file must still resolve links to sibling notes in
    # the parent dir (matching the canonical names a directory index makes),
    # not leave them dangling.
    (tmp_path / "a.md").write_text("# A\nlinks [[B]] and [c](b.md)")
    (tmp_path / "b.md").write_text("# B\nbody")
    col = collect_markdown(tmp_path / "a.md")
    # only a.md is chunked
    assert {c.payload["source"] for c in col.chunks} == {"a.md"}
    # but its links resolve to the real sibling b.md
    resolved = {(e.target, e.resolved) for e in col.edges if e.source == "a.md"}
    assert ("b.md", True) in resolved


def test_collect_indexes_uppercase_md_suffix(tmp_path):
    # README.MD must be indexed (rglob("*.md") alone would skip it on a
    # case-sensitive filesystem) and be resolvable as a link target.
    (tmp_path / "README.MD").write_text("# Readme\nbody text")
    (tmp_path / "a.md").write_text("see [[Readme]]")
    col = collect_markdown(tmp_path)
    assert "README.MD" in col.sources
    edge = next(e for e in col.edges if e.source == "a.md")
    assert edge.resolved is True and edge.target == "README.MD"


def test_collect_stringifies_non_string_frontmatter_keys(tmp_path):
    # A YAML key like `2026:` parses to an int; Qdrant payload fields must be
    # strings, so the key is coerced before it reaches the payload.
    (tmp_path / "n.md").write_text("---\n2026: release\n---\n# N\nbody text")
    col = collect_markdown(tmp_path)
    p = col.chunks[0].payload
    assert p.get("2026") == "release"
    assert all(isinstance(k, str) for k in p)


def test_collect_relative_link_escaping_corpus_is_not_resolved(tmp_path):
    # A root-level note links to ../foo.md. Even though foo.md exists inside the
    # root, the target points outside the corpus, so it must stay dangling —
    # not be mistaken for an internal edge.
    (tmp_path / "foo.md").write_text("# Foo")
    (tmp_path / "page.md").write_text("escape [x](../foo.md)")
    col = collect_markdown(tmp_path)
    edge = next(e for e in col.edges if e.source == "page.md")
    assert edge.resolved is False
    assert edge.target == "../foo"  # kept as a dangling out-of-corpus reference


def test_collect_ids_scoped_by_index_root(tmp_path):
    # Two roots with the same relative path and identical body (differing only
    # in frontmatter) must get distinct ids, or the collection-wide skip check
    # would drop the second corpus's chunk and its index_root/metadata.
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "note.md").write_text("---\nvault: A\n---\nsame body text here")
    (root_b / "note.md").write_text("---\nvault: B\n---\nsame body text here")
    ids_a = {c.id for c in collect_markdown(root_a, index_root=str(root_a)).chunks}
    ids_b = {c.id for c in collect_markdown(root_b, index_root=str(root_b)).chunks}
    assert ids_a and ids_b
    assert ids_a.isdisjoint(ids_b)


def test_collect_sources_include_empty_files(tmp_path):
    # A frontmatter-only / empty file produces no chunks but must appear in
    # sources so the caller can prune its old points and re-sync its links.
    (tmp_path / "full.md").write_text("# Full\nbody text")
    (tmp_path / "empty.md").write_text("---\ntitle: Only Meta\n---\n")
    col = collect_markdown(tmp_path)
    assert set(col.sources) == {"full.md", "empty.md"}
    assert "empty.md" not in {c.payload["source"] for c in col.chunks}
    assert col.files == 2


# ---------- GraphStore.sync_file_links ----------


def _graph_with_fake_session():
    from mnemostack.graph.store import GraphStore

    store = GraphStore.__new__(GraphStore)
    store.database = None
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    driver = MagicMock()
    driver.session.return_value = session
    store.driver = driver
    return store, session


def test_sync_file_links_deletes_then_adds():
    store, session = _graph_with_fake_session()

    n = store.sync_file_links("a.md", ["b.md", "sub/c.md"])
    assert n == 2
    cyphers = [c.args[0] for c in session.run.call_args_list]
    # first a DELETE of existing outgoing LINKS_TO, then MERGE edges
    assert any("DELETE r" in c and "LINKS_TO" in c for c in cyphers)
    assert sum("MERGE (s)-[r:LINKS_TO]->(o)" in c for c in cyphers) == 2


def test_sync_file_links_removed_all_links_still_deletes():
    # A file whose links were all removed: no MERGE edges, but the DELETE of
    # its stale LINKS_TO edges must still run so nothing is left dangling.
    store, session = _graph_with_fake_session()

    n = store.sync_file_links("a.md", [])
    assert n == 0
    cyphers = [c.args[0] for c in session.run.call_args_list]
    assert any("DELETE r" in c and "LINKS_TO" in c for c in cyphers)
    assert sum("MERGE (s)-[r:LINKS_TO]->(o)" in c for c in cyphers) == 0


def test_sync_file_links_scopes_file_nodes_by_index_root():
    # File nodes are keyed by (name, index_root) so two corpora sharing a
    # relative filename never collide. Every query must carry the root.
    store, session = _graph_with_fake_session()

    store.sync_file_links("index.md", ["other.md"], index_root="/vault-a")
    for call in session.run.call_args_list:
        assert "index_root" in call.args[0]
        assert call.kwargs.get("root") == "/vault-a"


def test_sync_file_links_sets_python_lowercased_name():
    # File nodes carry a Python-lowercased name_lower so MemgraphRetriever can
    # find non-ASCII names (Memgraph's toLower folds ASCII only).
    store, session = _graph_with_fake_session()

    store.sync_file_links("Ünïcöde.md", ["Tïtle.md"], index_root="/v")
    merge = next(
        c for c in session.run.call_args_list
        if "MERGE (s)-[r:LINKS_TO]->(o)" in c.args[0]
    )
    assert "name_lower" in merge.args[0]
    assert merge.kwargs["src_lower"] == "ünïcöde.md"
    assert merge.kwargs["dst_lower"] == "tïtle.md"


def test_file_link_sources_lists_names_for_root():
    store, session = _graph_with_fake_session()

    class _Rec:
        def __init__(self, name):
            self._name = name

        def __getitem__(self, key):
            return self._name

    session.run.return_value = [_Rec("a.md"), _Rec("gone.md")]
    names = store.file_link_sources(index_root="/v")
    assert names == ["a.md", "gone.md"]
    cypher = session.run.call_args.args[0]
    assert "LINKS_TO" in cypher and "index_root" in cypher
    assert session.run.call_args.kwargs.get("root") == "/v"


# ---------- CLI ----------


def test_cmd_index_markdown_indexes_and_writes_edges(tmp_path, monkeypatch, capsys):
    import mnemostack.cli as cli

    v = _vault(tmp_path)

    class _FakeProvider:
        dimension = 3

        def embed(self, text):
            return [0.1, 0.2, 0.3]

    class _FakeStore:
        def __init__(self, **_):
            self.upserts = []

        def collection_exists(self):
            return False

        def ensure_collection(self, recreate=False):
            return True

        def iter_ids(self):
            return []

        def upsert(self, cid, vec, payload):
            self.upserts.append((cid, payload))

    class _FakeGraph:
        instances = []

        def __init__(self, **_):
            self.links = {}
            self.roots = {}
            _FakeGraph.instances.append(self)

        def sync_file_links(self, source, targets, *, index_root=None):
            self.links[source] = list(targets)
            self.roots[source] = index_root
            return len(targets)

        def file_link_sources(self, *, index_root=None):
            return []

        def close(self):
            pass

    store = _FakeStore()
    monkeypatch.setattr(cli, "get_provider", lambda *_a, **_k: _FakeProvider())
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    monkeypatch.setattr("mnemostack.graph.GraphStore", _FakeGraph)

    import argparse

    args = argparse.Namespace(
        path=str(v), provider="fake", embedding_model=None,
        collection="c", qdrant="http://localhost:6333",
        chunk_size=1200, memgraph_uri="bolt://localhost:7687",
        graph_timeout=5.0, recreate=False, prune=False, yes=True,
    )
    rc = cli.cmd_index_markdown(args)
    assert rc == 0
    assert len(store.upserts) >= 3               # a, b, c chunks
    assert _FakeGraph.instances                  # graph opened
    graph = _FakeGraph.instances[-1]
    assert graph.links["a.md"] == ["b.md", "sub/c.md"]  # edges written per source
    # every source is synced (even b.md/sub/c.md whose links stay empty) so a
    # file with no links still gets its stale edges cleared
    assert graph.links.get("sub/c.md") == []
    # index_root is threaded through so File nodes are corpus-scoped
    assert graph.roots["a.md"] == str(v.resolve())


def test_cmd_index_markdown_without_graph_skips_edges(tmp_path, monkeypatch):
    import argparse

    import mnemostack.cli as cli

    class _FakeProvider:
        dimension = 3

        def embed(self, text):
            return [0.1, 0.2, 0.3]

    class _FakeStore:
        def __init__(self, **_):
            self.upserts = []

        def collection_exists(self):
            return False

        def ensure_collection(self, recreate=False):
            return True

        def iter_ids(self):
            return []

        def upsert(self, cid, vec, payload):
            self.upserts.append(cid)

    monkeypatch.setattr(cli, "get_provider", lambda *_a, **_k: _FakeProvider())
    monkeypatch.setattr(cli, "VectorStore", lambda **_: _FakeStore())
    # no memgraph_uri -> graph path skipped (would fail if GraphStore imported)
    args = argparse.Namespace(
        path=str(_vault(tmp_path)), provider="fake", embedding_model=None,
        collection="c", qdrant="http://localhost:6333",
        chunk_size=1200, memgraph_uri=None,
        graph_timeout=5.0, recreate=False, prune=False, yes=True,
    )
    assert cli.cmd_index_markdown(args) == 0


def test_cmd_index_markdown_rejects_non_positive_chunk_size(tmp_path, capsys):
    import argparse

    import mnemostack.cli as cli

    args = argparse.Namespace(
        path=str(_vault(tmp_path)), provider="fake", embedding_model=None,
        collection="c", qdrant="http://localhost:6333",
        chunk_size=0, memgraph_uri=None,
        graph_timeout=5.0, recreate=False, prune=False, yes=True,
    )
    # bad --chunk-size fails fast (exit 2) before touching the provider/store
    assert cli.cmd_index_markdown(args) == 2
    assert "chunk-size" in capsys.readouterr().err


def test_cmd_index_markdown_graph_write_failure_does_not_fail_run(
    tmp_path, monkeypatch, capsys
):
    import argparse

    import mnemostack.cli as cli

    class _FakeProvider:
        dimension = 3

        def embed(self, text):
            return [0.1, 0.2, 0.3]

    class _FakeStore:
        def __init__(self, **_):
            self.upserts = []

        def collection_exists(self):
            return False

        def ensure_collection(self, recreate=False):
            return True

        def iter_ids(self):
            return []

        def upsert(self, cid, vec, payload):
            self.upserts.append(cid)

    class _FlakyGraph:
        def __init__(self, **_):
            pass

        def file_link_sources(self, *, index_root=None):
            return []

        def sync_file_links(self, source, targets, *, index_root=None):
            raise RuntimeError("Memgraph disconnected mid-write")

        def close(self):
            pass

    store = _FakeStore()
    monkeypatch.setattr(cli, "get_provider", lambda *_a, **_k: _FakeProvider())
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    monkeypatch.setattr("mnemostack.graph.GraphStore", _FlakyGraph)

    args = argparse.Namespace(
        path=str(_vault(tmp_path)), provider="fake", embedding_model=None,
        collection="c", qdrant="http://localhost:6333",
        chunk_size=1200, memgraph_uri="bolt://localhost:7687",
        graph_timeout=5.0, recreate=False, prune=False, yes=True,
    )
    # vectors already upserted, so a graph outage must warn, not fail the run
    assert cli.cmd_index_markdown(args) == 0
    assert store.upserts                      # vectors were written
    assert "graph write failed" in capsys.readouterr().err


class _HitId:
    def __init__(self, id, payload):
        self.id = id
        self.payload = payload


def test_cmd_index_markdown_prune_removes_deleted_file_chunks(tmp_path, monkeypatch):
    import argparse

    import mnemostack.cli as cli

    v = _vault(tmp_path)
    root = str(v.resolve())

    class _FakeProvider:
        dimension = 3

        def embed(self, text):
            return [0.1, 0.2, 0.3]

    class _PruneStore:
        def __init__(self, **_):
            # a leftover point for a file that no longer exists on disk
            self._prior = [("id-gone", {"source": "gone.md", "index_root": root})]
            self.deleted = []

        def collection_exists(self):
            return True

        def ensure_collection(self, recreate=False):
            return True

        def upsert(self, cid, vec, payload):
            pass

        def iter_ids(self, filters=None):
            for pid, pl in self._prior:
                if not filters or all(pl.get(k) == val for k, val in filters.items()):
                    yield pid

        def scroll(self, filters=None):
            for pid, pl in self._prior:
                if not filters or all(pl.get(k) == val for k, val in filters.items()):
                    yield _HitId(pid, pl)

        def set_payload(self, cid, payload):
            pass

        def delete_payload_keys(self, cid, keys):
            pass

        def delete_points(self, ids):
            self.deleted.extend(ids)
            return len(ids)

    store = _PruneStore()
    monkeypatch.setattr(cli, "get_provider", lambda *_a, **_k: _FakeProvider())
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)

    args = argparse.Namespace(
        path=str(v), provider="fake", embedding_model=None,
        collection="c", qdrant="http://localhost:6333",
        chunk_size=1200, memgraph_uri=None,
        graph_timeout=5.0, recreate=False, prune=True, yes=True,
    )
    assert cli.cmd_index_markdown(args) == 0
    # the deleted file's stale point is pruned even though it isn't in the walk
    assert "id-gone" in store.deleted


def test_cmd_index_markdown_clears_graph_links_for_deleted_file(tmp_path, monkeypatch):
    import argparse

    import mnemostack.cli as cli

    v = _vault(tmp_path)

    class _FakeProvider:
        dimension = 3

        def embed(self, text):
            return [0.1, 0.2, 0.3]

    class _FakeStore:
        def __init__(self, **_):
            pass

        def collection_exists(self):
            return False

        def ensure_collection(self, recreate=False):
            return True

        def iter_ids(self, filters=None):
            return []

        def upsert(self, cid, vec, payload):
            pass

    class _Graph:
        def __init__(self, **_):
            self.synced = {}

        def file_link_sources(self, *, index_root=None):
            # a file that had links previously but is no longer on disk
            return ["gone.md", "a.md"]

        def sync_file_links(self, source, targets, *, index_root=None):
            self.synced[source] = list(targets)
            return len(targets)

        def close(self):
            pass

    graph = _Graph()
    monkeypatch.setattr(cli, "get_provider", lambda *_a, **_k: _FakeProvider())
    monkeypatch.setattr(cli, "VectorStore", lambda **_: _FakeStore())
    monkeypatch.setattr("mnemostack.graph.GraphStore", lambda **_: graph)

    args = argparse.Namespace(
        path=str(v), provider="fake", embedding_model=None,
        collection="c", qdrant="http://localhost:6333",
        chunk_size=1200, memgraph_uri="bolt://localhost:7687",
        graph_timeout=5.0, recreate=False, prune=False, yes=True,
    )
    assert cli.cmd_index_markdown(args) == 0
    # gone.md is no longer on disk -> its LINKS_TO edges are cleared (synced [])
    assert graph.synced.get("gone.md") == []


def test_cmd_index_markdown_recreate_validates_before_dropping(tmp_path, monkeypatch, capsys):
    import argparse

    import mnemostack.cli as cli

    empty = tmp_path / "empty"
    empty.mkdir()  # no .md files

    class _FakeProvider:
        dimension = 3

        def embed(self, text):
            return [0.1, 0.2, 0.3]

    class _Store:
        def __init__(self, **_):
            self.recreated = False

        def collection_exists(self):
            return True

        def count(self):
            return 5

        def ensure_collection(self, recreate=False):
            if recreate:
                self.recreated = True
            return True

        def iter_ids(self, filters=None):
            return []

        def upsert(self, cid, vec, payload):
            pass

    store = _Store()
    monkeypatch.setattr(cli, "get_provider", lambda *_a, **_k: _FakeProvider())
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)

    args = argparse.Namespace(
        path=str(empty), provider="fake", embedding_model=None,
        collection="c", qdrant="http://localhost:6333",
        chunk_size=1200, memgraph_uri=None,
        graph_timeout=5.0, recreate=True, prune=False, yes=True,
    )
    # no .md files -> exit 2 and the existing collection is NOT dropped
    assert cli.cmd_index_markdown(args) == 2
    assert store.recreated is False
    assert "no .md files" in capsys.readouterr().err


def test_cmd_index_markdown_refreshes_payload_on_frontmatter_change(tmp_path, monkeypatch):
    import argparse

    import mnemostack.cli as cli

    note = tmp_path / "n.md"
    note.write_text("---\ntag: old\ndrop_me: 1\n---\n# N\nstable body text")

    class _FakeProvider:
        dimension = 3

        def embed(self, text):
            return [0.1, 0.2, 0.3]

    # class-level state so the two VectorStore() instances (one per cmd run)
    # share the same simulated collection.
    class _Store:
        points: dict = {}

        def __init__(self, **_):
            pass

        def collection_exists(self):
            return bool(_Store.points)

        def ensure_collection(self, recreate=False):
            if recreate:
                _Store.points = {}
            return True

        def scroll(self, filters=None):
            for pid, pl in _Store.points.items():
                if not filters or all(pl.get(k) == v for k, v in filters.items()):
                    yield _HitId(pid, pl)

        def iter_ids(self, filters=None):
            return list(_Store.points)

        def upsert(self, cid, vec, payload):
            _Store.points[cid] = dict(payload)

        def set_payload(self, cid, payload):
            _Store.points.setdefault(cid, {}).update(payload)

        def delete_payload_keys(self, cid, keys):
            for k in keys:
                _Store.points.get(cid, {}).pop(k, None)

    monkeypatch.setattr(cli, "get_provider", lambda *_a, **_k: _FakeProvider())
    monkeypatch.setattr(cli, "VectorStore", lambda **_: _Store())

    def _args():
        return argparse.Namespace(
            path=str(tmp_path), provider="fake", embedding_model=None,
            collection="c", qdrant="http://localhost:6333",
            chunk_size=1200, memgraph_uri=None,
            graph_timeout=5.0, recreate=False, prune=False, yes=True,
        )

    assert cli.cmd_index_markdown(_args()) == 0          # initial index
    assert any(pl.get("tag") == "old" for pl in _Store.points.values())

    # simulate `mnemostack invalidate` on the stored chunk
    for pl in _Store.points.values():
        pl["invalidated_at"] = "2026-07-04T00:00:00Z"

    # change frontmatter (body/id unchanged) and re-index
    note.write_text("---\ntag: new\n---\n# N\nstable body text")
    assert cli.cmd_index_markdown(_args()) == 0
    # payload is refreshed in place: new value applied, removed key dropped
    payloads = list(_Store.points.values())
    assert any(pl.get("tag") == "new" for pl in payloads)
    assert all("drop_me" not in pl for pl in payloads)
    # ...but the system-owned invalidation marker is preserved (re-indexing must
    # not resurrect an intentionally-invalidated memory)
    assert all(pl.get("invalidated_at") == "2026-07-04T00:00:00Z" for pl in payloads)


def test_cmd_index_markdown_single_file_does_not_reconcile_siblings(tmp_path, monkeypatch):
    import argparse

    import mnemostack.cli as cli

    (tmp_path / "a.md").write_text("# A\nbody a")
    (tmp_path / "b.md").write_text("# B\nbody b")  # sibling, not indexed

    class _FakeProvider:
        dimension = 3

        def embed(self, text):
            return [0.1, 0.2, 0.3]

    root = str(tmp_path.resolve())

    class _Store:
        def __init__(self, **_):
            # a stored point for the sibling b.md under the same index_root
            self._points = [("id-b", {"source": "b.md", "index_root": root})]
            self.deleted = []

        def collection_exists(self):
            return True

        def ensure_collection(self, recreate=False):
            return True

        def scroll(self, filters=None):
            for pid, pl in self._points:
                if not filters or all(pl.get(k) == v for k, v in filters.items()):
                    yield _HitId(pid, pl)

        def iter_ids(self, filters=None):
            for pid, pl in self._points:
                if not filters or all(pl.get(k) == v for k, v in filters.items()):
                    yield pid

        def upsert(self, cid, vec, payload):
            pass

        def set_payload(self, cid, payload):
            pass

        def delete_payload_keys(self, cid, keys):
            pass

        def delete_points(self, ids):
            self.deleted.extend(ids)
            return len(ids)

    class _Graph:
        called_file_link_sources = False

        def __init__(self, **_):
            self.synced = {}

        def file_link_sources(self, *, index_root=None):
            _Graph.called_file_link_sources = True
            return ["b.md"]

        def sync_file_links(self, source, targets, *, index_root=None):
            self.synced[source] = list(targets)
            return len(targets)

        def close(self):
            pass

    store = _Store()
    graph = _Graph()
    monkeypatch.setattr(cli, "get_provider", lambda *_a, **_k: _FakeProvider())
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    monkeypatch.setattr("mnemostack.graph.GraphStore", lambda **_: graph)

    args = argparse.Namespace(
        path=str(tmp_path / "a.md"), provider="fake", embedding_model=None,
        collection="c", qdrant="http://localhost:6333",
        chunk_size=1200, memgraph_uri="bolt://localhost:7687",
        graph_timeout=5.0, recreate=False, prune=True, yes=True,
    )
    assert cli.cmd_index_markdown(args) == 0
    # single-file run must NOT prune the sibling's chunks or clear its links
    assert "id-b" not in store.deleted
    assert "b.md" not in graph.synced
    assert _Graph.called_file_link_sources is False
