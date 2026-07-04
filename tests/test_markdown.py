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


def test_links_deduped_first_seen_order():
    # First-seen wins: the wikilink [[A]] comes before the inline [b](A.md).
    links = extract_links("[[A]] [[A]] [b](A.md)")
    assert [(link.target, link.is_wikilink) for link in links] == [("A", True)]


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
