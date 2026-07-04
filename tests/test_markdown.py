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
    assert links == ["Target", "Other", "Third"]


def test_extract_markdown_links_notes_only():
    # .md and extensionless targets are notes; external, image (.png), embeds,
    # and pure anchors are not edges.
    text = "[a](notes/foo.md) [b](bare) [ext](https://x.com) [img](y.png) ![pic](z.png) [t](#top)"
    assert extract_links(text) == ["notes/foo", "bare"]


def test_links_deduped_first_seen_order():
    assert extract_links("[[A]] [[A]] [b](A.md)") == ["A"]


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


# ---------- GraphStore.sync_file_links ----------


def test_sync_file_links_deletes_then_adds():
    from mnemostack.graph.store import GraphStore

    store = GraphStore.__new__(GraphStore)
    store.database = None
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    driver = MagicMock()
    driver.session.return_value = session
    store.driver = driver

    n = store.sync_file_links("a.md", ["b.md", "sub/c.md"])
    assert n == 2
    cyphers = [c.args[0] for c in session.run.call_args_list]
    # first a DELETE of existing outgoing LINKS_TO, then MERGE edges
    assert any("DELETE r" in c and "LINKS_TO" in c for c in cyphers)
    assert sum("MERGE (s)-[r:LINKS_TO]->(o)" in c for c in cyphers) == 2


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
            _FakeGraph.instances.append(self)

        def sync_file_links(self, source, targets):
            self.links[source] = list(targets)
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
    links = _FakeGraph.instances[-1].links
    assert links["a.md"] == ["b.md", "sub/c.md"]  # edges written per source


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
