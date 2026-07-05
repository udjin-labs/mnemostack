"""MemgraphRetriever must honor (name, index_root) scoping on the read side.

The markdown indexer scopes ``:File`` nodes by ``(name, index_root)`` on write
(``GraphStore.sync_file_links``), so two corpora that share a relative filename
(both have ``index.md``) don't collide. The retriever used to group candidates
by ``name`` alone and expand relationships with ``MATCH (n {name: $name})`` — no
``index_root`` filter — so multi-root recall collapsed same-named files and
serialized ``LINKS_TO`` edges from every root. These tests pin the fix with a
fake driver (no live Memgraph needed).
"""

from __future__ import annotations

from typing import Any

from mnemostack.recall.retrievers import MemgraphRetriever


class _Result:
    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows

    def data(self):
        return self._rows


class _RecordingSession:
    """A fake neo4j session backed by an in-memory (name, index_root) graph.

    ``files`` maps ``(name, index_root)`` -> list of neighbor names for its
    LINKS_TO edges. ``rel_calls`` records the (name, root_key) each rel query
    was pinned to, so a test can assert the read side scoped correctly.
    """

    def __init__(self, files, rel_calls):
        self._files = files
        self.rel_calls = rel_calls

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, cypher, **params):
        if "labels(n)[0]" in cypher:  # a node probe
            w = params.get("w", "")
            rows = [
                {
                    "name": name,
                    "type": "File",
                    "mc": "",
                    "index_root": root,
                }
                for (name, root) in self._files
                if name.lower() == w
            ]
            # Honor the probe's LIMIT the way Memgraph would, so a regression to
            # a too-small cap (e.g. the old hardcoded 5) would drop roots here.
            probe_lim = params.get("probe_lim")
            if probe_lim is not None:
                rows = rows[:probe_lim]
            return _Result(rows)
        # rel expansion — pinned by (name, index_root)
        name = params["name"]
        root_key = params["root_key"]
        self.rel_calls.append((name, root_key))
        targets = self._files.get((name, root_key), [])
        rows = [
            {"from_n": name, "rel": "LINKS_TO", "to_n": t} for t in targets
        ]
        return _Result(rows)


class _Driver:
    def __init__(self, files, rel_calls):
        self._files = files
        self._rel_calls = rel_calls

    def session(self, **_):
        return _RecordingSession(self._files, self._rel_calls)


def _retriever(files, rel_calls):
    return MemgraphRetriever(uri="bolt://x", driver=_Driver(files, rel_calls))


def test_same_named_files_across_roots_stay_distinct():
    # Two roots each have an index.md linking to a different neighbor.
    files = {
        ("index.md", "/corpus/a"): ["a-notes.md"],
        ("index.md", "/corpus/b"): ["b-notes.md"],
    }
    rel_calls: list[tuple[str, str]] = []
    out = _retriever(files, rel_calls).search("index.md", include_invalidated=True)

    ids = {r.id for r in out}
    # Not collapsed into a single graph:index.md result.
    assert ids == {"graph:/corpus/a:index.md", "graph:/corpus/b:index.md"}

    # Each result serialized only its own root's edge, not the other root's.
    by_id = {r.id: r for r in out}
    assert "a-notes.md" in by_id["graph:/corpus/a:index.md"].text
    assert "b-notes.md" not in by_id["graph:/corpus/a:index.md"].text
    assert "b-notes.md" in by_id["graph:/corpus/b:index.md"].text
    assert "a-notes.md" not in by_id["graph:/corpus/b:index.md"].text

    # The rel query was pinned to each specific (name, root), never unscoped.
    assert set(rel_calls) == {("index.md", "/corpus/a"), ("index.md", "/corpus/b")}

    # Payload carries the root for downstream attribution.
    assert {r.payload["index_root"] for r in out} == {"/corpus/a", "/corpus/b"}


def test_more_than_five_roots_not_truncated():
    # More roots than the old hardcoded per-probe LIMIT 5: every root must still
    # be counted/expanded (the limit is tied to the candidate budget now).
    files = {("index.md", f"/corpus/{i}"): [f"note-{i}.md"] for i in range(8)}
    rel_calls: list[tuple[str, str]] = []
    out = _retriever(files, rel_calls).search("index.md", include_invalidated=True)

    ids = {r.id for r in out}
    assert len(ids) == 8, ids
    assert ids == {f"graph:/corpus/{i}:index.md" for i in range(8)}


def test_entity_node_without_index_root_keeps_plain_id():
    # An :Entity node has no index_root -> root_key "" -> plain graph:<name> id
    # and a rel query pinned with root_key "" (matches nodes lacking index_root).
    files = {("Alice", ""): ["Bob"]}
    rel_calls: list[tuple[str, str]] = []
    out = _retriever(files, rel_calls).search("alice", include_invalidated=True)

    assert [r.id for r in out] == ["graph:Alice"]
    assert out[0].payload["index_root"] is None
    assert rel_calls == [("Alice", "")]
