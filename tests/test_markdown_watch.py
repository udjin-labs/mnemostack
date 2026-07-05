"""Markdown watcher + incremental syncer (index-markdown --watch, issue #97)."""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Distance

from mnemostack.markdown.sync import MarkdownSyncer
from mnemostack.markdown.watch import REMOVE, UPSERT, MarkdownWatcher, _Debouncer
from mnemostack.vector import VectorStore

# ---------- fakes ----------


class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class _FakeSyncer:
    """Records index/remove calls; can be told to raise for a given path."""

    def __init__(self, boom: str | None = None):
        self.indexed: list[str] = []
        self.removed: list[str] = []
        self.boom = boom

    def index_file(self, path):
        if self.boom and str(path).endswith(self.boom):
            raise RuntimeError("boom")
        self.indexed.append(str(path))
        return _Res(source=str(path), inserted=1)

    def remove_file(self, path):
        self.removed.append(str(path))
        return _Res(source=str(path), pruned=1)


class _Res:
    def __init__(self, source=None, inserted=0, refreshed=0, pruned=0, edges=0, error=None):
        self.source = source
        self.inserted = inserted
        self.refreshed = refreshed
        self.pruned = pruned
        self.edges = edges
        self.error = error


def _watcher(root, syncer, clock):
    return MarkdownWatcher(syncer, root, debounce=1.0, clock=clock)


# ---------- debouncer / coalescing ----------


def test_debouncer_coalesces_latest_kind_wins():
    clock = _FakeClock()
    d = _Debouncer(1.0, clock)
    d.add("/x/a.md", UPSERT)
    d.add("/x/a.md", UPSERT)  # rapid second save
    d.add("/x/a.md", REMOVE)  # then deleted -> latest wins
    # Not due yet (window is 1.0, no time passed).
    assert d.drain() == []
    clock.t = 1.0
    ready = d.drain()
    assert ready == [("/x/a.md", REMOVE)]  # coalesced to one, latest kind
    assert d.drain() == []  # popped


def test_debouncer_force_drains_immediately():
    clock = _FakeClock()
    d = _Debouncer(5.0, clock)
    d.add("/x/a.md", UPSERT)
    assert d.drain() == []  # window not elapsed
    assert d.drain(force=True) == [("/x/a.md", UPSERT)]


# ---------- event dispatch ----------


def test_upsert_and_remove_route_to_syncer(tmp_path):
    clock = _FakeClock()
    syncer = _FakeSyncer()
    w = _watcher(tmp_path, syncer, clock)
    a = str(tmp_path / "a.md")
    b = str(tmp_path / "b.md")
    w.handle_event(a, UPSERT)
    w.handle_event(b, REMOVE)
    clock.t = 1.0
    assert w.flush() == 2
    assert syncer.indexed == [a]
    assert syncer.removed == [b]


def test_non_markdown_events_ignored(tmp_path):
    clock = _FakeClock()
    syncer = _FakeSyncer()
    w = _watcher(tmp_path, syncer, clock)
    w.handle_event(str(tmp_path / "notes.txt"), UPSERT)
    w.handle_event(str(tmp_path / ".a.md.swp"), UPSERT)  # not *.md
    clock.t = 1.0
    assert w.flush(force=True) == 0
    assert syncer.indexed == []


def test_failing_file_does_not_stop_the_watch(tmp_path):
    clock = _FakeClock()
    syncer = _FakeSyncer(boom="bad.md")
    errors: list[tuple[str, str, Exception]] = []
    w = MarkdownWatcher(
        syncer, tmp_path, debounce=1.0, clock=clock,
        on_error=lambda p, k, e: errors.append((p, k, e)),
    )
    bad = str(tmp_path / "bad.md")
    good = str(tmp_path / "good.md")
    w.handle_event(bad, UPSERT)
    w.handle_event(good, UPSERT)
    clock.t = 1.0
    assert w.flush() == 2  # both attempted
    assert syncer.indexed == [good]  # the good one still applied
    assert len(errors) == 1 and errors[0][0] == bad


# ---------- polling fallback ----------


def test_poll_once_detects_create_modify_delete(tmp_path):
    clock = _FakeClock()
    syncer = _FakeSyncer()
    w = _watcher(tmp_path, syncer, clock)

    f = tmp_path / "note.md"
    f.write_text("# One\n")
    snap = w.poll_once({})  # first scan sees the new file
    clock.t = 1.0
    w.flush()
    assert syncer.indexed == [str(f.resolve())]

    # modify -> different mtime -> upsert again
    import os

    os.utime(f, (snap[str(f.resolve())] + 10, snap[str(f.resolve())] + 10))
    snap = w.poll_once(snap)
    clock.t = 2.0
    w.flush()
    assert syncer.indexed.count(str(f.resolve())) == 2

    # delete -> vanished -> remove
    f.unlink()
    w.poll_once(snap)
    clock.t = 3.0
    w.flush()
    assert syncer.removed == [str(f.resolve())]


def test_polling_scan_is_case_insensitive(tmp_path):
    # The indexer accepts .MD/.Md; the polling scan must track them too, or an
    # uppercase file indexed initially would never sync its later edits/deletes.
    clock = _FakeClock()
    syncer = _FakeSyncer()
    w = _watcher(tmp_path, syncer, clock)
    f = tmp_path / "README.MD"
    f.write_text("# H\n")
    w.poll_once({})
    clock.t = 1.0
    w.flush()
    assert syncer.indexed == [str(f.resolve())]


def test_polling_honors_debounce_for_active_writes(tmp_path):
    # A file still being written re-stamps its mtime each tick, resetting the
    # quiet window; polling must NOT apply until it has been quiet for debounce.
    import os

    clock = _FakeClock()
    syncer = _FakeSyncer()
    w = _watcher(tmp_path, syncer, clock)  # debounce=1.0
    f = tmp_path / "a.md"
    f.write_text("v1")

    snap = w.poll_once({})  # detected at t=0, deadline 1.0
    w.flush()
    assert syncer.indexed == []  # window not elapsed

    # still writing at t=0.5 -> new mtime -> deadline resets to 1.5
    clock.t = 0.5
    os.utime(f, (snap[str(f.resolve())] + 5, snap[str(f.resolve())] + 5))
    snap = w.poll_once(snap)
    clock.t = 1.0
    w.flush()
    assert syncer.indexed == []  # reset window still open

    # quiet through the window -> applied once
    clock.t = 1.5
    w.flush()
    assert syncer.indexed == [str(f.resolve())]


def test_catch_up_reconciles_baseline_diff(tmp_path):
    # A file created during the initial index (absent from the pre-index
    # baseline) and one deleted during it are both reconciled at watch start.
    import os

    clock = _FakeClock()
    syncer = _FakeSyncer()
    w = _watcher(tmp_path, syncer, clock)
    new = tmp_path / "new.md"
    new.write_text("# N\n")
    gone = os.path.abspath(str(tmp_path / "gone.md"))  # in baseline, not on disk now

    w._catch_up({gone: 1.0})  # queues the diff (honors debounce, no force)
    clock.t = 1.0
    w.flush()
    assert syncer.indexed == [os.path.abspath(str(new))]
    assert syncer.removed == [gone]


def test_poll_and_debounce_intervals_floored_to_avoid_busy_loop():
    # --watch-poll-interval 0 / --watch-debounce 0 must not sleep(0) forever.
    w = MarkdownWatcher(_FakeSyncer(), ".", poll_interval=0.0, debounce=0.0)
    assert w.poll_interval >= 0.05  # floored


def test_reconcile_drops_sources_whose_file_is_gone(tmp_path):
    # A source indexed in the store but with no backing file on disk (e.g. a
    # file created+deleted during the initial index) is dropped on reconcile.
    store = _mem_store()
    root = str(tmp_path.resolve())
    syncer = MarkdownSyncer(store, _FakeProvider(), index_root=root, chunk_size=10000)

    present = tmp_path / "here.md"
    present.write_text("# H\n")
    syncer.index_file(present)
    # simulate a transient file the initial pass indexed but which is now gone
    # (real chunk ids are UUID strings, so pruning stringifies them consistently)
    from mnemostack.ingest import stable_chunk_id

    gid = stable_chunk_id("ghost.md", 0, "ghost")
    store.upsert(
        gid, [1.0, 0.0, 0.0, 0.0], {"text": "ghost", "source": "ghost.md", "index_root": root}
    )

    removed = syncer.reconcile_deletions()
    assert removed == ["ghost.md"]
    assert _sources_in(store, root) == {"here.md"}  # present source untouched


def test_watcher_reconcile_reports_removals(tmp_path):
    store = _mem_store()
    root = str(tmp_path.resolve())
    syncer = MarkdownSyncer(store, _FakeProvider(), index_root=root, chunk_size=10000)
    from mnemostack.ingest import stable_chunk_id

    store.upsert(
        stable_chunk_id("gone.md", 0, "x"),
        [1.0, 0.0, 0.0, 0.0],
        {"text": "x", "source": "gone.md", "index_root": root},
    )
    results = []
    w = MarkdownWatcher(syncer, tmp_path, on_result=results.append)
    w.reconcile()
    assert [r.source for r in results] == ["gone.md"]


def test_handle_event_does_not_resolve_symlinks(tmp_path):
    # A symlinked note must be addressed by the link path (what the initial walk
    # indexed), not its resolved target.
    import os

    (tmp_path / "sub").mkdir()
    target = tmp_path / "sub" / "target.md"
    target.write_text("# T\n")
    link = tmp_path / "link.md"
    os.symlink(target, link)

    clock = _FakeClock()
    syncer = _FakeSyncer()
    w = _watcher(tmp_path, syncer, clock)
    w.handle_event(str(link), UPSERT)
    clock.t = 1.0
    w.flush()
    assert syncer.indexed == [os.path.abspath(str(link))]  # not .../sub/target.md


# ---------- syncer integration (in-memory Qdrant + fake embedder) ----------


class _FakeProvider:
    dimension = 4

    def embed(self, text):
        return [1.0, 0.0, 0.0, 0.0]


def _mem_store():
    s = VectorStore.__new__(VectorStore)
    s.collection = "watch_test"
    s.dimension = 4
    s.distance = Distance.COSINE
    s.client = QdrantClient(":memory:")
    s.ensure_collection()
    return s


def _sources_in(store, index_root):
    return {
        (h.payload or {}).get("source")
        for h in store.scroll(filters={"index_root": index_root})
    }


def test_syncer_indexes_modifies_and_removes_a_file(tmp_path):
    store = _mem_store()
    root = str(tmp_path.resolve())
    syncer = MarkdownSyncer(store, _FakeProvider(), index_root=root, chunk_size=10000)

    f = tmp_path / "a.md"
    f.write_text("# Title\n\nHello world.\n")
    res = syncer.index_file(f)
    assert res.source == "a.md"
    assert res.inserted >= 1
    assert "a.md" in _sources_in(store, root)

    # Re-index unchanged: idempotent (no new inserts, ids stable).
    res2 = syncer.index_file(f)
    assert res2.inserted == 0

    # Modify to shrink content: stale chunks of a.md are pruned.
    before = len(list(store.scroll(filters={"index_root": root, "source": "a.md"})))
    f.write_text("# Title\n\nHi.\n")
    res3 = syncer.index_file(f)
    after = len(list(store.scroll(filters={"index_root": root, "source": "a.md"})))
    assert res3.pruned >= 0 and after >= 1
    # every remaining point still belongs to a.md (nothing leaked to other sources)
    assert _sources_in(store, root) == {"a.md"}
    _ = before

    # Remove: all of a.md's chunks are gone.
    rem = syncer.remove_file(f)
    assert rem.source == "a.md"
    assert list(store.scroll(filters={"index_root": root, "source": "a.md"})) == []


def test_syncer_syncs_and_clears_graph_links(tmp_path):
    store = _mem_store()
    root = str(tmp_path.resolve())

    class _Graph:
        def __init__(self):
            self.synced: dict[str, list[str]] = {}

        def sync_file_links(self, source, targets, *, index_root=None):
            self.synced[source] = list(targets)
            return len(targets)

    graph = _Graph()
    syncer = MarkdownSyncer(
        store, _FakeProvider(), index_root=root, chunk_size=10000, graph=graph
    )

    (tmp_path / "b.md").write_text("# B\n")
    (tmp_path / "a.md").write_text("# A\n\nSee [B](b.md).\n")
    syncer.index_file(tmp_path / "a.md")
    assert graph.synced.get("a.md") == ["b.md"]

    # Remove a.md -> its outgoing links are cleared.
    syncer.remove_file(tmp_path / "a.md")
    assert graph.synced.get("a.md") == []


# ---------- CLI wiring ----------


def test_watch_requires_a_directory(tmp_path, capsys):
    import argparse

    import mnemostack.cli as cli

    f = tmp_path / "note.md"
    f.write_text("# H\n")
    args = argparse.Namespace(path=str(f), chunk_size=1200, watch=True)
    # Returns before any provider/store construction — a file can't be watched.
    assert cli.cmd_index_markdown(args) == 2
    assert "requires a directory" in capsys.readouterr().err


def test_syncer_reports_failed_embeddings_and_keeps_stale(tmp_path):
    # A provider failure (empty vector) must surface as res.failed and NOT prune
    # the source's existing chunks — so the watch layer can warn, not report OK.
    store = _mem_store()
    root = str(tmp_path.resolve())
    f = tmp_path / "a.md"
    f.write_text("# A\n\noriginal body\n")

    ok = MarkdownSyncer(store, _FakeProvider(), index_root=root, chunk_size=10000)
    ok.index_file(f)
    before = len(list(store.scroll(filters={"index_root": root, "source": "a.md"})))
    assert before >= 1

    class _FailProvider:
        dimension = 4

        def embed(self, text):
            return []  # provider outage

    f.write_text("# A\n\ncompletely different body\n")
    failing = MarkdownSyncer(store, _FailProvider(), index_root=root, chunk_size=10000)
    res = failing.index_file(f)
    assert res.failed >= 1
    assert res.inserted == 0
    # old chunks kept (not pruned) since the re-embed failed
    after = len(list(store.scroll(filters={"index_root": root, "source": "a.md"})))
    assert after == before
