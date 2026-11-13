"""Startup failures retry without requiring an unchanged file to be edited."""
from pathlib import Path
from types import SimpleNamespace

import pytest

import mnemostack.cli as cli
import mnemostack.markdown.sync as sync
import mnemostack.markdown.watch as watch


@pytest.mark.parametrize("failed_passes", [0, 1])
@pytest.mark.parametrize("subtree", [False, True])
def test_initial_failed_source_retries_until_success(monkeypatch, tmp_path, failed_passes, subtree):
    root = tmp_path / "corpus"
    watched = root / "nested" if subtree else root
    watched.mkdir(parents=True)
    bad = watched / "failed.md"
    good = watched / "healthy.md"
    bad.write_text("synthetic failed source")
    good.write_text("synthetic healthy source")
    seen = []

    class Syncer:
        def __init__(self, *args, **kwargs):
            pass

        def index_file(self, path):
            seen.append(str(path))
            return SimpleNamespace(source=str(path), failed=int(len(seen) <= failed_passes), inserted=1, refreshed=0, pruned=0, edges=0, error=None)

    class Watcher(watch.MarkdownWatcher):
        def run(self, *, baseline):
            snapshot = baseline
            for _ in range(3):
                snapshot = self.poll_once(snapshot)
                self.flush(force=True)

    monkeypatch.setattr(sync, "MarkdownSyncer", Syncer)
    monkeypatch.setattr(watch, "MarkdownWatcher", Watcher)
    args = SimpleNamespace(chunk_size=800, watch_debounce=0, watch_poll_interval=1)
    rc = cli._watch_markdown(args, object(), object(), str(root), str(watched), None, watch._scan_mtimes(watched), failed_sources={str(bad.relative_to(root))})
    assert rc == 0
    assert seen == [str(bad)] * (failed_passes + 1)


def test_initial_retry_keeps_symlink_source_name(monkeypatch, tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    target = tmp_path / "outside.md"
    target.write_text("synthetic source")
    link = root / "link.md"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")
    seen = []

    class Syncer:
        def __init__(self, *args, **kwargs):
            pass

        def index_file(self, path):
            seen.append(str(path))
            return SimpleNamespace(source=None, failed=0, error=None)

    class Watcher(watch.MarkdownWatcher):
        def run(self, *, baseline):
            self.poll_once(baseline)
            self.flush(force=True)

    monkeypatch.setattr(sync, "MarkdownSyncer", Syncer)
    monkeypatch.setattr(watch, "MarkdownWatcher", Watcher)
    cli._watch_markdown(SimpleNamespace(chunk_size=800, watch_debounce=0, watch_poll_interval=1), object(), object(), str(root), str(root), None, watch._scan_mtimes(root), failed_sources={"link.md"})
    assert seen == [str(link)]
    assert Path(seen[0]) != target


def test_initial_retry_ignores_sources_outside_watched_subtree(monkeypatch, tmp_path):
    root = tmp_path / "corpus"
    watched = root / "nested"
    watched.mkdir(parents=True)
    (root / "sibling.md").write_text("not watched")

    class Syncer:
        def __init__(self, *args, **kwargs):
            pass

        def index_file(self, path):
            pytest.fail("must not index outside watched subtree")

    class Watcher(watch.MarkdownWatcher):
        def run(self, *, baseline):
            self.poll_once(baseline)
            assert self.flush(force=True) == 0

    monkeypatch.setattr(sync, "MarkdownSyncer", Syncer)
    monkeypatch.setattr(watch, "MarkdownWatcher", Watcher)
    cli._watch_markdown(SimpleNamespace(chunk_size=800, watch_debounce=0, watch_poll_interval=1), object(), object(), str(root), str(watched), None, watch._scan_mtimes(watched), failed_sources={"sibling.md"})


@pytest.mark.parametrize("retry_all", [False, True])
def test_normal_startup_is_idle_but_quota_skip_retries_all(monkeypatch, tmp_path, retry_all):
    first, second = tmp_path / "one.md", tmp_path / "two.md"
    first.write_text("first synthetic source")
    second.write_text("second synthetic source")
    seen = []

    class Syncer:
        def __init__(self, *args, **kwargs):
            pass

        def index_file(self, path):
            seen.append(str(path))
            return SimpleNamespace(source=None, failed=0, error=None)

    class Watcher(watch.MarkdownWatcher):
        def run(self, *, baseline):
            self.poll_once(baseline)
            self.flush(force=True)

    monkeypatch.setattr(sync, "MarkdownSyncer", Syncer)
    monkeypatch.setattr(watch, "MarkdownWatcher", Watcher)
    cli._watch_markdown(SimpleNamespace(chunk_size=800, watch_debounce=0, watch_poll_interval=1), object(), object(), str(tmp_path), str(tmp_path), None, watch._scan_mtimes(tmp_path), retry_all=retry_all)
    assert sorted(seen) == (sorted([str(first), str(second)]) if retry_all else [])
