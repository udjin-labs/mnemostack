"""Tests for markdown wrapper generation during ingest."""
from __future__ import annotations

from pathlib import Path

from mnemostack.ingest import IngestItem, Ingestor


class _FakeEmbedding:
    dimension = 3

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0] if text else []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class _FakeStore:
    def __init__(self):
        self.upserts: list[tuple[str, list[float], dict]] = []

    def upsert_batch(self, points):
        self.upserts.extend(points)
        return len(points)


class _FakeGraph:
    def __init__(self):
        self.file_tags: list[dict] = []

    def add_file_tags(self, **kwargs):
        self.file_tags.append(kwargs)


def _ingestor(wrapper_dir: Path, *, graph=None, skip_seen: bool = False) -> Ingestor:
    return Ingestor(
        embedding=_FakeEmbedding(),
        vector_store=_FakeStore(),
        wrapper_dir=wrapper_dir,
        graph=graph,
        skip_seen=skip_seen,
    )


def test_wrapper_generation_creates_markdown_file(tmp_path):
    wrapper_dir = tmp_path / "memory" / "files"
    ing = _ingestor(wrapper_dir)

    stats = ing.ingest([
        IngestItem(
            text="A research note about memory wrappers.",
            source="docs/report.pdf",
            tags=["research", "pdf"],
        )
    ])

    wrappers = list(wrapper_dir.glob("report-*.md"))
    assert len(wrappers) == 1
    content = wrappers[0].read_text(encoding="utf-8")
    assert stats.wrappers_created == 1
    assert stats.wrappers_updated == 0
    assert "title: report" in content
    assert "original_path: docs/report.pdf" in content
    assert "research" in content
    assert "A research note about memory wrappers." in content
    assert stats.ids[0] in content


def test_wrapper_update_is_idempotent_when_file_exists(tmp_path):
    ing = _ingestor(tmp_path)
    item = IngestItem(text="first version", source="notes/item.md")

    first = ing.ingest([item])
    second = ing.ingest([IngestItem(text="updated version", source="notes/item.md")])

    wrappers = list(tmp_path.glob("item-*.md"))
    assert len(wrappers) == 1
    assert first.wrappers_created == 1
    assert second.wrappers_updated == 1
    assert "updated version" in wrappers[0].read_text(encoding="utf-8")


def test_wrapper_filenames_are_collision_safe(tmp_path):
    ing = _ingestor(tmp_path)

    stats = ing.ingest([
        IngestItem(text="alpha", source="alpha/report.pdf"),
        IngestItem(text="beta", source="beta/report.pdf"),
    ])

    wrappers = sorted(path.name for path in tmp_path.glob("report-*.md"))
    assert stats.wrappers_created == 2
    assert len(wrappers) == 2
    assert wrappers[0] != wrappers[1]


def test_wrapper_generation_creates_missing_directory(tmp_path):
    wrapper_dir = tmp_path / "missing" / "nested"
    ing = _ingestor(wrapper_dir)

    stats = ing.ingest([IngestItem(text="hello", source="docs/hello.md")])

    assert wrapper_dir.is_dir()
    assert stats.wrappers_created == 1
    assert len(list(wrapper_dir.glob("hello-*.md"))) == 1


def test_wrapper_generation_uses_item_wrapper_dir(tmp_path):
    ing = Ingestor(embedding=_FakeEmbedding(), vector_store=_FakeStore(), skip_seen=False)
    item_dir = tmp_path / "per-item"

    stats = ing.ingest([IngestItem(text="hello", source="docs/hello.md", wrapper_dir=item_dir)])

    assert stats.wrappers_created == 1
    assert len(list(item_dir.glob("hello-*.md"))) == 1


def test_graph_integration_is_optional(tmp_path):
    graph = _FakeGraph()
    ing = _ingestor(tmp_path, graph=graph)

    stats = ing.ingest([IngestItem(text="hello", source="docs/hello.md", tags=["docs"])])

    assert stats.wrappers_created == 1
    assert graph.file_tags == [
        {
            "name": "hello.md",
            "path": "docs/hello.md",
            "indexed_date": graph.file_tags[0]["indexed_date"],
            "tags": ["docs"],
        }
    ]


def test_graph_sync_runs_without_wrapper_dir():
    graph = _FakeGraph()
    ing = Ingestor(
        embedding=_FakeEmbedding(),
        vector_store=_FakeStore(),
        graph=graph,
        skip_seen=False,
    )

    stats = ing.ingest([IngestItem(text="hello", source="docs/hello.md", tags=["docs"])])

    assert stats.wrappers_created == 0
    assert stats.wrappers_updated == 0
    assert graph.file_tags == [
        {
            "name": "hello.md",
            "path": "docs/hello.md",
            "indexed_date": graph.file_tags[0]["indexed_date"],
            "tags": ["docs"],
        }
    ]


def test_wrapper_failure_logs_warning_without_failing_ingest(tmp_path, caplog):
    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_text("blocks mkdir", encoding="utf-8")
    ing = _ingestor(blocking_file)

    stats = ing.ingest([IngestItem(text="hello", source="docs/hello.md")])

    assert stats.upserted == 1
    assert stats.wrappers_created == 0
    assert "failed to write markdown wrapper" in caplog.text
