"""Streaming ingest API for mnemostack.

Most callers don't want to shell out to the CLI, nor do they want to write
their own batching and dedup logic. They want: "here is a stream of items,
keep my Qdrant (and optionally Memgraph) in sync, don't duplicate anything,
tell me what actually changed."

That is what this module provides.

    from mnemostack.embeddings import get_provider
    from mnemostack.vector import VectorStore
    from mnemostack.ingest import Ingestor, IngestItem

    emb = get_provider("gemini")
    store = VectorStore(collection="my-memory", dimension=emb.dimension)
    store.ensure_collection()

    ingestor = Ingestor(embedding=emb, vector_store=store)
    stats = ingestor.ingest([
        IngestItem(text="alice joined acme on 2024-03-01", source="notes/alice.md"),
        IngestItem(text="alice left acme on 2025-06-15", source="notes/alice.md"),
    ])
    print(stats)  # -> IngestStats(seen=2, embedded=2, upserted=2, skipped=0, failed=0)

Re-running the same call is a no-op — the deterministic chunk id is the
same, embedding is skipped, Qdrant upsert replaces onto itself.

Typical server integration: call `ingest_one()` per incoming message. The
Ingestor keeps a small LRU cache of recently-seen ids so you don't hammer
Qdrant with existence probes inside a single process.
"""
from __future__ import annotations

import hashlib
import logging
import re
import uuid
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mnemostack.embeddings.base import EmbeddingProvider
from mnemostack.observability.recorder import counter, histogram
from mnemostack.vector import VectorStore

log = logging.getLogger(__name__)


@dataclass
class IngestItem:
    """A single item to ingest.

    `source` and `offset` together produce the deterministic chunk id. Supply
    them when ingesting chunks of a longer document; omit `offset` if each
    item is standalone.
    """
    text: str
    source: str = ""
    offset: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    wrapper_dir: str | Path | None = None


@dataclass
class IngestStats:
    seen: int = 0
    embedded: int = 0
    upserted: int = 0
    skipped: int = 0         # already-seen id, skipped embedding
    failed: int = 0
    wrappers_created: int = 0
    wrappers_updated: int = 0
    ids: list[str] = field(default_factory=list)

    def __iadd__(self, other: IngestStats) -> IngestStats:
        self.seen += other.seen
        self.embedded += other.embedded
        self.upserted += other.upserted
        self.skipped += other.skipped
        self.failed += other.failed
        self.wrappers_created += other.wrappers_created
        self.wrappers_updated += other.wrappers_updated
        self.ids.extend(other.ids)
        return self


def stable_chunk_id(source: str, offset: int, text: str) -> str:
    """Deterministic UUID-5 from an (source, offset, text) triple.

    Same inputs always produce the same id, so upsert replaces itself and
    re-indexing is idempotent. Also exported for callers that want to compute
    ids without going through the Ingestor (e.g. to delete an item).
    """
    digest = hashlib.sha256(f"{source}|{offset}|{text}".encode()).hexdigest()
    return str(uuid.UUID(digest[:32]))


def _item_tags(item: IngestItem) -> list[str]:
    raw_tags = item.tags or item.metadata.get("tags", [])
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    return [str(tag) for tag in raw_tags if str(tag)]


def _wrapper_filename(source: str) -> str:
    source_key = source or "item"
    basename = Path(source_key).stem or Path(source_key).name or "item"
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", basename).strip(".-_") or "item"
    digest = hashlib.sha256(source_key.encode()).hexdigest()[:12]
    return f"{safe_name}-{digest}.md"


def _wrapper_content(item: IngestItem, point_id: str, indexed_date: str) -> str:
    title = Path(item.source).stem or item.source or "Untitled"
    tags = _item_tags(item)
    tags_value = ", ".join(tags) if tags else ""
    summary = item.text[:200]
    return (
        "---\n"
        f"title: {title}\n"
        f"original_path: {item.source}\n"
        f"indexed_date: {indexed_date}\n"
        f"tags: [{tags_value}]\n"
        f"qdrant_point_id: {point_id}\n"
        "---\n\n"
        f"# {title}\n\n"
        f"**Original path:** `{item.source}`\n\n"
        f"**Indexed date:** {indexed_date}\n\n"
        f"**Tags:** {tags_value}\n\n"
        f"**Qdrant point ID:** `{point_id}`\n\n"
        "## Summary\n\n"
        f"{summary}\n"
    )


def _write_wrapper_file(wrapper_dir: Path, item: IngestItem, point_id: str) -> bool:
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    path = wrapper_dir / _wrapper_filename(item.source)
    existed = path.exists()
    indexed_date = datetime.now(timezone.utc).isoformat()
    path.write_text(_wrapper_content(item, point_id, indexed_date), encoding="utf-8")
    return existed


def _sync_wrapper_graph(graph: Any, item: IngestItem, point_id: str) -> None:
    tags = _item_tags(item)
    if not tags:
        return
    indexed_date = datetime.now(timezone.utc).isoformat()
    name = Path(item.source).name or item.source or point_id
    if hasattr(graph, "driver"):
        database = getattr(graph, "database", None)
        query = (
            "MERGE (f:File {path: $path}) "
            "SET f.name = $name, f.indexed_date = $indexed_date, f.point_id = $point_id "
            "WITH f "
            "UNWIND $tags AS tag "
            "MERGE (t:Tag {name: tag}) "
            "MERGE (f)-[:TAGGED]->(t)"
        )
        with graph.driver.session(database=database) as session:
            session.run(
                query,
                name=name,
                path=item.source,
                indexed_date=indexed_date,
                point_id=point_id,
                tags=tags,
            )
        return
    if hasattr(graph, "add_file_tags"):
        graph.add_file_tags(name=name, path=item.source, indexed_date=indexed_date, tags=tags)
        return
    for tag in tags:
        graph.add_triple(
            name,
            "TAGGED",
            tag,
            subject_label="File",
            obj_label="Tag",
            properties={"path": item.source, "indexed_date": indexed_date, "point_id": point_id},
        )


class _SeenCache:
    """Bounded LRU of point ids we've recently upserted in this process.

    A hit means we don't need to re-embed or re-probe Qdrant. The cache is
    soft — if you flush it, correctness is preserved (worst case one extra
    embedding call per item before Qdrant's own upsert-replace wins).
    """
    def __init__(self, max_size: int = 10_000):
        self.max_size = max_size
        self._data: OrderedDict[str, None] = OrderedDict()

    def __contains__(self, key: str) -> bool:
        if key in self._data:
            self._data.move_to_end(key)
            return True
        return False

    def add(self, key: str) -> None:
        if key in self._data:
            self._data.move_to_end(key)
            return
        self._data[key] = None
        if len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


class Ingestor:
    """Batch + streaming ingest into Qdrant (and optional Memgraph sync hook).

    Args:
        embedding: any EmbeddingProvider (Gemini, Ollama, HuggingFace)
        vector_store: an already-configured VectorStore
        batch_size: embed + upsert in batches of this many items
        skip_seen: if True, cache recently-upserted ids and skip re-embedding
            when the same chunk shows up again in the same process
        seen_cache_size: how many ids to keep in the LRU cache
        wrapper_dir: optional directory where markdown wrapper files are written
        graph: optional graph store used to link indexed files to tag nodes

    The ingestor does NOT create the Qdrant collection — call `store.ensure_collection()`
    yourself. This keeps the ingestor cheap to instantiate in servers where
    the collection is set up once at startup.
    """

    def __init__(
        self,
        embedding: EmbeddingProvider,
        vector_store: VectorStore,
        batch_size: int = 64,
        skip_seen: bool = True,
        seen_cache_size: int = 10_000,
        wrapper_dir: str | Path | None = None,
        graph: Any | None = None,
    ):
        self.embedding = embedding
        self.store = vector_store
        self.batch_size = batch_size
        self.skip_seen = skip_seen
        self.wrapper_dir = Path(wrapper_dir) if wrapper_dir is not None else None
        self.graph = graph
        self._seen = _SeenCache(seen_cache_size) if skip_seen else None

    # ---- Public API ----

    def ingest(self, items: Iterable[IngestItem]) -> IngestStats:
        """Ingest a batch of items. Returns aggregate stats.

        Items are chunked into `batch_size` groups for embedding + upsert.
        Safe to call repeatedly with overlapping data; deterministic ids mean
        duplicates upsert onto themselves.
        """
        stats = IngestStats()
        buffer: list[tuple[str, IngestItem]] = []
        for item in items:
            stats.seen += 1
            pid = stable_chunk_id(item.source, item.offset, item.text)
            if self.skip_seen and self._seen is not None and pid in self._seen:
                stats.skipped += 1
                continue
            buffer.append((pid, item))
            if len(buffer) >= self.batch_size:
                self._flush(buffer, stats)
                buffer.clear()
        if buffer:
            self._flush(buffer, stats)
        counter("mnemostack.ingest.items", stats.seen)
        counter("mnemostack.ingest.upserted", stats.upserted)
        counter("mnemostack.ingest.skipped", stats.skipped)
        counter("mnemostack.ingest.failed", stats.failed)
        return stats

    def ingest_one(self, item: IngestItem) -> IngestStats:
        """Convenience: ingest a single item. Same stats shape as `ingest`."""
        return self.ingest([item])

    def stream(self, item_iter: Iterable[IngestItem]) -> Iterator[IngestStats]:
        """Yield an IngestStats per flushed batch — useful for long feeds.

        Callers can log / monitor per-batch progress without waiting for the
        full stream to drain.
        """
        buffer: list[tuple[str, IngestItem]] = []
        total_seen = 0
        for item in item_iter:
            total_seen += 1
            pid = stable_chunk_id(item.source, item.offset, item.text)
            if self.skip_seen and self._seen is not None and pid in self._seen:
                continue
            buffer.append((pid, item))
            if len(buffer) >= self.batch_size:
                batch_stats = IngestStats(seen=len(buffer))
                self._flush(buffer, batch_stats)
                yield batch_stats
                buffer.clear()
        if buffer:
            batch_stats = IngestStats(seen=len(buffer))
            self._flush(buffer, batch_stats)
            yield batch_stats

    # ---- Internals ----

    def _flush(self, buffer: list[tuple[str, IngestItem]], stats: IngestStats) -> None:
        texts = [item.text for _, item in buffer]
        with histogram("mnemostack.ingest.embed_batch_ms"):
            try:
                vectors = self.embedding.embed_batch(texts)
            except AttributeError:
                # Provider without batch API — fall back to single-item
                vectors = [self.embedding.embed(t) for t in texts]
            except Exception as exc:
                log.warning("embed_batch failed (%s) — falling back to per-item", exc)
                vectors = [self._safe_embed_single(t) for t in texts]

        points = []
        for (pid, item), vec in zip(buffer, vectors, strict=False):
            if not vec:
                stats.failed += 1
                continue
            payload = {
                "text": item.text,
                "source": item.source,
                "offset": item.offset,
                **item.metadata,
            }
            tags = _item_tags(item)
            if tags:
                payload["tags"] = tags
            points.append((pid, vec, payload, item))

        if not points:
            return
        stats.embedded += len(points)
        with histogram("mnemostack.ingest.upsert_batch_ms"):
            try:
                self.store.upsert_batch([(pid, vec, payload) for pid, vec, payload, _item in points])
            except AttributeError:
                for pid, vec, payload, _item in points:
                    self.store.upsert(pid, vec, payload)
        stats.upserted += len(points)
        stats.ids.extend(p[0] for p in points)
        self._write_wrappers(points, stats)
        if self._seen is not None:
            for p in points:
                self._seen.add(p[0])

    def _safe_embed_single(self, text: str) -> list[float]:
        try:
            return self.embedding.embed(text)
        except Exception:
            return []

    def _write_wrappers(
        self,
        points: list[tuple[str, list[float], dict[str, Any], IngestItem]],
        stats: IngestStats,
    ) -> None:
        for pid, _vec, _payload, item in points:
            wrapper_dir = item.wrapper_dir or self.wrapper_dir
            if wrapper_dir is not None:
                try:
                    existed = _write_wrapper_file(Path(wrapper_dir), item, pid)
                    if existed:
                        stats.wrappers_updated += 1
                    else:
                        stats.wrappers_created += 1
                except Exception as exc:  # noqa: BLE001
                    log.warning("failed to write markdown wrapper for %s: %s", item.source, exc)
            if self.graph is not None:
                try:
                    _sync_wrapper_graph(self.graph, item, pid)
                except Exception as exc:  # noqa: BLE001
                    log.warning("failed to sync wrapper graph for %s: %s", item.source, exc)


__all__ = ["Ingestor", "IngestItem", "IngestStats", "stable_chunk_id"]
