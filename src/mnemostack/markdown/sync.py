"""Incremental markdown sync — index or remove a single file into a live index.

The one-shot ``index-markdown`` command walks a whole folder; the watcher
(``--watch``) needs to apply one file at a time as it changes. Both share the
same low-level steps here so a per-file update behaves identically to a full
walk for the file it touches: embed new chunks, refresh changed payloads, prune
chunks the file no longer produces, and re-sync its graph links.

A per-file update is deliberately NOT a full-root reconcile: it touches only the
source(s) it was given, never siblings (mirroring the one-shot command's
``full_root_walk`` gate).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..ingest import prune_stale_chunks
from . import collect_markdown

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .indexer import MarkdownCollection


@dataclass
class ChunkSyncResult:
    inserted: int = 0
    refreshed: int = 0
    failed: int = 0
    failed_sources: set[str] = field(default_factory=set)


def upsert_markdown_chunks(
    store: Any,
    provider: Any,
    chunks: list[tuple[str, str, dict]],
    existing_payloads: dict[str, dict],
) -> ChunkSyncResult:
    """Embed & upsert new chunks; refresh payloads of already-indexed ones.

    A chunk id already present is not re-embedded (the id is content-derived);
    its payload is refreshed so a frontmatter edit syncs without re-embedding,
    deleting only keys this indexer owned last run (``_md_keys``) that the file
    no longer produces — foreign payload fields (enrichment, validity markers)
    are preserved.
    """
    existing_ids = set(existing_payloads)
    res = ChunkSyncResult()
    for cid, text, payload in chunks:
        if cid in existing_ids:
            continue
        vec = provider.embed(text)
        if not vec:
            res.failed += 1
            res.failed_sources.add(payload["source"])
            continue
        store.upsert(cid, vec, payload)
        res.inserted += 1
    for cid, _text, payload in chunks:
        if cid not in existing_ids:
            continue
        old = existing_payloads.get(cid, {})
        owned = old.get("_md_keys") or []
        stale = [k for k in owned if k not in payload]
        if stale:
            store.delete_payload_keys(cid, stale)
        store.set_payload(cid, payload)
        res.refreshed += 1
    return res


def build_link_map(
    col: MarkdownCollection, failed_sources: set[str]
) -> dict[str, list[str]]:
    """Map each (non-failed) source to its de-duplicated outgoing link targets.

    Every visited source is seeded (even with no links) so ``sync_file_links``
    clears stale ``LINKS_TO`` edges of a file whose links were all removed.
    """
    by_source: dict[str, list[str]] = {
        s: [] for s in col.sources if s not in failed_sources
    }
    for edge in col.edges:
        if edge.source in failed_sources:
            continue
        if edge.target not in by_source[edge.source]:
            by_source[edge.source].append(edge.target)
    return by_source


@dataclass
class FileSyncResult:
    source: str | None = None
    inserted: int = 0
    refreshed: int = 0
    pruned: int = 0
    failed: int = 0
    edges: int = 0
    error: str | None = None


class MarkdownSyncer:
    """Applies single-file markdown changes to a live collection (+ graph).

    Holds the collection/embedder/graph so the watcher can call ``index_file`` /
    ``remove_file`` per event. ``graph`` is an optional open ``GraphStore``.
    """

    def __init__(
        self,
        store: Any,
        provider: Any,
        *,
        index_root: str,
        chunk_size: int,
        graph: Any = None,
    ):
        self.store = store
        self.provider = provider
        self.index_root = str(Path(index_root).resolve())
        self.chunk_size = chunk_size
        self.graph = graph

    def source_for(self, path: str | Path) -> str:
        """The corpus-relative source string the indexer stores for ``path``.

        Matches ``indexer._rel``: path relative to index_root, as posix. Uses an
        absolute path WITHOUT resolving symlinks, so a symlinked note keeps the
        link's source name (the one the initial walk indexed), not the target's.
        """
        return Path(os.path.abspath(str(path))).relative_to(self.index_root).as_posix()

    def index_file(self, path: str | Path) -> FileSyncResult:
        """Index/re-index one markdown file. Idempotent; touches only this file."""
        col = collect_markdown(
            Path(os.path.abspath(str(path))),  # do NOT resolve symlinks
            chunk_size=self.chunk_size,
            index_root=self.index_root,
            root_dir=self.index_root,
        )
        if col.files == 0:
            return FileSyncResult()  # not a .md file / vanished before we read it

        chunks = [(c.id, c.text, c.payload) for c in col.chunks]
        existing: dict[str, dict] = {}
        for source in col.sources:
            for hit in self.store.scroll(
                filters={"index_root": self.index_root, "source": source}
            ):
                existing[str(hit.id)] = hit.payload or {}

        cs = upsert_markdown_chunks(self.store, self.provider, chunks, existing)

        # Prune chunks this file no longer produces (shrunk / re-chunked). Skip a
        # source whose embeddings failed — its fresh ids never landed, so pruning
        # would drop live data with no replacement.
        fresh_by_source: dict[str, set[str]] = {
            s: set() for s in col.sources if s not in cs.failed_sources
        }
        for cid, _text, payload in chunks:
            src = payload["source"]
            if src in fresh_by_source:
                fresh_by_source[src].add(cid)
        pruned = prune_stale_chunks(
            self.store, fresh_by_source, index_root=self.index_root
        )

        edges = 0
        if self.graph is not None:
            by_source = build_link_map(col, cs.failed_sources)
            for src, targets in by_source.items():
                edges += self.graph.sync_file_links(
                    src, targets, index_root=self.index_root
                )

        first_source = col.sources[0] if col.sources else None
        return FileSyncResult(
            source=first_source,
            inserted=cs.inserted,
            refreshed=cs.refreshed,
            pruned=pruned,
            failed=cs.failed,
            edges=edges,
        )

    def remove_file(self, path: str | Path) -> FileSyncResult:
        """Drop a deleted file's chunks and clear its outgoing graph links."""
        source = self.source_for(path)
        pruned = prune_stale_chunks(
            self.store, {source: set()}, index_root=self.index_root
        )
        if self.graph is not None:
            self.graph.sync_file_links(source, [], index_root=self.index_root)
        return FileSyncResult(source=source, pruned=pruned)

    def reconcile_deletions(self) -> list[str]:
        """Remove sources under index_root whose file no longer exists on disk.

        A safety net for deletions the incremental event stream can miss: a file
        created and deleted during the initial index (never observed), or a
        directory removed as a single event. Enumerates the indexed sources and
        drops any whose backing file is gone.
        """
        # Collect the source set fully before removing anything — deleting points
        # mid-scroll would mutate the collection the lazy scroll is iterating.
        sources: set[str] = set()
        for hit in self.store.scroll(filters={"index_root": self.index_root}):
            source = (hit.payload or {}).get("source")
            if source:
                sources.add(source)
        removed: list[str] = []
        for source in sorted(sources):
            if not os.path.exists(os.path.join(self.index_root, source)):
                self.remove_file(os.path.join(self.index_root, source))
                removed.append(source)
        return removed
