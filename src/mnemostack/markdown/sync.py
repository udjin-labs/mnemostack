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

from ..quotas import enforce_points_quota
from .indexer import collect_markdown

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .indexer import MarkdownCollection


def _is_within(path: str, base: str) -> bool:
    """True when ``path`` is ``base`` itself or nested under it (no symlink resolve)."""
    try:
        return os.path.commonpath([os.path.abspath(path), base]) == base
    except ValueError:  # different drives on Windows, etc.
        return False


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
    *,
    tenant: str | None = None,
    max_points: int | None = None,
) -> ChunkSyncResult:
    """Embed & upsert new chunks; refresh payloads of already-indexed ones.

    A chunk id already present is not re-embedded (the id is content-derived);
    its payload is refreshed so a frontmatter edit syncs without re-embedding,
    deleting only keys this indexer owned last run (``_md_keys``) that the file
    no longer produces — foreign payload fields (enrichment, validity markers)
    are preserved.

    With ``tenant`` set, writes route through the store's tenant boundary
    (``tenant_id`` stamped server-side; a payload can't assert its own tenant,
    and refresh/delete are owner-guarded) so a markdown corpus is isolated. With
    ``max_points`` set too, the whole call is refused (``QuotaExceededError``)
    before any insert if its NEW chunks would push the tenant over its limit
    (refreshes of already-indexed chunks don't count — they don't grow the count).
    """
    tkw: dict[str, Any] = {"tenant": tenant} if tenant is not None else {}
    existing_ids = set(existing_payloads)
    if tenant is not None and max_points is not None:
        new_ids = {cid for cid, _t, _p in chunks}
        new_sources = {p.get("source") for _c, _t, p in chunks}
        num_new = len(new_ids - existing_ids)
        # A re-index REPLACES a source's chunks: existing markdown-owned chunks of
        # the sources being indexed that aren't in the new set get pruned, so they
        # offset the inserts. Count the NET change, or editing a file at the cap
        # (same chunk count, new ids) would be wrongly rejected as a pure add.
        num_replaced = sum(
            1
            for cid, pl in existing_payloads.items()
            if cid not in new_ids and pl.get("source") in new_sources and pl.get("_md_keys")
        )
        net = num_new - num_replaced
        if net > 0:
            enforce_points_quota(tenant, store.count(tenant=tenant), net, max_points)
    res = ChunkSyncResult()
    for cid, text, payload in chunks:
        if cid in existing_ids:
            continue
        vec = provider.embed(text)
        if not vec:
            res.failed += 1
            res.failed_sources.add(payload["source"])
            continue
        store.upsert(cid, vec, payload, **tkw)
        res.inserted += 1
    for cid, _text, payload in chunks:
        if cid not in existing_ids:
            continue
        old = existing_payloads.get(cid, {})
        owned = old.get("_md_keys") or []
        stale = [k for k in owned if k not in payload]
        if stale:
            store.delete_payload_keys(cid, stale, **tkw)
        store.set_payload(cid, payload, **tkw)
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
        subtree: str | None = None,
        tenant: str | None = None,
        max_points: int | None = None,
    ):
        self.store = store
        self.provider = provider
        self.index_root = str(Path(index_root).resolve())
        self.chunk_size = chunk_size
        self.graph = graph
        # Per-tenant storage quota, enforced on each file's new chunks (see
        # upsert_markdown_chunks). None = no limit.
        self.max_points = max_points
        # Scopes every store + graph read/write and the chunk-id derivation to one
        # tenant, so a markdown corpus is isolated (and tenant-scoped graph recall
        # can see its :File link nodes). None = unscoped (single-tenant), unchanged.
        self.tenant = tenant
        # Only forward tenant= to store/graph calls when set, so an unscoped
        # syncer stays byte-compatible with a store/graph that predates the kwarg.
        self._tkw: dict[str, Any] = {"tenant": tenant} if tenant is not None else {}
        # The watched subtree (may be narrower than index_root under a nested
        # --index-root watch). Referrer re-resolution stays inside it so a
        # sibling outside the watched tree is never rewritten.
        self.subtree = os.path.abspath(subtree) if subtree else self.index_root

    def _prune_markdown_stale(self, fresh_by_source: dict[str, set[str]]) -> int:
        """Prune stale chunks of the given sources — markdown-owned points only.

        Like :func:`prune_stale_chunks`, but skips points without the indexer's
        ``_md_keys`` record, so chunks the generic ``index`` command wrote for the
        same ``(source, index_root)`` are never deleted by the markdown watcher.
        """
        removed = 0
        for source, fresh_ids in fresh_by_source.items():
            stale: list[Any] = []
            for hit in self.store.scroll(
                filters={"source": source, "index_root": self.index_root},
                **self._tkw,
            ):
                payload = hit.payload or {}
                if payload.get("_md_keys") and str(hit.id) not in fresh_ids:
                    stale.append(hit.id)
            if stale:
                removed += self.store.delete_points(stale, **self._tkw)
        return removed

    def source_for(self, path: str | Path) -> str:
        """The corpus-relative source string the indexer stores for ``path``.

        Matches ``indexer._rel``: path relative to index_root, as posix. Uses an
        absolute path WITHOUT resolving symlinks, so a symlinked note keeps the
        link's source name (the one the initial walk indexed), not the target's.
        """
        return Path(os.path.abspath(str(path))).relative_to(self.index_root).as_posix()

    def index_file(self, path: str | Path) -> FileSyncResult:
        """Index/re-index one markdown file. Idempotent; touches only this file.

        When the file is NEW (had no chunks), it may satisfy a dangling
        ``[[wikilink]]`` in another note, so those referrers are re-resolved
        afterwards (see :meth:`_resync_referrers`).
        """
        col = collect_markdown(
            Path(os.path.abspath(str(path))),  # do NOT resolve symlinks
            chunk_size=self.chunk_size,
            index_root=self.index_root,
            root_dir=self.index_root,
            tenant=self.tenant,
        )
        if col.files == 0:
            return FileSyncResult()  # not a .md file / vanished before we read it

        chunks = [(c.id, c.text, c.payload) for c in col.chunks]
        existing: dict[str, dict] = {}
        for source in col.sources:
            for hit in self.store.scroll(
                filters={"index_root": self.index_root, "source": source},
                **self._tkw,
            ):
                existing[str(hit.id)] = hit.payload or {}

        cs = upsert_markdown_chunks(
            self.store, self.provider, chunks, existing,
            tenant=self.tenant, max_points=self.max_points,
        )

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
        pruned = self._prune_markdown_stale(fresh_by_source)

        edges = 0
        if self.graph is not None:
            by_source = build_link_map(col, cs.failed_sources)
            for src, targets in by_source.items():
                edges += self.graph.sync_file_links(
                    src, targets, index_root=self.index_root, **self._tkw
                )
            # A note can satisfy a dangling [[wikilink]] in another note; re-
            # resolve those referrers so their edge points at this note (a full
            # walk would). Run on every successful link-sync, not just first
            # index: a retry after a transient graph failure (chunks already
            # landed) must still heal the referrers. Cheap no-op when nothing
            # dangling matches.
            self._resync_referrers(col.sources, cs.failed_sources)

        first_source = col.sources[0] if col.sources else None
        return FileSyncResult(
            source=first_source,
            inserted=cs.inserted,
            refreshed=cs.refreshed,
            pruned=pruned,
            failed=cs.failed,
            edges=edges,
        )

    @staticmethod
    def _name_keys(rel: str) -> list[str]:
        """The name-based keys a link could use to resolve to ``rel``.

        The note's stem (basename without ``.md``) and its corpus-relative path
        without ``.md`` — mirrors the indexer's ``key_to_rel`` resolution keys.
        These match a dangling ``[[wikilink]]`` node (a bare name) or a same-
        directory inline ``[x](b.md)`` node. A still-dangling cross-directory or
        root-absolute inline link (``[x](../sub/b.md)`` → a ``../sub/b`` node) is
        NOT matched — its node name is source-relative, unknowable from ``rel``
        alone — so it re-resolves only on a full ``index-markdown`` reindex.
        """
        base = rel.rsplit("/", 1)[-1]
        stem = base[:-3] if base.lower().endswith(".md") else base
        rel_key = rel[:-3] if rel.lower().endswith(".md") else rel
        return list({stem, rel_key})

    def _resync_referrers(self, sources: list[str], failed: set[str]) -> None:
        """Re-resolve notes whose dangling links the given (new) sources satisfy."""
        if self.graph is None:
            return
        indexed = set(sources)
        referrers: set[str] = set()
        for src in sources:
            if src in failed:
                continue
            try:
                found = self.graph.referrers_of_dangling(
                    self._name_keys(src), index_root=self.index_root, **self._tkw
                )
            except Exception:  # noqa: BLE001 — best-effort; never fail the index
                continue
            referrers.update(r for r in found if r not in indexed)
        for referrer in referrers:
            path = os.path.join(self.index_root, referrer)
            if not _is_within(path, self.subtree):
                continue  # sibling outside the watched subtree — don't rewrite it
            try:
                self._resync_file_links(path)
            except Exception:  # noqa: BLE001 — a referrer fixup must not fail the new note's index
                continue

    def _resync_file_links(self, path: str | Path) -> None:
        """Re-resolve and re-sync just one file's outgoing graph links (no chunks).

        Used to fix up a referrer after a note it links to is created — cheap
        (no embedding) and does not itself trigger further referrer resolution.
        """
        if self.graph is None or not os.path.isfile(path):
            return
        col = collect_markdown(
            Path(os.path.abspath(str(path))),
            chunk_size=self.chunk_size,
            index_root=self.index_root,
            root_dir=self.index_root,
            tenant=self.tenant,
        )
        for src, targets in build_link_map(col, set()).items():
            self.graph.sync_file_links(
                src, targets, index_root=self.index_root, **self._tkw
            )

    def remove_file(self, path: str | Path) -> FileSyncResult:
        """Drop a deleted file's chunks and clear its outgoing graph links."""
        source = self.source_for(path)
        # Clear the graph edges BEFORE pruning the vector record. If the graph
        # write fails (a transient Memgraph outage), the exception propagates
        # with the vector chunks still present, so reconcile_deletions finds this
        # source again and retries — pruning the vector first would orphan the
        # graph edges with no record left to retry from.
        if self.graph is not None:
            self.graph.sync_file_links(
                source, [], index_root=self.index_root, **self._tkw
            )
        pruned = self._prune_markdown_stale({source: set()})
        return FileSyncResult(source=source, pruned=pruned)

    def reconcile_deletions(self, within: str | Path | None = None) -> list[str]:
        """Remove sources whose file no longer exists on disk. Returns removed.

        A safety net for deletions the incremental event stream can miss: a file
        created and deleted during the initial index (never observed), or a
        directory removed as a single event. ``within`` restricts reconciliation
        to sources under that subtree — the watcher passes the watched folder so
        a narrower ``--index-root`` watch never prunes siblings outside it (the
        one-shot path gates the same reconcile behind ``full_root_walk``).
        """
        scope = os.path.abspath(str(within)) if within is not None else self.index_root
        # Collect the source set fully before removing anything — deleting points
        # mid-scroll would mutate the collection the lazy scroll is iterating.
        # Only markdown-owned sources (payload carries the indexer's `_md_keys`
        # record) are eligible, so a markdown watcher never prunes chunks the
        # generic `index` command wrote under the same collection/root.
        sources: set[str] = set()
        for hit in self.store.scroll(
            filters={"index_root": self.index_root}, **self._tkw
        ):
            payload = hit.payload or {}
            source = payload.get("source")
            if source and payload.get("_md_keys"):
                sources.add(source)
        removed: list[str] = []
        for source in sorted(sources):
            path = os.path.join(self.index_root, source)
            if not _is_within(path, scope):
                continue  # outside the watched subtree — don't touch siblings
            # isfile (not exists): a directory that replaced the note at the same
            # path means the markdown source — a file — is gone. isfile follows
            # symlinks, so a live symlinked note still counts as present.
            if not os.path.isfile(path):
                self.remove_file(path)
                removed.append(source)
        return removed
