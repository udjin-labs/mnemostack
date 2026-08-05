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
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mnemostack.embeddings.base import EmbeddingProvider
from mnemostack.embeddings.roles import (
    EMBEDDING_SPACE_KEY,
    EmbeddingSpaceError,
    SpaceGuard,
    document_space_fingerprint_via,
    embed_documents_resilient,
)
from mnemostack.observability.recorder import counter, histogram
from mnemostack.quotas import enforce_points_quota
from mnemostack.vector import VectorStore

DEFAULT_WINDOW_SEPARATOR = "\n"

log = logging.getLogger(__name__)


@dataclass
class IngestItem:
    """A single item to ingest.

    `source` and `offset` together produce the deterministic chunk id. Supply
    them when ingesting chunks of a longer document; omit `offset` if each
    item is standalone.

    `timestamp` is the event time of the content (ISO-8601) — when the message
    was said or the note was written. It lands in `payload["timestamp"]` and
    drives temporal recall; without it, temporal questions cannot be answered
    for this chunk. Passing it via `metadata={"timestamp": ...}` still works;
    the explicit field wins when both are set.
    """

    text: str
    source: str = ""
    offset: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    wrapper_dir: str | Path | None = None
    timestamp: str | None = None


@dataclass
class IngestStats:
    seen: int = 0
    embedded: int = 0
    upserted: int = 0
    skipped: int = 0  # already-seen id, skipped embedding
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


def stable_chunk_id(source: str, offset: int, text: str, *, tenant: str | None = None) -> str:
    """Deterministic UUID-5 from an (source, offset, text) triple.

    Same inputs always produce the same id, so upsert replaces itself and
    re-indexing is idempotent. Also exported for callers that want to compute
    ids without going through the Ingestor (e.g. to delete an item).

    ``tenant`` scopes the id: two tenants ingesting the *same* (source, offset,
    text) into one collection get **different** ids, so one can't overwrite the
    other's point (a full-point upsert would otherwise destroy it). ``tenant``
    is prefixed so ``tenant=None`` reproduces the historical id exactly — legacy
    single-tenant ids are unchanged.
    """
    base = f"{source}|{offset}|{text}"
    if tenant is not None:
        base = f"{tenant}\x00{base}"
    digest = hashlib.sha256(base.encode()).hexdigest()
    return str(uuid.UUID(digest[:32]))


def _item_tags(item: IngestItem) -> list[str]:
    raw_tags = item.tags or item.metadata.get("tags", [])
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    return [str(tag) for tag in raw_tags if str(tag)]


# Keys an enricher may never override: text/source/offset feed
# stable_chunk_id, index_root scopes pruning, tenant_id is the isolation
# boundary (only the Ingestor's `tenant` may set it — see _flush), and the
# provenance snapshot pair backs `mnemostack resolve` verdicts (a fabricated
# hash/capture-time would corrupt citation verification).
_PROTECTED_PAYLOAD_KEYS = frozenset(
    {
        "text",
        "source",
        "offset",
        "index_root",
        "tenant_id",
        "source_content_hash",
        "source_captured_at",
        # Structural resolver keys: the windowed-point marker and the
        # id-scheme marker decide `mnemostack resolve` verdict paths.
        "_id_scheme",
        # Document-space fingerprint: identifies the embedding space the
        # point's vector belongs to — a planted value would defeat the
        # mixed-space guard.
        "_embedding_space",
        "chunk_kind",
        "chunk_window",
        "chunk_start_offset",
        "chunk_end_offset",
        "synthetic_prefix_len",
    }
)


def apply_enrichment(
    enrich: Callable[[IngestItem], dict[str, Any]] | None,
    item: IngestItem,
    payload: dict[str, Any],
) -> None:
    """Merge an enricher's output into *payload*, fail-open.

    The enricher is user-supplied (dates, amounts, entities — content
    extraction is corpus-specific and stays out of core); a raising or
    misbehaving hook logs a warning and the item is indexed without
    enrichment. Protected keys and an explicit item timestamp are never
    overridden; for other keys the enricher wins over `metadata` (it runs
    later in the pipeline).
    """
    if enrich is None:
        return
    try:
        extra = enrich(item)
    except Exception as exc:  # noqa: BLE001 — user hook must not break ingest
        counter("mnemostack.ingest.enrich_failed", 1)
        log.warning(
            "enrich hook failed for %s (%s) — indexing without enrichment",
            item.source,
            exc,
        )
        return
    if not isinstance(extra, dict):
        counter("mnemostack.ingest.enrich_failed", 1)
        log.warning(
            "enrich hook returned %s for %s — expected dict; ignored",
            type(extra).__name__,
            item.source,
        )
        return
    applied: list[str] = []
    for key, value in extra.items():
        if key in _PROTECTED_PAYLOAD_KEYS:
            continue
        if key == "timestamp" and item.timestamp:
            continue  # the explicit item timestamp is authoritative
        payload[key] = value
        applied.append(key)
    if applied:
        # Ownership record: which payload keys the enricher wrote. Payload
        # refresh uses it to delete keys a newer enricher no longer
        # produces, without touching fields written by other ingest paths.
        payload["_enrich_keys"] = sorted(applied)


def prune_stale_chunks(
    vector_store: VectorStore,
    fresh_ids_by_source: dict[str, set[str]],
    *,
    index_root: str | None = None,
    tenant: str | None = None,
) -> int:
    """Delete stale chunks of re-indexed sources. Returns count removed.

    For each source in *fresh_ids_by_source*, deletes points whose payload
    ``source`` matches but whose id is not in the fresh set — i.e. chunks
    that the source no longer produces (content edits shifted offsets, the
    document shrank, chunking parameters changed).

    Only the listed sources are touched; points ingested under other sources
    are never affected. The fresh set MUST contain every id the source
    currently produces (compute it via `stable_chunk_id` over all chunks,
    including ones skipped as already indexed) — an incomplete set would
    delete live data. Likewise, do NOT include a source whose chunks failed
    to embed or upsert in the current run: its fresh ids never landed, so
    pruning would delete the previous data without a replacement.

    Pass *index_root* when source names are relative and may collide across
    indexing roots (the CLI stores the resolved root as ``index_root`` in the
    payload): the delete is then scoped to points carrying the same root, so
    ``note.md`` from another root — or from a version that didn't record a
    root — is never touched.

    Pass *tenant* in a multi-tenant collection so only that tenant's points are
    considered for deletion — otherwise another tenant's same-``source`` chunks
    (which won't be in this tenant's fresh set) would be pruned as "stale".
    """
    # Only pass tenant when set, so a custom store without the parameter (and
    # the single-tenant path) is unaffected.
    tkw: dict[str, Any] = {"tenant": tenant} if tenant is not None else {}
    removed = 0
    for source, fresh_ids in fresh_ids_by_source.items():
        filters: dict[str, Any] = {"source": source}
        if index_root is not None:
            filters["index_root"] = index_root
        stale = [
            pid
            for pid in (str(p) for p in vector_store.iter_ids(filters=filters, **tkw))
            if pid not in fresh_ids
        ]
        if stale:
            removed += vector_store.delete_points(list(stale))
    counter("mnemostack.ingest.pruned", removed)
    return removed


#: Fallback discovery switches from narrow per-source indexed scans to one
#: root-scoped scroll above this many re-indexed sources. A selective
#: re-index (one file, a small subtree) costs a couple of cheap filtered
#: scans; paging every point of a large root for it would be orders of
#: magnitude more traffic. A bulk walk is the opposite: O(sources) scans
#: dwarf O(collection pages). The break-even depends on point counts the
#: client can't see, so a small constant keeps the worst case of either
#: mode bounded.
SELECTIVE_PRUNE_MAX_SOURCES = 16


def prune_stale_chunks_from_snapshot(
    vector_store: VectorStore,
    fresh_ids_by_source: dict[str, set[str]],
    existing: Iterable[tuple[str | int, dict[str, Any]]] | None = None,
    *,
    index_root: str | None = None,
    tenant: str | None = None,
    delete_batch_size: int = 256,
    selective_scan_max_sources: int = SELECTIVE_PRUNE_MAX_SOURCES,
) -> int:
    """Delete stale chunks of re-indexed sources from ONE point snapshot.

    Same contract as :func:`prune_stale_chunks` — only the listed sources are
    touched, each fresh set MUST be complete, and a source whose chunks failed
    to embed must not be listed — but discovery costs one pass instead of one
    filtered scan per source: over *existing*, an ``(id, payload)`` snapshot
    the caller already holds (both CLI paths scroll the collection's payloads
    anyway), or — when *existing* is None — over the store, adaptively: up to
    *selective_scan_max_sources* re-indexed sources keep the narrow per-source
    indexed scans (a one-file re-index into a large root must not page the
    whole root), a bulk map uses a single root-scoped scroll. Request count is
    O(min(sources, collection pages)), never unconditionally O(sources).

    Scoping mirrors the per-source filters exactly: with *index_root*, only
    points whose payload records the same root are considered, so a point from
    another root — or one with no recorded root — is never touched. The
    snapshot must already be confined to *tenant* (the callers load it through
    a tenant-scoped scroll); the stale ids are nevertheless re-validated by
    the tenant-aware delete, so even a wrongly-scoped snapshot cannot delete a
    foreign tenant's point.

    Snapshot semantics: a point created after the snapshot was taken is not
    seen and never deleted — this run's own upserts are exactly the fresh ids,
    and a concurrent writer's new points are left alone (the old live re-scan
    would have seeded a concurrently-created source for deletion on a
    full-root walk; working from the snapshot closes that hazard and keeps the
    prune consistent with the quota estimate computed from the same snapshot).
    """
    tkw: dict[str, Any] = {"tenant": tenant} if tenant is not None else {}
    if not fresh_ids_by_source:
        # Nothing can match — and the per-source implementation issued zero
        # scans here, so the fallback scroll must not run either.
        counter("mnemostack.ingest.prune_points_scanned", 0)
        counter("mnemostack.ingest.prune_delete_batches", 0)
        counter("mnemostack.ingest.pruned", 0)
        return 0
    scanned = 0
    removed = 0
    batches = 0
    stale: list[str | int] = []

    def _flush() -> None:
        nonlocal removed, batches
        if stale:
            removed += vector_store.delete_points(list(stale), **tkw)
            batches += 1
            stale.clear()

    if (
        existing is None
        and hasattr(vector_store, "iter_ids")
        and (
            not hasattr(vector_store, "scroll")
            or len(fresh_ids_by_source) <= selective_scan_max_sources
        )
    ):
        # Selective re-index (or a custom store exposing only iter_ids):
        # the narrow server-indexed scans, one per source — exactly the
        # historical discovery, but with bounded deletes, tenant-validated
        # removal and the scan counters of the snapshot path. A store
        # without iter_ids always takes the scroll path below, whatever
        # the map size.
        for source, fresh_ids in fresh_ids_by_source.items():
            filters: dict[str, Any] = {"source": source}
            if index_root is not None:
                filters["index_root"] = index_root
            for pid in (str(p) for p in vector_store.iter_ids(filters=filters, **tkw)):
                scanned += 1
                if pid not in fresh_ids:
                    stale.append(pid)
                    if len(stale) >= delete_batch_size:
                        _flush()
    else:
        if existing is None:
            root_filter = (
                {"index_root": index_root} if index_root is not None else None
            )
            existing = (
                (hit.id, hit.payload or {})
                for hit in vector_store.scroll(filters=root_filter, **tkw)
            )
        for point_id, payload in existing:
            scanned += 1
            src = payload.get("source")
            # A non-string source can never equal a fresh-map key (the
            # per-source MatchValue filter wouldn't have matched it either) —
            # and an unhashable one must not crash the membership test.
            if not isinstance(src, str) or src not in fresh_ids_by_source:
                continue
            if index_root is not None and payload.get("index_root") != index_root:
                continue
            if str(point_id) in fresh_ids_by_source[src]:
                continue
            stale.append(point_id)
            if len(stale) >= delete_batch_size:
                _flush()
    _flush()
    counter("mnemostack.ingest.prune_points_scanned", scanned)
    counter("mnemostack.ingest.prune_delete_batches", batches)
    counter("mnemostack.ingest.pruned", removed)
    return removed


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


def _accepts_kw(fn: Any, name: str) -> bool:
    """Whether ``fn`` accepts a keyword arg ``name`` (or **kwargs).

    Lets us thread ``tenant`` into a duck-typed graph adapter only when its
    signature supports it, so a legacy adapter isn't broken by an unexpected kwarg.
    """
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return False
    if name in params:
        return True
    return any(p.kind is p.VAR_KEYWORD for p in params.values())


def _sync_wrapper_graph(
    graph: Any, item: IngestItem, point_id: str, *, tenant: str | None = None
) -> None:
    tags = _item_tags(item)
    if not tags:
        return
    indexed_date = datetime.now(timezone.utc).isoformat()
    name = Path(item.source).name or item.source or point_id
    if hasattr(graph, "driver"):
        database = getattr(graph, "database", None)
        # Fold the tenant into the File/Tag node key and stamp it on the TAGGED
        # edge, so a scoped graph recall (which pins nodes AND edges to `tenant`)
        # traverses these. Unscoped keeps the legacy path-keyed write untouched.
        tk = ", tenant: $tenant" if tenant is not None else ""
        if tenant is not None:
            file_set = (
                "SET f.name = $name, f.indexed_date = $indexed_date, "
                "f.point_id = $point_id, f.tenant = $tenant "
            )
            # Fold the tenant into the TAGGED MERGE key so a scoped wrapper write
            # only matches/creates its own edge — never claims a foreign-tenant
            # TAGGED edge between these nodes (round-7 relationship-key pattern).
            tagged = "MERGE (f)-[r:TAGGED {tenant: $tenant}]->(t)"
        else:
            # Unscoped: the path-key subset-matches a tenant-owned :File node after
            # migration, so only write metadata when the node is tenant-less — an
            # unscoped wrapper ingest must not overwrite a tenant's point_id/date.
            # On a single-tenant graph the node has no tenant, so this always runs.
            file_set = (
                "FOREACH (_ IN CASE WHEN f.tenant IS NULL THEN [1] ELSE [] END | "
                "SET f.name = $name, f.indexed_date = $indexed_date, f.point_id = $point_id) "
            )
            tagged = "MERGE (f)-[r:TAGGED]->(t)"
        query = (
            f"MERGE (f:File {{path: $path{tk}}}) "
            f"{file_set}"
            "WITH f "
            "UNWIND $tags AS tag "
            f"MERGE (t:Tag {{name: tag{tk}}}) "
            f"{tagged}"
        )
        params: dict[str, Any] = {
            "name": name,
            "path": item.source,
            "indexed_date": indexed_date,
            "point_id": point_id,
            "tags": tags,
        }
        if tenant is not None:
            params["tenant"] = tenant
        with graph.driver.session(database=database) as session:
            session.run(query, **params)
        return
    if hasattr(graph, "add_file_tags") and (tenant is None or _accepts_kw(graph.add_file_tags, "tenant")):
        # Use the adapter's own hook, threading tenant only when it accepts it.
        fkw: dict[str, Any] = {"tenant": tenant} if tenant is not None else {}
        graph.add_file_tags(
            name=name, path=item.source, indexed_date=indexed_date, tags=tags, **fkw
        )
        return
    # Fallback (and: tenant set but add_file_tags can't scope it) → add_triple,
    # which threads tenant, so the tags land in the tenant's subgraph rather than
    # unscoped. Only pass tenant= when set, so a legacy add_triple signature (no
    # tenant kwarg) doesn't TypeError and silently drop tags in single-tenant use.
    tkw: dict[str, Any] = {"tenant": tenant} if tenant is not None else {}
    for tag in tags:
        graph.add_triple(
            name,
            "TAGGED",
            tag,
            subject_label="File",
            obj_label="Tag",
            properties={"path": item.source, "indexed_date": indexed_date, "point_id": point_id},
            **tkw,
        )


def _window_items(
    items: Sequence[IngestItem],
    window_size: int,
    separator: str = DEFAULT_WINDOW_SEPARATOR,
) -> list[IngestItem]:
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if window_size == 1 or len(items) < window_size:
        return list(items)

    expanded = list(items)
    group_start = 0
    while group_start < len(items):
        source = items[group_start].source
        group_end = group_start + 1
        while group_end < len(items) and items[group_end].source == source:
            group_end += 1

        group = items[group_start:group_end]
        if len(group) >= window_size:
            for start in range(0, len(group) - window_size + 1):
                window = group[start : start + window_size]
                expanded.append(_make_window_item(window, window_size, separator))
        group_start = group_end

    return expanded


def _effective_ts(item: IngestItem) -> str | None:
    return item.timestamp or item.metadata.get("timestamp")


def _make_window_item(
    window: Sequence[IngestItem],
    window_size: int,
    separator: str,
) -> IngestItem:
    middle = window[window_size // 2]
    metadata = dict(middle.metadata)
    metadata.update(
        {
            "chunk_window": window_size,
            "chunk_kind": "sliding_window",
            "chunk_start_offset": window[0].offset,
            "chunk_end_offset": window[-1].offset,
        }
    )
    # A window can span sessions; keep the full temporal range alongside the
    # middle item's timestamp so range-aware retrieval stays possible.
    start_ts = _effective_ts(window[0])
    end_ts = _effective_ts(window[-1])
    if start_ts:
        metadata["window_start_ts"] = start_ts
    if end_ts:
        metadata["window_end_ts"] = end_ts
    return IngestItem(
        text=separator.join(item.text for item in window),
        source=middle.source,
        offset=middle.offset,
        metadata=metadata,
        tags=list(middle.tags),
        wrapper_dir=middle.wrapper_dir,
        timestamp=middle.timestamp,
    )


def _iter_window_items(
    items: Iterable[IngestItem],
    window_size: int,
    separator: str = DEFAULT_WINDOW_SEPARATOR,
) -> Iterator[IngestItem]:
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if window_size == 1:
        yield from items
        return

    window: deque[IngestItem] = deque(maxlen=window_size)
    current_source: str | None = None
    for item in items:
        if item.source != current_source:
            window.clear()
            current_source = item.source
        yield item
        window.append(item)
        if len(window) == window_size:
            yield _make_window_item(list(window), window_size, separator)


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
        window_size: number of adjacent items to concatenate into overlapping
            context chunks. 1 preserves the current one-item-per-chunk behavior.
        enrich: optional payload enricher `callable(IngestItem) -> dict`,
            called for every final item (including assembled window chunks);
            the returned dict is merged into the payload. Fail-open: a
            raising hook logs a warning and the item is indexed without
            enrichment. See `apply_enrichment` for override rules.

    The ingestor does NOT create the Qdrant collection — call `store.ensure_collection()`
    yourself. This keeps the ingestor cheap to instantiate in servers where
    the collection is set up once at startup.

    Writes are space-guarded: every flush revalidates (TTL-bounded) that the
    collection's stamped embedding space matches this provider and stamps
    each point with the provider's document-space fingerprint. A conflict
    raises `EmbeddingSpaceError` instead of writing mixed-space vectors.
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
        window_size: int = 1,
        window_separator: str = DEFAULT_WINDOW_SEPARATOR,
        enrich: Callable[[IngestItem], dict[str, Any]] | None = None,
        tenant: str | None = None,
        max_points: int | None = None,
    ):
        self.embedding = embedding
        self.store = vector_store
        # When set, every ingested point is stamped with this tenant_id (the
        # write side of the multi-tenant isolation boundary). None = single-tenant.
        self.tenant = tenant
        # Per-tenant storage quota: refuse to flush a batch whose NEW points would
        # push this tenant over max_points. Enforced only when both tenant and
        # max_points are set (single-tenant / no-quota ingest is unaffected).
        # Precise on re-ingest: each flush checks the tenant's current count plus
        # only the batch ids not already stored (deterministic ids upsert onto
        # themselves and don't grow the count), so re-indexing a tenant at its
        # limit isn't falsely rejected. NOTE: an ingest spanning multiple flushes
        # commits earlier batches before a later one raises — an over-quota stream
        # is a partial write, and the caller loses the IngestStats on the raise.
        self.max_points = max_points
        self.batch_size = batch_size
        self.skip_seen = skip_seen
        self.wrapper_dir = Path(wrapper_dir) if wrapper_dir is not None else None
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.graph = graph
        self.window_size = window_size
        self.window_separator = window_separator
        self.enrich = enrich
        self._seen = _SeenCache(seen_cache_size) if skip_seen else None
        # Self-guarding writes: every flush revalidates UNCONDITIONALLY
        # (recheck_seconds=0, same policy as the markdown sync) that the
        # collection's stamped space matches this provider before any
        # embedding/upsert. Write-side staleness is not acceptable even
        # within a TTL: a repointed tag inside the window would stamp fresh
        # fingerprints next to old-space points and corrupt the collection
        # BEFORE any read-side revalidation could notice.
        self._space_guard = SpaceGuard(vector_store, embedding, recheck_seconds=0.0, fail_closed=True)

    # ---- Public API ----

    def ingest(self, items: Iterable[IngestItem]) -> IngestStats:
        """Ingest a batch of items. Returns aggregate stats.

        Items are chunked into `batch_size` groups for embedding + upsert.
        Safe to call repeatedly with overlapping data; deterministic ids mean
        duplicates upsert onto themselves.
        """
        stats = IngestStats()
        buffer: list[tuple[str, IngestItem]] = []
        for item in _iter_window_items(items, self.window_size, self.window_separator):
            stats.seen += 1
            pid = stable_chunk_id(item.source, item.offset, item.text, tenant=self.tenant)
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

    async def ingest_async(self, items: Iterable[IngestItem]) -> IngestStats:
        """Async wrapper around `ingest`.

        Runs the blocking work (embedding HTTP, Qdrant upserts, wrapper-file
        writes, optional graph sync) in a worker thread so asyncio services
        are not blocked. The items iterable is consumed inside that thread.

        Concurrency caveat: the skip-seen cache is per-instance and not
        synchronized — gather concurrent ingests on *separate* Ingestor
        instances, or run one ingest at a time per instance.
        """
        import asyncio

        return await asyncio.to_thread(self.ingest, items)

    async def ingest_one_async(self, item: IngestItem) -> IngestStats:
        """Async convenience: ingest a single item (see `ingest_async`)."""
        import asyncio

        return await asyncio.to_thread(self.ingest, [item])

    def stream(self, item_iter: Iterable[IngestItem]) -> Iterator[IngestStats]:
        """Yield an IngestStats per flushed batch — useful for long feeds.

        Callers can log / monitor per-batch progress without waiting for the
        full stream to drain.
        """
        buffer: list[tuple[str, IngestItem]] = []
        total_seen = 0
        for item in _iter_window_items(item_iter, self.window_size, self.window_separator):
            total_seen += 1
            pid = stable_chunk_id(item.source, item.offset, item.text, tenant=self.tenant)
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

    def _check_points_quota(self, point_ids: list[str]) -> None:
        """Raise QuotaExceededError if this batch's genuinely-new points would push
        the tenant over its point limit. No-op when unscoped or no limit set.

        Only ids NOT already stored count — a re-ingest upserts deterministic ids
        onto themselves and doesn't grow the tenant's footprint, so it's never
        falsely rejected (the store's current count already includes them).
        """
        if self.tenant is None or self.max_points is None:
            return
        # Tenant-scoped: an unowned/legacy id this tenant is about to adopt counts
        # as new growth (the scoped count() doesn't include it yet), so it can't
        # slip past the cap.
        existing = self.store.retrieve_existing_ids(list(point_ids), tenant=self.tenant)
        # UNIQUE new ids: a duplicated item in one flush (same id) upserts to one
        # point, so it must count once, not per occurrence.
        new = len({str(pid) for pid in point_ids} - existing)
        enforce_points_quota(
            self.tenant, self.store.count(tenant=self.tenant), new, self.max_points
        )

    def _flush(self, buffer: list[tuple[str, IngestItem]], stats: IngestStats) -> None:
        # Guard BEFORE embedding (raises EmbeddingSpaceError on conflict) and
        # stamp EXACTLY the fingerprint the guard validated — one resolution,
        # so a tag repointed between "check" and "stamp" cannot pass the
        # guard under space A and label the points space B. The fallback
        # resolution only runs for an unguardable pair (store without
        # scroll), where no verdict exists to race against.
        doc_space_fp = self._space_guard.ensure()
        if doc_space_fp is None:
            doc_space_fp = document_space_fingerprint_via(self.embedding)
        texts = [item.text for _, item in buffer]
        with histogram("mnemostack.ingest.embed_batch_ms"):
            # Shared degradation ladder (native batch → per-item on missing
            # batch API → guarded per-item on batch exceptions) — identical
            # semantics for Ingestor and the CLI/markdown group loops.
            vectors = embed_documents_resilient(self.embedding, texts)
        # SANDWICH (same policy as the CLI/markdown paths): the fingerprint
        # must still be the guarded one AFTER embedding — a tag repointed
        # during the embed call must not have its vectors stamped as the
        # pre-repoint space.
        if doc_space_fp is not None:
            current_fp = document_space_fingerprint_via(self.embedding)
            if current_fp != doc_space_fp:
                raise EmbeddingSpaceError(
                    "embedding space changed mid-flush (the model tag was "
                    "repointed) — aborting before any mixed-space write"
                )

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
            if item.timestamp:
                payload["timestamp"] = item.timestamp
            apply_enrichment(self.enrich, item, payload)
            # tenant_id is set only by the store from the Ingestor's `tenant`
            # (the write-side of the isolation boundary) — never by metadata or
            # an enrich hook. Drop any planted value so it can't be spoofed.
            payload.pop("tenant_id", None)
            # Same rule as tenant_id: the space stamp is set ONLY by this
            # pipeline. Dropped unconditionally so a caller-supplied value
            # can't survive when the provider is a duck type (no fingerprint
            # to overwrite it) and later forge space membership.
            payload.pop(EMBEDDING_SPACE_KEY, None)
            if doc_space_fp is not None:
                payload[EMBEDDING_SPACE_KEY] = doc_space_fp
            payload.setdefault("indexed_at", datetime.now(timezone.utc).isoformat())
            tags = _item_tags(item)
            if tags:
                payload["tags"] = tags
            points.append((pid, vec, payload, item))

        if not points:
            return
        stats.embedded += len(points)
        # Enforce the tenant's storage quota BEFORE upserting: raise (aborting the
        # ingest) if this batch's NEW points would push the tenant over its limit.
        self._check_points_quota([p[0] for p in points])
        # Only pass tenant when set, so a custom store without the parameter
        # (and the existing single-tenant path) is unaffected.
        tkw: dict[str, Any] = {"tenant": self.tenant} if self.tenant is not None else {}
        with histogram("mnemostack.ingest.upsert_batch_ms"):
            try:
                self.store.upsert_batch(
                    [(pid, vec, payload) for pid, vec, payload, _item in points],
                    **tkw,
                )
            except AttributeError:
                for pid, vec, payload, _item in points:
                    self.store.upsert(pid, vec, payload, **tkw)
        stats.upserted += len(points)
        stats.ids.extend(p[0] for p in points)
        # POST-COMMIT revalidation: Qdrant has no atomic "claim an empty
        # collection" primitive, so two processes bootstrapping the same
        # empty collection under different spaces could both pass the empty
        # pre-check. Re-sampling AFTER the write sees the other writer's
        # stamps and fails loud within the same flush — the exposure is
        # bounded to one interleaved batch per process, and every later
        # write is refused by the normal mismatch verdict.
        self._space_guard.ensure()
        self._write_wrappers(points, stats)
        if self._seen is not None:
            for p in points:
                self._seen.add(p[0])

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
                    _sync_wrapper_graph(self.graph, item, pid, tenant=self.tenant)
                except Exception as exc:  # noqa: BLE001
                    log.warning("failed to sync wrapper graph for %s: %s", item.source, exc)


__all__ = [
    "Ingestor",
    "IngestItem",
    "IngestStats",
    "prune_stale_chunks",
    "prune_stale_chunks_from_snapshot",
    "stable_chunk_id",
]
