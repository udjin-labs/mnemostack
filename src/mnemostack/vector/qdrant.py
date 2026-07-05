"""Qdrant vector store wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    DatetimeRange,
    Distance,
    FieldCondition,
    Filter,
    IsEmptyCondition,
    MatchValue,
    PayloadField,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    Range,
    VectorParams,
)

#: Payload key marking a fact as stale (system-time). Kept as a literal here so
#: the vector layer doesn't depend on the recall layer; mirrors
#: ``recall.validity.INVALIDATED_AT``.
_INVALIDATED_AT_KEY = "invalidated_at"


def _hide_invalidated_condition() -> IsEmptyCondition:
    """Qdrant predicate for a "current" fact — one with no ``invalidated_at``.

    This is the server-side push-down of the default validity view. It matches
    ``recall.validity.is_current`` for every marker ``VectorStore.invalidate``
    writes: invalidate always stamps a real UTC timestamp, so the only value that
    ``is_current`` treats as current but ``IsEmpty`` would not — a literal empty
    string — is never produced. Point-in-time (``as_of``) recall is deliberately
    NOT pushed down: its ``valid_from``/``valid_until`` bounds may be bare dates,
    and a server-side range that mis-orders those would silently drop valid facts
    (unrecoverable past the client-side filter), so it stays in Python.
    """
    return IsEmptyCondition(is_empty=PayloadField(key=_INVALIDATED_AT_KEY))


class DimensionMismatchError(ValueError):
    """Existing collection stores vectors of a different size than the provider produces."""


@dataclass
class Hit:
    """Single search result."""

    id: str | int
    score: float
    payload: dict[str, Any]


class VectorStore:
    """Thin wrapper around QdrantClient with typed search results.

    Handles collection setup, indexing payload fields for filtering, and
    consistent Hit dataclass output.
    """

    def __init__(
        self,
        collection: str,
        dimension: int,
        host: str = "http://localhost:6333",
        distance: Distance = Distance.COSINE,
        timeout: int = 30,
    ):
        self.collection = collection
        self.dimension = dimension
        self.distance = distance
        self.client = QdrantClient(url=host, timeout=timeout)

    # ---------- collection management ----------

    def collection_exists(self) -> bool:
        try:
            self.client.get_collection(self.collection)
            return True
        except ValueError as exc:
            if "not found" in str(exc).lower():
                return False
            raise
        except UnexpectedResponse as exc:
            if exc.status_code == 404:
                return False
            raise

    def ensure_collection(self, recreate: bool = False) -> bool:
        """Create collection if missing. If recreate=True, drop and recreate.

        Raises DimensionMismatchError when the collection already exists but
        stores vectors of a different size than this store's `dimension` —
        searching such a collection silently returns garbage otherwise.
        """
        exists = self.collection_exists()
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dimension, distance=self.distance),
            )
            return True
        self._validate_dimension()
        return False

    def _validate_dimension(self) -> None:
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        size = getattr(vectors, "size", None)  # named-vectors dict has no .size — skip
        if size is not None and int(size) != int(self.dimension):
            raise DimensionMismatchError(
                f"Collection '{self.collection}' stores {size}-dim vectors but the "
                f"embedding provider produces {self.dimension}-dim vectors. "
                f"Re-index with --recreate or switch the embedding model."
            )

    def index_payload_field(self, field: str, schema: PayloadSchemaType) -> None:
        """Create a payload index for filtering (e.g. timestamp as DATETIME)."""
        try:
            self.client.create_payload_index(
                collection_name=self.collection, field_name=field, field_schema=schema
            )
        except Exception:  # noqa: BLE001
            pass  # already indexed or collection not ready

    def count(self) -> int:
        info = self.client.get_collection(self.collection)
        return info.points_count or 0

    def delete(self) -> None:
        self.client.delete_collection(self.collection)

    # ---------- iteration ----------

    def scroll(
        self,
        batch_size: int = 256,
        filters: dict[str, Any] | None = None,
        with_vectors: bool = False,
    ):
        """Iterate over ALL points in the collection lazily.

        Memory-efficient: never loads the whole collection at once. Good for:
        - Re-indexing after schema changes
        - Bulk export / migration
        - Aggregation over entire corpus

        Yields `Hit` objects (score=1.0 since this isn't a similarity query).
        """
        qfilter = self._build_filter(filters) if filters else None
        next_offset: Any = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection,
                limit=batch_size,
                offset=next_offset,
                with_payload=True,
                with_vectors=with_vectors,
                scroll_filter=qfilter,
            )
            if not points:
                break
            for pt in points:
                pid = str(pt.id) if isinstance(pt.id, UUID) else pt.id
                yield Hit(id=pid, score=1.0, payload=pt.payload or {})
            if next_offset is None:
                break

    def iter_ids(
        self,
        batch_size: int = 1024,
        filters: dict[str, Any] | None = None,
    ):
        """Lightweight iteration returning only point IDs. Faster than scroll()."""
        qfilter = self._build_filter(filters) if filters else None
        next_offset: Any = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection,
                limit=batch_size,
                offset=next_offset,
                with_payload=False,
                with_vectors=False,
                scroll_filter=qfilter,
            )
            if not points:
                break
            for pt in points:
                yield pt.id
            if next_offset is None:
                break

    # ---------- write ----------

    def upsert(
        self,
        id: str | int,
        vector: list[float],
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=id, vector=vector, payload=payload or {})],
        )

    def upsert_batch(
        self,
        points: list[tuple[str | int, list[float], dict[str, Any]]],
        batch_size: int = 100,
    ) -> int:
        """Upsert a batch of (id, vector, payload) tuples. Returns count inserted."""
        total = 0
        for i in range(0, len(points), batch_size):
            chunk = points[i : i + batch_size]
            structs = [PointStruct(id=pid, vector=vec, payload=pl or {}) for pid, vec, pl in chunk]
            self.client.upsert(collection_name=self.collection, points=structs)
            total += len(structs)
        return total

    def set_payload(self, id: str | int, payload: dict[str, Any]) -> None:
        """Merge payload keys into an existing point without re-embedding.

        The vector is untouched — this is the cheap path for applying new
        payload fields (enrichment output, index_root) to already-indexed
        points. Merge semantics: keys not present in *payload* are kept.
        """
        self.client.set_payload(
            collection_name=self.collection,
            payload=payload,
            points=[id],
        )

    def delete_payload_keys(self, id: str | int, keys: list[str]) -> None:
        """Remove specific payload keys from a point (vector untouched)."""
        if not keys:
            return
        self.client.delete_payload(
            collection_name=self.collection,
            keys=keys,
            points=[id],
        )

    def delete_points(self, ids: list[str | int], batch_size: int = 1000) -> int:
        """Delete specific points by id. Returns count requested for deletion."""
        total = 0
        for i in range(0, len(ids), batch_size):
            chunk = ids[i : i + batch_size]
            self.client.delete(
                collection_name=self.collection,
                points_selector=PointIdsList(points=list(chunk)),
            )
            total += len(chunk)
        return total

    def invalidate(
        self,
        ids: str | int | list[str | int],
        *,
        invalidated_at: str | None = None,
        valid_until: str | None = None,
        index_root: str | None = None,
    ) -> int:
        """Mark chunks stale without deleting or re-embedding them.

        Sets ``invalidated_at`` (system-time; defaults to now, UTC ISO-8601)
        and optionally ``valid_until`` (world-time end) on each point's payload
        via a merge write — the vector is untouched and the chunk stays
        searchable via ``include_invalidated=True`` / point-in-time ``as_of``.
        Points that do not exist are skipped. When ``index_root`` is given, a
        point owned by a *different* root is left untouched (the same owner
        guard ``index --refresh-payloads`` uses), so one indexing root cannot
        invalidate another's chunks. Returns the number of points updated.

        Scope: this writes the Qdrant vector payload, so recall's vector and
        temporal retrievers (which read Qdrant live) honor it immediately. A
        BM25 retriever is an in-memory index built at startup — it reflects an
        invalidation only after its corpus is rebuilt, and only when that
        corpus was built from Qdrant payloads (``BM25Retriever.from_qdrant``).
        A file-backed BM25 corpus (``bm25_paths``) carries no validity keys and
        is not filtered; build BM25 from Qdrant if lexical search must respect
        invalidation.
        """
        if isinstance(ids, (str, int)):
            ids = [ids]
        ids = list(ids)
        if not ids:
            return 0
        payload: dict[str, Any] = {
            "invalidated_at": invalidated_at or datetime.now(timezone.utc).isoformat()
        }
        if valid_until is not None:
            payload["valid_until"] = valid_until
        existing = {
            str(pt.id): (pt.payload or {})
            for pt in self.client.retrieve(
                collection_name=self.collection, ids=ids, with_payload=True
            )
        }
        target: list[Any] = []
        for pid in ids:
            current = existing.get(str(pid))
            if current is None:
                continue  # point does not exist — nothing to invalidate
            if index_root is not None:
                owner = current.get("index_root")
                if owner is not None and owner != index_root:
                    continue  # foreign root — never touch another root's chunks
            target.append(pid)
        if not target:
            return 0
        self.client.set_payload(
            collection_name=self.collection, payload=payload, points=target
        )
        return len(target)

    # ---------- search ----------

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
        *,
        hide_invalidated: bool = False,
    ) -> list[Hit]:
        """Semantic search with optional payload filters.

        filters format (simple exact-match):
            {"memory_class": "decision", "source_file": "notes.md"}

        ``hide_invalidated`` pushes the default validity view into Qdrant so
        stale facts (those carrying an ``invalidated_at`` marker) are never
        fetched — cleaner and cheaper than fetching them and dropping them
        client-side. The client-side ``filter_by_validity`` remains the backstop.
        """
        must: list[Any] = list(self._build_filter(filters).must or []) if filters else []
        if hide_invalidated:
            must.append(_hide_invalidated_condition())
        qfilter = Filter(must=must) if must else None
        result = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=limit,
            query_filter=qfilter,
            with_payload=True,
        )
        hits = []
        for pt in result.points:
            if pt.score < min_score:
                continue
            pid = str(pt.id) if isinstance(pt.id, UUID) else pt.id
            hits.append(Hit(id=pid, score=pt.score, payload=pt.payload or {}))
        return hits

    def _build_filter(self, filters: dict[str, Any]) -> Filter:
        must: list[Any] = []
        for key, value in filters.items():
            if isinstance(value, dict) and ("gte" in value or "lte" in value):
                gte = value.get("gte")
                lte = value.get("lte")
                # Dispatch between numeric Range and DatetimeRange by
                # value type. Qdrant's Range(gte=, lte=) requires
                # numbers; string-valued ISO timestamps (as produced
                # by TemporalRetriever and any caller filtering a
                # DATETIME-indexed payload field) must go through
                # DatetimeRange instead. Before this split the
                # temporal path silently blew up inside pydantic
                # validation, which the caller's broad except
                # swallowed, yielding zero hits with no warning.
                if isinstance(gte, str) or isinstance(lte, str):
                    must.append(
                        FieldCondition(
                            key=key,
                            range=DatetimeRange(gte=gte, lte=lte),
                        )
                    )
                else:
                    must.append(
                        FieldCondition(
                            key=key,
                            range=Range(gte=gte, lte=lte),
                        )
                    )
            else:
                must.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=must)
