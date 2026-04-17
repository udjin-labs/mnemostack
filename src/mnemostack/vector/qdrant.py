"""Qdrant vector store wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)


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
        except Exception:  # noqa: BLE001
            return False

    def ensure_collection(self, recreate: bool = False) -> bool:
        """Create collection if missing. If recreate=True, drop and recreate."""
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
        return False

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
                yield Hit(id=pt.id, score=1.0, payload=pt.payload or {})
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
            structs = [
                PointStruct(id=pid, vector=vec, payload=pl or {})
                for pid, vec, pl in chunk
            ]
            self.client.upsert(collection_name=self.collection, points=structs)
            total += len(structs)
        return total

    # ---------- search ----------

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[Hit]:
        """Semantic search with optional payload filters.

        filters format (simple exact-match):
            {"memory_class": "decision", "source_file": "notes.md"}
        """
        qfilter = self._build_filter(filters) if filters else None
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
            hits.append(Hit(id=pt.id, score=pt.score, payload=pt.payload or {}))
        return hits

    def _build_filter(self, filters: dict[str, Any]) -> Filter:
        must = []
        for key, value in filters.items():
            if isinstance(value, dict) and ("gte" in value or "lte" in value):
                must.append(
                    FieldCondition(
                        key=key,
                        range=Range(
                            gte=value.get("gte"), lte=value.get("lte")
                        ),
                    )
                )
            else:
                must.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        return Filter(must=must)
