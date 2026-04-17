"""Async Qdrant vector store wrapper — for high-throughput servers.

Uses QdrantClient's async client variant. For single-shot applications the
sync VectorStore is simpler; use this for FastAPI/Starlette/async MCP servers
where blocking I/O would hurt concurrency.
"""
from __future__ import annotations

from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from .qdrant import Hit


class AsyncVectorStore:
    """Asyncio variant of VectorStore. Mirror of the sync API.

    Use inside coroutines:

        store = AsyncVectorStore(collection='...', dimension=768)
        await store.ensure_collection()
        await store.upsert(1, vec, {'text': '...'})
        hits = await store.search(query_vec, limit=10)
        await store.close()
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
        self.client = AsyncQdrantClient(url=host, timeout=timeout)

    async def close(self) -> None:
        await self.client.close()

    async def collection_exists(self) -> bool:
        try:
            await self.client.get_collection(self.collection)
            return True
        except Exception:  # noqa: BLE001
            return False

    async def ensure_collection(self, recreate: bool = False) -> bool:
        exists = await self.collection_exists()
        if exists and recreate:
            await self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dimension, distance=self.distance),
            )
            return True
        return False

    async def count(self) -> int:
        info = await self.client.get_collection(self.collection)
        return info.points_count or 0

    async def upsert(
        self,
        id: str | int,
        vector: list[float],
        payload: dict[str, Any] | None = None,
    ) -> None:
        await self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=id, vector=vector, payload=payload or {})],
        )

    async def upsert_batch(
        self,
        points: list[tuple[str | int, list[float], dict[str, Any]]],
        batch_size: int = 100,
    ) -> int:
        total = 0
        for i in range(0, len(points), batch_size):
            chunk = points[i : i + batch_size]
            structs = [
                PointStruct(id=pid, vector=vec, payload=pl or {})
                for pid, vec, pl in chunk
            ]
            await self.client.upsert(collection_name=self.collection, points=structs)
            total += len(structs)
        return total

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[Hit]:
        qfilter = self._build_filter(filters) if filters else None
        result = await self.client.query_points(
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

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> Filter:
        must = []
        for key, value in filters.items():
            if isinstance(value, dict) and ("gte" in value or "lte" in value):
                must.append(
                    FieldCondition(
                        key=key,
                        range=Range(gte=value.get("gte"), lte=value.get("lte")),
                    )
                )
            else:
                must.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=must)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
