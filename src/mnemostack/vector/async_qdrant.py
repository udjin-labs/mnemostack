"""Async Qdrant vector store wrapper — for high-throughput servers.

Uses QdrantClient's async client variant. For single-shot applications the
sync VectorStore is simpler; use this for FastAPI/Starlette/async MCP servers
where blocking I/O would hurt concurrency.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    DatetimeRange,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

from .qdrant import (
    TENANT_ID_KEY,
    DimensionMismatchError,
    Hit,
    TenantConflictError,
    _hide_invalidated_condition,
    _stamp_tenant,
    _tenant_condition,
)


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
        except ValueError as exc:
            if "not found" in str(exc).lower():
                return False
            raise
        except UnexpectedResponse as exc:
            if exc.status_code == 404:
                return False
            raise

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
            # Index tenant_id (the mandatory tenant filter) — mirror of the sync
            # store, so async-bootstrapped shared collections filter efficiently.
            try:
                await self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=TENANT_ID_KEY,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:  # noqa: BLE001
                pass  # already indexed or local Qdrant (payload indexes are a no-op there)
            return True
        await self._validate_dimension()
        return False

    async def _validate_dimension(self) -> None:
        info = await self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        size = getattr(vectors, "size", None)  # named-vectors dict has no .size — skip
        if size is not None and int(size) != int(self.dimension):
            raise DimensionMismatchError(
                f"Collection '{self.collection}' stores {size}-dim vectors but the "
                f"embedding provider produces {self.dimension}-dim vectors. "
                f"Re-index with --recreate or switch the embedding model."
            )

    async def count(self, tenant: str | None = None) -> int:
        if tenant is not None:
            result = await self.client.count(
                collection_name=self.collection,
                count_filter=Filter(must=[_tenant_condition(tenant)]),
            )
            return result.count
        info = await self.client.get_collection(self.collection)
        return info.points_count or 0

    async def _assert_tenant_owned(self, ids: list[str | int], tenant: str) -> None:
        """Refuse to overwrite points already owned by a *different* tenant
        (async mirror of ``VectorStore._assert_tenant_owned``)."""
        found = await self.client.retrieve(
            collection_name=self.collection, ids=list(ids), with_payload=[TENANT_ID_KEY]
        )
        conflicts = [
            p.id
            for p in found
            if (owner := (p.payload or {}).get(TENANT_ID_KEY)) is not None and owner != tenant
        ]
        if conflicts:
            raise TenantConflictError(
                f"{len(conflicts)} point id(s) already owned by another tenant "
                f"(e.g. {conflicts[0]!r}); tenant-scoped upsert will not overwrite them"
            )

    async def _existing_tenants(self, ids: list[str | int]) -> dict[Any, str]:
        """Map ``id -> tenant_id`` for the given ids that exist and carry an owner."""
        found = await self.client.retrieve(
            collection_name=self.collection, ids=list(ids), with_payload=[TENANT_ID_KEY]
        )
        return {p.id: t for p in found if (t := (p.payload or {}).get(TENANT_ID_KEY)) is not None}

    async def upsert(
        self,
        id: str | int,
        vector: list[float],
        payload: dict[str, Any] | None = None,
        *,
        tenant: str | None = None,
    ) -> None:
        stamped = _stamp_tenant(payload, tenant)
        if tenant is not None:
            await self._assert_tenant_owned([id], tenant)
        elif (owner := (await self._existing_tenants([id])).get(id)) is not None:
            stamped[TENANT_ID_KEY] = owner  # preserve existing owner on unscoped replace
        await self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=id, vector=vector, payload=stamped)],
        )

    async def upsert_batch(
        self,
        points: list[tuple[str | int, list[float], dict[str, Any]]],
        batch_size: int = 100,
        *,
        tenant: str | None = None,
    ) -> int:
        total = 0
        for i in range(0, len(points), batch_size):
            chunk = points[i : i + batch_size]
            if tenant is not None:
                await self._assert_tenant_owned([pid for pid, _, _ in chunk], tenant)
                structs = [
                    PointStruct(id=pid, vector=vec, payload=_stamp_tenant(pl, tenant))
                    for pid, vec, pl in chunk
                ]
            else:
                owners = await self._existing_tenants([pid for pid, _, _ in chunk])
                structs = []
                for pid, vec, pl in chunk:
                    p = _stamp_tenant(pl, None)
                    if pid in owners:
                        p[TENANT_ID_KEY] = owners[pid]
                    structs.append(PointStruct(id=pid, vector=vec, payload=p))
            await self.client.upsert(collection_name=self.collection, points=structs)
            total += len(structs)
        return total

    async def set_payload(
        self, id: str | int, payload: dict[str, Any], *, tenant: str | None = None
    ) -> None:
        """Merge payload keys into an existing point (vector untouched).

        With ``tenant`` set, the write is skipped unless the point belongs to
        that tenant; with no tenant a caller ``tenant_id`` is dropped from the
        merge so an unscoped refresh can't inject an owner (mirror of
        ``VectorStore.set_payload``)."""
        if tenant is not None:
            found = await self.client.retrieve(
                collection_name=self.collection, ids=[id], with_payload=True
            )
            if not found or (found[0].payload or {}).get(TENANT_ID_KEY) != tenant:
                return
            payload = {**payload, TENANT_ID_KEY: tenant}
        else:
            payload = {k: v for k, v in payload.items() if k != TENANT_ID_KEY}
        await self.client.set_payload(
            collection_name=self.collection,
            payload=payload,
            points=[id],
        )

    async def invalidate(
        self,
        ids: str | int | list[str | int],
        *,
        invalidated_at: str | None = None,
        valid_until: str | None = None,
        index_root: str | None = None,
        tenant: str | None = None,
    ) -> int:
        """Async mirror of ``VectorStore.invalidate``."""
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
        retrieved = await self.client.retrieve(
            collection_name=self.collection, ids=ids, with_payload=True
        )
        existing = {str(pt.id): (pt.payload or {}) for pt in retrieved}
        target: list[Any] = []
        for pid in ids:
            current = existing.get(str(pid))
            if current is None:
                continue
            if index_root is not None:
                owner = current.get("index_root")
                if owner is not None and owner != index_root:
                    continue
            if tenant is not None and current.get(TENANT_ID_KEY) != tenant:
                continue
            target.append(pid)
        if not target:
            return 0
        await self.client.set_payload(
            collection_name=self.collection, payload=payload, points=target
        )
        return len(target)

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
        *,
        hide_invalidated: bool = False,
        tenant: str | None = None,
    ) -> list[Hit]:
        must: list[Any] = list(self._build_filter(filters).must or []) if filters else []
        if tenant is not None:
            must.append(_tenant_condition(tenant))
        if hide_invalidated:
            must.append(_hide_invalidated_condition())
        qfilter = Filter(must=must) if must else None
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
            pid = str(pt.id) if isinstance(pt.id, UUID) else pt.id
            hits.append(Hit(id=pid, score=pt.score, payload=pt.payload or {}))
        return hits

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> Filter:
        must: list[Any] = []
        for key, value in filters.items():
            if isinstance(value, dict) and ("gte" in value or "lte" in value):
                gte = value.get("gte")
                lte = value.get("lte")
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

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
