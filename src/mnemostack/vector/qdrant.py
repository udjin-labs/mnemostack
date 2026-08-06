"""Qdrant vector store wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar, cast
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    DatetimeRange,
    DeletePayload,
    DeletePayloadOperation,
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    HasVectorCondition,
    IsEmptyCondition,
    MatchText,
    MatchValue,
    Modifier,
    PayloadField,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    Range,
    SetPayload,
    SetPayloadOperation,
    SparseVector,
    SparseVectorParams,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)

from .sparse import SPARSE_TEXT_VECTOR, SparseTextEncoder

#: Payload key marking a fact as stale (system-time). Kept as a literal here so
#: the vector layer doesn't depend on the recall layer; mirrors
#: ``recall.validity.INVALIDATED_AT``.
_INVALIDATED_AT_KEY = "invalidated_at"

#: Payload key carrying a point's tenant. When a ``tenant`` is passed to a read
#: it becomes a mandatory ``must`` filter (server-enforced isolation — see
#: docs/api-stability and the multi-tenant design); when passed to a write it is
#: stamped onto the payload, overriding any caller-supplied value so a client
#: can never write into another tenant's namespace.
TENANT_ID_KEY = "tenant_id"


def _tenant_condition(tenant: str) -> FieldCondition:
    """Qdrant predicate restricting a query to one tenant's points."""
    return FieldCondition(key=TENANT_ID_KEY, match=MatchValue(value=tenant))


def _stamp_tenant(payload: dict[str, Any] | None, tenant: str | None) -> dict[str, Any]:
    """Return the payload with a server-owned ``tenant_id``.

    With a tenant, the server value always wins over a caller-provided
    ``tenant_id`` so a write can't be redirected into another tenant. With **no**
    tenant the key is **stripped**: an unscoped write must never carry a caller
    ``tenant_id`` (e.g. from markdown frontmatter passed straight into the
    payload), or unauthenticated content could inject itself into a tenant's
    scoped reads. The explicit migration path (``stamp_tenant``) sets
    ``tenant_id`` directly, not through here.
    """
    base = dict(payload or {})
    if tenant is not None:
        base[TENANT_ID_KEY] = tenant
    else:
        base.pop(TENANT_ID_KEY, None)
    return base


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


def payload_index_type_name(field_info: Any) -> str:
    """Normalized type name of a payload-schema entry.

    Client versions expose the entry as an object with a ``data_type`` enum,
    a bare enum, or a bare string — one normalization shared by
    :meth:`VectorStore.payload_indexes` and doctor's listing, so the two can
    never disagree about the same collection.
    """
    data_type = getattr(field_info, "data_type", field_info)
    return str(getattr(data_type, "value", data_type)).rsplit(".", 1)[-1].lower()


class PayloadIndexConflictError(ValueError):
    """The field is already indexed with a DIFFERENT type.

    Deliberately distinct from a backend failure: a real server silently
    replaces an index re-created with another type, so the operator surface
    refuses before mutating — and its caller must be able to tell this
    refusal (a usage error) from an unreachable/failing backend.
    """


class TenantConflictError(ValueError):
    """A tenant-scoped write targets a point id already owned by another tenant.

    The tenant boundary is server-owned: a caller who guesses another tenant's
    (deterministic) point id must not be able to overwrite it. Raised instead of
    silently replacing the foreign point.
    """


@dataclass
class Hit:
    """Single search result."""

    id: str | int
    score: float
    payload: dict[str, Any]
    #: The stored embedding — populated only by ``scroll(with_vectors=True)``
    #: (export/migration); search results don't carry it.
    vector: list[float] | None = None


class VectorStore:
    """Thin wrapper around QdrantClient with typed search results.

    Handles collection setup, indexing payload fields for filtering, and
    consistent Hit dataclass output.
    """

    #: Class-level defaults so instances built without __init__ (the
    #: widespread __new__-style test/embedding-free construction) behave as
    #: plain dense stores with the standard schema.
    sparse_text = False
    text_key = "text"
    _sparse_encoder: SparseTextEncoder | None = None

    def __init__(
        self,
        collection: str,
        dimension: int,
        host: str = "http://localhost:6333",
        distance: Distance = Distance.COSINE,
        timeout: int = 30,
        *,
        sparse_text: bool = False,
        text_key: str = "text",
    ):
        self.collection = collection
        self.dimension = dimension
        self.distance = distance
        self.client = QdrantClient(url=host, timeout=timeout)
        #: Opt-in server-side lexical index: writes maintain a named sparse
        #: vector (see vector/sparse.py) next to the dense one, and
        #: ``sparse_search`` queries it. Off by default — collections and
        #: writes are byte-identical to before.
        self.sparse_text = sparse_text
        self.text_key = text_key
        self._sparse_encoder = SparseTextEncoder() if sparse_text else None

    def _point_vector(
        self, vector: list[float], payload: dict[str, Any] | None
    ) -> list[float] | dict[str, Any]:
        """The vector value for a PointStruct: plain dense (unchanged layout)
        unless sparse_text is on, in which case the named sparse encoding of
        the payload text rides along with the dense vector."""
        if not self.sparse_text or self._sparse_encoder is None:
            return vector
        text = str((payload or {}).get(self.text_key) or "")
        indices, values = self._sparse_encoder.encode_document(text)
        return {
            "": vector,
            SPARSE_TEXT_VECTOR: SparseVector(indices=indices, values=values),
        }

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
        sparse_cfg = (
            {SPARSE_TEXT_VECTOR: SparseVectorParams(modifier=Modifier.IDF)}
            if self.sparse_text
            else None
        )
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dimension, distance=self.distance),
                sparse_vectors_config=sparse_cfg,
            )
            # Index tenant_id: it's the mandatory filter on every tenant-scoped
            # read, so in a large shared collection a small-tenant search/count
            # would otherwise degrade to collection-wide filtered work.
            self.index_payload_field(TENANT_ID_KEY, PayloadSchemaType.KEYWORD)
            return True
        self._validate_dimension()
        if sparse_cfg is not None:
            self.ensure_sparse_space()
        return False

    def ensure_sparse_space(self, *, require_backfilled: bool = True) -> None:
        """Ensure the existing collection carries the sparse text space.

        A sparse-ONLY operation — no dense-dimension validation (the backfill
        CLI runs with a placeholder dimension). Adding the space post-hoc is
        attempted via a config update, but real Qdrant servers REFUSE adding a
        new sparse space to an existing collection (verified on v1.15.4) — the
        refusal surfaces LOUD with recreate/re-index guidance, because a
        silently missing sparse space would make every sparse_search return
        nothing. With ``require_backfilled`` (default), a collection whose
        space exists but whose points lack sparse vectors (dense-only writes,
        e.g. the async store) also refuses until backfilled — coverage is
        verified server-side, never assumed from the space existing."""
        info = self.client.get_collection(self.collection)
        existing_sparse = getattr(info.config.params, "sparse_vectors", None) or {}
        if SPARSE_TEXT_VECTOR in existing_sparse:
            # The SPACE existing says nothing about the POINTS: a first
            # migration attempt (or dense-only writes, e.g. the async store)
            # leaves uncovered points that sparse recall silently omits — a
            # retry must not sail past the backfill requirement.
            if require_backfilled and (gap := self.sparse_coverage_gap()) > 0:
                raise RuntimeError(
                    f"collection '{self.collection}' has the "
                    f"'{SPARSE_TEXT_VECTOR}' space but {gap} point(s) carry no "
                    "sparse vector — run `mnemostack sparse-backfill` (or a "
                    "full re-index) before using sparse text search"
                )
            return
        try:
            self.client.update_collection(
                collection_name=self.collection,
                sparse_vectors_config={
                    SPARSE_TEXT_VECTOR: SparseVectorParams(modifier=Modifier.IDF)
                },
            )
        except Exception as e:  # noqa: BLE001 — surface, don't guess
            raise RuntimeError(
                f"collection '{self.collection}' has no '{SPARSE_TEXT_VECTOR}' "
                "sparse space and the server refused to add one "
                f"({e}); re-create the collection (or re-index) to use "
                "sparse text search"
            ) from e
        if require_backfilled and self.count() > 0:
            raise RuntimeError(
                f"collection '{self.collection}' gained the "
                f"'{SPARSE_TEXT_VECTOR}' space, but its existing points "
                "have no sparse vectors yet — run "
                "`mnemostack sparse-backfill` (or a full re-index) "
                "before using sparse text search"
            )

    def sparse_coverage_gap(self) -> int:
        """How many points carry NO sparse text vector (0 = fully covered).
        The honesty check behind enabling/diagnosing sparse mode."""
        covered = self.client.count(
            collection_name=self.collection,
            count_filter=Filter(
                must=[HasVectorCondition(has_vector=SPARSE_TEXT_VECTOR)]
            ),
            exact=True,
        ).count
        total = self.client.count(collection_name=self.collection, exact=True).count
        return max(0, total - covered)

    def backfill_sparse_text(self, batch_size: int = 256) -> int:
        """Write the sparse text encoding onto EVERY existing point.

        Scrolls the whole collection and updates only the named sparse vector
        (``update_vectors`` — payloads and dense vectors untouched, nothing is
        re-embedded). Idempotent; returns the number of points updated. This
        is the migration path for enabling ``text_search: sparse`` on an
        already-populated collection."""
        if not self.sparse_text or self._sparse_encoder is None:
            raise RuntimeError("backfill_sparse_text requires VectorStore(sparse_text=True)")
        from qdrant_client.models import PointVectors

        total = 0
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=batch_size,
                offset=offset,
                with_payload=[self.text_key],
                with_vectors=False,
            )
            if not points:
                break
            updates = []
            for pt in points:
                text = str((pt.payload or {}).get(self.text_key) or "")
                indices, values = self._sparse_encoder.encode_document(text)
                updates.append(
                    PointVectors(
                        id=pt.id,
                        vector={
                            SPARSE_TEXT_VECTOR: SparseVector(indices=indices, values=values)
                        },
                    )
                )
            self.client.update_vectors(collection_name=self.collection, points=updates)
            total += len(updates)
            if offset is None:
                break
        return total

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

    def ensure_text_index(self, text_key: str | None = None) -> None:
        """Create the full-text payload index MatchText filtering REQUIRES on a
        real Qdrant server (word tokenizer, lowercased — matching the lexical
        arm's expectations). Idempotent; non-destructive on existing data, so
        it's safe to run against a pre-existing (mounted) collection. The local
        in-memory client matches text without an index, hence tests pass either
        way — production servers do not."""
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name=text_key or self.text_key,
            field_schema=TextIndexParams(
                type=TextIndexType.TEXT, tokenizer=TokenizerType.WORD, lowercase=True
            ),
        )

    def index_payload_field(self, field: str, schema: PayloadSchemaType) -> None:
        """Create a payload index for filtering (e.g. timestamp as DATETIME)."""
        try:
            self.client.create_payload_index(
                collection_name=self.collection, field_name=field, field_schema=schema
            )
        except Exception:  # noqa: BLE001
            pass  # already indexed or collection not ready

    #: Schema names accepted by :meth:`ensure_payload_index`. `text` is
    #: deliberately absent — full-text indexes have their own command and
    #: semantics (`ensure_text_index` / `mnemostack text-index`).
    PAYLOAD_INDEX_SCHEMAS: ClassVar[dict[str, PayloadSchemaType]] = {
        "keyword": PayloadSchemaType.KEYWORD,
        "integer": PayloadSchemaType.INTEGER,
        "float": PayloadSchemaType.FLOAT,
        "bool": PayloadSchemaType.BOOL,
        "datetime": PayloadSchemaType.DATETIME,
    }

    def payload_indexes(self) -> dict[str, str]:
        """Existing payload indexes as ``{field: schema-type-name}``.

        Read-only. The local in-memory client never records indexes (they
        have no effect there), so this returns ``{}`` for it — a real server
        reports the collection's actual index schema.
        """
        info = self.client.get_collection(self.collection)
        schema = getattr(info, "payload_schema", None) or {}
        return {name: payload_index_type_name(fi) for name, fi in schema.items()}

    def ensure_payload_index(self, field: str, schema: str) -> str:
        """Create a payload index for filtered recall; returns the type name.

        LOUD on purpose — this backs the operator command (`mnemostack
        payload-index`), unlike the fail-open :meth:`index_payload_field`
        used opportunistically on write paths: a real server evaluates a
        filter on an unindexed field by scanning candidate payloads, and the
        operator asking for the index needs the failure, not a shrug.
        Idempotent for an existing index of the same type (no server call).
        A CONFLICTING type is refused before any mutation: a real server
        silently REPLACES the index on re-create (verified live on v1.18.3),
        which would swap the field's filtering semantics behind the
        operator's back.
        """
        try:
            schema_type = self.PAYLOAD_INDEX_SCHEMAS[schema]
        except KeyError:
            allowed = ", ".join(sorted(self.PAYLOAD_INDEX_SCHEMAS))
            raise ValueError(
                f"unknown payload index schema '{schema}' (one of: {allowed})"
            ) from None
        existing = self.payload_indexes().get(field)
        if existing is not None:
            if existing == schema:
                return existing  # already indexed as requested — nothing to do
            raise PayloadIndexConflictError(
                f"field '{field}' is already indexed as '{existing}' — the "
                f"server would silently replace it with '{schema}'; drop the "
                "existing index first if the type change is intended"
            )
        self.client.create_payload_index(
            collection_name=self.collection, field_name=field, field_schema=schema_type
        )
        # Report the type the collection actually records; local mode keeps
        # no record, so fall back to the requested name.
        return self.payload_indexes().get(field, schema)

    def count(self, tenant: str | None = None) -> int:
        if tenant is not None:
            # Scoped count: only this tenant's points (collection-wide
            # points_count can't be filtered).
            return self.client.count(
                collection_name=self.collection,
                count_filter=Filter(must=[_tenant_condition(tenant)]),
            ).count
        info = self.client.get_collection(self.collection)
        return info.points_count or 0

    def retrieve_payload(
        self, point_id: str | int, *, tenant: str | None = None
    ) -> dict[str, Any] | None:
        """Full payload of one point by id, or ``None`` if absent.

        ``tenant`` scopes the lookup like every other read: a point belonging
        to a different tenant resolves to ``None`` (indistinguishable from
        absent — existence must not leak across the boundary)."""
        found = self.client.retrieve(
            collection_name=self.collection,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        if not found:
            return None
        payload = dict(found[0].payload or {})
        if tenant is not None and payload.get(TENANT_ID_KEY) != tenant:
            return None
        return payload

    def retrieve_existing_ids(
        self, ids: list[str | int], *, tenant: str | None = None
    ) -> set[str]:
        """Which of ``ids`` already exist (as string ids), for re-upsert detection.

        Lets a caller distinguish genuinely-new points from re-upserts of
        already-stored ones (chunk ids are deterministic, so a re-ingest upserts
        onto itself and doesn't grow the count) — used by the ingest quota check.

        With ``tenant`` set, only ids **already owned by that tenant** count as
        existing: an unowned/legacy or other-tenant point sharing an id is treated
        as new, because a tenant-scoped upsert adopts an unowned id (stamping
        ``tenant_id``) and so grows the tenant's count. This keeps the "new" tally
        consistent with the tenant-scoped ``count()`` the quota checks against —
        otherwise adopting an unowned point would slip past the cap uncounted.
        """
        if not ids:
            return set()
        found = self.client.retrieve(
            collection_name=self.collection,
            ids=ids,
            with_payload=[TENANT_ID_KEY] if tenant is not None else False,
            with_vectors=False,
        )
        if tenant is None:
            return {str(p.id) for p in found}
        return {
            str(p.id) for p in found if (p.payload or {}).get(TENANT_ID_KEY) == tenant
        }

    def delete(self) -> None:
        self.client.delete_collection(self.collection)

    # ---------- iteration ----------

    def scroll(
        self,
        batch_size: int = 256,
        filters: dict[str, Any] | None = None,
        with_vectors: bool = False,
        *,
        tenant: str | None = None,
    ):
        """Iterate over points in the collection lazily.

        Memory-efficient: never loads the whole collection at once. Good for:
        - Re-indexing after schema changes
        - Bulk export / migration
        - Aggregation over entire corpus

        With ``tenant`` set, only that tenant's points are yielded. Yields `Hit`
        objects (score=1.0 since this isn't a similarity query).
        """
        must: list[Any] = list(self._build_filter(filters).must or []) if filters else []
        if tenant is not None:
            must.append(_tenant_condition(tenant))
        qfilter = Filter(must=must) if must else None
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
                # qdrant-client types .vector as a union covering named/multi
                # vectors; a single-vector collection yields a flat float list,
                # while a sparse_text collection yields a NAMED mapping whose
                # "" entry is the dense vector — a backup (tenant-export) must
                # not silently lose it.
                raw_vec: Any = pt.vector
                if isinstance(raw_vec, dict):
                    raw_vec = raw_vec.get("")
                vec = (
                    cast("list[float]", list(raw_vec))
                    if with_vectors and isinstance(raw_vec, list)
                    else None
                )
                yield Hit(id=pid, score=1.0, payload=pt.payload or {}, vector=vec)
            if next_offset is None:
                break

    def iter_ids(
        self,
        batch_size: int = 1024,
        filters: dict[str, Any] | None = None,
        *,
        tenant: str | None = None,
    ):
        """Lightweight iteration returning only point IDs. Faster than scroll()."""
        must: list[Any] = list(self._build_filter(filters).must or []) if filters else []
        if tenant is not None:
            must.append(_tenant_condition(tenant))
        qfilter = Filter(must=must) if must else None
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

    def _assert_tenant_owned(self, ids: list[str | int], tenant: str) -> None:
        """Refuse to overwrite points already owned by a *different* tenant.

        A full-point ``upsert`` replaces the whole point, so without this guard a
        tenant-scoped caller who supplies another tenant's id would take it over
        and re-stamp it. Points that don't exist or are unowned (no ``tenant_id``,
        e.g. pre-migration legacy data) are free to write; a point stamped with a
        different tenant raises ``TenantConflictError``.

        Concurrency: this is a check-then-write and Qdrant has no compare-and-swap,
        so a precisely-timed concurrent write to the *same* id by two tenants can
        race past the check. The primary defense is the id scheme, not this guard —
        ``stable_chunk_id(..., tenant=)`` prefixes the tenant, so honest
        tenant-scoped writers never compute the same id and never collide; the race
        exists only for an attacker deliberately targeting another tenant's guessed
        id (a tamper, not a read leak). A per-process lock would not close it for
        the real multi-tenant deployment (multiple server processes on one Qdrant)
        and would only add contention — for a hard guarantee under adversarial
        concurrent writes, serialize writes per collection or front the store with a
        compare-and-swap-capable layer.
        """
        found = self.client.retrieve(
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

    def _existing_tenants(self, ids: list[str | int]) -> dict[Any, str]:
        """Map ``id -> tenant_id`` for the given ids that exist and carry an owner."""
        found = self.client.retrieve(
            collection_name=self.collection, ids=list(ids), with_payload=[TENANT_ID_KEY]
        )
        return {p.id: t for p in found if (t := (p.payload or {}).get(TENANT_ID_KEY)) is not None}

    def upsert(
        self,
        id: str | int,
        vector: list[float],
        payload: dict[str, Any] | None = None,
        *,
        tenant: str | None = None,
    ) -> None:
        stamped = _stamp_tenant(payload, tenant)
        if tenant is not None:
            self._assert_tenant_owned([id], tenant)
        elif (owner := self._existing_tenants([id]).get(id)) is not None:
            # Unscoped upsert is a full replace: preserve an existing owner so an
            # unscoped re-index of a migrated point doesn't silently orphan it
            # (the caller's own tenant_id was already stripped by _stamp_tenant).
            stamped[TENANT_ID_KEY] = owner
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=id, vector=self._point_vector(vector, stamped), payload=stamped)],
        )

    def upsert_batch(
        self,
        points: list[tuple[str | int, list[float], dict[str, Any]]],
        batch_size: int = 100,
        *,
        tenant: str | None = None,
    ) -> int:
        """Upsert a batch of (id, vector, payload) tuples. Returns count inserted.

        With ``tenant`` set, every point's payload is stamped with it (the tenant
        wins over any caller-supplied ``tenant_id``), and any id already owned by
        a different tenant raises ``TenantConflictError`` before the chunk is
        written (the batch is not partially applied past the offending chunk).
        Unscoped (``tenant=None``) strips a caller ``tenant_id`` but preserves an
        existing point's owner (one ownership read per chunk) so an unscoped
        re-index doesn't orphan migrated points."""
        total = 0
        for i in range(0, len(points), batch_size):
            chunk = points[i : i + batch_size]
            if tenant is not None:
                self._assert_tenant_owned([pid for pid, _, _ in chunk], tenant)
                structs = []
                for pid, vec, pl in chunk:
                    stamped_pl = _stamp_tenant(pl, tenant)
                    structs.append(
                        PointStruct(
                            id=pid,
                            vector=self._point_vector(vec, stamped_pl),
                            payload=stamped_pl,
                        )
                    )
            else:
                owners = self._existing_tenants([pid for pid, _, _ in chunk])
                structs = []
                for pid, vec, pl in chunk:
                    p = _stamp_tenant(pl, None)
                    if pid in owners:
                        p[TENANT_ID_KEY] = owners[pid]
                    structs.append(
                        PointStruct(id=pid, vector=self._point_vector(vec, p), payload=p)
                    )
            self.client.upsert(collection_name=self.collection, points=structs)
            total += len(structs)
        return total

    def set_payload(
        self, id: str | int, payload: dict[str, Any], *, tenant: str | None = None
    ) -> None:
        """Merge payload keys into an existing point without re-embedding.

        Under ``sparse_text=True``, editing the text field here leaves the
        point's SPARSE encoding stale too (like the dense vector, it is only
        rewritten on upsert) — re-upsert the point or run
        ``backfill_sparse_text`` after bulk text edits.

        The vector is untouched — this is the cheap path for applying new
        payload fields (enrichment output, index_root) to already-indexed
        points. Merge semantics: keys not present in *payload* are kept.

        With ``tenant`` set, the write is skipped unless the point belongs to
        that tenant — so a caller who knows another tenant's (deterministic)
        point id still can't relabel or overwrite it. ``payload`` also can't
        change ``tenant_id`` under a tenant: the server-owned value is restored.
        With **no** tenant a caller ``tenant_id`` is dropped from the merge (an
        unscoped refresh e.g. of markdown frontmatter must not inject an owner);
        merge semantics leave any existing owner intact.
        """
        if tenant is not None:
            found = self.client.retrieve(
                collection_name=self.collection, ids=[id], with_payload=True
            )
            if not found or (found[0].payload or {}).get(TENANT_ID_KEY) != tenant:
                return  # point missing or owned by another tenant — never touch it
            payload = {**payload, TENANT_ID_KEY: tenant}  # payload can't reassign tenant
        else:
            payload = {k: v for k, v in payload.items() if k != TENANT_ID_KEY}
        self.client.set_payload(
            collection_name=self.collection,
            payload=payload,
            points=[id],
        )

    def stamp_tenant(self, tenant: str, *, only_missing: bool = True) -> int:
        """Assign ``tenant_id`` to existing points (multi-tenant migration).

        Moves a legacy single-tenant corpus into one named tenant with a
        server-side payload-only merge — **ids and vectors are untouched**. With
        ``only_missing`` (default), points that already carry a ``tenant_id`` are
        left untouched, so it's safe to re-run and safe against a partially
        migrated collection. Returns the number of points stamped.

        Idempotency note: a tenant-aware re-ingest of the *same* content via
        ``Ingestor(tenant=...)`` computes a tenant-scoped ``stable_chunk_id`` that
        differs from a migrated point's legacy id, so a plain re-ingest would add a
        second (tenant-scoped) point beside the migrated one. Reconcile by running
        the first post-migration re-ingest with ``--prune`` (tenant-scoped), which
        drops the legacy stragglers. Re-keying here is deliberately avoided: id
        derivation differs per indexer (the markdown indexer folds ``index_root``
        into the hash and is not tenant-scoped), so the store can't reproduce every
        ingester's id from payload alone — pruning is the reliable reconciliation.
        """
        must: list[Any] = []
        if only_missing:
            must.append(IsEmptyCondition(is_empty=PayloadField(key=TENANT_ID_KEY)))
        sel = Filter(must=must)
        n = self.client.count(collection_name=self.collection, count_filter=sel).count
        if n:
            self.client.set_payload(
                collection_name=self.collection,
                payload={TENANT_ID_KEY: tenant},
                points=FilterSelector(filter=sel),
            )
        # A pre-existing collection was created before tenant_id existed, so
        # ensure_collection() never indexed it. Add the KEYWORD index here so
        # post-migration tenant-scoped reads on this large collection stay fast.
        self.index_payload_field(TENANT_ID_KEY, PayloadSchemaType.KEYWORD)
        return n

    def delete_payload_keys(
        self, id: str | int, keys: list[str], *, tenant: str | None = None
    ) -> None:
        """Remove specific payload keys from a point (vector untouched).

        ``tenant_id`` is the server-owned isolation field and is **silently
        skipped** — it's never removed this way (stripping it would make the point
        vanish from its owner's scoped reads), so a payload refresh that lists it
        among stale keys drops the rest without aborting. With ``tenant`` set, the
        delete is skipped unless the point belongs to that tenant.
        """
        keys = [k for k in keys if k != TENANT_ID_KEY]
        if not keys:
            return
        if tenant is not None:
            found = self.client.retrieve(
                collection_name=self.collection, ids=[id], with_payload=[TENANT_ID_KEY]
            )
            if not found or (found[0].payload or {}).get(TENANT_ID_KEY) != tenant:
                return  # point missing or owned by another tenant — never touch it
        self.client.delete_payload(
            collection_name=self.collection,
            keys=keys,
            points=[id],
        )

    def apply_payload_patches(
        self,
        patches: Sequence[Any],
        *,
        tenant: str | None = None,
        batch_size: int = 100,
    ) -> int:
        """Apply per-point payload patches in bounded batch operations.

        Each patch (see :class:`mnemostack.vector.patch.PayloadPatch`)
        carries the point id, the full payload mapping to merge-set and the
        stale keys to delete — the batched equivalent of the scalar
        ``delete_payload_keys`` + ``set_payload`` pair, submitted as ONE
        ``batch_update_points`` round-trip per bounded group instead of two
        HTTP requests per point.

        Semantics mirror the scalar guarded writes exactly:

        - existence is validated for the WHOLE group BEFORE any mutation: a
          real server 404s a batch that names a since-deleted point (and may
          have applied the earlier operations by then — the local in-memory
          client silently no-ops instead, which is why only a live server
          shows this), so a point deleted concurrently is silently skipped
          rather than crashing the refresh; the residual check-then-write
          race is closed by re-filtering and retrying (idempotent
          operations), never by ignoring the error;
        - with ``tenant`` set, the same pre-check also validates ownership;
          foreign-owned points are silently skipped, and the set side
          restores the server-owned ``tenant_id``;
        - unscoped patches can neither set nor delete ``tenant_id``;
        - per-point operation order (delete-then-set) and input order are
          preserved within a submitted batch;
        - any other backend rejection raises — a partial/ambiguous batch must
          never look committed, so callers do not advance their checkpoints.

        Returns the number of points actually patched (tenant-skipped and
        vanished points are not counted).
        """
        applied = 0
        for start in range(0, len(patches), max(1, batch_size)):
            group = list(patches[start : start + max(1, batch_size)])
            last_404: Exception | None = None
            while True:
                found = self.client.retrieve(
                    collection_name=self.collection,
                    ids=[p.id for p in group],
                    with_payload=[TENANT_ID_KEY] if tenant is not None else False,
                )
                if tenant is not None:
                    keep = {
                        str(pt.id)
                        for pt in found
                        if (pt.payload or {}).get(TENANT_ID_KEY) == tenant
                    }
                else:
                    keep = {str(pt.id) for pt in found}
                survivors = [p for p in group if str(p.id) in keep]
                if last_404 is not None and len(survivors) == len(group):
                    # The 404 was not explained by a vanished point — an
                    # ambiguous rejection must propagate, not spin.
                    raise last_404
                group = survivors
                operations: list[Any] = []
                for patch in group:
                    delete_keys = [k for k in patch.delete_keys if k != TENANT_ID_KEY]
                    if delete_keys:
                        operations.append(
                            DeletePayloadOperation(
                                delete_payload=DeletePayload(
                                    keys=delete_keys, points=[patch.id]
                                )
                            )
                        )
                    if tenant is not None:
                        values = {**dict(patch.set_values), TENANT_ID_KEY: tenant}
                    else:
                        values = {
                            k: v
                            for k, v in patch.set_values.items()
                            if k != TENANT_ID_KEY
                        }
                    if values:
                        operations.append(
                            SetPayloadOperation(
                                set_payload=SetPayload(payload=values, points=[patch.id])
                            )
                        )
                if not operations:
                    break
                try:
                    # wait=True: the scalar methods are synchronous — the
                    # batched path must not return before the backend
                    # confirms, or a caller's checkpoint could outrun an
                    # uncommitted mutation.
                    self.client.batch_update_points(
                        collection_name=self.collection,
                        update_operations=operations,
                        wait=True,
                    )
                    break
                except UnexpectedResponse as exc:
                    if getattr(exc, "status_code", None) != 404:
                        raise
                    # A point vanished between the existence check and the
                    # batch; the server may have applied the earlier
                    # operations before failing. Re-filter and re-apply the
                    # survivors — delete-then-merge-set per point is
                    # idempotent, and the group strictly shrinks (checked
                    # above), so this terminates.
                    last_404 = exc
            applied += len(group)
        return applied

    def delete_points(
        self, ids: list[str | int], batch_size: int = 1000, *, tenant: str | None = None
    ) -> int:
        """Delete specific points by id. Returns the number of points deleted.

        With ``tenant`` set, only points owned by that tenant are deleted: a
        caller who knows another tenant's deterministic point id can't delete it
        by id (Qdrant deletes by id without applying the ``tenant_id`` filter, so
        the ownership check is enforced here — the vector layer is the isolation
        boundary). Without ``tenant`` every requested id is deleted (legacy).
        """
        total = 0
        for i in range(0, len(ids), batch_size):
            chunk: list[Any] = list(ids[i : i + batch_size])
            if tenant is not None:
                found = self.client.retrieve(
                    collection_name=self.collection, ids=chunk, with_payload=[TENANT_ID_KEY]
                )
                chunk = [p.id for p in found if (p.payload or {}).get(TENANT_ID_KEY) == tenant]
                if not chunk:
                    continue
            self.client.delete(
                collection_name=self.collection,
                points_selector=PointIdsList(points=chunk),
            )
            total += len(chunk)
        return total

    def delete_tenant(self, tenant: str) -> int:
        """Delete EVERY point owned by ``tenant`` (offboarding). Returns the count.

        Filter-based (server-side), so it never touches unscoped/legacy points or
        another tenant's — the mandatory ``tenant_id`` condition is the selector,
        not a caller-supplied id list. ``tenant`` must be a non-empty string: an
        empty value must never silently select (and drop) the whole collection.
        """
        if not tenant or not isinstance(tenant, str):
            raise ValueError("delete_tenant requires a non-empty tenant")
        sel = Filter(must=[_tenant_condition(tenant)])
        # Pre-delete snapshot count (Qdrant's filter-delete doesn't return a
        # deleted count). Exact for the single-writer offboarding case; under a
        # concurrent writer the actual delete may differ by a few.
        n = self.client.count(collection_name=self.collection, count_filter=sel).count
        # ALWAYS issue the (idempotent) filter-delete, even when the snapshot
        # count is 0: a concurrent writer could create the tenant's first point
        # between the count and here, and gating the delete on the stale count
        # would leave it behind while tenant-rm reports "fully removed".
        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(filter=sel),
        )
        return n

    def invalidate(
        self,
        ids: str | int | list[str | int],
        *,
        invalidated_at: str | None = None,
        valid_until: str | None = None,
        index_root: str | None = None,
        tenant: str | None = None,
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
            if tenant is not None and current.get(TENANT_ID_KEY) != tenant:
                continue  # foreign tenant — never invalidate another tenant's chunks
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
        tenant: str | None = None,
        text_any: list[str] | None = None,
        text_any_key: str | None = None,
    ) -> list[Hit]:
        """Semantic search with optional payload filters.

        filters format (simple exact-match):
            {"memory_class": "decision", "source_file": "notes.md"}

        ``hide_invalidated`` pushes the default validity view into Qdrant so
        stale facts (those carrying an ``invalidated_at`` marker) are never
        fetched — cleaner and cheaper than fetching them and dropping them
        client-side. The client-side ``filter_by_validity`` remains the backstop.
        """
        qfilter = self._assemble_filter(
            filters,
            tenant=tenant,
            hide_invalidated=hide_invalidated,
            text_any=text_any,
            text_any_key=text_any_key,
        )
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

    def _assemble_filter(
        self,
        filters: dict[str, Any] | None,
        *,
        tenant: str | None,
        hide_invalidated: bool,
        text_any: list[str] | None = None,
        text_any_key: str | None = None,
    ) -> Filter | None:
        """The full server-side filter: caller filters + tenant boundary +
        validity view + optional lexical gate. ``text_any`` restricts to points
        whose text field full-text-matches AT LEAST ONE token — expressed as a
        nested should-clause inside the outer must, so it composes with the
        other conditions without weakening them (a bare top-level ``should``
        next to ``must`` would become optional)."""
        must: list[Any] = list(self._build_filter(filters).must or []) if filters else []
        if tenant is not None:
            must.append(_tenant_condition(tenant))
        if hide_invalidated:
            must.append(_hide_invalidated_condition())
        if text_any is not None and not text_any:
            # An explicitly EMPTY gate is a caller bug: "match at least one of
            # zero tokens" silently degrades to match-all — refuse loudly.
            raise ValueError("text_any must contain at least one token (or be None)")
        if text_any:
            gate_key = text_any_key or self.text_key
            must.append(
                Filter(
                    should=[
                        FieldCondition(key=gate_key, match=MatchText(text=token))
                        for token in text_any
                    ]
                )
            )
        return Filter(must=must) if must else None

    def sparse_search(
        self,
        query_text: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
        *,
        hide_invalidated: bool = False,
        tenant: str | None = None,
    ) -> list[Hit]:
        """Lexical search over the ``text_sparse`` space (sparse_text=True).

        The query rides as a sparse vector of its distinct tokens; the server
        applies collection-wide IDF (``Modifier.IDF``), so scoring is tf·idf-
        like without any client-side corpus. Same tenant/validity/filter
        semantics as :meth:`search`."""
        if not self.sparse_text or self._sparse_encoder is None:
            raise RuntimeError(
                "sparse_search requires VectorStore(sparse_text=True) — this store "
                "does not maintain the sparse text space"
            )
        indices, values = self._sparse_encoder.encode_query(query_text)
        if not indices:
            return []
        qfilter = self._assemble_filter(
            filters, tenant=tenant, hide_invalidated=hide_invalidated
        )
        result = self.client.query_points(
            collection_name=self.collection,
            query=SparseVector(indices=indices, values=values),
            using=SPARSE_TEXT_VECTOR,
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
