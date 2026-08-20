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
import threading
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

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
        # Code-metadata ownership record: names the keys a --code refresh
        # deletes when the chunk stops being code — a planted list would
        # mark unrelated payload fields for deletion.
        "_code_keys",
        # Graph filter-attribution proof marker: honored by the post-pipeline
        # filter backstop for graph-sourced results — a planted value on a
        # vector point must never exist (the sources gate blocks it anyway;
        # this keeps the key retriever-owned everywhere).
        "_attributed_filters",
    }
)


# ---------------------------------------------------------------- remote ingest
#
# The write surface for REMOTE clients (HTTP `POST /memories`, MCP
# `mnemostack_remember`). Distinct from the operator paths (CLI / library):
# the caller is untrusted, so its metadata is validated against the reserved
# namespace, its work is bounded by explicit caps, and duplicates are
# detected against the STORE (not a per-process cache) so retries and
# repeated content never re-embed — cost discipline, not just idempotency.

#: Hard caps on one remote ingest call. Bounds embedding cost and payload
#: size for a caller-controlled request; an operator who needs more runs the
#: CLI next to the stores.
REMOTE_MAX_ITEMS = 64
REMOTE_MAX_TEXT_CHARS = 32_768
REMOTE_MAX_SOURCE_CHARS = 1_024
REMOTE_MAX_TIMESTAMP_CHARS = 64
REMOTE_MAX_TAGS = 32
REMOTE_MAX_TAG_CHARS = 128
REMOTE_MAX_METADATA_KEYS = 32
REMOTE_MAX_METADATA_CHARS = 16_384

#: Server-side chunking of long documents (`chunk: true` items): the raw
#: text may be larger, and it is split into fixed character windows of
#: ``REMOTE_CHUNK_SIZE`` at offsets 0, size, 2*size... — the split
#: `mnemostack index` applies to prose files at its DEFAULT ``--chunk-size``
#: (kept equal to ``VectorConfig.chunk_size``, pinned by test), so with
#: default settings a document POSTed here and the same file indexed on the
#: box produce identical chunk ids. A deployment indexing with a custom
#: ``--chunk-size`` (or via the section-aware markdown indexer) produces
#: different boundaries — cross-path dedup holds only for the matching
#: split. The TOTAL number of chunks one request may produce stays bounded:
#: embedding work is the resource a caller-controlled request must not
#: scale.
REMOTE_MAX_DOC_CHARS = 262_144
REMOTE_CHUNK_SIZE = 800
REMOTE_MAX_CHUNKS_PER_REQUEST = 128

#: Ceiling on a caller-supplied offset: 2^53-1 — exactly representable in
#: an IEEE double (JSON/JS interop) and well inside Qdrant's int64 payload
#: domain. An unbounded Python int (2**100) would pay for embedding first
#: and only then be rejected — or silently lose precision — at the store.
REMOTE_MAX_OFFSET = 2**53 - 1

#: Metadata keys a remote caller may never supply, beyond the underscore
#: namespace (every "_"-prefixed key is server-structural by convention).
#: `indexed_at` is server-stamped write time; the protected set covers the
#: id-material trio, isolation and provenance keys. `timestamp` and `tags`
#: are ALSO reserved here: they have dedicated request fields with their own
#: caps and type checks, and the ingest pipeline reads them from metadata as
#: a library-era fallback — a remote value smuggled through metadata would
#: bypass every one of those caps. (Library/CLI callers are unaffected:
#: this reservation applies only to the remote validator.)
_REMOTE_RESERVED_METADATA_KEYS = _PROTECTED_PAYLOAD_KEYS | {
    "indexed_at",
    "tags",
    "timestamp",
    # Server-owned lifecycle marker: recall treats any payload carrying it
    # as stale, so a remote caller planting it would store memories that
    # default recall immediately hides ("stored" with a lie inside).
    # Retraction goes through the invalidate API, never through ingest.
    "invalidated_at",
}


def _parse_iso_timestamp(value: str):
    """`datetime` for an ISO-8601 string, or None when it doesn't parse.

    Tolerates a trailing ``Z`` (Python 3.10's ``fromisoformat`` doesn't).
    """
    from datetime import datetime as _dt

    try:
        return _dt.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


#: Per-tenant write serialization for THIS process. The quota preflight and
#: the store-backed duplicate check are read-then-write sequences; without
#: serialization two concurrent same-tenant requests could both pass a
#: near-cap preflight (exceeding the cap) or both embed the same new item.
#: One lock per tenant keeps unrelated tenants fully parallel. Cross-PROCESS
#: writers (multi-worker deployments) remain best-effort — the quota is a
#: guardrail, not a security boundary (documented since the quota feature
#: shipped), and duplicate ids still collapse to one stored point.
_TENANT_WRITE_LOCKS: dict[str | None, threading.Lock] = {}
_TENANT_WRITE_LOCKS_GUARD = threading.Lock()


def _tenant_write_lock(tenant: str | None) -> threading.Lock:
    with _TENANT_WRITE_LOCKS_GUARD:
        lock = _TENANT_WRITE_LOCKS.get(tenant)
        if lock is None:
            lock = threading.Lock()
            _TENANT_WRITE_LOCKS[tenant] = lock
        return lock


def _normalized_metadata_key(key: str) -> str:
    """NFKC-fold a metadata key for reserved-namespace matching.

    Exact-string matching alone would let `Tenant_Id` or a full-width
    underscore variant sail through and sit in the stored payload visually
    impersonating a structural field. Normalization is for MATCHING only —
    the original key is what gets stored when it is clean.
    """
    import unicodedata

    return unicodedata.normalize("NFKC", key).casefold()


def _instants_not_increasing(start, end) -> bool:
    """True when [start, end) is empty — the validity predicate is
    ``valid_from <= as_of < valid_until``, so start >= end can never match.
    Naive datetimes are compared as UTC (the stack convention)."""
    from datetime import timezone as _tz

    if start.tzinfo is None:
        start = start.replace(tzinfo=_tz.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=_tz.utc)
    return start >= end


def _utf8_encodable(value: str) -> bool:
    """Whether the string survives UTF-8 encoding (JSON permits lone
    surrogates like "\\ud800"; deterministic ids and the store transport
    do not — they must be a 400, never a 500)."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def reserved_metadata_keys(metadata: dict[str, Any]) -> list[str]:
    """Names in *metadata* a remote caller is not allowed to set.

    Returns a sorted list (empty = clean). Rejection is loud by design:
    silently stripping would let a client believe a forged `tenant_id` or
    `_id_scheme` was stored.
    """
    bad = set()
    for key in metadata:
        if not isinstance(key, str):
            bad.add(str(key))
            continue
        normalized = _normalized_metadata_key(key)
        if normalized.startswith("_") or normalized in _REMOTE_RESERVED_METADATA_KEYS:
            bad.add(key)
    return sorted(bad)


#: Qdrant's integer payload domain is signed int64 — an ASYMMETRIC range:
#: [-2**63, 2**63-1]. Python/JSON integers beyond it would embed first and
#: only fail (or silently lose precision) at upsert.
_STORE_INT_MIN = -(2**63)
_STORE_INT_MAX = 2**63 - 1

#: Nesting ceiling for the metadata walk: a few kilobytes of pathologically
#: nested lists would otherwise blow Python's recursion limit inside the
#: VALIDATOR itself — an uncaught 500 instead of a clean rejection. Real
#: payload metadata is a handful of levels; 32 is generous.
_METADATA_MAX_DEPTH = 32


def _find_unrepresentable_number(
    value: Any, path: str = "metadata", depth: int = 0
) -> str | None:
    """First metadata number the store cannot represent, or None.

    Walks nested dicts/lists with a hard depth ceiling (the walk must never
    be the thing that crashes on caller-shaped input). Rejects integers
    outside signed int64 and non-finite floats (json.dumps emits
    NaN/Infinity by default — invalid JSON for the store and undefined for
    range filters).
    """
    import math

    if depth > _METADATA_MAX_DEPTH:
        return f"{path} exceeds {_METADATA_MAX_DEPTH} nesting levels"
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        if value < _STORE_INT_MIN or value > _STORE_INT_MAX:
            return f"{path} integer exceeds the store's 64-bit domain"
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return f"{path} must be a finite number"
        return None
    if isinstance(value, dict):
        for k, v in value.items():
            found = _find_unrepresentable_number(v, f"{path}.{k}", depth + 1)
            if found:
                return found
        return None
    if isinstance(value, list):
        for i, v in enumerate(value):
            found = _find_unrepresentable_number(v, f"{path}[{i}]", depth + 1)
            if found:
                return found
        return None
    return None


def validate_remote_item(
    text: str,
    source: str,
    timestamp: str | None,
    tags: list[str],
    metadata: dict[str, Any],
    *,
    offset: int = 0,
    chunk: bool = False,
    reserved_extra: frozenset[str] | set[str] | None = None,
) -> str | None:
    """First violated constraint of one remote item, or None when clean.

    Shared by the HTTP endpoint and the MCP tool so both surfaces enforce
    the identical contract (caps + reserved namespace). ``chunk=True``
    raises the text ceiling to the document cap (the server splits it) but
    requires a non-empty ``source`` — chunk ids are (source, offset)-keyed,
    and an unnamed multi-chunk document would collide at offset positions
    with every other unnamed document. ``reserved_extra`` lets a deployment
    reserve additional payload keys (its configured text/timestamp keys) so
    metadata can't shadow what recall actually reads."""
    if not isinstance(text, str) or not text.strip():
        return "text must be a non-empty string"
    if not _utf8_encodable(text):
        return "text must be valid UTF-8 (no lone surrogates)"
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        return "offset must be a non-negative integer"
    if offset > REMOTE_MAX_OFFSET:
        return f"offset exceeds {REMOTE_MAX_OFFSET} (must fit the store's integer domain)"
    if chunk:
        if len(text) > REMOTE_MAX_DOC_CHARS:
            return f"text exceeds {REMOTE_MAX_DOC_CHARS} characters (chunked cap)"
        if not isinstance(source, str) or not source.strip():
            return "chunked items require a non-empty source"
        if offset != 0:
            # The server assigns window offsets for chunked documents; a
            # caller-supplied base would silently shift ids/positions.
            return "chunked items must leave offset at 0"
    elif len(text) > REMOTE_MAX_TEXT_CHARS:
        return f"text exceeds {REMOTE_MAX_TEXT_CHARS} characters (set chunk=true for documents)"
    if not isinstance(source, str):
        return "source must be a string"
    if len(source) > REMOTE_MAX_SOURCE_CHARS:
        return f"source exceeds {REMOTE_MAX_SOURCE_CHARS} characters"
    if not _utf8_encodable(source):
        return "source must be valid UTF-8 (no lone surrogates)"
    if "|" in source or any(ord(ch) < 0x20 for ch in source):
        # `stable_chunk_id` joins (source, offset, text) with "|" (and the
        # tenant with NUL): ("a", 0, "0|X") and ("a|0", 0, "X") would hash
        # identically, letting one memory masquerade as a duplicate of
        # unrelated content. The provenance verifier already treats
        # pipe-bearing sources as ambiguous — the untrusted surface rejects
        # them outright.
        return "source must not contain '|' or control characters"
    if timestamp is not None:
        if not isinstance(timestamp, str) or len(timestamp) > REMOTE_MAX_TIMESTAMP_CHARS:
            return "timestamp must be an ISO-8601 string"
        if _parse_iso_timestamp(timestamp) is None:
            # A stored-but-unparseable event time would silently drop the
            # memory out of temporal recall while reporting it stored.
            return "timestamp must be an ISO-8601 string"
    if not isinstance(tags, list) or len(tags) > REMOTE_MAX_TAGS:
        return f"tags must be a list of at most {REMOTE_MAX_TAGS} strings"
    for tag in tags:
        if not isinstance(tag, str) or len(tag) > REMOTE_MAX_TAG_CHARS:
            return f"each tag must be a string of at most {REMOTE_MAX_TAG_CHARS} characters"
        if not _utf8_encodable(tag):
            return "tags must be valid UTF-8 (no lone surrogates)"
    if not isinstance(metadata, dict):
        return "metadata must be an object"
    if len(metadata) > REMOTE_MAX_METADATA_KEYS:
        return f"metadata exceeds {REMOTE_MAX_METADATA_KEYS} keys"
    non_string = sorted(str(k) for k in metadata if not isinstance(k, str))
    if non_string:
        # Distinct from the reserved-namespace rejection: "reserved" tells a
        # caller to rename the key; this tells them the key TYPE is wrong.
        return "metadata keys must be strings: " + ", ".join(non_string)
    reserved = reserved_metadata_keys(metadata)
    if reserved_extra:
        # Both sides through the SAME normalization: a configured key stored
        # in a non-NFKC form must still catch its normalized client variant.
        normalized_extra = {_normalized_metadata_key(e) for e in reserved_extra}
        extra_hits = sorted(
            k for k in metadata if _normalized_metadata_key(k) in normalized_extra
        )
        reserved = sorted(set(reserved) | set(extra_hits))
    if reserved:
        return "metadata uses reserved key(s): " + ", ".join(reserved)
    # Lifecycle validity bounds are ALLOWED (legitimate world-time content)
    # but must parse: an unparseable bound degrades point-in-time recall to
    # lexicographic comparison — the memory joins/leaves history at
    # unrelated as_of instants (same rule as /triples).
    vf = metadata.get("valid_from")
    if vf is not None and (not isinstance(vf, str) or _parse_iso_timestamp(vf) is None):
        return "metadata.valid_from must be ISO-8601"
    vu = metadata.get("valid_until")
    if vu is not None and vu != "current":
        if not isinstance(vu, str) or _parse_iso_timestamp(vu) is None:
            return "metadata.valid_until must be ISO-8601 or 'current'"
        if vf is not None:
            vf_dt, vu_dt = _parse_iso_timestamp(vf), _parse_iso_timestamp(vu)
            if vf_dt is not None and vu_dt is not None and _instants_not_increasing(vf_dt, vu_dt):
                # An empty [from, until) window: the memory would be stored
                # but invisible to every point-in-time query.
                return "metadata.valid_from must precede metadata.valid_until"
    bad_number = _find_unrepresentable_number(metadata)
    if bad_number is not None:
        return bad_number
    try:
        import json

        encoded = json.dumps(metadata, ensure_ascii=False, default=None)
    except (TypeError, ValueError):
        return "metadata must be JSON-serializable"
    if len(encoded) > REMOTE_MAX_METADATA_CHARS:
        return f"metadata exceeds {REMOTE_MAX_METADATA_CHARS} serialized characters"
    if not _utf8_encodable(encoded):
        # A lone surrogate (valid JSON escape) survives json.dumps with
        # ensure_ascii=False and would crash UTF-8 encoding downstream —
        # id generation for text/source, the HTTP layer for metadata.
        return "metadata strings must be valid UTF-8 (no lone surrogates)"
    return None


@dataclass
class RemoteMemoryResult:
    """Per-item outcome of a remote ingest, in input order."""

    id: str
    status: Literal["stored", "duplicate", "failed"]
    #: Whether this item was sent to the embedding provider. False for
    #: duplicates AND for reactivations (a re-remember of an invalidated
    #: point reuses the stored vector — status "stored", zero embed cost),
    #: so metering can attribute actual provider spend, not statuses.
    embed_attempted: bool = False


class RemoteRequestTooLarge(ValueError):
    """A remote request expands past the per-request chunk budget."""


def expand_remote_items(
    entries: list[tuple[IngestItem, bool]],
    *,
    chunk_size: int = REMOTE_CHUNK_SIZE,
) -> tuple[list[IngestItem], list[int]]:
    """Expand ``(item, chunk?)`` pairs into flat ingest items.

    Chunked items are split into fixed character windows at offsets
    0, chunk_size, 2*chunk_size... — the exact split the CLI's prose
    indexer applies, so a document POSTed here and the same file indexed
    on the box produce identical chunk ids. Non-chunked items pass through
    with their caller-supplied offset. Returns the flat items plus, for
    each, the index of the request item it came from. Raises
    :class:`RemoteRequestTooLarge` when the expansion exceeds
    ``REMOTE_MAX_CHUNKS_PER_REQUEST`` — embedding work per request is
    bounded; the caller splits the request instead.
    """
    flat: list[IngestItem] = []
    origins: list[int] = []
    for idx, (item, do_chunk) in enumerate(entries):
        if do_chunk:
            pieces = [
                (start, item.text[start : start + chunk_size])
                for start in range(0, len(item.text), chunk_size)
            ]
        else:
            pieces = [(item.offset, item.text)]
        for start, piece in pieces:
            if not piece.strip():
                continue  # whitespace-only window: nothing to embed
            if len(flat) >= REMOTE_MAX_CHUNKS_PER_REQUEST:
                # Checked per PIECE so the budget is a hard boundary, not a
                # per-item overshoot window that widens with cap tuning.
                raise RemoteRequestTooLarge(
                    f"request expands to more than {REMOTE_MAX_CHUNKS_PER_REQUEST} "
                    "chunks — split it into smaller requests"
                )
            flat.append(
                IngestItem(
                    text=piece,
                    source=item.source,
                    offset=start,
                    metadata=dict(item.metadata),
                    tags=list(item.tags),
                    timestamp=item.timestamp,
                )
            )
            origins.append(idx)
    return flat, origins


#: Every payload key the ingest pipeline itself writes — a configured
#: schema key colliding with ANY of these corrupts a downstream step
#: (text_key="source" overwrites provenance; timestamp_key="tags" feeds a
#: float to the tag materializer and 500s every timestamped write).
_PIPELINE_PAYLOAD_KEYS = _PROTECTED_PAYLOAD_KEYS | {
    "tags",
    "timestamp",
    "indexed_at",
    # Lifecycle keys the reactivation/invalidate machinery reads and writes:
    # text_key="invalidated_at" would stamp every stored point with a truthy
    # stale marker — writes report "stored" while default recall hides them
    # ALL, silently and permanently. valid_from/valid_until would corrupt
    # as_of recall the same way (lexicographic fallback on non-ISO values).
    "invalidated_at",
    "valid_from",
    "valid_until",
}


def _valid_remote_predicate(predicate: str) -> bool:
    """Predicate contract for remote graph writes: a LETTER (any script —
    the store's sanitizer is Unicode-aware and keeps non-ASCII letters
    losslessly, so "работает_в" is as legitimate as "works_on"), then
    letters, decimal digits, or underscores. The store uppercases
    relationship types, so case variants of one predicate intentionally
    merge; what this EXCLUDES are the punctuation/space variants
    ("works-at", "works at") that would silently collapse into one edge
    type while both writes report success, the leading-digit forms whose
    sanitized shape a caller can never legally submit, and Unicode
    number-but-not-digit characters (superscripts ², Roman numerals Ⅳ,
    circled digits ① — categories No/Nl) that a regex ``\\w`` admits but
    that either get silently underscore-mangled by the store's sanitizer
    (leading position, both categories) or are rejected by Cypher's
    unescaped-identifier grammar (category No, any position); a
    non-leading Nl would actually survive both, but the whole class is
    rejected uniformly — one rule, no positional carve-outs. Accepted
    characters pass the sanitizer without substitution; uppercasing is
    the only transformation (a handful of letters uppercase into
    decomposed forms — ``ǰ`` → ``J̌`` — which the store and Cypher both
    accept).
    """
    return predicate[0].isalpha() and all(
        c.isalpha() or c.isdecimal() or c == "_" for c in predicate
    )

#: Bounds for one remote triple — enforced in the SHARED validator so the
#: MCP surface is capped identically to HTTP's pydantic schema (an
#: unbounded predicate would flow into a Cypher relationship token).
REMOTE_MAX_TRIPLE_CHARS = 512
REMOTE_MAX_VALIDITY_CHARS = 64


def validate_remote_triple(
    subject: str,
    predicate: str,
    obj: str,
    valid_from: str | None,
    valid_until: str | None,
) -> str | None:
    """First violated constraint of one remote graph triple, or None.

    Shared by POST /triples and the MCP graph_add_triple tool so both
    surfaces enforce the identical contract.
    """
    for field_name, value in (("subject", subject), ("predicate", predicate), ("object", obj)):
        if not isinstance(value, str) or not value.strip():
            return f"{field_name} must be a non-blank string"
        if len(value) > REMOTE_MAX_TRIPLE_CHARS:
            return f"{field_name} exceeds {REMOTE_MAX_TRIPLE_CHARS} characters"
        if not _utf8_encodable(value):
            return f"{field_name} must be valid UTF-8"
    if not _valid_remote_predicate(predicate):
        return (
            "predicate must be a relation identifier (letters, decimal digits, "
            "underscores; starting with a letter) — the store uppercases it"
        )
    vf_dt = None
    if valid_from is not None:
        if not isinstance(valid_from, str) or len(valid_from) > REMOTE_MAX_VALIDITY_CHARS:
            return "valid_from must be ISO-8601"
        vf_dt = _parse_iso_timestamp(valid_from)
        if vf_dt is None:
            return "valid_from must be ISO-8601"
    if valid_until is not None and valid_until != "current":
        if not isinstance(valid_until, str) or len(valid_until) > REMOTE_MAX_VALIDITY_CHARS:
            return "valid_until must be ISO-8601 or 'current'"
        vu_dt = _parse_iso_timestamp(valid_until)
        if vu_dt is None:
            return "valid_until must be ISO-8601 or 'current'"
        if vf_dt is not None and _instants_not_increasing(vf_dt, vu_dt):
            return "valid_from must precede valid_until"
    return None


#: Cap for one remote lifecycle request (invalidate / hard delete). Ids are
#: cheap to validate but every one costs a store round-trip in the ownership
#: check — bounded like every other remote request.
REMOTE_MAX_IDS = 256
#: Owner-guard path cap, shared with the HTTP models' max_length.
REMOTE_MAX_INDEX_ROOT_CHARS = 4096
#: Qdrant's point id domain: an unsigned 64-bit integer or a UUID. Anything
#: else is not an id this system could have produced (stable_chunk_id emits
#: UUID-shaped strings) and would surface as an opaque backend error instead
#: of the promised 400 — reject it up front.
_QDRANT_ID_MAX = 2**64 - 1
_REMOTE_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _is_numeric_id_string(s: str) -> bool:
    # isascii() matters twice: non-ASCII decimals ('٧') would silently
    # convert to a different-looking numeric id, and category-No digits
    # ('²') pass isdigit() but CRASH int(). ASCII digits only.
    return s.isascii() and s.isdigit()


def coerce_point_ids(ids: Sequence[str | int]) -> list[str | int]:
    """Digit-only string ids become ints so numeric-id collections match.

    Qdrant stores integer ids as integers; a JSON caller often sends them
    as strings. UUID ids contain hyphens, so they stay strings but are
    LOWERCASED: the store canonicalizes UUIDs to lowercase, and the
    ownership/existence checks compare ids as case-sensitive strings — an
    uppercase spelling of an existing id would silently no-op. Shared by
    the HTTP lifecycle endpoints and the MCP invalidate tool. The length
    gate mirrors validate_remote_ids: past CPython's int-from-str digit
    limit int() RAISES, and a library caller may not have validated first.
    """
    out: list[str | int] = []
    for x in ids:
        if isinstance(x, str):
            if _is_numeric_id_string(x):
                # Significant digits only — leading zeros don't add range.
                # The magnitude check keeps this helper safe standalone: an
                # unvalidated over-u64 digit string passes through as a
                # string instead of becoming an out-of-range int.
                digits = x.lstrip("0") or "0"
                if len(digits) <= 20 and int(digits) <= _QDRANT_ID_MAX:
                    out.append(int(digits))
                    continue
            if _REMOTE_UUID_RE.fullmatch(x):
                out.append(x.lower())
                continue
        out.append(x)
    return out


def validate_remote_ids(ids: Sequence[Any]) -> str | None:
    """First violated constraint of a remote id list, or None."""
    if not isinstance(ids, (list, tuple)) or not ids:
        return "ids must be a non-empty list"
    if len(ids) > REMOTE_MAX_IDS:
        return f"at most {REMOTE_MAX_IDS} ids per request"
    for i, pid in enumerate(ids):
        # bool is an int subclass — True would silently target point id 1.
        # (The HTTP/MCP schemas use StrictInt so a JSON boolean never even
        # coerces this far; this guard covers library callers.)
        if isinstance(pid, bool) or not isinstance(pid, (str, int)):
            return f"ids[{i}] must be a string or integer"
        if isinstance(pid, int):
            if pid < 0 or pid > _QDRANT_ID_MAX:
                return f"ids[{i}] must fit an unsigned 64-bit point id"
        elif _is_numeric_id_string(pid):
            # Same magnitude bound as literal ints — coerce_point_ids will
            # convert this string, and an over-range int is a backend error.
            # Length gate BEFORE int(): CPython's int-from-str digit limit
            # (~4300, sys.get_int_max_str_digits) makes int() itself RAISE
            # on a long enough digit string — a 500, not the promised 400.
            # u64 needs at most 20 SIGNIFICANT digits — leading zeros are
            # stripped first ("007" is documented as point 7, so a
            # zero-padded 21-char spelling of a valid id must not bounce).
            digits = pid.lstrip("0") or "0"
            if len(digits) > 20 or int(digits) > _QDRANT_ID_MAX:
                return f"ids[{i}] must fit an unsigned 64-bit point id"
        elif not _REMOTE_UUID_RE.fullmatch(pid):
            return f"ids[{i}] must be a UUID or an unsigned 64-bit integer"
    return None


def validate_remote_invalidate(
    ids: Sequence[Any],
    invalidated_at: str | None,
    valid_until: str | None,
    index_root: str | None = None,
) -> str | None:
    """First violated constraint of one remote invalidate call, or None.

    Shared by POST /invalidate and the MCP mnemostack_invalidate tool so
    both surfaces enforce the identical contract. The two timestamps live
    on different axes (system-time vs world-time), so no ordering between
    them is required.
    """
    problem = validate_remote_ids(ids)
    if problem:
        return problem
    for field_name, value in (
        ("invalidated_at", invalidated_at),
        ("valid_until", valid_until),
    ):
        if value is None:
            continue
        if (
            not isinstance(value, str)
            or len(value) > REMOTE_MAX_VALIDITY_CHARS
            or _parse_iso_timestamp(value) is None
        ):
            return f"{field_name} must be ISO-8601"
    if index_root is not None:
        if not isinstance(index_root, str) or not index_root.strip():
            # A blank owner guard matches NO owner — every id would be
            # silently skipped while the response reads like a no-op.
            return "index_root must be a non-blank string"
        if len(index_root) > REMOTE_MAX_INDEX_ROOT_CHARS:
            # Same bound as the HTTP models — the MCP surface relies
            # solely on this validator for the cap.
            return f"index_root exceeds {REMOTE_MAX_INDEX_ROOT_CHARS} characters"
        if not _utf8_encodable(index_root):
            return "index_root must be valid UTF-8"
    return None


def ensure_remote_schema_keys(text_key: str, timestamp_key: str) -> None:
    """Fail loud on a schema-key configuration the write path cannot honor.

    Called at SERVICE BOOT by both surfaces (a misconfigured deployment
    must not start and then 500 on every write) and defensively at the
    ingest boundary for library callers.
    """
    if not isinstance(text_key, str) or not text_key.strip():
        raise ValueError("text_key must be a non-blank string")
    if not isinstance(timestamp_key, str) or not timestamp_key.strip():
        raise ValueError("timestamp_key must be a non-blank string")
    if text_key != "text" and text_key.startswith("_"):
        # The whole underscore namespace is server-structural by convention
        # (ownership markers like _enrich_keys/_md_keys included — a text
        # mirror there would make refresh iterate garbage or delete
        # unrelated fields). Same rule client metadata already obeys.
        raise ValueError(f"text_key {text_key!r} is in the reserved underscore namespace")
    if timestamp_key != "timestamp" and timestamp_key.startswith("_"):
        raise ValueError(
            f"timestamp_key {timestamp_key!r} is in the reserved underscore namespace"
        )
    if text_key != "text" and text_key in _PIPELINE_PAYLOAD_KEYS:
        raise ValueError(
            f"text_key {text_key!r} collides with a payload field the ingest "
            "pipeline writes"
        )
    if timestamp_key != "timestamp" and timestamp_key in _PIPELINE_PAYLOAD_KEYS:
        raise ValueError(
            f"timestamp_key {timestamp_key!r} collides with a payload field "
            "the ingest pipeline writes"
        )
    if text_key == timestamp_key and text_key != "text":
        raise ValueError(
            "text_key and timestamp_key must differ — one field cannot carry both"
        )


def ingest_remote_items(
    embedding: EmbeddingProvider,
    store: VectorStore,
    items: list[IngestItem],
    *,
    tenant: str | None = None,
    max_points: int | None = None,
    text_key: str = "text",
    timestamp_key: str = "timestamp",
    timestamp_format: str = "iso",
) -> list[RemoteMemoryResult]:
    """Ingest client-supplied items with store-backed duplicate detection.

    The deterministic id of every item is computed up front; ids already in
    the store (tenant-scoped when scoped) — and repeats within the request —
    are reported as ``duplicate`` WITHOUT embedding, so a client retry or a
    re-sent conversation costs zero provider calls. The rest go through a
    fresh :class:`Ingestor` (space guard, quota check before upsert,
    resilient batch embedding). Raises ``QuotaExceededError`` /
    ``EmbeddingSpaceError`` for the caller's surface to map; per-item
    embedding failures are reported as ``failed``, never raised.

    ``text_key``/``timestamp_key``/``timestamp_format`` mirror the
    deployment's recall schema: on a collection with non-default keys the
    payload additionally carries the text under ``text_key`` and the event
    time under ``timestamp_key`` (converted into the collection's own
    domain — iso/epoch/epoch_ms), so remotely written memories are
    readable and temporally recallable exactly like operator-indexed ones.

    The read-then-write sequences (quota preflight, duplicate detection)
    are serialized per tenant within this process; cross-process writers
    remain best-effort (the quota is a guardrail — see deployment docs).

    Validation (`validate_remote_item`) is the CALLER's obligation — this
    function trusts its items are within caps and clean of reserved keys.
    """
    ensure_remote_schema_keys(text_key, timestamp_key)
    # First remote write on a fresh deployment must not require an operator
    # ingest to have created the collection. Bootstrap runs ONCE per store
    # instance: on a sparse-aware store ensure_collection re-verifies sparse
    # coverage with collection-wide counts, which must not be paid on every
    # small write. ensure_collection is check-then-create: two concurrent
    # FIRST writers (any tenants — the per-tenant lock below cannot cover
    # this) can both see "missing" and race the create; the loser checks
    # whether the winner's collection now exists instead of retrying blind —
    # a genuine failure (store down, dimension mismatch) re-raises without
    # doubling load.
    if not getattr(store, "_remote_bootstrap_done", False):
        # Existence is sampled BEFORE ensure: the create race has exactly one
        # shape — absent before, present after. An ensure failure on a
        # PRE-EXISTING collection is validation (dimension mismatch, missing
        # sparse space) and must propagate untouched; treating "exists now"
        # alone as proof of a race would swallow exactly those guards.
        # Hook PRESENCE is probed with getattr, never `except
        # AttributeError` around the call: an AttributeError raised INSIDE
        # an implemented hook is a genuine failure (broken adapter, wire
        # format change) and must propagate — not be mistaken for a duck
        # store without the hook and silently skipped.
        exists_fn = getattr(store, "collection_exists", None)
        ensure_fn = getattr(store, "ensure_collection", None)
        existed_before: bool | None = (
            exists_fn() if callable(exists_fn) else None
        )  # duck store without the hook: cannot discriminate a race
        if callable(ensure_fn):
            try:
                ensure_fn()
            except Exception:
                appeared = False
                if existed_before is False and callable(exists_fn):
                    try:
                        appeared = bool(exists_fn())
                    except Exception:  # noqa: BLE001 — keep the ORIGINAL error
                        appeared = False
                if not appeared:
                    raise  # genuine failure (or undiscriminable duck): loud
                # Lost the concurrent create race — the winner's collection
                # is up, but it still has to pass OUR validation (dimension,
                # sparse space): re-run ensure and let it raise honestly.
                ensure_fn()
        try:
            # Instance-level marker, deliberately not part of the store
            # protocol (a slotted/frozen duck store just re-runs bootstrap).
            store._remote_bootstrap_done = True  # type: ignore[attr-defined]
        except AttributeError:
            pass
    with _tenant_write_lock(tenant):
        return _ingest_remote_items_locked(
            embedding,
            store,
            items,
            tenant=tenant,
            max_points=max_points,
            text_key=text_key,
            timestamp_key=timestamp_key,
            timestamp_format=timestamp_format,
        )


def _apply_timestamp_domain(
    items: list[IngestItem], timestamp_key: str, timestamp_format: str
) -> None:
    """Map each item's ISO event time onto the collection's schema, in place.

    Conversion is keyed to the FORMAT, not the key name: a deployment with
    the default ``timestamp`` key but a numeric format still needs its epoch
    value. The converted value travels via ``metadata`` (the payload merge),
    and for the default key the explicit field is cleared — the pipeline
    treats an explicit item timestamp as authoritative, which would shadow
    the conversion. Naive ISO inputs are treated as UTC (the recall validity
    convention) — ``datetime.timestamp()`` on a naive value would otherwise
    use the server's LOCAL zone and silently skew every stored instant.
    """
    from datetime import timezone as _tz

    if timestamp_format not in ("epoch", "epoch_ms"):
        # "iso" — and, deliberately, ANY unrecognized format: the safe
        # default is ISO passthrough (the codebase-wide convention —
        # _emit_epoch_bound, validity utils). Converting on an unvalidated
        # format string would let a typo'd config silently replace every
        # ISO event time with bogus numbers. The service surfaces validate
        # the format at boot, so this branch is their "iso" path and the
        # library caller's safety net.
        if timestamp_key != "timestamp":
            for item in items:
                if item.timestamp:
                    # Mirror under the configured key; the historical
                    # `timestamp` field keeps ISO via the explicit field.
                    item.metadata[timestamp_key] = item.timestamp
        return
    for item in items:
        if not item.timestamp:
            continue
        dt = _parse_iso_timestamp(item.timestamp)
        if dt is None:
            continue  # validated surfaces never get here; fail soft for ducks
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_tz.utc)
        value = dt.timestamp() * (1000 if timestamp_format == "epoch_ms" else 1)
        item.metadata[timestamp_key] = value
        if timestamp_key == "timestamp":
            item.timestamp = None  # would shadow the numeric value otherwise


def _ingest_remote_items_locked(
    embedding: EmbeddingProvider,
    store: VectorStore,
    items: list[IngestItem],
    *,
    tenant: str | None,
    max_points: int | None,
    text_key: str,
    timestamp_key: str,
    timestamp_format: str,
) -> list[RemoteMemoryResult]:
    ids = [
        stable_chunk_id(item.source, item.offset, item.text, tenant=tenant)
        for item in items
    ]
    tkw: dict[str, Any] = {"tenant": tenant} if tenant is not None else {}
    # getattr, not `except AttributeError` around the call: an internal
    # AttributeError from an implemented probe must propagate — swallowed,
    # it would re-embed every duplicate and 507 tenants at their quota.
    probe = getattr(store, "retrieve_existing_ids", None)
    existing = (
        probe(list(ids), **tkw)
        if callable(probe)
        else set()  # duck store without the hook: everything embeds
    )
    to_ingest: list[IngestItem] = []
    seen_now: set[str] = set()
    for pid, item in zip(ids, items, strict=True):
        if pid in existing or pid in seen_now:
            continue
        seen_now.add(pid)
        to_ingest.append(item)
    # QUOTA FIRST: every effect of this call — new points AND lifecycle
    # reactivations below — must sit behind the preflight, or a
    # 507-rejected request would still have un-retracted memories
    # (commit-nothing means nothing).
    if to_ingest and tenant is not None and max_points is not None:
        enforce_points_quota(
            tenant, store.count(tenant=tenant), len(to_ingest), max_points
        )
    stored: set[str] = set()
    if to_ingest:
        # Map event times and the text mirror onto the collection's schema
        # BEFORE the pipeline: conversion happens only for items actually
        # being stored (ids are (source, offset, text)-derived — neither
        # timestamps nor the mirror are id material). The mirror travels via
        # metadata — a STRUCTURAL payload field — never via the enrichment
        # hook: apply_enrichment records its keys in _enrich_keys, and a
        # later `index --refresh-payloads` run WITHOUT an enricher would
        # treat the schema field as stale enrichment and delete it, blanking
        # recall for deployments with a custom text_key.
        if text_key != "text":
            # Server-injected structural field: deliberately OUTSIDE the
            # client metadata caps (those bound caller-controlled data; the
            # mirror is server-owned and bounded by the text caps).
            for item in to_ingest:
                item.metadata[text_key] = item.text
        _apply_timestamp_domain(to_ingest, timestamp_key, timestamp_format)
        # (Quota was preflighted above, before ANY effect of this call —
        # including reactivations. Deliberately CONSERVATIVE: items that
        # will later fail embedding still count, since which one fails is
        # unknowable pre-embed — a mixed batch at the cap edge is rejected
        # whole rather than the commit-nothing invariant weakened.)
        def _meter_spend() -> None:
            # Per-tenant embedding-spend attribution — not from result
            # statuses: duplicates and reactivations never reach this
            # branch (zero cost), per-item retries inside the resilience
            # ladder count once (retry amplification is provider health,
            # not tenant behavior), and chars are the provider-agnostic
            # token proxy. Emitted in the shared layer, so the MCP
            # remember tool attributes identically into its process
            # recorder.
            if tenant is None:
                return
            counter(
                "mnemostack.tenant.embedded_chunks",
                len(to_ingest),
                labels={"tenant": tenant},
            )
            counter(
                "mnemostack.tenant.embedded_chars",
                sum(len(item.text) for item in to_ingest),
                labels={"tenant": tenant},
            )

        ingestor = Ingestor(
            embedding,
            store,
            # One flush for the whole request (expansion is capped below this),
            # so the Ingestor's own per-flush quota re-check — kept as
            # defense-in-depth against concurrent writers — can never split
            # the request into a committed half and a rejected half.
            batch_size=REMOTE_MAX_CHUNKS_PER_REQUEST,
            skip_seen=False,  # duplicates were resolved against the STORE above
            tenant=tenant,
            max_points=max_points,
        )
        # Emission is EXCEPTION-AWARE, after the attempt: the space guard
        # aborts BEFORE any provider call (EmbeddingSpaceError, the 503
        # deployment-misconfig shape — billing it would inflate the meters
        # on every retry for the whole incident), while every OTHER ingest
        # failure (upsert, storage, quota re-check) happens after embedding
        # — that spend is real and must be attributed even though the
        # request 5xxes. `except Exception`, NOT BaseException: interrupt/
        # cancellation signals (KeyboardInterrupt, SystemExit,
        # CancelledError on client disconnect) can fire pre-embed and must
        # propagate unbilled. Documented residuals, both bounded to one
        # request: a post-embed EmbeddingSpaceError from the sandwich/
        # revalidation checks (concurrent space flip) undercounts; a hard
        # crash mid-batch in a per-item embedding fallback bills the whole
        # request as an UPPER BOUND (exact per-item accounting on crash
        # paths would mean threading counts out of the embedding layer —
        # deliberately not done for a monitoring proxy).
        try:
            stats = ingestor.ingest(to_ingest)
        except Exception as exc:
            if not isinstance(exc, EmbeddingSpaceError):
                _meter_spend()
            raise
        _meter_spend()
        stored = {str(pid) for pid in stats.ids}
    # Lifecycle: an existing id may be a RETRACTED memory (invalidated_at
    # set). Re-remembering the same fact must make it recallable again —
    # reporting a hidden point as "duplicate" would claim success while
    # default recall stays empty. Reactivate via a payload patch clearing
    # ONLY invalidated_at: valid_until is dual-use (the invalidate call
    # MAY set it, but it is equally legitimate ingest-declared expiry, and
    # payloads carry no provenance to tell the two apart) — deleting it
    # would destroy client content, so it is PRESERVED and the as_of
    # caveat documented instead. Content is byte-identical, so the stored
    # vector stays valid — no re-embedding. Duck stores without the hooks
    # keep the historical duplicate semantics; a failing patch propagates.
    # ORDER: this runs AFTER the new-item ingest above — embedding calls an
    # external provider and is the failure-prone step; were the store-local
    # patch applied first, an embedding/space failure would 5xx the request
    # with the reactivation already visible. The residual (patch raising
    # after a successful upsert) shares the store the upsert just wrote —
    # correlated availability — and still propagates loudly.
    reactivated: set[str] = set()
    if existing:
        # Same getattr-not-except discipline as the duplicate probe above.
        # `collection` is part of the probed SHAPE too: a duck store with a
        # client but no collection attribute keeps the historical duplicate
        # semantics instead of raising on the direct attribute access.
        retrieve_fn = getattr(getattr(store, "client", None), "retrieve", None)
        collection = getattr(store, "collection", None)
        stale_ids: list[str] = []
        if callable(retrieve_fn) and collection is not None:
            points = retrieve_fn(collection, ids=list(existing), with_payload=True)
            stale_ids = [
                str(pt.id)
                for pt in points
                if (getattr(pt, "payload", None) or {}).get("invalidated_at")
            ]
        # else: duck store — historical duplicate semantics
        if stale_ids:
            from mnemostack.vector.patch import PayloadPatch

            patched = store.apply_payload_patches(
                [
                    PayloadPatch(id=pid, delete_keys=("invalidated_at",))
                    for pid in stale_ids
                ],
                **tkw,
            )
            if patched == len(stale_ids):
                reactivated = set(stale_ids)
            else:
                # The patch silently skips points that vanished mid-flight
                # (concurrent prune/delete) — re-verify instead of reporting
                # "stored" for a memory that no longer exists; unverified
                # ids drop out of `existing` so they surface as failed.
                # (stale_ids non-empty implies retrieve_fn/collection were
                # present — the guard is for the type checker.)
                verify = (
                    retrieve_fn(collection, ids=stale_ids, with_payload=True)
                    if callable(retrieve_fn)
                    else []
                )
                cleared = {
                    str(pt.id)
                    for pt in verify
                    if not (getattr(pt, "payload", None) or {}).get("invalidated_at")
                }
                reactivated = cleared
                for pid in stale_ids:
                    if pid not in cleared:
                        existing.discard(pid)
    results: list[RemoteMemoryResult] = []
    first_seen: set[str] = set()
    for pid in ids:
        status: Literal["stored", "duplicate", "failed"]
        if pid in first_seen:
            # An in-request repeat mirrors its first occurrence: duplicate
            # only when the content actually exists — a repeat of a FAILED
            # item must not read as stored-elsewhere.
            status = "duplicate" if (pid in stored or pid in existing) else "failed"
        elif pid in stored or pid in reactivated:
            # Reactivated = the memory became recallable again: the caller's
            # intent succeeded, and "duplicate" would undersell the change.
            status = "stored"
        elif pid in existing:
            status = "duplicate"
        else:
            status = "failed"
        # seen_now = exactly the ids handed to the Ingestor (the embedding
        # provider was called for them, stored or failed); reactivations,
        # store-duplicates, and in-request repeats never were.
        attempted = pid in seen_now and pid not in first_seen
        first_seen.add(pid)
        results.append(
            RemoteMemoryResult(id=pid, status=status, embed_attempted=attempted)
        )
    counter("mnemostack.ingest.remote_items", len(items))
    counter("mnemostack.ingest.remote_stored", sum(r.status == "stored" for r in results))
    return results


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

    has_scroll = hasattr(vector_store, "scroll")
    if (
        existing is None
        and has_scroll
        and len(fresh_ids_by_source) <= selective_scan_max_sources
    ):
        # Selective re-index: narrow server-indexed scans, one per source —
        # but payload-bearing, because a Qdrant MatchValue filter ALSO
        # matches array payloads containing the value ({"source": ["a.md"]}
        # matches source="a.md"), and such points must be revalidated and
        # skipped exactly like the snapshot path skips them, never deleted.
        for source, fresh_ids in fresh_ids_by_source.items():
            filters: dict[str, Any] = {"source": source}
            if index_root is not None:
                filters["index_root"] = index_root
            for hit in vector_store.scroll(filters=filters, **tkw):
                scanned += 1
                payload = hit.payload or {}
                if payload.get("source") != source:
                    continue  # array/malformed source merely matched the filter
                if index_root is not None and payload.get("index_root") != index_root:
                    continue
                # str() ONLY for the fresh-set membership check — the delete
                # must carry the backend's raw id: an integer point 2 is not
                # deletable as the string "2", and a stringified delete would
                # still be COUNTED while leaving the point searchable.
                if str(hit.id) not in fresh_ids:
                    stale.append(hit.id)
                    if len(stale) >= delete_batch_size:
                        _flush()
    elif existing is None and not has_scroll:
        # A custom store exposing only iter_ids/delete_points: keep the
        # historical per-source discovery verbatim (its filters carry its
        # own exact-match semantics), with bounded tenant-validated deletes.
        for source, fresh_ids in fresh_ids_by_source.items():
            filters = {"source": source}
            if index_root is not None:
                filters["index_root"] = index_root
            for raw_id in vector_store.iter_ids(filters=filters, **tkw):
                scanned += 1
                if str(raw_id) not in fresh_ids:
                    stale.append(raw_id)
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
            # A non-string source can never equal a fresh-map key — and an
            # unhashable one must not crash the membership test. (A Qdrant
            # source filter WOULD match an array containing the value; both
            # discovery modes deliberately skip such points.)
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
