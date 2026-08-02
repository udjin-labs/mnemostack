"""Verifiable citations — resolve a chunk id back to its source fragment.

A recall citation (``[id:...]``) has always been a *deterministic label*:
``stable_chunk_id`` commits to ``(source, offset, text)`` cryptographically and
the payload stores all three fields. This module adds the missing half of the
provenance contract — the *verification*: re-read the source document, locate
the fragment, and return an honest verdict instead of trusting the label.

Verdicts (``Resolution.verdict``):

- ``intact``          — the source still contains the fragment at its recorded
  position. When the stored document snapshot hash matches, this is implied by
  ingest determinism without any search.
- ``source_changed``  — the fragment is still at its recorded position, but the
  document changed elsewhere (snapshot hash mismatch). The citation is still
  supported; its surrounding context drifted.
- ``moved``           — the exact fragment text exists in the document, but at
  a different position (edits above it shifted offsets). Still supported.
- ``changed``         — the document exists but no longer contains the exact
  fragment text. The citation is NOT supported by the current source.
- ``missing``         — the source document cannot be found (deleted/renamed).
- ``unresolvable``    — the payload carries no source, or the source is not a
  local document this resolver can read. An honest "cannot verify", not an
  error.

The resolver never runs inside the recall path — recall latency is untouched.
Verdicts are about the CURRENT state of the source; they never mutate stored
memory. ``supported`` is the one-bit summary callers usually want: does the
current source still evidence this citation?
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Payload keys written at ingest for the document snapshot. Additive and
#: optional: points ingested before this feature simply resolve with
#: ``snapshot="absent"`` — no migration.
SOURCE_HASH_KEY = "source_content_hash"
SOURCE_CAPTURED_KEY = "source_captured_at"


def source_content_hash(text: str) -> str:
    """Snapshot hash of a source document's decoded text.

    Hashes the *decoded* text (the same form the indexer reads with
    ``errors="ignore"``), so ingest-side and resolve-side hashes compare the
    identical byte stream regardless of on-disk encoding quirks."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_snapshot(text: str) -> dict[str, str]:
    """The two snapshot payload fields for a document being ingested."""
    return {
        SOURCE_HASH_KEY: source_content_hash(text),
        SOURCE_CAPTURED_KEY: datetime.now(timezone.utc).isoformat(),
    }


@dataclass
class Resolution:
    """The outcome of resolving one citation against its current source."""

    chunk_id: str
    verdict: str
    #: Whether the CURRENT source still evidences the citation.
    supported: bool
    source: str
    resolved_path: str | None
    #: "match" | "mismatch" | "absent" — stored snapshot hash vs current.
    snapshot: str
    stored_offset: int | None
    found_offset: int | None
    #: The fragment as read from the CURRENT source (when locatable).
    fragment: str | None
    detail: str
    captured_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SUPPORTED_VERDICTS = frozenset({"intact", "source_changed", "moved"})


def _resolution(chunk_id: str, payload: dict[str, Any], **kw: Any) -> Resolution:
    kw.setdefault("source", str(payload.get("source", "")))
    kw.setdefault("resolved_path", None)
    kw.setdefault("snapshot", "absent")
    kw.setdefault("stored_offset", None)
    kw.setdefault("found_offset", None)
    kw.setdefault("fragment", None)
    kw.setdefault("captured_at", payload.get(SOURCE_CAPTURED_KEY))
    kw["supported"] = kw["verdict"] in _SUPPORTED_VERDICTS
    return Resolution(chunk_id=chunk_id, **kw)


#: Refuse to read source documents larger than this (bytes). Verification is
#: a metadata operation — an unbounded read_text would let one resolve call
#: pull an arbitrarily large file into memory.
MAX_RESOLVE_BYTES = 32 * 1024 * 1024


def _candidate_paths(
    source: str, index_root: str | None, root: str | None, allow_unrooted: bool
) -> tuple[list[tuple[Path | None, Path]], bool]:
    """(base, candidate) pairs the source may live at, plus an escape flag.

    ``source`` is an UNTRUSTED label (any ingest caller writes it) that this
    module is the first to open as a filesystem path — so candidates are
    CONFINED: a root-joined candidate must resolve inside its root
    (``.resolve()`` + ``is_relative_to``), which kills ``..`` traversal and
    symlink escapes. An explicit ``root`` SELECTS the corpus — no fallback to
    the stored index_root (a source deleted from the selected root must
    report missing, not resolve intact against the stale copy). A bare
    ``source`` path (absolute, or relative to the process cwd) is inherently
    unconfined and is only ever tried for ``allow_unrooted`` callers (the
    operator CLI) when no root is known at all."""
    pairs: list[tuple[Path | None, Path]] = []
    escaped = False
    base_str = root or index_root
    if base_str:
        base = Path(base_str)
        candidate = base / source
        try:
            inside = candidate.resolve().is_relative_to(base.resolve())
        except OSError:
            inside = False
        if inside:
            pairs.append((base, candidate))
        else:
            escaped = True
    elif allow_unrooted:
        pairs.append((None, Path(source)))
    return pairs, escaped


_URI_SOURCE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")


def _root_allowed(base: str, allowed_roots: list[str]) -> bool:
    """Whether ``base`` (the corpus root about to be read) lies inside one of
    the operator-configured allowlist directories."""
    try:
        resolved = Path(base).resolve()
        return any(resolved.is_relative_to(Path(ar).resolve()) for ar in allowed_roots)
    except OSError:
        return False


def _search_texts(raw: str, payload: dict[str, Any]) -> list[str]:
    """The texts a stored offset may index into.

    Markdown chunk offsets are relative to the BODY (frontmatter stripped by
    the indexer), so for markdown points the body is searched FIRST — a
    ``found_offset`` should live in the same coordinate system as the stored
    offset whenever possible. Plain ingest offsets index the raw document."""
    texts = [raw]
    if "_md_keys" in payload or "heading_path" in payload:
        try:
            from .markdown.parse import parse_frontmatter

            _meta, body = parse_frontmatter(raw)
            if body != raw:
                texts.insert(0, body)
        except Exception:  # noqa: BLE001 - fall back to raw-only search
            pass
    return texts


def _fragment_variants(text: str, payload: dict[str, Any]) -> list[str]:
    """The stored text as it may appear IN the source document.

    The markdown chunker prepends synthetic heading context
    (``[Parent > Path]\\n<body>``) to some chunks for embedding quality — that
    prefix does not exist in the source. Try both readings rather than guess:
    a chunk whose real text begins with ``[`` keeps its full-text variant, a
    prefixed chunk matches via the stripped one."""
    variants = [text]
    if payload.get("heading_path") and text.startswith("["):
        _prefix, sep, rest = text.partition("]\n")
        if sep and rest:
            variants.append(rest)
    return variants


def _at_offset(candidate: str, offset: int, fragment: str) -> bool:
    """Fragment present at ``offset``, modulo stripped leading whitespace.

    Exact match at the offset first. Otherwise: the chunker strips segment
    slices before recording them, so the recorded offset can point at
    whitespace directly preceding the fragment — a whitespace-only gap is
    still "the recorded position"; any real inserted text is not. Both sides
    are lstripped for that comparison, because a mid-section window fragment
    can itself begin with whitespace."""
    if candidate[offset : offset + len(fragment)] == fragment:
        return True
    window = candidate[offset : offset + len(fragment) + 4096]
    lean = fragment.lstrip()
    return bool(lean) and window.lstrip()[: len(lean)] == lean


def resolve_payload(
    chunk_id: str,
    payload: dict[str, Any],
    *,
    root: str | None = None,
    allow_unrooted: bool = False,
    text_key: str = "text",
    allowed_roots: list[str] | None = None,
) -> Resolution:
    """Resolve a citation given its point payload (see module docstring).

    ``root`` overrides where sources are looked up (default: the payload's
    own ``index_root``); resolution is CONFINED to that root. Bare source
    paths (no root known) are only tried when ``allow_unrooted=True`` — the
    operator-CLI trust level; service surfaces keep the default.

    ``text_key`` is the payload key holding the chunk text (the deployment's
    configured payload schema — a mounted foreign collection keeps its own
    field names).

    ``allowed_roots`` is the operator-configured resolution allowlist for
    SERVICE surfaces: the corpus root actually used must live inside one of
    these directories. The stored ``index_root`` is payload data — writable
    by whoever ingests — so it cannot be its own security boundary; ``None``
    (operator/CLI trust) skips the check, an empty list disables resolution
    entirely (the fail-closed service default)."""
    text = payload.get(text_key)
    source = payload.get("source")
    if not source or not isinstance(source, str) or not isinstance(text, str) or not text:
        return _resolution(
            chunk_id,
            payload,
            verdict="unresolvable",
            detail=f"payload carries no resolvable source/{text_key} pair",
        )
    if _URI_SOURCE.match(source):
        return _resolution(
            chunk_id,
            payload,
            verdict="unresolvable",
            detail="source is not a local document (URI-style source)",
        )
    index_root = payload.get("index_root")
    base_str = root or (index_root if isinstance(index_root, str) else None)
    if allowed_roots is not None:
        if not allowed_roots:
            return _resolution(
                chunk_id,
                payload,
                verdict="unresolvable",
                detail=(
                    "resolution is not enabled on this surface — set "
                    "MNEMOSTACK_RESOLVE_ROOTS to the corpus directories this "
                    "process may read"
                ),
            )
        if not base_str or not _root_allowed(base_str, allowed_roots):
            return _resolution(
                chunk_id,
                payload,
                verdict="unresolvable",
                detail=(
                    "the point's corpus root is not in the operator-configured "
                    "resolution allowlist"
                ),
            )
    candidates, escaped = _candidate_paths(
        source, index_root if isinstance(index_root, str) else None, root, allow_unrooted
    )
    if escaped:
        return _resolution(
            chunk_id,
            payload,
            verdict="unresolvable",
            detail="source path escapes the corpus root — refusing to read it",
        )
    if not candidates:
        return _resolution(
            chunk_id,
            payload,
            verdict="unresolvable",
            detail=(
                "no corpus root is known for this point and unrooted "
                "resolution is not allowed on this surface"
            ),
        )
    path = next((p for _base, p in candidates if p.is_file()), None)
    if path is None:
        return _resolution(
            chunk_id,
            payload,
            verdict="missing",
            detail="source document not found under the known corpus root",
        )
    try:
        if path.stat().st_size > MAX_RESOLVE_BYTES:
            return _resolution(
                chunk_id,
                payload,
                verdict="unresolvable",
                resolved_path=str(path),
                detail=f"source larger than the {MAX_RESOLVE_BYTES}-byte verification cap",
            )
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        return _resolution(
            chunk_id,
            payload,
            verdict="unresolvable",
            resolved_path=str(path),
            detail=f"source exists but cannot be read: {e}",
        )

    stored_hash = payload.get(SOURCE_HASH_KEY)
    current_hash = source_content_hash(raw)
    if isinstance(stored_hash, str) and stored_hash:
        snapshot = "match" if stored_hash == current_hash else "mismatch"
    else:
        snapshot = "absent"
    offset = payload.get("offset")
    stored_offset = offset if isinstance(offset, int) and offset >= 0 else None

    texts = _search_texts(raw, payload)
    fragments = _fragment_variants(text, payload)

    if payload.get("chunk_kind") == "sliding_window":
        # Windowed points store a SYNTHETIC concatenation (constituent chunks
        # joined with a separator) — that exact blob never exists in the
        # source, so text search cannot verify it. The snapshot hash CAN:
        # a byte-identical document implies the derived text by ingest
        # determinism. No fragment is returned either way (it would be
        # embedding text, not source text).
        if snapshot == "match":
            return _resolution(
                chunk_id,
                payload,
                verdict="intact",
                resolved_path=str(path),
                snapshot=snapshot,
                stored_offset=stored_offset,
                found_offset=stored_offset,
                detail="sliding-window point: source snapshot hash matches the ingested document",
            )
        return _resolution(
            chunk_id,
            payload,
            verdict="unresolvable",
            resolved_path=str(path),
            snapshot=snapshot,
            stored_offset=stored_offset,
            detail=(
                "sliding-window point: the stored text is a synthetic "
                "concatenation that cannot be text-verified against the "
                "changed source (snapshot hash no longer matches)"
            ),
        )

    # Locate the fragment in the CURRENT document. A matching snapshot hash
    # authenticates the FILE, not the point: text/offset live in a mutable
    # payload (set_payload, external writers), so `intact` is only issued
    # when the fragment is actually found — a hash match must never launder
    # arbitrary stored text into a supported citation.
    located: str | None = None
    if stored_offset is not None:
        for candidate in texts:
            located = next(
                (f for f in fragments if _at_offset(candidate, stored_offset, f)), None
            )
            if located is not None:
                break
    if located is not None:
        if snapshot == "match":
            detail = "fragment at its recorded position; source snapshot hash matches"
            verdict = "intact"
        elif snapshot == "absent":
            detail = "fragment at its recorded position (no ingest snapshot to compare)"
            verdict = "intact"
        else:
            detail = "fragment intact at its recorded position; document changed elsewhere"
            verdict = "source_changed"
        return _resolution(
            chunk_id,
            payload,
            verdict=verdict,
            resolved_path=str(path),
            snapshot=snapshot,
            stored_offset=stored_offset,
            found_offset=stored_offset,
            fragment=located,
            detail=detail,
        )
    found: tuple[int, str] | None = None
    for candidate in texts:
        for fragment in fragments:
            idx = candidate.find(fragment)
            if idx != -1:
                found = (idx, fragment)
                break
        if found:
            break
    if found is not None:
        idx, fragment = found
        if stored_offset is None:
            # Without a recorded position, presence alone cannot distinguish
            # "moved by edits" from "planted text that happens to occur".
            # A snapshot-verified document is the exception: the fragment
            # provably belongs to the file that was ingested.
            if snapshot == "match":
                return _resolution(
                    chunk_id,
                    payload,
                    verdict="intact",
                    resolved_path=str(path),
                    snapshot=snapshot,
                    stored_offset=None,
                    found_offset=idx,
                    fragment=fragment,
                    detail="fragment present in the snapshot-verified document (no recorded position)",
                )
            return _resolution(
                chunk_id,
                payload,
                verdict="unresolvable",
                resolved_path=str(path),
                snapshot=snapshot,
                stored_offset=None,
                found_offset=idx,
                detail=(
                    "point has no recorded offset — presence without position "
                    "cannot be verified against a changed document"
                ),
            )
        return _resolution(
            chunk_id,
            payload,
            verdict="moved",
            resolved_path=str(path),
            snapshot=snapshot,
            stored_offset=stored_offset,
            found_offset=idx,
            fragment=fragment,
            detail="exact fragment found at a different position",
        )
    # No fragment on `changed` ON PURPOSE: every other verdict returns text
    # the caller already possesses (the cited fragment); echoing what the
    # CURRENT file holds at an offset the payload controls would turn the
    # resolver into a read oracle over the corpus root.
    detail = (
        "document matches the ingest snapshot but never contained this "
        "fragment — the point's text/offset does not belong to it"
        if snapshot == "match"
        else "source exists but no longer contains the exact fragment"
    )
    return _resolution(
        chunk_id,
        payload,
        verdict="changed",
        resolved_path=str(path),
        snapshot=snapshot,
        stored_offset=stored_offset,
        found_offset=None,
        fragment=None,
        detail=detail,
    )


def resolve_citation(
    store: Any,
    chunk_id: str,
    *,
    root: str | None = None,
    tenant: str | None = None,
    allow_unrooted: bool = False,
    text_key: str = "text",
    allowed_roots: list[str] | None = None,
) -> Resolution:
    """Resolve a citation by chunk id against ``store`` (a ``VectorStore``).

    ``tenant`` scopes the lookup: another tenant's point resolves exactly like
    an absent one (existence must not leak across the boundary).
    ``allow_unrooted`` extends resolution to bare source paths — operator-CLI
    trust level only; service surfaces keep the default. ``text_key`` and
    ``allowed_roots``: see :func:`resolve_payload`."""
    # Integer-id collections render citations as decimal strings — hand
    # Qdrant the integer back, or the lookup misses its own point.
    lookup_id: str | int = chunk_id
    if isinstance(chunk_id, str) and chunk_id.isdigit():
        lookup_id = int(chunk_id)
    try:
        payload = store.retrieve_payload(lookup_id, tenant=tenant)
    except Exception as e:  # noqa: BLE001 — an invalid handle (e.g. a graph
        # id) or a store error is "cannot verify", never a crashing surface.
        return _resolution(
            chunk_id,
            {},
            verdict="unresolvable",
            detail=f"point lookup failed: {e}",
        )
    if payload is None:
        return _resolution(
            chunk_id,
            {},
            verdict="unresolvable",
            detail="no such point in the collection",
        )
    return resolve_payload(
        chunk_id,
        payload,
        root=root,
        allow_unrooted=allow_unrooted,
        text_key=text_key,
        allowed_roots=allowed_roots,
    )
