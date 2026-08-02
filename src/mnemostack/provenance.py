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


def _search_texts(raw: str, payload: dict[str, Any]) -> list[str]:
    """The texts a stored offset may index into.

    Markdown chunk offsets are relative to the BODY (frontmatter stripped by
    the indexer), while plain ingest offsets index the raw document — check
    both rather than encode per-format knowledge into the verdicts."""
    texts = [raw]
    if "_md_keys" in payload or "heading_path" in payload:
        try:
            from .markdown.parse import parse_frontmatter

            _meta, body = parse_frontmatter(raw)
            if body != raw:
                texts.append(body)
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
) -> Resolution:
    """Resolve a citation given its point payload (see module docstring).

    ``root`` overrides where sources are looked up (default: the payload's
    own ``index_root``); resolution is CONFINED to that root. Bare source
    paths (no root known) are only tried when ``allow_unrooted=True`` — the
    operator-CLI trust level; service surfaces keep the default."""
    text = payload.get("text")
    source = payload.get("source")
    if not source or not isinstance(source, str) or not isinstance(text, str) or not text:
        return _resolution(
            chunk_id,
            payload,
            verdict="unresolvable",
            detail="payload carries no resolvable source/text pair",
        )
    index_root = payload.get("index_root")
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

    if snapshot == "match":
        # The document is byte-identical to what was indexed — the fragment's
        # presence at its recorded position is implied by ingest determinism.
        # The returned fragment must be SOURCE text, not the stored embedding
        # text (which may carry the chunker's synthetic heading prefix): pick
        # the variant actually present at the recorded position.
        fragment = text
        if stored_offset is not None:
            for candidate in texts:
                located = next(
                    (f for f in fragments if _at_offset(candidate, stored_offset, f)), None
                )
                if located is not None:
                    fragment = located
                    break
        return _resolution(
            chunk_id,
            payload,
            verdict="intact",
            resolved_path=str(path),
            snapshot=snapshot,
            stored_offset=stored_offset,
            found_offset=stored_offset,
            fragment=fragment,
            detail="source snapshot hash matches the ingested document",
        )

    if payload.get("chunk_kind") == "sliding_window":
        # Windowed points store a SYNTHETIC concatenation (constituent chunks
        # joined with a separator) — that exact blob never exists in the
        # source, so text search would report a false `changed` on an
        # untouched document. With the hash equality path exhausted, be
        # honest: the synthetic text cannot be verified against the source.
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
    if stored_offset is not None:
        for candidate in texts:
            for fragment in fragments:
                if _at_offset(candidate, stored_offset, fragment):
                    verdict = "source_changed" if snapshot == "mismatch" else "intact"
                    detail = (
                        "fragment intact at its recorded position; document changed elsewhere"
                        if snapshot == "mismatch"
                        else "fragment at its recorded position (no ingest snapshot to compare)"
                    )
                    return _resolution(
                        chunk_id,
                        payload,
                        verdict=verdict,
                        resolved_path=str(path),
                        snapshot=snapshot,
                        stored_offset=stored_offset,
                        found_offset=stored_offset,
                        fragment=fragment,
                        detail=detail,
                    )
    for candidate in texts:
        for fragment in fragments:
            idx = candidate.find(fragment)
            if idx != -1:
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
    return _resolution(
        chunk_id,
        payload,
        verdict="changed",
        resolved_path=str(path),
        snapshot=snapshot,
        stored_offset=stored_offset,
        found_offset=None,
        fragment=None,
        detail="source exists but no longer contains the exact fragment",
    )


def resolve_citation(
    store: Any,
    chunk_id: str,
    *,
    root: str | None = None,
    tenant: str | None = None,
    allow_unrooted: bool = False,
) -> Resolution:
    """Resolve a citation by chunk id against ``store`` (a ``VectorStore``).

    ``tenant`` scopes the lookup: another tenant's point resolves exactly like
    an absent one (existence must not leak across the boundary).
    ``allow_unrooted`` extends resolution to bare source paths — operator-CLI
    trust level only; service surfaces keep the default."""
    payload = store.retrieve_payload(chunk_id, tenant=tenant)
    if payload is None:
        return _resolution(
            chunk_id,
            {},
            verdict="unresolvable",
            detail="no such point in the collection",
        )
    return resolve_payload(chunk_id, payload, root=root, allow_unrooted=allow_unrooted)
