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
import os
import re
import stat as stat_module
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Payload keys written at ingest for the document snapshot. Additive and
#: optional: points ingested before this feature simply resolve with
#: ``snapshot="absent"`` — no migration.
SOURCE_HASH_KEY = "source_content_hash"
SOURCE_CAPTURED_KEY = "source_captured_at"

#: Explicit id-scheme marker: ONLY payloads carrying this value have their
#: deterministic ``stable_chunk_id`` commitment enforced. The snapshot hash
#: alone must not imply the scheme — a mounted collection or custom indexer
#: may legitimately use the documented ``source_snapshot()`` helper with its
#: own point ids. Stamped by the built-in indexers; reserved at ingest so
#: frontmatter/enrichers cannot plant or clobber it.
ID_SCHEME_KEY = "_id_scheme"
STABLE_ID_SCHEME = "stable_chunk_id"


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
    # Foreign/external payloads may carry anything here — the field's contract
    # (and the HTTP response model) is `str | None`, never a raw object.
    captured = payload.get(SOURCE_CAPTURED_KEY)
    kw.setdefault("captured_at", captured if isinstance(captured, str) else None)
    kw["supported"] = kw["verdict"] in _SUPPORTED_VERDICTS
    return Resolution(chunk_id=chunk_id, **kw)


#: Refuse to read source documents larger than this (bytes). Verification is
#: a metadata operation — an unbounded read_text would let one resolve call
#: pull an arbitrarily large file into memory.
MAX_RESOLVE_BYTES = 32 * 1024 * 1024


def _candidate_paths(
    source: str, bases: list[str], allow_unrooted: bool
) -> tuple[list[tuple[Path | None, Path]], bool]:
    """(base, candidate) pairs the source may live at, plus an escape flag.

    ``source`` is an UNTRUSTED label (any ingest caller writes it) that this
    module is the first to open as a filesystem path — so candidates are
    CONFINED: every candidate must resolve inside its base (``.resolve()`` +
    ``is_relative_to``), which kills ``..`` traversal and symlink escapes; an
    absolute ``source`` is only accepted when it already lies inside a base.
    A bare ``source`` path (no base known) is inherently unconfined and is
    only ever tried for ``allow_unrooted`` callers (the operator CLI).
    Malformed labels (e.g. an embedded NUL) simply produce no candidate."""
    pairs: list[tuple[Path | None, Path]] = []
    escaped = False
    for base_str in bases:
        try:
            base = Path(base_str)
            candidate = Path(source) if Path(source).is_absolute() else base / source
            inside = candidate.resolve().is_relative_to(base.resolve())
        except (OSError, ValueError):
            escaped = True
            continue
        if inside:
            pairs.append((base, candidate))
        else:
            escaped = True
    if not bases and allow_unrooted:
        try:
            pairs.append((None, Path(source)))
        except ValueError:
            escaped = True
    return pairs, escaped


_URI_SOURCE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")


_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd


def _open_beneath(base: Path, source: str) -> int:
    """Open ``source`` beneath ``base`` with EVERY component refusing symlinks.

    ``O_NOFOLLOW`` on a single open only protects the final component — a
    corpus writer racing the resolver could swap an intermediate directory
    for a symlink after the containment check and divert the read outside
    the root. So walk with ``openat`` semantics (``dir_fd``), each component
    ``O_NOFOLLOW`` (the same primitive the audit trail uses). Deliberate
    trade-off, identical to the audit trail's: a legitimately symlinked
    entry inside the corpus is ALSO refused, loudly — index real paths.
    Raises OSError on refusal."""
    # Platform-correct component split: on POSIX a backslash is a LEGAL
    # filename character, not a separator — splitting it would break a file
    # literally named "a\\b.md". Windows accepts both separators.
    separators = r"[/\\]+" if os.name == "nt" else r"/+"
    parts = [p for p in re.split(separators, source) if p and p != "."]
    if not parts or ".." in parts:
        raise OSError("source path escapes the corpus root")
    final_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if not _SUPPORTS_DIR_FD:
        # No openat walk (Windows, where O_NOFOLLOW is also absent): refuse
        # symlinks by lstat-walking first — check-then-open, best effort on a
        # secondary platform where symlink creation needs elevation anyway.
        probe = base
        for comp in parts:
            probe = probe / comp
            if probe.is_symlink():
                raise OSError(f"source path component {comp!r} is a symlink — refused")
        return os.open(str(base.joinpath(*parts)), final_flags)
    dfd = os.open(str(base), os.O_RDONLY | os.O_DIRECTORY)
    try:
        for comp in parts[:-1]:
            ndfd = os.open(comp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dfd)
            os.close(dfd)
            dfd = ndfd
        return os.open(parts[-1], final_flags, dir_fd=dfd)
    finally:
        os.close(dfd)


def _read_source(path: Path, base: Path | None, source: str) -> tuple[str | None, str]:
    """Read the source through ONE file descriptor, TOCTOU-hardened.

    Confined reads (a base is known) go through the symlink-refusing
    ``openat`` walk. Unconfined reads (operator CLI, no root known — or an
    absolute source validated against its base) fall back to resolve +
    final-component ``O_NOFOLLOW``. Size cap and bytes both come from the
    SAME descriptor via ``fstat``/``read``. Returns ``(text, "")`` or
    ``(None, reason)``."""
    try:
        if base is not None:
            resolved_base = base.resolve()
            if Path(source).is_absolute():
                # A confined ABSOLUTE source walks too: convert it to a
                # root-relative path first, so intermediate components get
                # the same no-symlink treatment as relative sources.
                target = path.resolve()
                if not target.is_relative_to(resolved_base):
                    return None, "source path escapes the corpus root — refusing to read it"
                walk_source = target.relative_to(resolved_base).as_posix()
            else:
                walk_source = source
            fd = _open_beneath(resolved_base, walk_source)
        else:
            fd = os.open(
                str(path.resolve()), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
    except OSError as e:
        return None, f"source exists but cannot be opened: {e}"
    try:
        st = os.fstat(fd)
        if not stat_module.S_ISREG(st.st_mode):
            return None, "source is not a regular file"
        if st.st_size > MAX_RESOLVE_BYTES:
            return None, f"source larger than the {MAX_RESOLVE_BYTES}-byte verification cap"
        chunks: list[bytes] = []
        remaining = MAX_RESOLVE_BYTES + 1
        while remaining > 0:
            block = os.read(fd, min(remaining, 1 << 20))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        data = b"".join(chunks)
        if len(data) > MAX_RESOLVE_BYTES:
            return None, f"source larger than the {MAX_RESOLVE_BYTES}-byte verification cap"
    except OSError as e:
        return None, f"source exists but cannot be read: {e}"
    finally:
        os.close(fd)
    # Universal-newline translation, exactly as the ingest-side
    # ``Path.read_text()`` applied it: the stored snapshot hash, offsets and
    # chunk text all use normalized \n — decoding raw bytes without this
    # would make every untouched CRLF document a snapshot mismatch whose
    # multiline fragments then fail exact matching.
    text = data.decode("utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")
    return text, ""


def _root_allowed(base: str, allowed_roots: list[str]) -> bool:
    """Whether ``base`` (the corpus root about to be read) lies inside one of
    the operator-configured allowlist directories."""
    try:
        resolved = Path(base).resolve()
        return any(resolved.is_relative_to(Path(ar).resolve()) for ar in allowed_roots)
    except (OSError, ValueError):  # ValueError: e.g. an embedded NUL byte
        return False


def _id_commitment_holds(chunk_id: str, payload: dict[str, Any], text: str) -> bool:
    """Whether a FIRST-PARTY payload still matches its deterministic id.

    Built-in indexers derive the point id from ``stable_chunk_id`` over
    (source[, index_root], offset, text[, tenant]) — so an in-place payload
    edit (or a payload copied from another genuine point) breaks the
    commitment even when the replacement triple verifies against the file.
    Only enforced for payloads explicitly marked ``_id_scheme:
    stable_chunk_id``: foreign/mounted collections use their own id schemes
    and are exempt. This is defense-in-depth against corruption, not against
    a hostile store — a writer who controls payloads controls the memory."""
    from .ingest import stable_chunk_id

    source = payload.get("source")
    offset = payload.get("offset")
    if not isinstance(source, str) or not isinstance(offset, int):
        return False
    index_root = payload.get("index_root")
    id_sources = [source]
    if isinstance(index_root, str):
        id_sources.insert(0, f"{index_root}\x00{source}")
    tenant = payload.get("tenant_id")
    tenants = [tenant, None] if isinstance(tenant, str) else [None]
    return any(
        stable_chunk_id(s, offset, text, tenant=t) == str(chunk_id)
        for s in id_sources
        for t in tenants
    )


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
        # The MARKDOWN chunker records offsets into STRIPPED segments (a
        # headingless note or pre-heading lead-in is `.strip()`-ed before
        # windowing) — search the stripped renderings too, right after their
        # unstripped originals. Only for markdown points: plain `index`
        # offsets are raw-file coordinates, and a stripped candidate would
        # recreate a stale coordinate system (leading whitespace added to a
        # plain file must read as `moved`, not `source_changed`).
        for candidate in list(texts):
            stripped = candidate.strip()
            if stripped != candidate:
                texts.insert(texts.index(candidate) + 1, stripped)
    return texts


def _is_windowed(payload: dict[str, Any]) -> bool:
    """Whether the payload carries the FULL sliding-window convention.

    ``chunk_kind`` alone is not trusted — a library-ingested item or mounted
    collection may use that key for unrelated metadata, and classifying its
    source-native text as synthetic would falsely refuse a verifiable
    citation. The built-in window writers always stamp all four structural
    fields together."""
    return (
        payload.get("chunk_kind") == "sliding_window"
        and isinstance(payload.get("chunk_window"), int)
        and isinstance(payload.get("chunk_start_offset"), int)
        and isinstance(payload.get("chunk_end_offset"), int)
    )


def _fragment_variants(text: str, payload: dict[str, Any]) -> list[str]:
    """The stored text as it may appear IN the source document.

    The markdown chunker prepends synthetic heading context
    (``[Parent > Path]\\n<body>``) to some chunks for embedding quality — and
    records EXACTLY how many characters it added (``synthetic_prefix_len``)
    at ingest. Stripping uses only that ingest-recorded length, with shape
    sanity checks — never a reconstruction from other (mutable) metadata
    like ``heading_path``, which could be edited to make the resolver strip
    a REAL cited line and misreport a deleted fragment as supported."""
    variants = [text]
    n = payload.get("synthetic_prefix_len")
    if (
        isinstance(n, int)
        and 0 < n < len(text)
        and text.startswith("[")
        and text[n - 1] == "\n"
    ):
        variants.append(text[n:])
    return variants


def _at_offset(candidate: str, offset: int, fragment: str) -> bool:
    """Fragment present EXACTLY at ``offset`` in this candidate rendering.

    No whitespace slack: a lenient comparison would bless fragments whose
    own (source-significant) indentation changed. Coordinate-base drift from
    the chunker's segment stripping is handled by searching the stripped
    RENDERINGS of the document (see ``_search_texts``), not by loosening the
    match — and when the ingest snapshot hash matches, position is not
    consulted at all (ingest determinism)."""
    return candidate[offset : offset + len(fragment)] == fragment


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
    if payload.get(ID_SCHEME_KEY) == STABLE_ID_SCHEME and not _id_commitment_holds(
        chunk_id, payload, text
    ):
        # First-party payloads (explicitly marked with the id scheme) commit
        # to (source, offset, text) through their deterministic id — an edited
        # or copied payload must not be validated under the original citation.
        return _resolution(
            chunk_id,
            payload,
            verdict="unresolvable",
            detail=(
                "payload does not match the point's id commitment "
                "(edited in place or copied from another point?)"
            ),
        )
    index_root = payload.get("index_root")
    if root and isinstance(index_root, str) and index_root:
        # Relocation contract: an explicit root REPLACES the recorded one.
        # A point that stored its source as an ABSOLUTE path under the old
        # root must be rebased onto the override, or the confinement check
        # would reject the relocated corpus as an escape.
        try:
            if Path(source).is_absolute():
                source = Path(source).relative_to(Path(index_root)).as_posix()
        except ValueError:
            pass  # absolute source outside the old root — handled below as-is
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
        if base_str:
            if not _root_allowed(base_str, allowed_roots):
                return _resolution(
                    chunk_id,
                    payload,
                    verdict="unresolvable",
                    detail=(
                        "the point's corpus root is not in the operator-configured "
                        "resolution allowlist"
                    ),
                )
            bases = [base_str]
        else:
            # Rootless point (legacy / library-ingested / mounted): the
            # operator-configured roots themselves are the trusted bases —
            # the surface offers no per-call override, so without this such
            # points could never resolve at all. First root wins on a
            # relative-path collision (the operator controls the order).
            bases = list(allowed_roots)
    else:
        bases = [base_str] if base_str else []
    candidates, escaped = _candidate_paths(source, bases, allow_unrooted)
    if not candidates:
        if escaped:
            return _resolution(
                chunk_id,
                payload,
                verdict="unresolvable",
                detail="source path escapes the corpus root — refusing to read it",
            )
        return _resolution(
            chunk_id,
            payload,
            verdict="unresolvable",
            detail=(
                "no corpus root is known for this point and unrooted "
                "resolution is not allowed on this surface"
            ),
        )
    path: Path | None = None
    base_of_path: Path | None = None
    access_denied = False
    for cand_base, p in candidates:
        try:
            if p.is_file():
                path = p
                base_of_path = cand_base
                break
        except OSError:
            # pathlib re-raises EACCES and friends — an untraversable
            # candidate is "cannot verify", never a 500.
            access_denied = True
    if path is None:
        if access_denied:
            return _resolution(
                chunk_id,
                payload,
                verdict="unresolvable",
                detail="source path cannot be accessed by this process",
            )
        return _resolution(
            chunk_id,
            payload,
            verdict="missing",
            detail="source document not found under the known corpus root",
        )
    read_base = base_of_path
    read_source_path = source
    if allowed_roots is not None and base_of_path is not None:
        # Anchor the descriptor walk at the OPERATOR-configured root, not at
        # the payload-supplied (possibly nested) index_root: a writer able to
        # rename that nested directory could swap it for an out-of-root
        # symlink after the allowlist check. Walking the nested prefix AND
        # the source beneath the allowed root's descriptor keeps every
        # component O_NOFOLLOW-protected relative to operator-trusted ground.
        anchor = next(
            (a for a in allowed_roots if _root_allowed(str(base_of_path), [a])), None
        )
        if anchor is not None:
            try:
                anchor_path = Path(anchor)
                prefix = (
                    base_of_path.resolve().relative_to(anchor_path.resolve()).as_posix()
                )
            except (OSError, ValueError):
                return _resolution(
                    chunk_id,
                    payload,
                    verdict="unresolvable",
                    detail=(
                        "the point's corpus root is not in the operator-configured "
                        "resolution allowlist"
                    ),
                )
            read_base = anchor_path
            if not Path(source).is_absolute() and prefix not in (".", ""):
                read_source_path = f"{prefix}/{source}"
    raw, read_error = _read_source(path, read_base, read_source_path)
    if raw is None:
        return _resolution(
            chunk_id,
            payload,
            verdict="unresolvable",
            resolved_path=str(path),
            detail=read_error,
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

    if _is_windowed(payload):
        # Windowed points store a SYNTHETIC concatenation (constituent chunks
        # joined with a separator) — that exact blob never exists in the
        # source, so text search cannot verify it. Nor may a matching
        # snapshot hash stand in: the hash authenticates the FILE, not the
        # mutable payload, and `chunk_kind` itself is payload data — trusting
        # it would let any writer mark planted text as windowed and have it
        # laundered into `intact`. Honest answer: not verifiable; resolve the
        # constituent chunks instead (the snapshot comparison is still
        # reported for whatever it is worth).
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
                "source — resolve the constituent chunks instead"
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
    found: tuple[int, str] | None = None
    if located is None:
        for candidate in texts:
            for fragment in fragments:
                idx = candidate.find(fragment)
                if idx != -1:
                    found = (idx, fragment)
                    break
            if found:
                break

    if snapshot == "match":
        # The document is byte-identical to the one that was ingested, so a
        # fragment present ANYWHERE provably belongs to it — position is a
        # coordinate-system detail (the chunker records offsets into stripped
        # segment renderings), not evidence. Absent entirely → the point's
        # text never came from this document.
        if located is not None:
            return _resolution(
                chunk_id,
                payload,
                verdict="intact",
                resolved_path=str(path),
                snapshot=snapshot,
                stored_offset=stored_offset,
                found_offset=stored_offset,
                fragment=located,
                detail="fragment at its recorded position; source snapshot hash matches",
            )
        if found is not None:
            idx, fragment = found
            return _resolution(
                chunk_id,
                payload,
                verdict="intact",
                resolved_path=str(path),
                snapshot=snapshot,
                stored_offset=stored_offset,
                found_offset=idx,
                fragment=fragment,
                detail="fragment present in the snapshot-verified document",
            )
    elif located is not None:
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
            fragment=located,
            detail=detail,
        )
    elif found is not None:
        idx, fragment = found
        if stored_offset is None:
            # Without a recorded position, presence alone cannot distinguish
            # "moved by edits" from "planted text that happens to occur" in a
            # document that no longer matches its ingest snapshot.
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
    # Qdrant the integer back, or the lookup misses its own point. But a
    # 32-digit simple-form UUID is also all digits and must STAY a string
    # (Qdrant distinguishes unsigned-int ids from UUID strings).
    lookup_id: str | int = chunk_id
    if isinstance(chunk_id, str) and chunk_id.isdigit():
        try:
            uuid.UUID(chunk_id)
        except ValueError:
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
