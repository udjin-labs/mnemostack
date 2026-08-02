"""Verifiable citations — resolving chunk ids back to source fragments.

Contract under test: `mnemostack.provenance` turns the citation label
(`stable_chunk_id` already commits to (source, offset, text); the payload
already stores all three) into a VERIFIED link: re-read the current source,
compare the ingest-time snapshot hash, and return an honest verdict for every
way a source can drift — edited elsewhere, fragment shifted, fragment edited,
file deleted or renamed, non-file source, legacy point without a snapshot.
"""

from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance

from mnemostack.markdown import collect_markdown
from mnemostack.provenance import (
    ID_SCHEME_KEY,
    SOURCE_CAPTURED_KEY,
    SOURCE_HASH_KEY,
    Resolution,
    resolve_citation,
    resolve_payload,
    source_content_hash,
)
from mnemostack.vector import VectorStore

_V = [1.0, 0.0, 0.0, 0.0]

_DOC = """# Alpha runbook

First paragraph about postgres backups and verification.

Second paragraph about kubernetes ingress and certificates.
"""

_FM_DOC = """---
title: Beta note
---
Body line one about memgraph timeouts.

Body line two about qdrant snapshots.
"""


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "alpha.md").write_text(_DOC)
    (root / "beta.md").write_text(_FM_DOC)
    return root


def _index(root: Path) -> tuple[VectorStore, list]:
    s = VectorStore.__new__(VectorStore)
    s.collection = "prov"
    s.dimension = 4
    s.distance = Distance.COSINE
    s.client = QdrantClient(":memory:")
    s.sparse_text = False
    s.text_key = "text"
    s._sparse_encoder = None
    s.ensure_collection()
    col = collect_markdown(root, index_root=str(root))
    for chunk in col.chunks:
        s.upsert(chunk.id, _V, chunk.payload)
    return s, col.chunks


def _resolve_all(store: VectorStore, chunks, root: Path | None = None) -> dict[str, Resolution]:
    kw = {"root": str(root)} if root else {}
    return {c.id: resolve_citation(store, c.id, **kw) for c in chunks}


# ---------- ingest side: snapshot fields ----------


def test_markdown_payload_carries_source_snapshot(tmp_path):
    root = _corpus(tmp_path)
    _, chunks = _index(root)
    assert chunks
    for c in chunks:
        raw = (root / c.payload["source"]).read_text()
        assert c.payload[SOURCE_HASH_KEY] == source_content_hash(raw)
        assert c.payload[SOURCE_CAPTURED_KEY]
        # Ownership record includes the snapshot keys, so a payload refresh
        # re-owns (and re-stamps) them instead of orphaning stale hashes.
        assert SOURCE_HASH_KEY in c.payload["_md_keys"]
        assert SOURCE_CAPTURED_KEY in c.payload["_md_keys"]


# ---------- verdicts, one per mutation type ----------


def test_untouched_corpus_resolves_intact(tmp_path):
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    for res in _resolve_all(store, chunks).values():
        assert res.verdict == "intact" and res.supported
        assert res.snapshot == "match"


def test_edit_elsewhere_gives_source_changed(tmp_path):
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    # Append at the END: earlier fragments keep their offsets, hash drifts.
    (root / "alpha.md").write_text(_DOC + "\nNew trailing paragraph.\n")
    for c in chunks:
        if c.payload["source"] != "alpha.md":
            continue
        res = resolve_citation(store, c.id)
        assert res.verdict == "source_changed" and res.supported
        assert res.snapshot == "mismatch"


def test_insert_above_gives_moved(tmp_path):
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    (root / "alpha.md").write_text("An inserted preamble line.\n\n" + _DOC)
    for c in chunks:
        if c.payload["source"] != "alpha.md":
            continue
        res = resolve_citation(store, c.id)
        assert res.verdict == "moved" and res.supported
        assert res.found_offset is not None and res.found_offset != c.payload["offset"]


def test_edited_fragment_gives_changed(tmp_path):
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    (root / "alpha.md").write_text(_DOC.replace("postgres", "mysql"))
    changed = [
        resolve_citation(store, c.id)
        for c in chunks
        if c.payload["source"] == "alpha.md" and "postgres" in c.text
    ]
    assert changed
    for res in changed:
        assert res.verdict == "changed" and not res.supported
        # No fragment on `changed`: echoing what the CURRENT file holds at a
        # payload-controlled offset would be a read oracle.
        assert res.fragment is None


def test_deleted_and_renamed_give_missing(tmp_path):
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    (root / "alpha.md").rename(root / "alpha-renamed.md")
    for c in chunks:
        if c.payload["source"] != "alpha.md":
            continue
        res = resolve_citation(store, c.id)
        assert res.verdict == "missing" and not res.supported


def test_frontmatter_offsets_resolve_against_body(tmp_path):
    # Markdown offsets are body-relative (frontmatter stripped) — position
    # checks must succeed via the parsed body, not just the raw file.
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    # Change frontmatter ONLY: body intact, raw hash drifts.
    (root / "beta.md").write_text(_FM_DOC.replace("Beta note", "Beta note v2"))
    for c in chunks:
        if c.payload["source"] != "beta.md":
            continue
        res = resolve_citation(store, c.id)
        assert res.verdict == "source_changed" and res.supported


def test_legacy_point_without_snapshot_resolves_via_position(tmp_path):
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    c = next(c for c in chunks if c.payload["source"] == "alpha.md")
    legacy = {k: v for k, v in c.payload.items() if k not in (SOURCE_HASH_KEY, SOURCE_CAPTURED_KEY, ID_SCHEME_KEY)}
    res = resolve_payload(c.id, legacy)
    assert res.verdict == "intact" and res.supported
    assert res.snapshot == "absent"


def test_unresolvable_cases(tmp_path):
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    # No such point in the collection.
    res = resolve_citation(store, "00000000-0000-0000-0000-000000000000")
    assert res.verdict == "unresolvable" and not res.supported
    # Payload without a source (library ingest of standalone items).
    res2 = resolve_payload("x", {"text": "abc", "source": ""})
    assert res2.verdict == "unresolvable"


def test_allowed_roots_gate_service_surfaces(tmp_path):
    # The stored index_root is payload data — writable by whoever ingests —
    # so it cannot be its own security boundary. Service surfaces pass an
    # operator allowlist: empty = fail closed, and a planted index_root
    # outside the allowlist ("/") is refused even for real files.
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    c = chunks[0]
    # Empty allowlist (the service default): resolution disabled.
    res = resolve_citation(store, c.id, allowed_roots=[])
    assert res.verdict == "unresolvable" and "not enabled" in res.detail
    # Allowlist covering the corpus: resolves normally.
    assert resolve_citation(store, c.id, allowed_roots=[str(tmp_path)]).verdict == "intact"
    # Planted root outside the allowlist: refused (foreign-shaped payload so
    # the allowlist check itself is what fires, not the id commitment).
    planted = {
        k: v
        for k, v in c.payload.items()
        if k not in (SOURCE_HASH_KEY, SOURCE_CAPTURED_KEY, ID_SCHEME_KEY)
    }
    planted["index_root"] = "/"
    planted["source"] = "etc/hostname"
    res = resolve_payload(c.id, planted, allowed_roots=[str(tmp_path)])
    assert res.verdict == "unresolvable" and "allowlist" in res.detail


def test_configured_text_key_is_honored(tmp_path):
    root = _corpus(tmp_path)
    doc = root / "alpha.md"
    payload = {
        "content": "First paragraph about postgres backups and verification.",
        "source": "alpha.md",
        "index_root": str(root),
        "offset": _DOC.index("First paragraph"),
    }
    assert resolve_payload("x", payload, text_key="content").verdict == "intact"
    # The default key reports the pair as unresolvable, naming the key.
    res = resolve_payload("x", payload)
    assert res.verdict == "unresolvable" and "text" in res.detail
    assert doc.is_file()


def test_uri_sources_are_unresolvable(tmp_path):
    for src in ("https://example.com/doc", "urn:example:note", "mailto:u@example.com", "x:note"):
        res = resolve_payload(
            "x",
            {"text": "abc", "source": src, "offset": 0},
            allow_unrooted=True,
        )
        assert res.verdict == "unresolvable" and "URI" in res.detail, src


def test_authority_uris_stay_nonlocal_even_with_a_root(tmp_path):
    # "https://..." is unambiguous: a rooted lookup must NOT read a decoy at
    # <root>/https:/example.com/doc.
    root = _corpus(tmp_path)
    decoy = root / "https:" / "example.com"
    decoy.mkdir(parents=True)
    (decoy / "doc").write_text("decoy body")
    res = resolve_payload(
        "x",
        {"text": "decoy body", "source": "https://example.com/doc", "index_root": str(root), "offset": 0},
    )
    assert res.verdict == "unresolvable" and "URI" in res.detail


def test_one_letter_authority_uris_stay_nonlocal(tmp_path):
    import os as _os

    if _os.name == "nt":
        return
    root = _corpus(tmp_path)
    decoy = root / "x:" / "remote"
    decoy.mkdir(parents=True)
    (decoy / "doc").write_text("decoy body")
    res = resolve_payload(
        "x",
        {"text": "decoy body", "source": "x://remote/doc", "index_root": str(root), "offset": 0},
    )
    assert res.verdict == "unresolvable" and "URI" in res.detail


def test_md_commitment_binds_the_root(tmp_path):
    # Markdown ids fold index_root into the commitment — repointing the
    # payload root at another corpus breaks it.
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    other = tmp_path / "other"
    other.mkdir()
    (other / "alpha.md").write_text(_DOC)
    c = next(c for c in chunks if c.payload["source"] == "alpha.md")
    repointed = dict(c.payload)
    repointed["index_root"] = str(other)
    res = resolve_payload(c.id, repointed)
    assert res.verdict == "unresolvable" and "commitment" in res.detail


def test_rooted_colon_filenames_are_not_uris(tmp_path):
    import os as _os

    if _os.name == "nt":
        return
    # On POSIX a colon is a legal filename character: a source under a known
    # corpus root is a FILENAME, never a URI scheme.
    root = _corpus(tmp_path)
    (root / "notes:2026.md").write_text("Colon-named body text.\n")
    payload = {
        "text": "Colon-named body text.",
        "source": "notes:2026.md",
        "index_root": str(root),
        "offset": 0,
    }
    res = resolve_payload("x", payload)
    assert res.verdict == "intact" and res.supported


def test_hash_match_does_not_launder_planted_text(tmp_path):
    # A matching snapshot hash authenticates the FILE, not the point.
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    # First-party-shaped payload (carries the snapshot): a swapped text
    # breaks the deterministic id commitment before anything is read.
    planted = dict(chunks[0].payload)
    planted["text"] = "totally fabricated claim"
    res = resolve_payload(chunks[0].id, planted)
    assert res.verdict == "unresolvable" and not res.supported
    assert "commitment" in res.detail
    # Foreign-shaped payload (no snapshot marker): the planted text is
    # searched for honestly and is simply not there.
    foreign = {
        k: v for k, v in planted.items() if k not in (SOURCE_HASH_KEY, SOURCE_CAPTURED_KEY, ID_SCHEME_KEY)
    }
    res2 = resolve_payload(chunks[0].id, foreign)
    assert res2.verdict == "changed" and not res2.supported
    assert res2.fragment is None


def test_no_offset_points_cannot_claim_position(tmp_path):
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    c = next(c for c in chunks if c.payload["source"] == "alpha.md")
    # A first-party-shaped payload (snapshot present) without an offset
    # cannot satisfy its id commitment — refused up front.
    no_offset = {k: v for k, v in c.payload.items() if k != "offset"}
    assert resolve_payload(c.id, no_offset).verdict == "unresolvable"
    # A foreign-shaped payload without an offset: presence alone cannot
    # distinguish moved from planted once the document has no snapshot.
    foreign = {
        k: v
        for k, v in no_offset.items()
        if k not in (SOURCE_HASH_KEY, SOURCE_CAPTURED_KEY, ID_SCHEME_KEY)
    }
    res = resolve_payload(c.id, foreign)
    assert res.verdict == "unresolvable" and "no recorded offset" in res.detail


def test_exotic_digit_ids_resolve_gracefully(tmp_path):
    # Unicode digits pass str.isdigit() but fail int(); huge digit strings
    # exceed integer-conversion limits — both must produce a verdict, not a
    # 500. And a 21+ digit string is not a Qdrant u64, so it stays a string.
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    for weird in ("\u00b2\u00b3", "9" * 5000, "1" * 21):
        res = resolve_citation(store, weird)
        assert res.verdict == "unresolvable", weird


def test_integer_and_invalid_ids_resolve_gracefully(tmp_path):
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    c = chunks[0]
    copy = {
        k: v
        for k, v in c.payload.items()
        if k not in (SOURCE_HASH_KEY, SOURCE_CAPTURED_KEY, ID_SCHEME_KEY)
    }
    store.upsert(41, [1.0, 0.0, 0.0, 0.0], copy)
    # A decimal-string citation reaches the integer point.
    assert resolve_citation(store, "41").verdict == "intact"
    # A handle the store rejects (e.g. a graph id) is a verdict, not a crash.
    res = resolve_citation(store, "graph:acme:node-7")
    assert res.verdict == "unresolvable"


def test_moved_offset_is_body_relative_for_markdown(tmp_path):
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    # Insert into the BODY of the frontmatter doc: stored offsets are
    # body-relative, so found_offset must be too (body searched first).
    (root / "beta.md").write_text(
        _FM_DOC.replace("Body line one", "Inserted paragraph.\n\nBody line one")
    )
    c = next(
        c
        for c in chunks
        if c.payload["source"] == "beta.md" and "Body line one" in c.text
    )
    res = resolve_citation(store, c.id)
    assert res.verdict == "moved"
    from mnemostack.markdown.parse import parse_frontmatter

    _meta, body = parse_frontmatter((root / "beta.md").read_text())
    assert res.found_offset == body.index(res.fragment)


def test_crlf_sources_verify_like_ingest_read_them(tmp_path):
    # Ingest reads with Path.read_text() (universal newlines: CRLF -> \n);
    # the fd-based resolver read must normalize identically or every
    # untouched CRLF document becomes a snapshot mismatch whose multiline
    # fragments then fail exact matching.
    root = _corpus(tmp_path)
    (root / "windows.md").write_bytes(
        b"# CRLF note\r\n\r\nLine one about backups.\r\n\r\nLine two about restores.\r\n"
    )
    store, chunks = _index(root)
    crlf = [c for c in chunks if c.payload["source"] == "windows.md"]
    assert crlf
    for c in crlf:
        res = resolve_citation(store, c.id)
        assert res.verdict == "intact" and res.snapshot == "match", res.detail


def test_chunk_kind_alone_is_not_a_window_marker(tmp_path):
    # A library-ingested item may carry metadata chunk_kind="sliding_window"
    # for its own reasons — without the full structural convention
    # (chunk_window + both offsets) its source-native text must resolve
    # normally, not be refused as synthetic.
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    c = next(c for c in chunks if c.payload["source"] == "alpha.md")
    quirky = {
        k: v
        for k, v in c.payload.items()
        if k not in (SOURCE_HASH_KEY, SOURCE_CAPTURED_KEY, ID_SCHEME_KEY)
    }
    quirky["chunk_kind"] = "sliding_window"  # no chunk_window/offsets
    res = resolve_payload(c.id, quirky)
    assert res.verdict == "intact" and res.supported


def test_frontmatter_cannot_plant_index_root(tmp_path):
    # collect_markdown(index_root=None): a frontmatter index_root must not
    # survive into the payload and redirect resolution to a decoy directory.
    root = _corpus(tmp_path)
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    (root / "redir.md").write_text(
        f"---\nindex_root: {decoy}\n---\nRedirect body text.\n"
    )
    col = collect_markdown(root)  # no index_root supplied
    redir = [c for c in col.chunks if c.payload["source"] == "redir.md"]
    assert redir
    for c in redir:
        assert "index_root" not in c.payload


def test_frontmatter_cannot_plant_structural_keys(tmp_path):
    # A note whose frontmatter claims `chunk_kind: sliding_window` (or an
    # id-scheme / snapshot key) must not have its chunks misclassified as
    # windowed or tampered — structural resolver keys are reserved.
    root = _corpus(tmp_path)
    (root / "tricky.md").write_text(
        "---\nchunk_kind: sliding_window\n_id_scheme: bogus\n"
        "source_content_hash: deadbeef\n---\nPerfectly ordinary body text.\n"
    )
    store, chunks = _index(root)
    tricky = [c for c in chunks if c.payload["source"] == "tricky.md"]
    assert tricky
    for c in tricky:
        assert c.payload.get("chunk_kind") != "sliding_window"
        res = resolve_citation(store, c.id)
        assert res.verdict == "intact", res.detail


def test_pipe_bearing_sources_refuse_the_commitment(tmp_path):
    # stable_chunk_id's tuple encoding is only unambiguous for pipe-free
    # sources — a pipe-bearing one could collide with a different tuple, so
    # the commitment gate refuses conservatively.
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    c = chunks[0]
    piped = dict(c.payload)
    piped["source"] = "weird|name.md"
    res = resolve_payload(c.id, piped)
    assert res.verdict == "unresolvable" and "commitment" in res.detail


def test_snapshot_helper_without_id_scheme_is_not_first_party(tmp_path):
    # A mounted collection may use the documented source_snapshot() helper
    # with its OWN point ids — the hash field alone must not drag it under
    # the stable_chunk_id commitment.
    from mnemostack.provenance import source_snapshot

    root = _corpus(tmp_path)
    raw = (root / "alpha.md").read_text()
    payload = {
        "text": "First paragraph about postgres backups and verification.",
        "source": "alpha.md",
        "index_root": str(root),
        "offset": raw.index("First paragraph"),
        **source_snapshot(raw),
    }
    res = resolve_payload("my-own-id-scheme-0001", payload)
    assert res.verdict == "intact" and res.supported


def test_access_denied_candidate_is_unresolvable(tmp_path, monkeypatch):
    # EACCES during the protected open must produce a verdict, never a 500 —
    # and it must NOT read as `missing` (that split would leak existence).
    import os as _os

    root = _corpus(tmp_path)
    store, chunks = _index(root)
    real_open = _os.open

    def _denied(*a, **kw):
        raise PermissionError("denied")

    monkeypatch.setattr(_os, "open", _denied)
    res = resolve_citation(store, chunks[0].id)
    monkeypatch.setattr(_os, "open", real_open)
    assert res.verdict == "unresolvable" and not res.supported


def test_symlinked_source_is_refused_in_confined_reads(tmp_path):
    # The openat walk refuses symlinks on EVERY component (the audit trail's
    # documented trade-off): a corpus writer racing the resolver could
    # otherwise swap a component and divert the read outside the root.
    # Legitimately symlinked corpus entries are refused loudly — index real
    # paths.
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    link = root / "gamma.md"
    try:
        link.symlink_to(root / "alpha.md")
    except OSError:  # filesystem without symlink support
        return
    c = next(c for c in chunks if c.payload["source"] == "alpha.md")
    aliased = {
        k: v
        for k, v in c.payload.items()
        if k not in (SOURCE_HASH_KEY, SOURCE_CAPTURED_KEY, ID_SCHEME_KEY)
    }
    aliased["source"] = "gamma.md"
    res = resolve_payload(c.id, aliased)
    assert res.verdict == "unresolvable" and not res.supported


def test_symlinked_intermediate_directory_is_refused(tmp_path):
    root = _corpus(tmp_path)
    real = root / "real"
    real.mkdir()
    (real / "doc.md").write_text("Nested body text about backups.\n")
    link_dir = root / "sub"
    try:
        link_dir.symlink_to(real, target_is_directory=True)
    except OSError:
        return
    payload = {
        "text": "Nested body text about backups.",
        "source": "sub/doc.md",
        "index_root": str(root),
        "offset": 0,
    }
    res = resolve_payload("x", payload)
    assert res.verdict == "unresolvable" and not res.supported


def test_nested_index_root_walks_from_the_allowed_root(tmp_path):
    # index_root nested beneath an allowlisted corpus: the walk anchors at
    # the OPERATOR root, so the nested prefix is O_NOFOLLOW-protected too.
    allowed = tmp_path / "allowed"
    nested = allowed / "team" / "docs"
    nested.mkdir(parents=True)
    (nested / "note.md").write_text("Nested note body.\n")
    payload = {
        "text": "Nested note body.",
        "source": "note.md",
        "index_root": str(nested),
        "offset": 0,
    }
    res = resolve_payload("x", payload, allowed_roots=[str(allowed)])
    assert res.verdict == "intact" and res.supported
    # Swap the nested prefix for an out-of-root symlink: refused by the walk.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "docs").mkdir()
    (outside / "docs" / "note.md").write_text("Nested note body.\n")
    import shutil

    shutil.rmtree(allowed / "team")
    try:
        (allowed / "team").symlink_to(outside, target_is_directory=True)
    except OSError:
        return
    res2 = resolve_payload("x", payload, allowed_roots=[str(allowed)])
    assert res2.verdict == "unresolvable" and not res2.supported


def test_plain_index_offsets_are_raw_coordinates(tmp_path):
    # Non-markdown-chunker points (plain `index`): offsets are raw-file
    # coordinates — leading whitespace added to the file must read as
    # `moved`, never `source_changed` at a stale stripped coordinate.
    root = _corpus(tmp_path)
    doc = root / "plain.md"
    doc.write_text("hello world of plain indexing.\n")
    payload = {
        "text": "hello world",
        "source": "plain.md",
        "index_root": str(root),
        "offset": 0,
    }
    assert resolve_payload("x", payload).verdict == "intact"
    doc.write_text("\n\n" + "hello world of plain indexing.\n")
    res = resolve_payload("x", payload)
    assert res.verdict == "moved" and res.found_offset == 2


def test_absolute_confined_source_goes_through_the_walk(tmp_path):
    root = _corpus(tmp_path)
    # Absolute source INSIDE the root: resolves (via the walk), and a
    # symlinked intermediate directory on the way is refused like for
    # relative sources.
    payload = {
        "text": "First paragraph about postgres backups and verification.",
        "source": str(root / "alpha.md"),
        "index_root": str(root),
        "offset": _DOC.index("First paragraph"),
    }
    assert resolve_payload("x", payload).verdict == "intact"
    real = root / "real"
    real.mkdir()
    (real / "doc.md").write_text("Nested body.\n")
    link_dir = root / "sub"
    try:
        link_dir.symlink_to(real, target_is_directory=True)
    except OSError:
        return
    via_link = {
        "text": "Nested body.",
        "source": str(root / "sub" / "doc.md"),
        "index_root": str(root),
        "offset": 0,
    }
    res = resolve_payload("x", via_link)
    # resolve() collapses the symlink to real/doc.md (inside root) — the
    # WALK then reads the real components; either the collapsed path
    # verifies or a planted symlink refuses. Both are confined outcomes.
    assert res.verdict in ("intact", "unresolvable")


def test_posix_backslash_filenames_survive(tmp_path):
    import os as _os

    if _os.name == "nt":
        return
    root = _corpus(tmp_path)
    weird = root / "a\\b.md"
    weird.write_text("Backslash-named body text.\n")
    payload = {
        "text": "Backslash-named body text.",
        "source": "a\\b.md",
        "index_root": str(root),
        "offset": 0,
    }
    res = resolve_payload("x", payload)
    assert res.verdict == "intact" and res.supported


def test_inserted_blank_line_reads_as_moved_not_in_position(tmp_path):
    # The whitespace gap covers ONLY the chunker's heading-indent strip
    # (<= 3 spaces): an inserted blank line before a fragment is a real
    # shift and must read as moved with the true offset.
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    (root / "alpha.md").write_text("\n" + _DOC)
    for c in chunks:
        if c.payload["source"] != "alpha.md":
            continue
        res = resolve_citation(store, c.id)
        assert res.verdict == "moved", (c.payload["offset"], res.verdict)


def test_indented_heading_sections_keep_their_position(tmp_path):
    # A CommonMark heading indented 1-3 spaces: the chunker strips the
    # section, so the stored text starts AFTER the indent while the offset
    # points at it. Under a drifted snapshot the position check must still
    # hold (whitespace-only gap), yielding source_changed - not moved.
    root = _corpus(tmp_path)
    (root / "indent.md").write_text(
        "# Top\n\nIntro before.\n\n   ## Indented section\n\nIndented body line.\n"
    )
    store, chunks = _index(root)
    mine = [c for c in chunks if c.payload["source"] == "indent.md"]
    assert mine
    p = root / "indent.md"
    p.write_text(p.read_text() + "\nTrailing edit.\n")
    for c in mine:
        if c.payload.get("synthetic_prefix_len", 0) > 0:
            continue  # prefixed chunks are ambiguous under mismatch by contract
        res = resolve_citation(store, c.id)
        assert res.verdict == "source_changed", (c.payload["offset"], res.verdict, res.detail)


def test_frontmatter_text_is_not_search_material(tmp_path):
    # A phrase deleted from the BODY but surviving in YAML frontmatter (a
    # title) must not resolve as `moved` — frontmatter was never indexed.
    root = _corpus(tmp_path)
    (root / "dup.md").write_text(
        "---\ntitle: duplicated exact phrase\n---\nduplicated exact phrase in body.\n"
    )
    store, chunks = _index(root)
    c = next(c for c in chunks if c.payload["source"] == "dup.md")
    (root / "dup.md").write_text(
        "---\ntitle: duplicated exact phrase\n---\nentirely different body now.\n"
    )
    res = resolve_citation(store, c.id)
    assert res.verdict == "changed" and not res.supported


def test_suffix_match_without_snapshot_is_ambiguous(tmp_path):
    # The prefix-length marker is mutable payload (not covered by the id
    # commitment): a suffix-only match against a CHANGED document cannot
    # distinguish a synthetic prefix from a deleted cited first line — the
    # safe verdict is unresolvable, never supported.
    root = _corpus(tmp_path)
    (root / "nested.md").write_text(
        "# Top\n\nIntro paragraph.\n\n## Inner section\n\nDeep content line.\n"
    )
    store, chunks = _index(root)
    prefixed = [
        c
        for c in chunks
        if c.payload["source"] == "nested.md" and c.payload.get("synthetic_prefix_len", 0) > 0
    ]
    assert prefixed
    # Under a matching snapshot the stripped reading is provably synthetic.
    for c in prefixed:
        assert resolve_citation(store, c.id).verdict == "intact"
    # Document drifts: suffix-only evidence downgrades to unresolvable.
    p = root / "nested.md"
    p.write_text(p.read_text() + "\nTrailing edit.\n")
    for c in prefixed:
        res = resolve_citation(store, c.id)
        assert res.verdict == "unresolvable" and not res.supported
        assert "synthetic-prefix" in res.detail
    # The tampering scenario itself: a first-party-shaped point (carries a
    # snapshot hash) whose marker was planted after its real first line was
    # deleted from the source — snapshot mismatches, and the suffix-only
    # match must NOT come back supported.
    doc = root / "plainline.md"
    doc.write_text("bar\n")
    planted = {
        "text": "[Foo]\nbar\n",
        "source": "plainline.md",
        "index_root": str(root),
        "offset": 0,
        "synthetic_prefix_len": 6,
        SOURCE_HASH_KEY: source_content_hash("[Foo]\nbar\n"),  # ingest-era doc
    }
    res = resolve_payload("x", planted)
    assert res.verdict == "unresolvable" and not res.supported
    # A pre-feature (snapshot-absent) prefixed point keeps its positional
    # legacy semantics — the whole legacy path is position-trust already.
    legacy_prefixed = {k: v for k, v in planted.items() if k != SOURCE_HASH_KEY}
    doc.write_text("bar\n")
    assert resolve_payload("x", legacy_prefixed).verdict == "intact"


def test_only_ingest_recorded_prefixes_are_stripped():
    from mnemostack.provenance import _fragment_variants

    # The ingest-recorded length is the ONLY strip authority.
    assert _fragment_variants("[A]\nbody", {"synthetic_prefix_len": 4}) == [
        "[A]\nbody",
        "body",
    ]
    # Marker PRESENT (even zero): it is the only authority — heading_path
    # has no say.
    assert _fragment_variants(
        "[Foo]\nrest", {"synthetic_prefix_len": 0, "heading_path": ["Foo"]}
    ) == ["[Foo]\nrest"]
    # Marker ABSENT (legacy pre-feature payload): the chunker-derived
    # heading-path prefix is stripped for compatibility...
    assert _fragment_variants("[Foo]\nrest", {"heading_path": ["Foo"]}) == [
        "[Foo]\nrest",
        "rest",
    ]
    # ...but a source-native bracketed Setext heading ("[Foo]" is the TITLE,
    # so heading_path holds "[Foo]") still never matches the derivation.
    assert _fragment_variants("[Foo]\n=====\nrest", {"heading_path": ["[Foo]"]}) == [
        "[Foo]\n=====\nrest"
    ]
    # Shape sanity: a bogus length that does not end on the "]\n" boundary
    # or covers the whole text is ignored.
    assert _fragment_variants("[A]\nbody", {"synthetic_prefix_len": 3}) == ["[A]\nbody"]
    assert _fragment_variants("[A]\n", {"synthetic_prefix_len": 4}) == ["[A]\n"]


def test_root_override_rebases_absolute_sources(tmp_path):
    # A point that recorded an ABSOLUTE source under its old root must
    # follow --root to the relocated corpus (the override replaces the
    # recorded root, docs promise relocation works).
    root = _corpus(tmp_path)
    payload = {
        "text": "First paragraph about postgres backups and verification.",
        "source": str(root / "alpha.md"),
        "index_root": str(root),
        "offset": _DOC.index("First paragraph"),
    }
    moved_root = tmp_path / "relocated"
    root.rename(moved_root)
    assert resolve_payload("x", payload).verdict == "missing"
    res = resolve_payload("x", payload, root=str(moved_root))
    assert res.verdict == "intact" and res.supported


def test_traversal_source_is_refused(tmp_path):
    # `source` is an untrusted label — a ../ escape under the corpus root
    # must be refused outright, even when the target file exists.
    root = _corpus(tmp_path)
    (tmp_path / "outside.md").write_text("secret contents")
    res = resolve_payload(
        "x",
        {"text": "secret contents", "source": "../outside.md", "index_root": str(root), "offset": 0},
    )
    assert res.verdict == "unresolvable" and "escapes" in res.detail
    assert res.fragment is None


def test_symlink_escape_is_refused(tmp_path):
    root = _corpus(tmp_path)
    (tmp_path / "outside.md").write_text("secret contents")
    link = root / "link.md"
    try:
        link.symlink_to(tmp_path / "outside.md")
    except OSError:  # filesystem without symlink support
        return
    res = resolve_payload(
        "x", {"text": "secret contents", "source": "link.md", "index_root": str(root), "offset": 0}
    )
    assert res.verdict == "unresolvable" and "escapes" in res.detail


def test_bare_paths_only_resolve_for_the_operator_surface(tmp_path):
    doc = tmp_path / "standalone.md"
    doc.write_text("standalone body text")
    payload = {"text": "standalone body text", "source": str(doc), "offset": 0}
    # Service surfaces (default): a payload with no corpus root never touches
    # the filesystem via a bare (absolute or cwd-relative) path.
    assert resolve_payload("x", payload).verdict == "unresolvable"
    # Operator CLI: explicitly allowed.
    res = resolve_payload("x", payload, allow_unrooted=True)
    assert res.verdict == "intact" and res.supported


def test_oversized_source_is_refused(tmp_path, monkeypatch):
    import mnemostack.provenance as prov

    root = _corpus(tmp_path)
    store, chunks = _index(root)
    monkeypatch.setattr(prov, "MAX_RESOLVE_BYTES", 8)
    res = resolve_citation(store, chunks[0].id)
    assert res.verdict == "unresolvable" and "cap" in res.detail


def test_missing_detail_leaks_no_paths(tmp_path):
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    (root / "alpha.md").unlink()
    c = next(c for c in chunks if c.payload["source"] == "alpha.md")
    res = resolve_citation(store, c.id)
    assert res.verdict == "missing"
    # Reconnaissance hygiene: the detail names no absolute server paths.
    assert str(root) not in res.detail


def test_tenant_scope_hides_foreign_points(tmp_path):
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    # Stamp one point with a tenant, then resolve as another tenant.
    target = chunks[0]
    payload = dict(target.payload)
    store.upsert(target.id, _V, payload, tenant="acme")
    assert resolve_citation(store, target.id, tenant="acme").verdict != "unresolvable"
    res = resolve_citation(store, target.id, tenant="globex")
    assert res.verdict == "unresolvable"
    assert "no such point" in res.detail  # indistinguishable from absent


def test_explicit_root_fully_overrides_stored_root(tmp_path):
    # With the OLD corpus still on disk, a source deleted from the explicitly
    # selected root must report missing — not resolve intact via the stale
    # index_root copy.
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    new_root = tmp_path / "selected"
    new_root.mkdir()
    (new_root / "beta.md").write_text(_FM_DOC)  # alpha.md absent here
    c = next(c for c in chunks if c.payload["source"] == "alpha.md")
    res = resolve_citation(store, c.id, root=str(new_root))
    assert res.verdict == "missing"


def test_sliding_window_points_are_honest(tmp_path):
    # Windowed text is synthetic BY DESIGN, and chunk_kind is payload data —
    # a matching document hash must not launder it into `intact` (a writer
    # could mark planted text as windowed to skip fragment verification).
    # The only honest verdict is unresolvable, hash match or not.
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    c = next(c for c in chunks if c.payload["source"] == "alpha.md")
    # Tampered first-party payload marked windowed: the id commitment
    # refuses it before the windowed special case is even consulted.
    windowed = dict(c.payload)
    windowed["chunk_kind"] = "sliding_window"
    windowed["chunk_window"] = 3
    windowed["chunk_start_offset"] = 0
    windowed["chunk_end_offset"] = 2
    windowed["text"] = "synthetic\n" + c.payload["text"] + "\njoined"
    res_match = resolve_payload(c.id, windowed)
    assert res_match.verdict == "unresolvable" and not res_match.supported
    # A genuine windowed point (no snapshot fields — cmd_index does not
    # stamp them on windows) is honestly unverifiable, hash or no hash.
    genuine = {
        k: v for k, v in windowed.items() if k not in (SOURCE_HASH_KEY, SOURCE_CAPTURED_KEY, ID_SCHEME_KEY)
    }
    res = resolve_payload(c.id, genuine)
    assert res.verdict == "unresolvable" and "constituent" in res.detail
    (root / "alpha.md").write_text(_DOC + "\ntail edit\n")
    assert resolve_payload(c.id, genuine).verdict == "unresolvable"


def test_all_digit_uuid_ids_stay_strings(tmp_path):
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    digit_uuid = "11111111111111111111111111111111"  # valid simple-form UUID
    # Foreign-shaped copy (no snapshot): this test exercises ID handling,
    # not the first-party id commitment.
    copy = {
        k: v
        for k, v in chunks[0].payload.items()
        if k not in (SOURCE_HASH_KEY, SOURCE_CAPTURED_KEY, ID_SCHEME_KEY)
    }
    store.upsert(digit_uuid, [1.0, 0.0, 0.0, 0.0], copy)
    assert resolve_citation(store, digit_uuid).verdict == "intact"


def test_non_string_captured_at_normalizes_to_none(tmp_path):
    from mnemostack.provenance import SOURCE_CAPTURED_KEY as CK

    root = _corpus(tmp_path)
    store, chunks = _index(root)
    weird = dict(chunks[0].payload)
    weird[CK] = 1722600000  # epoch number from an external writer
    res = resolve_payload(chunks[0].id, weird)
    assert res.verdict == "intact"
    assert res.captured_at is None  # response contract is str | None


def test_leading_whitespace_document_offsets_still_verify(tmp_path):
    # A headingless note starting with whitespace: the chunker strips the
    # text before windowing, so stored offsets are stripped-coordinates —
    # an unchanged later chunk must verify at its offset, not misreport
    # `moved`.
    root = _corpus(tmp_path)
    para = "word " * 80  # ~400 chars, forces multiple 1200-char windows
    (root / "lead.md").write_text("\n\n\n" + "\n\n".join(para for _ in range(6)))
    store, chunks = _index(root)
    lead = [c for c in chunks if c.payload["source"] == "lead.md"]
    assert len(lead) > 1  # windowed, with a stripped coordinate base
    # Snapshot mismatch forces the position path (the hash shortcut would
    # mask a coordinate bug): touch the file's END only.
    p = root / "lead.md"
    p.write_text(p.read_text() + "\n\ntail")
    for c in lead:
        res = resolve_citation(store, c.id)
        assert res.verdict == "source_changed", (c.payload["offset"], res.verdict)


def test_hash_match_returns_source_native_fragment(tmp_path):
    root = _corpus(tmp_path)
    # Nested headings force the chunker's synthetic "[parent]\n" prefix.
    (root / "nested.md").write_text(
        "# Top\n\nIntro paragraph.\n\n## Inner section\n\nDeep content line.\n"
    )
    store, chunks = _index(root)
    prefixed = [c for c in chunks if c.text.startswith("[") and "]\n" in c.text]
    assert prefixed  # the fixture must actually exercise the prefix path
    for c in prefixed:
        res = resolve_citation(store, c.id)
        assert res.verdict == "intact"
        # The fragment is what the SOURCE contains — no synthetic prefix.
        raw = (root / c.payload["source"]).read_text()
        assert res.fragment in raw


def test_root_override_wins_over_payload_root(tmp_path):
    root = _corpus(tmp_path)
    store, chunks = _index(root)
    moved_root = tmp_path / "relocated"
    root.rename(moved_root)
    c = chunks[0]
    # Payload's index_root is now stale — explicit root recovers resolution.
    assert resolve_citation(store, c.id).verdict == "missing"
    res = resolve_citation(store, c.id, root=str(moved_root))
    assert res.verdict == "intact" and res.supported


# ---------- CLI ----------


def test_cli_resolve_exit_codes(tmp_path, monkeypatch, capsys):
    import argparse

    import mnemostack.cli as cli

    root = _corpus(tmp_path)
    store, chunks = _index(root)
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)

    def _args(chunk_id):
        return argparse.Namespace(
            collection="prov", qdrant="http://x", chunk_id=chunk_id,
            root=None, tenant=None, json=False,
        )

    assert cli.cmd_resolve(_args(chunks[0].id)) == 0
    assert "intact" in capsys.readouterr().out
    (root / "alpha.md").write_text(_DOC.replace("postgres", "mysql"))
    bad = next(c for c in chunks if c.payload["source"] == "alpha.md" and "postgres" in c.text)
    assert cli.cmd_resolve(_args(bad.id)) == 1
    assert cli.cmd_resolve(_args("00000000-0000-0000-0000-000000000000")) == 2
