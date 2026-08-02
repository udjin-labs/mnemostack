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
    res = resolve_payload(
        "x",
        {"text": "abc", "source": "https://example.com/doc", "offset": 0},
        allow_unrooted=True,
    )
    assert res.verdict == "unresolvable" and "URI" in res.detail


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
