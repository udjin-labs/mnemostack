"""Owned-payload diffing: warm syncs must cost zero mutation requests."""

from __future__ import annotations

from mnemostack.embeddings.base import EmbeddingProvider
from mnemostack.markdown.sync import upsert_markdown_chunks
from mnemostack.vector.patch import PayloadPatch, diff_payload

# ------------------------------------------------------------------- diff


def test_identical_owned_payload_yields_no_patch():
    old = {"text": "t", "source": "a.md", "_md_keys": ["title"], "title": "X"}
    new = {"text": "t", "source": "a.md", "_md_keys": ["title"], "title": "X"}
    assert diff_payload(old, new, point_id="p1") is None


def test_container_flavor_and_key_order_do_not_create_patches():
    old = {"tags": ["a", "b"], "meta": {"x": 1, "y": 2}}
    new = {"tags": ("a", "b"), "meta": {"y": 2, "x": 1}}
    assert diff_payload(old, new, point_id="p1") is None


def test_changed_value_rewrites_the_full_owned_payload():
    # Write-or-skip: any owned change triggers the FULL merge-write (single
    # writer stays coherent under overlap), never a per-key fragment.
    old = {"text": "t", "title": "Old", "source": "a.md"}
    new = {"text": "t", "title": "New", "source": "a.md"}
    patch = diff_payload(old, new, point_id="p1")
    assert patch == PayloadPatch(id="p1", set_values=new)


def test_stale_owned_key_produces_delete_only_when_present():
    old = {"text": "t", "title": "Old"}
    new = {"text": "t"}
    patch = diff_payload(old, new, point_id="p1", stale_keys=["title", "ghost"])
    assert patch is not None
    # The delete list passes through verbatim (historical semantics —
    # deleting an absent key was always an idempotent no-op downstream).
    assert patch.delete_keys == ("title", "ghost")
    assert patch.set_values == new


def test_foreign_keys_never_enter_the_patch():
    old = {"text": "t", "validity": "current", "enriched_topic": "x"}
    new = {"text": "t"}
    # Foreign keys differ from the new payload but are neither set nor
    # deleted — they are simply outside the owned universe.
    assert diff_payload(old, new, point_id="p1") is None


def test_new_key_triggers_a_write():
    patch = diff_payload({"text": "t"}, {"text": "t", "title": "T"}, point_id="p1")
    assert patch == PayloadPatch(id="p1", set_values={"text": "t", "title": "T"})


def test_mixed_yaml_key_types_do_not_crash_and_compare_effectively():
    # {1: ..., "x": ...} is valid YAML; a JSON backend stores keys as
    # strings — comparison must neither crash on the sort nor see a change.
    old = {"meta": {"1": "one", "x": "two"}}
    new = {"meta": {1: "one", "x": "two"}}
    assert diff_payload(old, new, point_id="p1") is None


# ------------------------------------------------------- markdown warm sync


class _Provider(EmbeddingProvider):
    def embed(self, text):
        return [0.1, 0.2]

    def embed_batch(self, texts):
        return [[0.1, 0.2] for _ in texts]

    @property
    def dimension(self):
        return 2

    @property
    def name(self):
        return "custom:diff-test"


class _CountingStore:
    def __init__(self):
        self.set_payloads: list[tuple[str, dict]] = []
        self.deletes: list[tuple[str, list]] = []
        self.upserts = 0

    def upsert(self, cid, vec, payload, **kw):
        self.upserts += 1

    def upsert_batch(self, points, **kw):
        self.upserts += len(points)

    def set_payload(self, cid, payload, **kw):
        self.set_payloads.append((cid, payload))

    def delete_payload_keys(self, cid, keys, **kw):
        self.deletes.append((cid, keys))


def _payload(title: str) -> dict:
    return {
        "text": "body",
        "source": "a.md",
        "_md_keys": ["title"],
        "title": title,
    }


def test_unchanged_warm_sync_issues_zero_mutation_requests():
    store = _CountingStore()
    provider = _Provider()
    # A truly warm point already carries the current space stamp — the very
    # first post-upgrade sync adopts it (one legitimate patch), every run
    # after that must be free.
    warm = dict(_payload("T"), _embedding_space=provider.document_space_fingerprint())
    res = upsert_markdown_chunks(
        store,
        provider,
        [("c1", "body", _payload("T"))],
        existing_payloads={"c1": warm},
    )
    assert res.inserted == 0
    assert res.compared == 1
    assert res.unchanged == 1
    assert res.refreshed == 0
    assert store.set_payloads == [] and store.deletes == [] and store.upserts == 0


def test_changed_frontmatter_patches_only_the_difference():
    store = _CountingStore()
    res = upsert_markdown_chunks(
        store,
        _Provider(),
        [("c1", "body", _payload("New"))],
        existing_payloads={"c1": _payload("Old")},
    )
    assert res.refreshed == 1 and res.unchanged == 0
    assert len(store.set_payloads) == 1
    cid, values = store.set_payloads[0]
    assert cid == "c1"
    # Full merge-write on change (last-writer coherence), including the
    # space stamp the provider contributes on first contact.
    assert values.get("title") == "New"
    assert values.get("text") == "body" and values.get("source") == "a.md"


def test_ownership_transition_deletes_stale_owned_key():
    store = _CountingStore()
    old = dict(_payload("T"), subtitle="S")
    old["_md_keys"] = ["title", "subtitle"]
    new = _payload("T")  # subtitle no longer produced
    res = upsert_markdown_chunks(
        store,
        _Provider(),
        [("c1", "body", new)],
        existing_payloads={"c1": old},
    )
    assert res.refreshed == 1
    assert store.deletes and store.deletes[0][1] == ["subtitle"]


def test_snapshot_capture_time_parity_keeps_warm_runs_free():
    """A re-run stamps a fresh capture timestamp; with an unchanged content
    hash it must NOT register as a change — or every warm run would rewrite
    every point for the timestamp alone."""
    store = _CountingStore()
    provider = _Provider()
    fp = provider.document_space_fingerprint()
    old = dict(
        _payload("T"),
        _embedding_space=fp,
        source_content_hash="sha256:same",
        source_captured_at="2026-08-01T00:00:00+00:00",
    )
    fresh = dict(
        _payload("T"),
        source_content_hash="sha256:same",
        source_captured_at="2026-08-05T12:00:00+00:00",  # this run's stamp
    )
    res = upsert_markdown_chunks(
        store, provider, [("c1", "body", fresh)], existing_payloads={"c1": old}
    )
    assert res.unchanged == 1 and res.refreshed == 0
    assert store.set_payloads == []

    # A CHANGED hash is a real re-capture: the fresh timestamp is written.
    store2 = _CountingStore()
    changed = dict(
        _payload("T"),
        source_content_hash="sha256:different",
        source_captured_at="2026-08-05T12:00:00+00:00",
    )
    res2 = upsert_markdown_chunks(
        store2, provider, [("c1", "body", changed)], existing_payloads={"c1": old}
    )
    assert res2.refreshed == 1
    assert store2.set_payloads[0][1]["source_captured_at"] == "2026-08-05T12:00:00+00:00"


def test_date_frontmatter_objects_compare_against_stored_iso_strings():
    # PyYAML resolves `date: 2024-01-01` to datetime.date; the backend stores
    # the ISO string — a warm run must not see that as a change.
    import datetime

    old = {"date": "2024-01-01"}
    new = {"date": datetime.date(2024, 1, 1)}
    assert diff_payload(old, new, point_id="p1") is None


def test_cmd_index_warm_second_run_issues_zero_payload_writes(tmp_path, monkeypatch, capsys):
    import mnemostack.cli as cli

    (tmp_path / "doc.md").write_text("stable content")

    class _P(_Provider):
        pass

    provider = _P()

    class _Store:
        def __init__(self):
            self.points: dict[str, dict] = {}
            self.set_payloads: list = []
            self.deletes: list = []

        def collection_exists(self):
            return True

        def ensure_collection(self, recreate=False):
            pass

        def count(self):
            return len(self.points)

        def iter_ids(self):
            return iter(self.points)

        def scroll(self, *a, **kw):
            from types import SimpleNamespace

            return iter(
                SimpleNamespace(id=cid, payload=dict(p)) for cid, p in self.points.items()
            )

        def upsert(self, cid, vec, payload, **kw):
            self.points[cid] = dict(payload)

        def upsert_batch(self, points, **kw):
            for cid, vec, payload in points:
                self.upsert(cid, vec, payload)

        def set_payload(self, cid, payload, **kw):
            self.set_payloads.append((cid, payload))
            self.points[cid].update(payload)

        def delete_payload_keys(self, cid, keys, **kw):
            self.deletes.append((cid, keys))

    store = _Store()
    monkeypatch.setattr(cli, "get_provider", lambda *a, **kw: provider)
    monkeypatch.setattr(cli, "VectorStore", lambda **kw: store)
    argv = [
        "index",
        str(tmp_path),
        "--chunk-size",
        "2000",
        "--refresh-payloads",
    ]
    assert cli.main(argv) == 0
    first_points = {k: dict(v) for k, v in store.points.items()}
    capsys.readouterr()

    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    # Second run: nothing embedded, nothing patched — and the summary says so.
    assert store.set_payloads == [] and store.deletes == []
    assert store.points == first_points
    assert "1 compared / 1 unchanged / 0 patched" in out
