"""Source-scoped lifecycle (`source` on /invalidate and DELETE /memories)
and the reconciliation listing (`GET /memories?source=`).

The operation every client actually has is "this file was rewritten (or
this session reset) — forget what came from it", and the check every
client needs is "what do you hold from it, and does it match what I
have". Neither was expressible with id lists alone.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from test_remote_ingest import _ingest_app

from mnemostack.ingest import REMOTE_SOURCE_BATCH, find_source_points
from mnemostack.provenance import SOURCE_HASH_KEY


def _store_items(client, key, items):
    r = client.post("/memories", json={"items": items}, headers={"X-API-Key": key})
    assert r.status_code == 200, r.text
    return [row["id"] for row in r.json()["results"]]


# ------------------------------------------------------ source retraction


def test_invalidate_by_source_retracts_the_whole_document(monkeypatch, tmp_path):
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    hdr = {"X-API-Key": keys["write"]}
    kept = _store_items(client, keys["write"], [{"text": "other doc", "source": "b.md"}])
    ids = _store_items(
        client,
        keys["write"],
        [
            {"text": "first chunk of the doc", "source": "a.md", "offset": 0},
            {"text": "second chunk of the doc", "source": "a.md", "offset": 1},
        ],
    )
    r = client.post("/invalidate", json={"source": "a.md"}, headers=hdr)
    assert r.status_code == 200
    body = r.json()
    assert body == {"requested": 2, "invalidated": 2, "complete": True}
    for pid in ids:
        point = store.client.retrieve(store.collection, ids=[pid], with_payload=True)[0]
        assert point.payload.get("invalidated_at")
    # A different source is untouched.
    point = store.client.retrieve(store.collection, ids=kept, with_payload=True)[0]
    assert "invalidated_at" not in (point.payload or {})


def test_delete_by_source_erases_the_whole_document(monkeypatch, tmp_path):
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    hdr = {"X-API-Key": keys["write"]}
    ids = _store_items(
        client,
        keys["write"],
        [
            {"text": "chunk one of the file", "source": "doc.md", "offset": 0},
            {"text": "chunk two of the file", "source": "doc.md", "offset": 1},
        ],
    )
    r = client.request("DELETE", "/memories", json={"source": "doc.md"}, headers=hdr)
    assert r.json() == {"requested": 2, "deleted": 2, "complete": True}
    assert store.client.retrieve(store.collection, ids=ids, with_payload=False) == []
    # Idempotent: nothing left to erase.
    r = client.request("DELETE", "/memories", json={"source": "doc.md"}, headers=hdr)
    assert r.json() == {"requested": 0, "deleted": 0, "complete": True}


def test_source_selector_is_tenant_scoped(monkeypatch, tmp_path):
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    alpha_ids = _store_items(
        client, keys["write"], [{"text": "alpha's copy", "source": "shared.md"}]
    )
    beta_ids = _store_items(
        client, keys["beta_write"], [{"text": "beta's copy", "source": "shared.md"}]
    )
    r = client.request(
        "DELETE",
        "/memories",
        json={"source": "shared.md"},
        headers={"X-API-Key": keys["beta_write"]},
    )
    assert r.json()["deleted"] == 1  # only beta's
    assert store.client.retrieve(store.collection, ids=beta_ids, with_payload=False) == []
    assert len(store.client.retrieve(store.collection, ids=alpha_ids, with_payload=False)) == 1


def test_exactly_one_selector_is_required(monkeypatch, tmp_path):
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    hdr = {"X-API-Key": keys["write"]}
    for payload in (
        {},  # neither
        {"ids": ["7"], "source": "a.md"},  # both
    ):
        r = client.post("/invalidate", json=payload, headers=hdr)
        assert r.status_code == 400 and "exactly one" in r.json()["detail"]
        r = client.request("DELETE", "/memories", json=payload, headers=hdr)
        assert r.status_code == 400 and "exactly one" in r.json()["detail"]
    r = client.post("/invalidate", json={"source": "   "}, headers=hdr)
    assert r.status_code == 400 and "non-blank" in r.json()["detail"]


def test_array_payload_source_is_not_a_match(monkeypatch, tmp_path):
    """Qdrant's MatchValue also matches an ARRAY payload containing the
    value — a point whose source is ['a.md','b.md'] must NOT be erased by
    source='a.md' (the rule the selective-prune fix established)."""
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    (pid,) = _store_items(
        client, keys["write"], [{"text": "multi-source chunk", "source": "a.md"}]
    )
    store.client.set_payload(
        collection_name=store.collection,
        payload={"source": ["a.md", "b.md"]},
        points=[pid],
    )
    r = client.request(
        "DELETE",
        "/memories",
        json={"source": "a.md"},
        headers={"X-API-Key": keys["write"]},
    )
    assert r.json() == {"requested": 0, "deleted": 0, "complete": True}
    assert len(store.client.retrieve(store.collection, ids=[pid], with_payload=False)) == 1


def test_source_batch_reports_incomplete(monkeypatch, tmp_path):
    """A big document is processed in bounded batches; the caller repeats
    until the service says it finished."""
    import mnemostack.server as srv

    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    hdr = {"X-API-Key": keys["write"]}
    _store_items(
        client,
        keys["write"],
        [
            {"text": f"chunk number {i} of a long doc", "source": "big.md", "offset": i}
            for i in range(5)
        ],
    )
    monkeypatch.setattr(srv, "REMOTE_SOURCE_BATCH", 2)
    seen = 0
    for _ in range(5):
        body = client.request(
            "DELETE", "/memories", json={"source": "big.md"}, headers=hdr
        ).json()
        seen += body["deleted"]
        if body["complete"]:
            break
    assert seen == 5


# ----------------------------------------------------------- listing


def test_listing_returns_ids_and_hashes_never_text(monkeypatch, tmp_path):
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    ids = _store_items(
        client,
        keys["write"],
        [
            {"text": "first chunk here", "source": "a.md", "offset": 0},
            {"text": "second chunk here", "source": "a.md", "offset": 1},
        ],
    )
    store.client.set_payload(
        collection_name=store.collection,
        payload={SOURCE_HASH_KEY: "deadbeef"},
        points=[ids[0]],
    )
    r = client.get(
        "/memories", params={"source": "a.md"}, headers={"X-API-Key": keys["read"]}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["complete"] is True
    assert {row["id"] for row in body["items"]} == set(ids)
    assert "chunk here" not in r.text  # no memory text on this surface
    hashes = {row["id"]: row["content_hash"] for row in body["items"]}
    assert hashes[ids[0]] == "deadbeef" and hashes[ids[1]] is None
    assert all(row["indexed_at"] for row in body["items"])


def test_listing_is_tenant_scoped_and_read_gated(monkeypatch, tmp_path):
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    _store_items(client, keys["write"], [{"text": "alpha only", "source": "s.md"}])
    r = client.get("/memories", params={"source": "s.md"})
    assert r.status_code == 401
    # write-only key: the listing is read-gated
    r = client.get(
        "/memories", params={"source": "s.md"}, headers={"X-API-Key": keys["beta_write"]}
    )
    assert r.status_code == 403
    # another tenant's read key sees nothing of alpha's
    r = client.get(
        "/memories", params={"source": "s.md"}, headers={"X-API-Key": keys["beta_read"]}
    )
    assert r.status_code == 200 and r.json()["items"] == []
    # alpha's own read key sees it
    r = client.get(
        "/memories", params={"source": "s.md"}, headers={"X-API-Key": keys["read"]}
    )
    assert r.status_code == 200 and len(r.json()["items"]) == 1


def test_listing_paginates_by_id(monkeypatch, tmp_path):
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    ids = _store_items(
        client,
        keys["write"],
        [
            {"text": f"paged chunk {i}", "source": "p.md", "offset": i}
            for i in range(5)
        ],
    )
    hdr = {"X-API-Key": keys["read"]}
    seen: list[str] = []
    after = None
    for _ in range(5):
        params = {"source": "p.md", "limit": 2}
        if after:
            params["after"] = after
        body = client.get("/memories", params=params, headers=hdr).json()
        seen.extend(row["id"] for row in body["items"])
        if body["complete"]:
            break
        after = body["items"][-1]["id"]
    assert sorted(seen) == sorted(ids)
    assert len(seen) == len(set(seen))  # no duplicates across pages


def test_listing_rejects_bad_input(monkeypatch, tmp_path):
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    hdr = {"X-API-Key": keys["read"]}
    r = client.get("/memories", params={"source": "  "}, headers=hdr)
    assert r.status_code == 400
    r = client.get("/memories", params={"source": "a.md", "limit": 0}, headers=hdr)
    assert r.status_code == 400
    r = client.get("/memories", params={"source": "a.md", "limit": 100000}, headers=hdr)
    assert r.status_code == 400


def test_finder_is_a_pure_helper_without_a_scroll_hook():
    """Duck stores without scroll degrade to 'nothing found', not a crash."""

    class _NoScroll:
        pass

    assert find_source_points(_NoScroll(), "a.md") == ([], [], False)
    assert REMOTE_SOURCE_BATCH > 0


def test_batched_source_invalidate_reaches_the_tail(monkeypatch, tmp_path):
    """R1 (codex P1): an invalidated point keeps its source, so without
    excluding it the next call re-selects the same first batch forever
    and a document larger than one batch is never fully retracted."""
    import mnemostack.server as srv

    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    hdr = {"X-API-Key": keys["write"]}
    ids = _store_items(
        client,
        keys["write"],
        [
            {"text": f"tail chunk {i} of the doc", "source": "long.md", "offset": i}
            for i in range(5)
        ],
    )
    monkeypatch.setattr(srv, "REMOTE_SOURCE_BATCH", 2)
    calls = 0
    while True:
        body = client.post("/invalidate", json={"source": "long.md"}, headers=hdr).json()
        calls += 1
        assert calls <= 6, "batches are not making progress"
        if body["complete"]:
            break
    for pid in ids:  # every point, not just the first batch
        point = store.client.retrieve(store.collection, ids=[pid], with_payload=True)[0]
        assert point.payload.get("invalidated_at"), pid
    # A finished retraction reports complete immediately, with nothing left.
    body = client.post("/invalidate", json={"source": "long.md"}, headers=hdr).json()
    assert body == {"requested": 0, "invalidated": 0, "complete": True}


def test_listing_page_costs_the_page_not_the_source(monkeypatch, tmp_path):
    """R1 (codex P2): every page used to scan the whole source and sort it
    before slicing. Cost must follow the page, not the source — checked
    with a source larger than one store batch, or the two are
    indistinguishable."""
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    total = 400
    for start in range(0, total, 50):
        _store_items(
            client,
            keys["write"],
            [
                {"text": f"scanned chunk {i}", "source": "wide.md", "offset": i}
                for i in range(start, start + 50)
            ],
        )
    seen_points = 0
    orig_scroll = store.client.scroll

    def _counting(*a, **kw):
        nonlocal seen_points
        points, offset = orig_scroll(*a, **kw)
        seen_points += len(points)
        return points, offset

    store.client.scroll = _counting  # type: ignore[method-assign]
    hdr = {"X-API-Key": keys["read"]}
    body = client.get(
        "/memories", params={"source": "wide.md", "limit": 3}, headers=hdr
    ).json()
    assert len(body["items"]) == 3 and body["complete"] is False
    first_page_cost = seen_points
    assert first_page_cost < total, first_page_cost  # not a full scan

    # The next page resumes instead of re-walking what was returned.
    seen_points = 0
    body2 = client.get(
        "/memories",
        params={"source": "wide.md", "limit": 3, "after": body["items"][-1]["id"]},
        headers=hdr,
    ).json()
    assert len(body2["items"]) == 3
    assert {r["id"] for r in body2["items"]}.isdisjoint(
        {r["id"] for r in body["items"]}
    )
    assert seen_points < total, seen_points


def _put_int_points(store, source: str, count: int, **extra) -> list[int]:
    """Integer point ids — Qdrant's other supported id domain, and the one
    a JSON cursor round-trip loses (the response prints "7", the store
    distinguishes 7 from "7")."""
    ids = list(range(1, count + 1))
    for pid in ids:
        store.upsert(
            pid,
            [0.1, 0.2, 0.3],
            {"text": f"chunk {pid}", "source": source, "offset": pid, **extra},
            tenant="alpha",  # the key the listing reads with
        )
    return ids


def test_listing_paginates_integer_point_ids(monkeypatch, tmp_path):
    """R2 (codex P2): the cursor comes back as the STRING the response
    printed. Passed through uncoerced it resumes from an id that does not
    exist, so page 2 is empty and the listing reports itself complete
    while points remain."""
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    ids = _put_int_points(store, "int.md", 5)
    hdr = {"X-API-Key": keys["read"]}
    seen: list[str] = []
    after = None
    for _ in range(5):
        params = {"source": "int.md", "limit": 2}
        if after:
            params["after"] = after
        body = client.get("/memories", params=params, headers=hdr).json()
        seen.extend(row["id"] for row in body["items"])
        if body["complete"]:
            break
        after = body["items"][-1]["id"]
    assert sorted(seen, key=int) == [str(i) for i in ids]
    assert len(seen) == len(set(seen))


def test_listing_returns_a_numeric_timestamp_domain(monkeypatch, tmp_path):
    """R2 (codex P2): under `recall.timestamp_format: epoch` the payload's
    timestamp is a NUMBER. A str-only response field made listing any
    timestamped memory a 500."""
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    _put_int_points(store, "epoch.md", 1, timestamp=1755600000.0)
    hdr = {"X-API-Key": keys["read"]}
    r = client.get("/memories", params={"source": "epoch.md"}, headers=hdr)
    assert r.status_code == 200, r.text
    assert r.json()["items"][0]["timestamp"] == 1755600000.0


def test_listing_metadata_of_a_foreign_type_does_not_fail_the_page(monkeypatch, tmp_path):
    """One point with a junk value under a reserved key must not 500 the
    whole page — the field is an integrity hint, reported absent instead."""
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    _put_int_points(store, "junk.md", 1, **{SOURCE_HASH_KEY: 12345, "indexed_at": 7})
    hdr = {"X-API-Key": keys["read"]}
    r = client.get("/memories", params={"source": "junk.md"}, headers=hdr)
    assert r.status_code == 200, r.text
    item = r.json()["items"][0]
    assert item["content_hash"] is None and item["indexed_at"] is None


def test_listing_rejects_a_malformed_cursor(monkeypatch, tmp_path):
    """A cursor IS a point id: same domain, same 400 — and notably the
    digit-limit guard, since a long enough digit string makes int() itself
    raise (a 500 on malformed input)."""
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    hdr = {"X-API-Key": keys["read"]}
    for bad in ("not-a-point-id", "1" * 5000, "-3"):
        r = client.get(
            "/memories", params={"source": "a.md", "after": bad}, headers=hdr
        )
        assert r.status_code == 400, (bad, r.status_code)


def test_listing_takes_the_index_root_owner_guard(monkeypatch, tmp_path):
    """R2 (review agent P3): one source name can exist under several
    indexing roots. The lifecycle endpoints scope by `index_root`; the
    listing silently ignored the parameter, so a client reconciling ONE
    root saw another root's points mixed in with no warning."""
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    store.upsert(1, [0.1, 0.2, 0.3], {"source": "shared.md", "index_root": "/a"}, tenant="alpha")
    store.upsert(2, [0.1, 0.2, 0.3], {"source": "shared.md", "index_root": "/b"}, tenant="alpha")
    store.upsert(3, [0.1, 0.2, 0.3], {"source": "shared.md"}, tenant="alpha")  # untagged
    hdr = {"X-API-Key": keys["read"]}
    body = client.get(
        "/memories", params={"source": "shared.md", "index_root": "/a"}, headers=hdr
    ).json()
    # The guard EXCLUDES a different root; it does not require one, so the
    # untagged point stays listed — the semantics the siblings document.
    assert sorted(row["id"] for row in body["items"]) == ["1", "3"]
    body = client.get("/memories", params={"source": "shared.md"}, headers=hdr).json()
    assert len(body["items"]) == 3  # no guard: everything
    r = client.get(
        "/memories", params={"source": "shared.md", "index_root": "  "}, headers=hdr
    )
    assert r.status_code == 400  # a blank guard matches no owner


def test_the_unix_epoch_is_a_timestamp_not_an_absence(monkeypatch, tmp_path):
    """R3 (codex P2): under `timestamp_format: epoch` the Unix epoch is a
    VALID timestamp stored as numeric 0, and a falsy test silently
    substituted the legacy mirror for it."""
    app, store, _emb, keys = _ingest_app(
        monkeypatch, tmp_path, cfg_extra={"timestamp_key": "event_time"}
    )
    client = TestClient(app)
    # The CONFIGURED key holds the epoch; the legacy mirror holds something
    # else entirely, so a falsy test is visible instead of coincidental.
    store.upsert(
        1,
        [0.1, 0.2, 0.3],
        {"source": "epoch0.md", "event_time": 0, "timestamp": "1999-01-01T00:00:00+00:00"},
        tenant="alpha",
    )
    r = client.get(
        "/memories", params={"source": "epoch0.md"}, headers={"X-API-Key": keys["read"]}
    )
    assert r.status_code == 200, r.text
    assert r.json()["items"][0]["timestamp"] == 0


def test_source_retraction_is_first_retraction_wins(monkeypatch, tmp_path):
    """R3 (codex P2): a point already invalidated is skipped by the source
    path — that is what makes batching terminate. The contract is
    documented as first-retraction-wins: `invalidated_at` records WHEN a
    memory was retracted, so a later source call must not falsify it."""
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    hdr = {"X-API-Key": keys["write"]}
    ids = _store_items(
        client,
        keys["write"],
        [
            {"text": "first chunk of the doc", "source": "d.md", "offset": 0},
            {"text": "second chunk of the doc", "source": "d.md", "offset": 1},
        ],
    )
    r = client.post(
        "/invalidate",
        json={"ids": [ids[0]], "invalidated_at": "2020-01-01T00:00:00+00:00"},
        headers=hdr,
    )
    assert r.json()["invalidated"] == 1
    # Now retract the whole document with a world-time bound.
    r = client.post(
        "/invalidate",
        json={"source": "d.md", "valid_until": "2026-01-01T00:00:00+00:00"},
        headers=hdr,
    )
    # Only the still-active point is requested and touched; the response
    # does not claim to have restamped the one already retracted.
    assert r.json() == {"requested": 1, "invalidated": 1, "complete": True}
    first = store.client.retrieve(store.collection, ids=[ids[0]], with_payload=True)[0]
    second = store.client.retrieve(store.collection, ids=[ids[1]], with_payload=True)[0]
    assert first.payload["invalidated_at"] == "2020-01-01T00:00:00+00:00"
    assert "valid_until" not in first.payload  # the earlier retraction stands
    assert second.payload["valid_until"] == "2026-01-01T00:00:00+00:00"


def test_the_index_root_guard_is_pushed_down_to_the_store(monkeypatch, tmp_path):
    """R3 (codex P2): the limit counts KEPT points, so a guard applied only
    in Python lets a bounded page scroll every point of the OTHER roots to
    fill itself — the collection-scale request this surface exists to
    avoid. The guard must reach the backend filter."""
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    # One sparse root among many points of another root.
    for pid in range(1, 51):
        store.upsert(
            pid,
            [0.1, 0.2, 0.3],
            {"source": "multi.md", "index_root": "/bulk"},
            tenant="alpha",
        )
    store.upsert(
        99, [0.1, 0.2, 0.3], {"source": "multi.md", "index_root": "/sparse"}, tenant="alpha"
    )
    scanned = 0
    real_scroll = store.scroll

    def _counting(*args, **kwargs):
        nonlocal scanned
        for hit in real_scroll(*args, **kwargs):
            scanned += 1
            yield hit

    monkeypatch.setattr(store, "scroll", _counting)
    body = client.get(
        "/memories",
        params={"source": "multi.md", "index_root": "/sparse", "limit": 5},
        headers={"X-API-Key": keys["read"]},
    ).json()
    assert [row["id"] for row in body["items"]] == ["99"]
    # The backend handed us only the guard's own points — not all 51.
    assert scanned == 1, scanned


class _LegacyStore:
    """A custom store on the historical `scroll(filters=, tenant=)`
    signature — no hide_invalidated, no start_after, no index_root_guard."""

    def __init__(self, points):
        self.points = points  # [(id, payload)]
        self.calls = 0

    def scroll(self, batch_size=256, filters=None, with_vectors=False, *, tenant=None):
        self.calls += 1
        for pid, payload in self.points:
            if filters and payload.get("source") != filters.get("source"):
                continue
            yield type("Hit", (), {"id": pid, "payload": dict(payload)})()


def test_legacy_store_gets_the_semantics_not_just_the_call(monkeypatch, tmp_path):
    """R4 (codex P2): a store on the older scroll signature used to raise
    TypeError. Retrying with the keyword merely DROPPED would be worse than
    the crash — dropping hide_invalidated resurrects the non-terminating
    retraction, and dropping start_after makes every page return the same
    points forever. Each is emulated here with identical semantics."""
    points = [
        (1, {"source": "a.md"}),
        (2, {"source": "a.md", "invalidated_at": "2020-01-01T00:00:00+00:00"}),
        (3, {"source": "a.md", "index_root": "/other"}),
        (4, {"source": "b.md"}),
    ]
    store = _LegacyStore(points)
    # hide_invalidated emulated: the stale point is excluded, so a batched
    # retraction makes progress instead of re-selecting it forever.
    ids, _p, more = find_source_points(store, "a.md", skip_invalidated=True)
    assert ids == [1, 3] and more is False
    # start_after emulated: resume after id 1, do not restart the page.
    ids, _p, _more = find_source_points(store, "a.md", start_after=1)
    assert ids == [2, 3]
    # index_root guard still applies (in Python — it is an optimization).
    ids, _p, _more = find_source_points(store, "a.md", index_root="/other")
    assert ids == [1, 2, 3]  # untagged points are not protected by a guard
    ids, _p, _more = find_source_points(store, "a.md", index_root="/nowhere")
    assert ids == [1, 2]


def test_legacy_store_refuses_a_vanished_cursor_instead_of_lying(monkeypatch, tmp_path):
    """Emulated resume cannot tell "the tail is empty" from "the cursor is
    gone", and an empty page would report the listing COMPLETE while points
    remain — the exact lie this surface exists to prevent."""
    store = _LegacyStore([(1, {"source": "a.md"}), (2, {"source": "a.md"})])
    with pytest.raises(ValueError, match="no longer present"):
        find_source_points(store, "a.md", start_after=999)
