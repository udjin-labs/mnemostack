"""Remote lifecycle surface: POST /invalidate + DELETE /memories.

The GDPR/right-to-erasure counterpart of the remote write surface: the
non-destructive retraction (HTTP twin of mnemostack_invalidate) and the
hard delete, both tenant-guarded server-side.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from test_remote_ingest import _ingest_app

from mnemostack.ingest import (
    REMOTE_MAX_IDS,
    coerce_point_ids,
    validate_remote_ids,
    validate_remote_invalidate,
)

# ------------------------------------------------------------ validators


def test_validate_remote_ids_contract():
    assert validate_remote_ids([]) is not None
    assert validate_remote_ids("not-a-list") is not None
    assert "at most" in validate_remote_ids(list(range(REMOTE_MAX_IDS + 1)))
    assert validate_remote_ids([True]) is not None  # bool is not point id 1
    assert validate_remote_ids([-1]) is not None
    assert validate_remote_ids([2**64]) is not None
    assert validate_remote_ids([""]) is not None
    assert validate_remote_ids(["  "]) is not None
    assert validate_remote_ids(["x" * 129]) is not None
    assert validate_remote_ids(["a\ud800"]) is not None  # lone surrogate
    assert validate_remote_ids([{"id": 1}]) is not None
    # Qdrant id domain: UUIDs or unsigned 64-bit ints only — arbitrary
    # strings would surface as opaque backend errors instead of a 400.
    assert "UUID" in validate_remote_ids(["not-a-uuid"])
    assert validate_remote_ids(["9" * 25]) is not None  # digit string > u64
    # Past CPython's int-from-str digit limit (~4300) int() RAISES — the
    # validator must return a message, never propagate ValueError (500).
    assert "64-bit" in validate_remote_ids(["9" * 5000])
    assert coerce_point_ids(["9" * 5000]) == ["9" * 5000]  # no int() crash
    # Exact len==20 boundary still coerces (u64 max is 20 digits).
    assert coerce_point_ids([str(2**64 - 1)]) == [2**64 - 1]
    # Leading zeros don't add range: a 21-char zero-padded spelling of a
    # valid id is that id, not a length violation ("007" is point 7).
    assert validate_remote_ids(["0" * 20 + "1"]) is None
    assert coerce_point_ids(["0" * 20 + "1"]) == [1]
    assert coerce_point_ids(["0" * 10000]) == [0]  # linear, resolves to 0
    # Standalone safety: an unvalidated over-u64 digit string passes
    # through as a string, never as an out-of-range int.
    assert coerce_point_ids(["0" + "9" * 20]) == ["0" + "9" * 20]


def test_lifecycle_on_fresh_deployment_is_empty_not_500(monkeypatch, tmp_path):
    """Codex-R5: before the first write the collection doesn't exist (lazy
    bootstrap) — lifecycle calls must report zero effect, not 500."""
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    hdr = {"X-API-Key": keys["write"]}
    store.client.delete_collection(store.collection)  # pre-bootstrap state
    pid = "d9428888-122b-11e1-b85c-61cd3cbb3210"
    r = client.post("/invalidate", json={"ids": [pid]}, headers=hdr)
    assert r.status_code == 200
    assert r.json() == {"requested": 1, "invalidated": 0, "complete": True}
    r = client.request("DELETE", "/memories", json={"ids": [pid]}, headers=hdr)
    assert r.status_code == 200
    assert r.json() == {"requested": 1, "deleted": 0, "complete": True}
    # UUIDs canonicalize to lowercase — the store compares ids as
    # case-sensitive strings, so an uppercase spelling would no-op.
    assert coerce_point_ids(["D9428888-122B-11E1-B85C-61CD3CBB3210"]) == [
        "d9428888-122b-11e1-b85c-61cd3cbb3210"
    ]


def test_invalidate_accepts_uppercase_uuid_spelling(monkeypatch, tmp_path):
    """Codex-R3: uppercase UUID of an existing lowercase id must hit the
    point, not silently report invalidated=0."""
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    pid = _stored_id(client, keys["write"])
    r = client.post(
        "/invalidate",
        json={"ids": [pid.upper()]},
        headers={"X-API-Key": keys["write"]},
    )
    assert r.status_code == 200 and r.json()["invalidated"] == 1
    point = store.client.retrieve(store.collection, ids=[pid], with_payload=True)[0]
    assert point.payload.get("invalidated_at")
    assert validate_remote_ids(["²"]) is not None  # isdigit() but int() crashes
    assert validate_remote_ids(["٧"]) is not None  # non-ASCII decimal
    assert validate_remote_ids(
        ["c7751834-6a7d-0516-5d84-032e6e92d50f", 7, "123", str(2**64 - 1)]
    ) is None


def test_validate_remote_invalidate_contract():
    ok = ["7"]
    assert validate_remote_invalidate(ok, None, None) is None
    assert validate_remote_invalidate(ok, "2026-01-01T00:00:00Z", "2026-02-01") is None
    assert "invalidated_at" in validate_remote_invalidate(ok, "not-a-date", None)
    assert "valid_until" in validate_remote_invalidate(ok, None, "not-a-date")
    assert "valid_until" in validate_remote_invalidate(ok, None, "x" * 65)
    # index_root guard: blank would silently skip every id.
    assert "index_root" in validate_remote_invalidate(ok, None, None, "")
    assert "index_root" in validate_remote_invalidate(ok, None, None, "   ")
    # The 4096 cap lives in the SHARED validator — the MCP surface has no
    # pydantic model to enforce it.
    assert "index_root" in validate_remote_invalidate(ok, None, None, "x" * 4097)
    assert validate_remote_invalidate(ok, None, None, "/srv/corpus") is None


def test_coerce_point_ids():
    assert coerce_point_ids(["123", "a-b", 7, "007"]) == [123, "a-b", 7, 7]
    # '²'.isdigit() is True but int('²') raises — must pass through, not
    # crash; non-ASCII decimals ('٧') must not silently become numeric ids.
    assert coerce_point_ids(["²", "٧"]) == ["²", "٧"]


# ------------------------------------------------------------ POST /invalidate


def _stored_id(client, key, text="the sky is green", source="chat"):
    r = client.post(
        "/memories",
        json={"items": [{"text": text, "source": source}]},
        headers={"X-API-Key": key},
    )
    assert r.status_code == 200, r.text
    return r.json()["results"][0]["id"]


def test_invalidate_requires_write_scope(monkeypatch, tmp_path):
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    r = client.post("/invalidate", json={"ids": ["x"]})
    assert r.status_code == 401
    r = client.post(
        "/invalidate", json={"ids": ["x"]}, headers={"X-API-Key": keys["read"]}
    )
    assert r.status_code == 403


def test_invalidate_marks_own_point_stale(monkeypatch, tmp_path):
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    pid = _stored_id(client, keys["write"])
    r = client.post(
        "/invalidate",
        json={"ids": [pid], "valid_until": "2026-01-01"},
        headers={"X-API-Key": keys["write"]},
    )
    assert r.status_code == 200
    assert r.json() == {"requested": 1, "invalidated": 1, "complete": True}
    point = store.client.retrieve(store.collection, ids=[pid], with_payload=True)[0]
    assert point.payload.get("invalidated_at")
    assert point.payload.get("valid_until") == "2026-01-01"


def test_invalidate_never_touches_foreign_tenant(monkeypatch, tmp_path):
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    pid = _stored_id(client, keys["write"])  # owned by alpha
    r = client.post(
        "/invalidate", json={"ids": [pid]}, headers={"X-API-Key": keys["beta_write"]}
    )
    assert r.status_code == 200
    # Skipped indistinguishably from a missing id — no existence oracle.
    assert r.json() == {"requested": 1, "invalidated": 0, "complete": True}
    point = store.client.retrieve(store.collection, ids=[pid], with_payload=True)[0]
    assert "invalidated_at" not in (point.payload or {})


def test_invalidate_rejects_bad_input(monkeypatch, tmp_path):
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    hdr = {"X-API-Key": keys["write"]}
    r = client.post(
        "/invalidate", json={"ids": ["7"], "valid_until": "garbage"}, headers=hdr
    )
    assert r.status_code == 400 and "valid_until" in r.json()["detail"]
    r = client.post(
        "/invalidate", json={"ids": ["7"], "invalidated_at": "garbage"}, headers=hdr
    )
    assert r.status_code == 400 and "invalidated_at" in r.json()["detail"]
    r = client.post(
        "/invalidate", json={"ids": ["7"], "index_root": "  "}, headers=hdr
    )
    assert r.status_code == 400 and "index_root" in r.json()["detail"]
    r = client.post("/invalidate", json={"ids": ["not-a-uuid"]}, headers=hdr)
    assert r.status_code == 400 and "UUID" in r.json()["detail"]
    r = client.post("/invalidate", json={"ids": ["9" * 25]}, headers=hdr)
    assert r.status_code == 400  # digit string past the u64 domain
    r = client.post("/invalidate", json={"ids": ["9" * 5000]}, headers=hdr)
    assert r.status_code == 400  # past int()'s digit limit — 400, not 500
    r = client.request(
        "DELETE", "/memories", json={"ids": ["9" * 5000]}, headers=hdr
    )
    assert r.status_code == 400
    r = client.post(
        "/invalidate",
        json={"ids": ["7"] * (REMOTE_MAX_IDS + 1)},
        headers=hdr,
    )
    assert r.status_code in (400, 422)  # pydantic cap or shared validator
    r = client.post("/invalidate", json={"ids": []}, headers=hdr)
    assert r.status_code in (400, 422)
    # JSON booleans must not coerce onto numeric point ids 1/0 (StrictInt).
    r = client.post("/invalidate", json={"ids": [True]}, headers=hdr)
    assert r.status_code == 422
    r = client.request("DELETE", "/memories", json={"ids": [False]}, headers=hdr)
    assert r.status_code == 422


def test_invalidate_respects_index_root_guard(monkeypatch, tmp_path):
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    pid = _stored_id(client, keys["write"])
    store.client.set_payload(
        collection_name=store.collection,
        payload={"index_root": "/srv/other"},
        points=[pid],
    )
    r = client.post(
        "/invalidate",
        json={"ids": [pid], "index_root": "/srv/mine"},
        headers={"X-API-Key": keys["write"]},
    )
    assert r.status_code == 200 and r.json()["invalidated"] == 0
    point = store.client.retrieve(store.collection, ids=[pid], with_payload=True)[0]
    assert "invalidated_at" not in (point.payload or {})


# ------------------------------------------------------------ DELETE /memories


def test_delete_requires_write_scope(monkeypatch, tmp_path):
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    r = client.request("DELETE", "/memories", json={"ids": ["x"]})
    assert r.status_code == 401
    r = client.request(
        "DELETE", "/memories", json={"ids": ["x"]}, headers={"X-API-Key": keys["read"]}
    )
    assert r.status_code == 403


def test_delete_removes_own_point(monkeypatch, tmp_path):
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    pid = _stored_id(client, keys["write"])
    r = client.request(
        "DELETE", "/memories", json={"ids": [pid]}, headers={"X-API-Key": keys["write"]}
    )
    assert r.status_code == 200
    assert r.json() == {"requested": 1, "deleted": 1, "complete": True}
    assert store.client.retrieve(store.collection, ids=[pid], with_payload=True) == []
    # Idempotent retry: already gone.
    r = client.request(
        "DELETE", "/memories", json={"ids": [pid]}, headers={"X-API-Key": keys["write"]}
    )
    assert r.status_code == 200 and r.json()["deleted"] == 0


def test_delete_never_touches_foreign_tenant(monkeypatch, tmp_path):
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    pid = _stored_id(client, keys["write"])  # owned by alpha
    r = client.request(
        "DELETE",
        "/memories",
        json={"ids": [pid]},
        headers={"X-API-Key": keys["beta_write"]},
    )
    assert r.status_code == 200
    assert r.json() == {"requested": 1, "deleted": 0, "complete": True}  # no oracle, no effect
    assert len(store.client.retrieve(store.collection, ids=[pid], with_payload=True)) == 1


def test_delete_then_rememeber_stores_fresh(monkeypatch, tmp_path):
    """Erase → re-send = a fresh store, not a duplicate of the erased copy."""
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    hdr = {"X-API-Key": keys["write"]}
    pid = _stored_id(client, keys["write"])
    client.request("DELETE", "/memories", json={"ids": [pid]}, headers=hdr)
    r = client.post(
        "/memories",
        json={"items": [{"text": "the sky is green", "source": "chat"}]},
        headers=hdr,
    )
    assert r.status_code == 200
    assert r.json()["results"][0]["status"] == "stored"


def test_delete_rejects_bad_input(monkeypatch, tmp_path):
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    hdr = {"X-API-Key": keys["write"]}
    r = client.request("DELETE", "/memories", json={"ids": [""]}, headers=hdr)
    assert r.status_code == 400
    r = client.request(
        "DELETE",
        "/memories",
        json={"ids": ["x"] * (REMOTE_MAX_IDS + 1)},
        headers=hdr,
    )
    assert r.status_code in (400, 422)


def test_lifecycle_unscoped_legacy_mode(monkeypatch, tmp_path):
    """Auth off = historical single-tenant service: no tenant guard."""
    app, store, _emb, _keys = _ingest_app(monkeypatch, tmp_path, auth=False)
    client = TestClient(app)
    r = client.post(
        "/memories", json={"items": [{"text": "legacy fact", "source": "s"}]}
    )
    pid = r.json()["results"][0]["id"]
    r = client.post("/invalidate", json={"ids": [pid]})
    assert r.status_code == 200 and r.json()["invalidated"] == 1
    r = client.request("DELETE", "/memories", json={"ids": [pid]})
    assert r.status_code == 200 and r.json()["deleted"] == 1
    assert store.client.retrieve(store.collection, ids=[pid], with_payload=True) == []


def test_invalidated_memory_hidden_from_default_recall_semantics(monkeypatch, tmp_path):
    """The endpoint writes the exact marker the recall layer filters on."""
    from mnemostack.recall import is_current

    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    pid = _stored_id(client, keys["write"])
    client.post(
        "/invalidate", json={"ids": [pid]}, headers={"X-API-Key": keys["write"]}
    )
    point = store.client.retrieve(store.collection, ids=[pid], with_payload=True)[0]
    assert not is_current(point.payload)


# ------------------------------------------- review round-1 batch pins


def test_invalid_input_never_reaches_the_store(monkeypatch, tmp_path):
    """Agent-R1: a 400-class problem must not cost ANY store call — pin it
    at the endpoint level, not against the pure validator."""
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    hdr = {"X-API-Key": keys["write"]}

    def _boom(*a, **kw):  # pragma: no cover — must not be reached
        raise AssertionError("store must not be touched for invalid input")

    monkeypatch.setattr(store, "invalidate", _boom)
    monkeypatch.setattr(store, "delete_points", _boom)
    monkeypatch.setattr(store.client, "retrieve", _boom)
    r = client.post("/invalidate", json={"ids": ["not-a-uuid"]}, headers=hdr)
    assert r.status_code == 400
    r = client.request("DELETE", "/memories", json={"ids": [""]}, headers=hdr)
    assert r.status_code == 400


def test_duplicate_ids_count_once(monkeypatch, tmp_path):
    """Agent-R1 + codex-R1: [x, x] must not report 2 for one touched point
    — `invalidated`/`deleted` are documented as points actually affected."""
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    hdr = {"X-API-Key": keys["write"]}
    pid = _stored_id(client, keys["write"])
    r = client.post("/invalidate", json={"ids": [pid, pid]}, headers=hdr)
    assert r.json() == {"requested": 2, "invalidated": 1, "complete": True}
    r = client.request(
        "DELETE", "/memories", json={"ids": [pid, pid]}, headers=hdr
    )
    assert r.json() == {"requested": 2, "deleted": 1, "complete": True}


def test_unscoped_delete_counts_actual_removals(monkeypatch, tmp_path):
    """Codex-R1: without auth delete_points skips the existence check, so
    unknown ids / retries / duplicates would be reported as deleted."""
    app, store, _emb, _keys = _ingest_app(monkeypatch, tmp_path, auth=False)
    client = TestClient(app)
    r = client.post(
        "/memories", json={"items": [{"text": "legacy fact", "source": "s"}]}
    )
    pid = r.json()["results"][0]["id"]
    missing = "00000000-0000-0000-0000-000000000000"
    r = client.request("DELETE", "/memories", json={"ids": [pid, pid, missing]})
    assert r.status_code == 200
    assert r.json() == {"requested": 3, "deleted": 1, "complete": True}
    # Retry: everything already gone.
    r = client.request("DELETE", "/memories", json={"ids": [pid]})
    assert r.json()["deleted"] == 0


def test_delete_respects_index_root_guard(monkeypatch, tmp_path):
    """Agent-R1: the irreversible endpoint gets the same defense-in-depth
    scoping knob as /invalidate; untagged points pass (documented)."""
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    hdr = {"X-API-Key": keys["write"]}
    owned = _stored_id(client, keys["write"], text="mine", source="a")
    untagged = _stored_id(client, keys["write"], text="untagged", source="b")
    store.client.set_payload(
        collection_name=store.collection,
        payload={"index_root": "/srv/other"},
        points=[owned],
    )
    r = client.request(
        "DELETE",
        "/memories",
        json={"ids": [owned, untagged], "index_root": "/srv/mine"},
        headers=hdr,
    )
    assert r.status_code == 200
    # Foreign-root point survives; the untagged one is NOT protected.
    assert r.json() == {"requested": 2, "deleted": 1, "complete": True}
    assert len(store.client.retrieve(store.collection, ids=[owned], with_payload=True)) == 1
    assert store.client.retrieve(store.collection, ids=[untagged], with_payload=True) == []
    r = client.request(
        "DELETE", "/memories", json={"ids": [owned], "index_root": "  "}, headers=hdr
    )
    assert r.status_code == 400 and "index_root" in r.json()["detail"]
