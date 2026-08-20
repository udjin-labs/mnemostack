"""Remote lifecycle surface: POST /invalidate + DELETE /memories.

The GDPR/right-to-erasure counterpart of the remote write surface: the
non-destructive retraction (HTTP twin of mnemostack_invalidate) and the
hard delete, both tenant-guarded server-side.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from test_remote_ingest import _CountingEmbedding, _ingest_app, _mem_store

from mnemostack.ingest import (
    REMOTE_MAX_IDS,
    IngestItem,
    coerce_point_ids,
    ingest_remote_items,
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
    assert validate_remote_ids(["uuid-like", 7, "123"]) is None


def test_validate_remote_invalidate_contract():
    ok = ["some-id"]
    assert validate_remote_invalidate(ok, None, None) is None
    assert validate_remote_invalidate(ok, "2026-01-01T00:00:00Z", "2026-02-01") is None
    assert "invalidated_at" in validate_remote_invalidate(ok, "not-a-date", None)
    assert "valid_until" in validate_remote_invalidate(ok, None, "not-a-date")
    assert "valid_until" in validate_remote_invalidate(ok, None, "x" * 65)
    # index_root guard: blank would silently skip every id.
    assert "index_root" in validate_remote_invalidate(ok, None, None, "")
    assert "index_root" in validate_remote_invalidate(ok, None, None, "   ")
    assert validate_remote_invalidate(ok, None, None, "/srv/corpus") is None


def test_coerce_point_ids():
    assert coerce_point_ids(["123", "a-b", 7, "007"]) == [123, "a-b", 7, 7]


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
    assert r.json() == {"requested": 1, "invalidated": 1}
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
    assert r.json() == {"requested": 1, "invalidated": 0}
    point = store.client.retrieve(store.collection, ids=[pid], with_payload=True)[0]
    assert "invalidated_at" not in (point.payload or {})


def test_invalidate_rejects_bad_input(monkeypatch, tmp_path):
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    hdr = {"X-API-Key": keys["write"]}
    r = client.post(
        "/invalidate", json={"ids": ["x"], "valid_until": "garbage"}, headers=hdr
    )
    assert r.status_code == 400 and "valid_until" in r.json()["detail"]
    r = client.post(
        "/invalidate", json={"ids": ["x"], "invalidated_at": "garbage"}, headers=hdr
    )
    assert r.status_code == 400 and "invalidated_at" in r.json()["detail"]
    r = client.post(
        "/invalidate", json={"ids": ["x"], "index_root": "  "}, headers=hdr
    )
    assert r.status_code == 400 and "index_root" in r.json()["detail"]
    r = client.post(
        "/invalidate",
        json={"ids": ["x"] * (REMOTE_MAX_IDS + 1)},
        headers=hdr,
    )
    assert r.status_code in (400, 422)  # pydantic cap or shared validator
    r = client.post("/invalidate", json={"ids": []}, headers=hdr)
    assert r.status_code in (400, 422)


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
    assert r.json() == {"requested": 1, "deleted": 1}
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
    assert r.json() == {"requested": 1, "deleted": 0}  # no oracle, no effect
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


# ------------------------------------------------------------ store-level unit


def test_ids_validation_precedes_store_roundtrip():
    """A 400-class problem must not cost a store call (validated up front)."""
    emb, store = _CountingEmbedding(), _mem_store("val")
    (res,) = ingest_remote_items(
        emb, store, [IngestItem(text="x", source="s")], tenant="a"
    )
    calls: list = []
    orig = store.client.retrieve

    def _counting(*a, **kw):
        calls.append(a)
        return orig(*a, **kw)

    store.client.retrieve = _counting  # type: ignore[method-assign]
    assert validate_remote_ids([""]) is not None  # rejected without touching store
    assert calls == []
    assert res.status == "stored"
