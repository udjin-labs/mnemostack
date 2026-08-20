"""Per-tenant metering (/metrics) + key-revocation immediacy pins."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from test_remote_ingest import _ingest_app

from mnemostack.observability.recorder import (
    InMemoryRecorder,
    get_recorder,
)


def _rec() -> InMemoryRecorder:
    rec = get_recorder()
    assert isinstance(rec, InMemoryRecorder)  # build_app installs it
    return rec


def test_tenant_requests_counter_labels(monkeypatch, tmp_path):
    """Every authenticated call increments requests{tenant, endpoint} —
    endpoint is the route TEMPLATE, so cardinality stays bounded."""
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    rec = _rec()
    base = rec.counter_value(
        "mnemostack.tenant.requests",
        labels={"tenant": "alpha", "endpoint": "POST /memories"},
    )
    client.post(
        "/memories",
        json={"items": [{"text": "metered fact", "source": "s"}]},
        headers={"X-API-Key": keys["write"]},
    )
    assert rec.counter_value(
        "mnemostack.tenant.requests",
        labels={"tenant": "alpha", "endpoint": "POST /memories"},
    ) == base + 1
    # A rejected (401/403) call must NOT count as a tenant request.
    unauth = rec.snapshot_counters()
    client.post("/memories", json={"items": [{"text": "x", "source": "s"}]})
    client.post(
        "/memories",
        json={"items": [{"text": "x", "source": "s"}]},
        headers={"X-API-Key": keys["read"]},
    )
    assert rec.snapshot_counters() == unauth


def test_tenant_embedding_cost_attribution(monkeypatch, tmp_path):
    """embedded_chunks/chars count only items that paid a provider call —
    a duplicate re-send costs zero and must not inflate the meter."""
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    rec = _rec()
    hdr = {"X-API-Key": keys["write"]}
    text = "the metered sky is green"
    client.post("/memories", json={"items": [{"text": text, "source": "s"}]}, headers=hdr)
    chunks = rec.counter_value(
        "mnemostack.tenant.embedded_chunks", labels={"tenant": "alpha"}
    )
    chars = rec.counter_value(
        "mnemostack.tenant.embedded_chars", labels={"tenant": "alpha"}
    )
    assert chunks == 1 and chars == len(text)
    # Duplicate: no embedding paid, meters unchanged.
    client.post("/memories", json={"items": [{"text": text, "source": "s"}]}, headers=hdr)
    assert rec.counter_value(
        "mnemostack.tenant.embedded_chunks", labels={"tenant": "alpha"}
    ) == chunks
    assert rec.counter_value(
        "mnemostack.tenant.embedded_chars", labels={"tenant": "alpha"}
    ) == chars


def test_tenant_quota_rejection_is_counted(monkeypatch, tmp_path):
    app, _store, _emb, keys = _ingest_app(
        monkeypatch, tmp_path, quotas={"alpha": 1}
    )
    client = TestClient(app)
    rec = _rec()
    hdr = {"X-API-Key": keys["write"]}
    client.post("/memories", json={"items": [{"text": "one", "source": "s"}]}, headers=hdr)
    r = client.post(
        "/memories", json={"items": [{"text": "two", "source": "s"}]}, headers=hdr
    )
    assert r.status_code == 507
    assert rec.counter_value(
        "mnemostack.tenant.quota_rejected", labels={"tenant": "alpha"}
    ) == 1


def test_unscoped_mode_emits_no_tenant_metrics(monkeypatch, tmp_path):
    app, _store, _emb, _keys = _ingest_app(monkeypatch, tmp_path, auth=False)
    client = TestClient(app)
    rec = _rec()
    client.post("/memories", json={"items": [{"text": "solo", "source": "s"}]})
    assert not any(
        key[0].startswith("mnemostack.tenant.")
        for key in rec.snapshot_counters()
    )


def test_http_key_revocation_is_immediate(monkeypatch, tmp_path):
    """#112 design pin for the HTTP surface: verify() re-reads the store
    per request, so a revoked key dies on the very next call."""
    from mnemostack.auth import FileKeyStore

    app, _store, _emb, _keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    ks = FileKeyStore(tmp_path / "keys.json")
    kid, key = ks.issue("gamma", ["write"])
    hdr = {"X-API-Key": key}
    ok = client.post(
        "/memories", json={"items": [{"text": "alive", "source": "s"}]}, headers=hdr
    )
    assert ok.status_code == 200
    assert ks.revoke(kid) is True
    dead = client.post(
        "/memories", json={"items": [{"text": "dead", "source": "s"}]}, headers=hdr
    )
    assert dead.status_code == 401  # no restart, no session grace
