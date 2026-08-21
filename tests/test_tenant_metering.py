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
    assert (
        rec.counter_value(
            "mnemostack.tenant.requests",
            labels={"tenant": "alpha", "endpoint": "POST /memories"},
        )
        == base + 1
    )
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
    chunks = rec.counter_value("mnemostack.tenant.embedded_chunks", labels={"tenant": "alpha"})
    chars = rec.counter_value("mnemostack.tenant.embedded_chars", labels={"tenant": "alpha"})
    assert chunks == 1 and chars == len(text)
    # Duplicate: no embedding paid, meters unchanged.
    client.post("/memories", json={"items": [{"text": text, "source": "s"}]}, headers=hdr)
    assert (
        rec.counter_value("mnemostack.tenant.embedded_chunks", labels={"tenant": "alpha"}) == chunks
    )
    assert (
        rec.counter_value("mnemostack.tenant.embedded_chars", labels={"tenant": "alpha"}) == chars
    )


def test_reactivation_is_not_billed_as_embedding(monkeypatch, tmp_path):
    """R1 (agent P1 + codex): re-remember of an invalidated point reuses
    the stored vector — status 'stored' but ZERO provider calls; the cost
    meters must not move (attribution follows embed_attempted, not status)."""
    app, _store, emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    rec = _rec()
    hdr = {"X-API-Key": keys["write"]}
    text = "reactivated fact"
    r = client.post("/memories", json={"items": [{"text": text, "source": "s"}]}, headers=hdr)
    pid = r.json()["results"][0]["id"]
    chunks = rec.counter_value("mnemostack.tenant.embedded_chunks", labels={"tenant": "alpha"})
    chars = rec.counter_value("mnemostack.tenant.embedded_chars", labels={"tenant": "alpha"})
    embeds_before = len(emb.embedded)
    client.post("/invalidate", json={"ids": [pid]}, headers=hdr)
    r = client.post("/memories", json={"items": [{"text": text, "source": "s"}]}, headers=hdr)
    assert r.json()["results"][0]["status"] == "stored"  # reactivated
    assert len(emb.embedded) == embeds_before  # zero provider calls...
    assert (
        rec.counter_value("mnemostack.tenant.embedded_chunks", labels={"tenant": "alpha"}) == chunks
    )  # ...and zero billed cost
    assert (
        rec.counter_value("mnemostack.tenant.embedded_chars", labels={"tenant": "alpha"}) == chars
    )


def test_rate_limited_requests_still_count_as_requests(monkeypatch, tmp_path):
    """R1 (codex): `requests` is documented as EVERY authenticated request
    — the counter increments before the limiter, so the 429 ratio is
    rate_limited / requests."""
    from mnemostack.quotas import FileQuotaStore

    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path, quotas={"alpha": 1000})
    # Tighten the rate AFTER boot but before first resolve (config cache
    # is lazy): 1 request per 2s, burst 1 → the second request 429s.
    FileQuotaStore(tmp_path / "quotas.json").set("alpha", max_rps=0.5)
    client = TestClient(app)
    rec = _rec()
    hdr = {"X-API-Key": keys["read"]}
    before = sum(
        v for k, v in rec.snapshot_counters().items() if k[0] == "mnemostack.tenant.requests"
    )
    ok = client.post("/recall", json={"query": "q"}, headers=hdr)
    limited = client.post("/recall", json={"query": "q"}, headers=hdr)
    assert limited.status_code == 429, (ok.status_code, limited.status_code)
    after = sum(
        v for k, v in rec.snapshot_counters().items() if k[0] == "mnemostack.tenant.requests"
    )
    assert after - before == 2  # the 429'd request is still a request
    assert rec.counter_value("mnemostack.tenant.rate_limited", labels={"tenant": "alpha"}) == 1


def test_embedding_spend_counted_even_when_ingest_fails(monkeypatch, tmp_path):
    """R2 (codex): the provider round trip is paid even when a later
    upsert/space-check fails the request — meters emit at SUBMISSION, so a
    5xx response still attributes the spend."""
    import mnemostack.ingest as ingest_mod

    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    rec = _rec()

    def _boom(self, items):
        raise RuntimeError("upsert exploded after embedding")

    monkeypatch.setattr(ingest_mod.Ingestor, "ingest", _boom)
    text = "spent but not stored"
    r = client.post(
        "/memories",
        json={"items": [{"text": text, "source": "s"}]},
        headers={"X-API-Key": keys["write"]},
    )
    assert r.status_code == 500
    assert rec.counter_value("mnemostack.tenant.embedded_chunks", labels={"tenant": "alpha"}) == 1
    assert rec.counter_value("mnemostack.tenant.embedded_chars", labels={"tenant": "alpha"}) == len(
        text
    )


def test_space_guard_rejection_is_not_billed(monkeypatch, tmp_path):
    """R3 (both reviewers): the space guard aborts BEFORE any provider
    call (503 misconfig) — billing it would inflate the meters on every
    retry for the whole incident."""
    import mnemostack.ingest as ingest_mod
    from mnemostack.embeddings.roles import EmbeddingSpaceError

    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    rec = _rec()

    def _guard_reject(self, items):
        raise EmbeddingSpaceError("collection stamped with a different space")

    monkeypatch.setattr(ingest_mod.Ingestor, "ingest", _guard_reject)
    r = client.post(
        "/memories",
        json={"items": [{"text": "never embedded", "source": "s"}]},
        headers={"X-API-Key": keys["write"]},
    )
    assert r.status_code == 503
    assert rec.counter_value("mnemostack.tenant.embedded_chunks", labels={"tenant": "alpha"}) == 0
    assert rec.counter_value("mnemostack.tenant.embedded_chars", labels={"tenant": "alpha"}) == 0


def test_embed_attempted_field_attribution(monkeypatch, tmp_path):
    """The additive RemoteMemoryResult.embed_attempted field carries the
    same per-item attribution for API consumers: True only for items the
    provider actually saw."""
    from test_remote_ingest import _CountingEmbedding, _mem_store

    from mnemostack.ingest import IngestItem, ingest_remote_items

    emb, store = _CountingEmbedding(), _mem_store("attr")
    (fresh,) = ingest_remote_items(emb, store, [IngestItem(text="f", source="s")], tenant="a")
    assert fresh.embed_attempted is True
    (dup,) = ingest_remote_items(emb, store, [IngestItem(text="f", source="s")], tenant="a")
    assert dup.status == "duplicate" and dup.embed_attempted is False
    store.invalidate([fresh.id], tenant="a")
    (react,) = ingest_remote_items(emb, store, [IngestItem(text="f", source="s")], tenant="a")
    assert react.status == "stored" and react.embed_attempted is False


def test_tenant_quota_rejection_is_counted(monkeypatch, tmp_path):
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path, quotas={"alpha": 1})
    client = TestClient(app)
    rec = _rec()
    hdr = {"X-API-Key": keys["write"]}
    client.post("/memories", json={"items": [{"text": "one", "source": "s"}]}, headers=hdr)
    r = client.post("/memories", json={"items": [{"text": "two", "source": "s"}]}, headers=hdr)
    assert r.status_code == 507
    assert rec.counter_value("mnemostack.tenant.quota_rejected", labels={"tenant": "alpha"}) == 1


def test_unscoped_mode_emits_no_tenant_metrics(monkeypatch, tmp_path):
    app, _store, _emb, _keys = _ingest_app(monkeypatch, tmp_path, auth=False)
    client = TestClient(app)
    rec = _rec()
    client.post("/memories", json={"items": [{"text": "solo", "source": "s"}]})
    assert not any(key[0].startswith("mnemostack.tenant.") for key in rec.snapshot_counters())


def test_http_key_revocation_is_immediate(monkeypatch, tmp_path):
    """#112 design pin for the HTTP surface: verify() re-reads the store
    per request, so a revoked key dies on the very next call."""
    from mnemostack.auth import FileKeyStore

    app, _store, _emb, _keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    ks = FileKeyStore(tmp_path / "keys.json")
    kid, key = ks.issue("gamma", ["write"])
    hdr = {"X-API-Key": key}
    ok = client.post("/memories", json={"items": [{"text": "alive", "source": "s"}]}, headers=hdr)
    assert ok.status_code == 200
    assert ks.revoke(kid) is True
    dead = client.post("/memories", json={"items": [{"text": "dead", "source": "s"}]}, headers=hdr)
    assert dead.status_code == 401  # no restart, no session grace
