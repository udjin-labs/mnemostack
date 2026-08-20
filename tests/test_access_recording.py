"""Server-side access accounting (`serve --record-access`).

The freshness stage has always READ `access_count`/`last_accessed` —
each recorded access extends a memory's effective half-life — but nothing
in the stack wrote them, so reinforcement only worked if every client
stamped the payloads itself. The service sees every retrieval; this
records the access there.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from test_remote_ingest import _ingest_app

from mnemostack.access import (
    ACCESS_COUNT_KEY,
    LAST_ACCESSED_KEY,
    MAX_STORED_ACCESS_COUNT,
    record_access,
)


class _Hit:
    """Enough of a recall result for both the recorder and the response
    serializer (the endpoint renders text/score/sources)."""

    def __init__(self, pid, payload=None, text="a memory", score=0.9):
        self.id = pid
        self.payload = payload or {}
        self.text = text
        self.score = score
        self.sources = ["vector"]


def _payload(store, pid):
    return store.client.retrieve(store.collection, ids=[pid], with_payload=True)[0].payload


def _seed(store, pid, payload=None, tenant="alpha"):
    store.upsert(pid, [0.1, 0.2, 0.3], {"source": "a.md", **(payload or {})}, tenant=tenant)
    return pid


def test_first_access_starts_the_count(monkeypatch, tmp_path):
    _app, store, _emb, _keys = _ingest_app(monkeypatch, tmp_path)
    _seed(store, 1)
    assert record_access(store, [_Hit(1)], tenant="alpha") == 1
    payload = _payload(store, 1)
    assert payload[ACCESS_COUNT_KEY] == 1
    assert payload[LAST_ACCESSED_KEY]
    assert payload["source"] == "a.md"  # a merge, not a payload replacement


def test_repeated_access_accumulates(monkeypatch, tmp_path):
    _app, store, _emb, _keys = _ingest_app(monkeypatch, tmp_path)
    _seed(store, 1, {ACCESS_COUNT_KEY: 4})
    record_access(store, [_Hit(1, _payload(store, 1))], tenant="alpha")
    assert _payload(store, 1)[ACCESS_COUNT_KEY] == 5


def test_a_junk_counter_is_treated_as_zero(monkeypatch, tmp_path):
    """The key is client-writable on deployments that stamped it
    themselves, so it can hold anything. None of it may raise after a
    successful recall."""
    _app, store, _emb, _keys = _ingest_app(monkeypatch, tmp_path)
    for pid, junk in ((1, "many"), (2, 3.7), (3, True), (4, -5), (5, None)):
        _seed(store, pid, {ACCESS_COUNT_KEY: junk})
        record_access(store, [_Hit(pid, _payload(store, pid))], tenant="alpha")
        assert _payload(store, pid)[ACCESS_COUNT_KEY] == 1, junk


def test_the_stored_counter_is_bounded(monkeypatch, tmp_path):
    _app, store, _emb, _keys = _ingest_app(monkeypatch, tmp_path)
    _seed(store, 1, {ACCESS_COUNT_KEY: MAX_STORED_ACCESS_COUNT})
    record_access(store, [_Hit(1, _payload(store, 1))], tenant="alpha")
    assert _payload(store, 1)[ACCESS_COUNT_KEY] == MAX_STORED_ACCESS_COUNT


def test_one_recall_is_one_increment_per_point(monkeypatch, tmp_path):
    """A point can arrive from several retrievers in one fused result."""
    _app, store, _emb, _keys = _ingest_app(monkeypatch, tmp_path)
    _seed(store, 1)
    assert record_access(store, [_Hit(1), _Hit("1"), _Hit(1)], tenant="alpha") == 1
    assert _payload(store, 1)[ACCESS_COUNT_KEY] == 1


def test_a_non_point_id_does_not_cost_the_others_their_bookkeeping(monkeypatch, tmp_path):
    """A knowledge-graph hit is NAMED, not addressed by point id. Handing
    that name to the store would fail the whole batch — so it is filtered
    before the write, not discovered by the backend."""
    _app, store, _emb, _keys = _ingest_app(monkeypatch, tmp_path)
    _seed(store, 1)
    _seed(store, 2)
    # Assert on what is HANDED to the store, not on the outcome: the
    # in-memory client tolerates a nonsense id (returns nothing), while a
    # real server 400s the whole batch — so an outcome-only assertion would
    # pass with the filter removed. (The same in-memory-vs-server gap that
    # hid a batch-update defect before.)
    handed: list[list] = []
    orig = store.apply_payload_patches

    def _spy(patches, **kwargs):
        handed.append([p.id for p in patches])
        return orig(patches, **kwargs)

    store.apply_payload_patches = _spy  # type: ignore[method-assign]
    hits = [_Hit(1), _Hit("deploy-window"), _Hit(2), _Hit(None), _Hit(True)]
    assert record_access(store, hits, tenant="alpha") == 2
    assert handed == [[1, 2]]
    assert _payload(store, 1)[ACCESS_COUNT_KEY] == 1
    assert _payload(store, 2)[ACCESS_COUNT_KEY] == 1


def test_another_tenants_point_is_not_stamped(monkeypatch, tmp_path):
    _app, store, _emb, _keys = _ingest_app(monkeypatch, tmp_path)
    _seed(store, 1, tenant="beta")
    assert record_access(store, [_Hit(1)], tenant="alpha") == 0
    assert ACCESS_COUNT_KEY not in _payload(store, 1)


def test_a_failing_store_never_reaches_the_caller(monkeypatch, tmp_path):
    """Bookkeeping must not fail a recall whose results the caller already
    has: the only observable difference is the counter."""
    _app, store, _emb, _keys = _ingest_app(monkeypatch, tmp_path)
    _seed(store, 1)

    def _boom(*_a, **_k):
        raise RuntimeError("qdrant is down")

    monkeypatch.setattr(store, "apply_payload_patches", _boom)
    assert record_access(store, [_Hit(1)], tenant="alpha") == 0


def test_recall_records_only_when_the_operator_asked(monkeypatch, tmp_path):
    """Off by default — it turns reads into writes."""
    import mnemostack.server as srv

    seen: list[tuple] = []
    monkeypatch.setattr(
        srv, "record_access", lambda *a, **k: seen.append((a, k)) or 0
    )
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    _seed(store, 1)
    monkeypatch.setattr(
        srv, "recall_flow", lambda *_a, **_k: [_Hit(1)]
    )
    client = TestClient(app)
    r = client.post(
        "/recall", json={"query": "anything", "limit": 5}, headers={"X-API-Key": keys["read"]}
    )
    assert r.status_code == 200
    assert seen == []


def test_recall_records_when_enabled(monkeypatch, tmp_path):
    import mnemostack.server as srv

    app, store, _emb, keys = _ingest_app(
        monkeypatch, tmp_path, cfg_extra={"record_access": True}
    )
    _seed(store, 1)
    monkeypatch.setattr(srv, "recall_flow", lambda *_a, **_k: [_Hit(1)])
    client = TestClient(app)
    r = client.post(
        "/recall", json={"query": "anything", "limit": 5}, headers={"X-API-Key": keys["read"]}
    )
    assert r.status_code == 200
    # The key's tenant, not one the client asserted.
    assert _payload(store, 1)[ACCESS_COUNT_KEY] == 1
