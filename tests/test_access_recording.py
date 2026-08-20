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


def _answer_app(monkeypatch, tmp_path, generator):
    """The ingest app, but with an answer generator wired in.

    `_ingest_app` deliberately has no LLM, and `answer_gen` is resolved ONCE
    at build time — so /answer there is a 503 and a later patch cannot
    change that. Rather than patch around the helper, build on top of it and
    replace the generator before the app is constructed.
    """
    import mnemostack.server as srv

    monkeypatch.setattr(srv, "AnswerGenerator", lambda *_a, **_k: generator)
    app, store, emb, keys = _ingest_app(
        monkeypatch,
        tmp_path,
        cfg_extra={"record_access": True},
        llm=object(),  # any truthy LLM: AnswerGenerator is stubbed above
    )
    return app, store, emb, keys


def test_answer_does_not_record_when_generation_fails(monkeypatch, tmp_path):
    """R1 (codex P2): /answer's recall can succeed and its generation fail —
    the caller gets a 500 and no memories, so those points must not be
    counted as accessed. The claim was in the commit message before it was
    in the code."""
    import mnemostack.server as srv

    class _Boom:
        def generate(self, *_a, **_k):
            raise RuntimeError("llm exploded")

    app, store, _emb, keys = _answer_app(monkeypatch, tmp_path, _Boom())
    _seed(store, 1)
    monkeypatch.setattr(srv, "recall_flow", lambda *_a, **_k: [_Hit(1)])
    client = TestClient(app)
    r = client.post(
        "/answer", json={"query": "anything"}, headers={"X-API-Key": keys["read"]}
    )
    assert r.status_code == 500, r.text
    assert ACCESS_COUNT_KEY not in _payload(store, 1)


def test_answer_records_when_generation_succeeds(monkeypatch, tmp_path):
    """The mirror image: a delivered answer DID hand those memories over."""
    import mnemostack.server as srv

    class _Ok:
        def generate(self, *_a, **_k):
            return type(
                "Ans", (), {"text": "an answer", "confidence": 0.9, "sources": []}
            )()

    app, store, _emb, keys = _answer_app(monkeypatch, tmp_path, _Ok())
    _seed(store, 1)
    monkeypatch.setattr(srv, "recall_flow", lambda *_a, **_k: [_Hit(1)])
    client = TestClient(app)
    r = client.post(
        "/answer", json={"query": "anything"}, headers={"X-API-Key": keys["read"]}
    )
    assert r.status_code == 200, r.text
    assert _payload(store, 1)[ACCESS_COUNT_KEY] == 1


def test_a_stale_hit_payload_does_not_freeze_the_counter(monkeypatch, tmp_path):
    """R2 (codex P2): a lexical (BM25) hit carries the in-process corpus
    SNAPSHOT taken at startup, not current store state. Incrementing that
    would write 1 forever — the counter would never accumulate for a
    deployment whose recalls come from the lexical arm."""
    _app, store, _emb, _keys = _ingest_app(monkeypatch, tmp_path)
    _seed(store, 1, {ACCESS_COUNT_KEY: 7})
    # The hit still carries what the snapshot held when the server started.
    stale = _Hit(1, {ACCESS_COUNT_KEY: 0, "source": "a.md"})
    assert record_access(store, [stale], tenant="alpha") == 1
    assert _payload(store, 1)[ACCESS_COUNT_KEY] == 8


def test_a_stale_hit_payload_cannot_walk_the_counter_backwards(monkeypatch, tmp_path):
    """The same snapshot in a FUSED result: whichever arm's payload wins,
    the larger of stored-and-hit is the base, so a count never decreases."""
    _app, store, _emb, _keys = _ingest_app(monkeypatch, tmp_path)
    _seed(store, 1, {ACCESS_COUNT_KEY: 20})
    record_access(store, [_Hit(1, {ACCESS_COUNT_KEY: 2})], tenant="alpha")
    assert _payload(store, 1)[ACCESS_COUNT_KEY] == 21


def test_the_counter_read_costs_one_round_trip(monkeypatch, tmp_path):
    """Per-point reads would make an enabled deployment pay `limit` extra
    requests on every recall."""
    _app, store, _emb, _keys = _ingest_app(monkeypatch, tmp_path)
    for pid in range(1, 11):
        _seed(store, pid)
    calls: list[int] = []
    orig = store.retrieve_payload_fields

    def _counting(ids, keys, **kwargs):
        calls.append(len(list(ids)))
        return orig(ids, keys, **kwargs)

    store.retrieve_payload_fields = _counting  # type: ignore[method-assign]
    record_access(store, [_Hit(pid) for pid in range(1, 11)], tenant="alpha")
    assert calls == [10]


def test_a_store_without_the_batch_reader_still_records(monkeypatch, tmp_path):
    """Duck stores keep working — the hit's payload is the fallback."""
    _app, store, _emb, _keys = _ingest_app(monkeypatch, tmp_path)
    _seed(store, 1, {ACCESS_COUNT_KEY: 3})
    monkeypatch.delattr(type(store), "retrieve_payload_fields", raising=False)
    assert record_access(store, [_Hit(1, {ACCESS_COUNT_KEY: 3})], tenant="alpha") == 1
    assert _payload(store, 1)[ACCESS_COUNT_KEY] == 4


def test_a_failing_counter_read_stamps_the_time_but_not_the_count(monkeypatch, tmp_path):
    """R3 (codex P2): falling back to the hit's payload when the READ fails
    reintroduces the very defect the read was added to prevent — a stale
    lexical snapshot overwriting a stored 7 with 1. A failed read stamps
    the timestamp (the decay stage still gets its input) and leaves the
    counter exactly where it was."""
    _app, store, _emb, _keys = _ingest_app(monkeypatch, tmp_path)
    _seed(store, 1, {ACCESS_COUNT_KEY: 7})

    def _boom(*_a, **_k):
        raise RuntimeError("read failed")

    monkeypatch.setattr(store, "retrieve_payload_fields", _boom)
    stale = _Hit(1, {ACCESS_COUNT_KEY: 0})
    assert record_access(store, [stale], tenant="alpha") == 1
    payload = _payload(store, 1)
    assert payload[ACCESS_COUNT_KEY] == 7  # untouched, not decreased
    assert payload[LAST_ACCESSED_KEY]  # and the time IS recorded


def test_the_counter_read_is_tenant_scoped(monkeypatch, tmp_path):
    """A foreign point must not even reveal its counter to the read."""
    _app, store, _emb, _keys = _ingest_app(monkeypatch, tmp_path)
    _seed(store, 1, {ACCESS_COUNT_KEY: 9}, tenant="beta")
    assert store.retrieve_payload_fields([1], [ACCESS_COUNT_KEY], tenant="alpha") == {}
    assert store.retrieve_payload_fields([1], [ACCESS_COUNT_KEY], tenant="beta") == {
        "1": {ACCESS_COUNT_KEY: 9}
    }


def test_fail_open_covers_the_whole_body_not_just_the_write(monkeypatch, tmp_path):
    """R6 (review agent P3): the guard used to wrap only the store write,
    so the contract "record_access must never raise" held by construction
    rather than by structure — a future fallible step in the patch-building
    block would have broken it silently."""
    import mnemostack.access as acc

    _app, store, _emb, _keys = _ingest_app(monkeypatch, tmp_path)
    _seed(store, 1)

    def _boom(*_a, **_k):
        raise RuntimeError("a future step blew up")

    monkeypatch.setattr(acc, "_recordable_ids", _boom)
    assert record_access(store, [_Hit(1)], tenant="alpha") == 0
