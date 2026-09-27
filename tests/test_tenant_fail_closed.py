"""Tenantless recall over a multi-tenant collection fails closed (issue #193).

The library used to treat ``tenant=None`` as "all tenants": a consumer wired
straight to a shared collection that forgot the argument silently searched
every tenant at once and answered with full confidence. These tests pin the
guard (fail-closed over verifiably multi-tenant collections, permissive over
single-tenant/legacy/unknowable ones), the construction-time scope
(``default_tenant``), the BM25 corpus stamp, the arm-skip warning, and the
serve startup gate.
"""

from __future__ import annotations

import logging

import pytest

from mnemostack.recall import CrossTenantRecallError, Recaller
from mnemostack.recall.bm25 import BM25Doc
from mnemostack.recall.recaller import RecallResult
from mnemostack.recall.retrievers import BM25Retriever, Retriever


class _FakeStore:
    """Minimal vector-store stand-in with a controllable tenant facet."""

    def __init__(self, tenants: int | None):
        self._tenants = tenants
        self.probes = 0

    def distinct_tenant_count(self, limit: int = 2):
        self.probes += 1
        return self._tenants


class _StaticArm(Retriever):
    """Arm returning canned results; optionally tenant-capable."""

    def __init__(self, name, results, *, tenant_capable=False, store=None):
        self.name = name
        self._results = results
        if tenant_capable:
            self.accepts_tenant = True
        if store is not None:
            self.vector_store = store
        self.seen_tenants: list[str | None] = []

    def search(self, query, limit=20, filters=None, **kw):
        self.seen_tenants.append(kw.get("tenant"))
        return list(self._results)


def _hit(id, tenant=None, score=1.0):
    payload = {"text": f"text {id}"}
    if tenant is not None:
        payload["tenant_id"] = tenant
    return RecallResult(id=id, text=payload["text"], score=score, payload=payload)


# ------------------------------------------------------- fail-closed guard


def test_tenantless_recall_over_multi_tenant_collection_raises():
    store = _FakeStore(tenants=2)
    arm = _StaticArm("vector", [_hit("a", "t1")], store=store)
    r = Recaller(retrievers=[arm])
    with pytest.raises(CrossTenantRecallError, match="MNEMOSTACK_TENANT"):
        r.recall("q")
    assert arm.seen_tenants == []  # refused before any arm ran


def test_probe_runs_once_and_is_cached():
    store = _FakeStore(tenants=2)
    r = Recaller(retrievers=[_StaticArm("vector", [], store=store)])
    for _ in range(3):
        with pytest.raises(CrossTenantRecallError):
            r.recall("q")
    assert store.probes == 1


@pytest.mark.parametrize("tenants", [0, 1, None])
def test_single_tenant_legacy_and_unknown_collections_stay_open(tenants):
    store = _FakeStore(tenants=tenants)
    arm = _StaticArm("vector", [_hit("a")], store=store)
    r = Recaller(retrievers=[arm])
    assert [x.id for x in r.recall("q")] == ["a"]


def test_store_without_probe_capability_stays_open():
    class _Bare:
        pass

    arm = _StaticArm("vector", [_hit("a")], store=_Bare())
    r = Recaller(retrievers=[arm])
    assert [x.id for x in r.recall("q")] == ["a"]


def test_allow_cross_tenant_opts_out_and_skips_the_probe():
    store = _FakeStore(tenants=5)
    arm = _StaticArm("vector", [_hit("a", "t1"), _hit("b", "t2")], store=store)
    r = Recaller(retrievers=[arm], allow_cross_tenant=True)
    assert len(r.recall("q")) == 2
    assert store.probes == 0


def test_explicit_per_call_tenant_bypasses_the_guard():
    store = _FakeStore(tenants=2)
    arm = _StaticArm("vector", [_hit("a", "t1")], tenant_capable=True, store=store)
    r = Recaller(retrievers=[arm])
    assert [x.id for x in r.recall("q", tenant="t1")] == ["a"]
    assert store.probes == 0


# --------------------------------------------------- construction-time scope


def test_default_tenant_scopes_every_call():
    arm = _StaticArm("vector", [_hit("a", "t1"), _hit("b", "t2")], tenant_capable=True)
    r = Recaller(retrievers=[arm], default_tenant="t1")
    results = r.recall("q")
    assert [x.id for x in results] == ["a"]
    assert arm.seen_tenants == ["t1"]


def test_per_call_tenant_wins_over_default():
    arm = _StaticArm("vector", [_hit("a", "t1"), _hit("b", "t2")], tenant_capable=True)
    r = Recaller(retrievers=[arm], default_tenant="t1")
    results = r.recall("q", tenant="t2")
    assert [x.id for x in results] == ["b"]
    assert arm.seen_tenants == ["t2"]


# ----------------------------------------- cross-tenant sweep across arms


def test_write_to_a_recall_as_b_is_empty_across_arms():
    """Tenant A's data must never surface in tenant B's recall, whichever
    arm produced it — including an arm that ignores the tenant it was
    handed (the post-fusion backstop is the guarantee)."""
    leaky = _StaticArm("leaky", [_hit("a1", "A"), _hit("a2", "A")], tenant_capable=True)
    honest = _StaticArm("honest", [_hit("b1", "B")], tenant_capable=True)
    r = Recaller(retrievers=[leaky, honest], default_tenant="B")
    results = r.recall("q")
    assert [x.id for x in results] == ["b1"]
    assert all((x.payload or {}).get("tenant_id") == "B" for x in results)


# ------------------------------------------------------- BM25 tenant stamp


def test_bm25_file_corpus_stamped_with_tenant_survives_scoped_recall():
    docs = [BM25Doc(id="d1", text="the rollout decision")]
    bm25 = BM25Retriever(docs, tenant="t1")
    assert getattr(bm25, "accepts_tenant", False) is True
    r = Recaller(retrievers=[bm25], default_tenant="t1")
    results = r.recall("rollout decision")
    assert [x.id for x in results] == ["d1"]


def test_bm25_stamp_never_relabels_a_foreign_doc():
    docs = [BM25Doc(id="d1", text="alpha", payload={"tenant_id": "other"})]
    bm25 = BM25Retriever(docs, tenant="t1")
    r = Recaller(retrievers=[bm25], default_tenant="t1")
    assert r.recall("alpha") == []


def test_unstamped_bm25_is_skipped_under_tenant_with_one_warning(caplog):
    docs = [BM25Doc(id="d1", text="alpha")]
    bm25 = BM25Retriever(docs)  # no tenant stamp -> not tenant-capable
    r = Recaller(retrievers=[bm25], default_tenant="t1")
    with caplog.at_level(logging.WARNING, logger="mnemostack.recall.recaller"):
        assert r.recall("alpha") == []
        assert r.recall("alpha") == []
    # The dedup marker is the contract (immune to global logging state other
    # tests may leave behind); the record count is asserted as an upper bound.
    assert r._tenant_skip_warned == {"bm25"}
    warnings = [rec for rec in caplog.records if "cannot enforce a tenant filter" in rec.message]
    assert len(warnings) <= 1


# ------------------------------------------------------- serve startup gate


def test_unauth_serve_refuses_a_multi_tenant_collection(monkeypatch):
    pytest.importorskip("fastapi")
    import mnemostack.server as server_mod

    monkeypatch.setattr(
        server_mod, "get_provider", lambda *a, **k: type("P", (), {"dimension": 8})()
    )
    monkeypatch.setattr(
        server_mod.VectorStore,
        "distinct_tenant_count",
        lambda self, limit=2: 3,
        raising=False,
    )
    monkeypatch.setattr(server_mod.VectorStore, "__init__", lambda self, **kw: None, raising=False)
    cfg = server_mod.ServerConfig(provider_name="gemini")
    with pytest.raises(CrossTenantRecallError, match="refusing to serve without auth"):
        server_mod.build_app(cfg)


def test_unauth_serve_with_default_tenant_starts(monkeypatch):
    pytest.importorskip("fastapi")
    import mnemostack.server as server_mod

    monkeypatch.setattr(
        server_mod, "get_provider", lambda *a, **k: type("P", (), {"dimension": 8})()
    )
    calls = []
    monkeypatch.setattr(
        server_mod.VectorStore,
        "distinct_tenant_count",
        lambda self, limit=2: calls.append(1) or 3,
        raising=False,
    )
    monkeypatch.setattr(server_mod.VectorStore, "__init__", lambda self, **kw: None, raising=False)
    cfg = server_mod.ServerConfig(provider_name="gemini", default_tenant="t1")
    try:
        server_mod.build_app(cfg)
    except CrossTenantRecallError:  # must not be the tenant gate
        pytest.fail("a scoped server must not hit the tenant gate")
    assert calls == []  # scoped server never needs the probe


# ------------------------------------------------- synthesis clone carries


def test_synthesis_source_filter_keeps_the_tenant_scope():
    """_filter_recaller rebuilds the recaller for a source filter; the clone
    must carry the tenant scope — a source filter must not silently widen a
    tenant-scoped recaller to all tenants."""
    from mnemostack.synthesis import _filter_recaller

    arm = _StaticArm("vector", [], tenant_capable=True)
    src = Recaller(retrievers=[arm], default_tenant="t1", allow_cross_tenant=True)
    clone = _filter_recaller(src, {"vector"})
    assert clone.default_tenant == "t1"
    assert clone.allow_cross_tenant is True


# ------------------------------------------------------------ config layer


def test_mnemostack_tenant_env_reaches_recall_config(monkeypatch):
    from mnemostack.config import Config

    monkeypatch.setenv("MNEMOSTACK_TENANT", "t-env")
    cfg = Config.load(path=None)
    assert cfg.recall.tenant == "t-env"


# ============================================== review round 1 regressions


def test_gate_raises_the_typed_refusal(monkeypatch):
    """The unauth-serve startup gate raises CrossTenantRecallError (the CLI
    turns that into a clean `error:` + exit 2, not a traceback)."""
    pytest.importorskip("fastapi")
    import mnemostack.server as server_mod

    monkeypatch.setattr(
        server_mod, "get_provider", lambda *a, **k: type("P", (), {"dimension": 8})()
    )
    monkeypatch.setattr(
        server_mod.VectorStore, "distinct_tenant_count", lambda self, limit=2: 2, raising=False
    )
    monkeypatch.setattr(server_mod.VectorStore, "__init__", lambda self, **kw: None, raising=False)
    with pytest.raises(CrossTenantRecallError):
        server_mod.build_app(server_mod.ServerConfig(provider_name="gemini"))


def test_recall_flow_applies_default_tenant_to_pipeline_and_backstop():
    """The default tenant must reach the pipeline stages and the post-
    pipeline backstop, not only the retrieval step: a stage injecting an
    unscoped record (graph resurrection does exactly that) must be dropped."""
    from mnemostack.recall import recall_flow
    from mnemostack.recall.pipeline import Pipeline, Stage

    seen = {}

    class _Inject(Stage):
        @property
        def name(self):
            return "inject"

        def apply(self, context, results):
            seen["tenant"] = context.extras.get("tenant")
            return list(results) + [RecallResult(id="foreign", text="x", score=9.0, payload={})]

    arm = _StaticArm("vector", [_hit("mine", "t1")], tenant_capable=True)
    r = Recaller(retrievers=[arm], default_tenant="t1")
    out = recall_flow(r, "q", limit=5, pipeline=Pipeline([_Inject()]))
    assert [x.id for x in out] == ["mine"]
    assert seen["tenant"] == "t1"


def test_search_many_resolves_default_tenant_and_guards():
    calls = []

    class _Store:
        def distinct_tenant_count(self, limit=2):
            return 2

        def search(self, vector, limit, filters=None, hide_invalidated=None, **kw):
            calls.append(kw.get("tenant"))
            return []

    scoped = Recaller(vector_store=_Store(), default_tenant="t1")
    scoped._ensure_space_compat = lambda tenant=None: None
    scoped.search_many([[0.1]], limit=3)
    assert calls == ["t1"]

    unscoped = Recaller(vector_store=_Store())
    unscoped._ensure_space_compat = lambda tenant=None: None
    with pytest.raises(CrossTenantRecallError):
        unscoped.search_many([[0.1]], limit=3)


def test_negative_probe_expires_and_a_new_tenant_trips_the_guard(monkeypatch):
    store = _FakeStore(tenants=1)
    arm = _StaticArm("vector", [_hit("a")], store=store)
    r = Recaller(retrievers=[arm])
    r.recall("q")  # single tenant: open
    store._tenants = 2  # a second tenant is ingested later
    monkeypatch.setattr(Recaller, "_PROBE_TTL_S", 0.0)
    with pytest.raises(CrossTenantRecallError):
        r.recall("q")


def test_positive_probe_is_permanent(monkeypatch):
    store = _FakeStore(tenants=2)
    r = Recaller(retrievers=[_StaticArm("vector", [], store=store)])
    monkeypatch.setattr(Recaller, "_PROBE_TTL_S", 0.0)
    for _ in range(3):
        with pytest.raises(CrossTenantRecallError):
            r.recall("q")
    assert store.probes == 1


def test_unknown_probe_is_retried_and_warns_once_per_process(monkeypatch):
    import mnemostack.recall.recaller as rec_mod

    messages = []
    monkeypatch.setattr(rec_mod.logger, "warning", lambda msg, *a, **k: messages.append(msg))
    monkeypatch.setattr(rec_mod, "_PROBE_UNKNOWN_WARNED", False)
    monkeypatch.setattr(Recaller, "_PROBE_TTL_S", 0.0)
    store = _FakeStore(tenants=None)
    for _ in range(3):  # fresh recallers, as synthesis builds them per call
        Recaller(retrievers=[_StaticArm("vector", [_hit("a")], store=store)]).recall("q")
    assert store.probes == 3  # "unknown" is never cached for good
    assert sum("could not determine" in m for m in messages) == 1


def test_nothing_to_probe_is_silent_and_open(monkeypatch):
    """No collection-backed arm at all (file BM25 only): nothing can be
    multi-tenant, so no probe, no warning, no refusal."""
    import mnemostack.recall.recaller as rec_mod

    messages = []
    monkeypatch.setattr(rec_mod.logger, "warning", lambda msg, *a, **k: messages.append(msg))
    monkeypatch.setattr(rec_mod, "_PROBE_UNKNOWN_WARNED", False)
    bm25 = BM25Retriever([BM25Doc(id="d1", text="alpha")])
    assert [x.id for x in Recaller(retrievers=[bm25]).recall("alpha")] == ["d1"]
    assert not any("could not determine" in m for m in messages)


def test_synthesize_surfaces_the_refusal_instead_of_an_empty_report():
    from mnemostack.synthesis import synthesize

    store = _FakeStore(tenants=2)
    r = Recaller(retrievers=[_StaticArm("vector", [_hit("a", "t1")], store=store)])
    with pytest.raises(CrossTenantRecallError):
        synthesize("alpha", recaller=r)


def test_cli_honours_the_cross_tenant_env(monkeypatch):
    import argparse

    from mnemostack.cli import _allow_cross_tenant

    ns = argparse.Namespace(allow_cross_tenant=False)
    monkeypatch.delenv("MNEMOSTACK_ALLOW_CROSS_TENANT", raising=False)
    assert _allow_cross_tenant(ns) is False
    monkeypatch.setenv("MNEMOSTACK_ALLOW_CROSS_TENANT", "1")
    assert _allow_cross_tenant(ns) is True


def test_bm25_stamp_does_not_mutate_the_callers_docs():
    doc = BM25Doc(id="d1", text="alpha")
    BM25Retriever([doc], tenant="t1")
    assert "tenant_id" not in doc.payload


@pytest.mark.parametrize("blank", ["", None])
def test_blank_default_tenant_is_unscoped_and_guarded(blank):
    store = _FakeStore(tenants=2)
    r = Recaller(retrievers=[_StaticArm("vector", [], store=store)], default_tenant=blank)
    assert r.default_tenant is None
    with pytest.raises(CrossTenantRecallError):
        r.recall("q")


def test_unauth_scoped_server_stamps_writes_with_its_tenant(monkeypatch, tmp_path):
    """Auth off + a configured scope: the server's one tenant-resolution
    point hands that tenant to writes too, so a scoped deployment can read
    back what it writes."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from test_remote_ingest import _ingest_app

    app, store, _emb, _keys = _ingest_app(
        monkeypatch, tmp_path, auth=False, cfg_extra={"default_tenant": "alpha"}
    )
    resp = TestClient(app).post("/memories", json={"items": [{"text": "hello there"}]})
    assert resp.status_code == 200, resp.text
    pts, _ = store.client.scroll(collection_name=store.collection, limit=10)
    assert pts and all(p.payload.get("tenant_id") == "alpha" for p in pts)


# ============================================== review round 2 regressions


def _real_store(points):
    """In-memory Qdrant with the given (id, tenant_or_None) points — the
    probe is exercised against a real query engine, not a fake scroll."""
    from qdrant_client import QdrantClient

    from mnemostack.vector.qdrant import VectorStore

    s = VectorStore(collection="probe", dimension=3)
    s.client = QdrantClient(":memory:")
    s.ensure_collection()
    for pid, tenant in points:
        payload = {"text": f"p{pid}"}
        s.client.upsert(
            collection_name="probe",
            points=[
                __import__("qdrant_client").models.PointStruct(
                    id=pid,
                    vector=[0.1, 0.2, 0.3],
                    payload={**payload, **({"tenant_id": tenant} if tenant else {})},
                )
            ],
        )
    return s


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        ([], 0),
        ([(1, None), (2, None)], 0),  # legacy only
        ([(1, "a"), (2, "a")], 1),
        ([(1, None), (2, "a"), (3, "a")], 1),  # one tenant + legacy stays open
        ([(1, "a"), (2, "b")], 2),
        ([(1, None)] * 1 + [(i, "a") for i in range(2, 60)] + [(99, "b")], 2),
    ],
)
def test_exact_probe_on_a_real_query_engine(points, expected):
    """Two limit=1 scrolls answer exactly, with no cap: the second tenant is
    found even when it sits behind many points of the first."""
    assert _real_store(points).distinct_tenant_count() == expected


def test_count_unstamped_counts_points_without_a_tenant():
    s = _real_store([(1, None), (2, None), (3, "a")])
    assert s.count_unstamped() == 2


def test_probe_refresh_is_single_flight(monkeypatch):
    """A stale cache under concurrent traffic re-probes once, not once per
    thread."""
    import threading
    import time as _time

    class _SlowStore(_FakeStore):
        def distinct_tenant_count(self, limit=2):
            _time.sleep(0.05)  # long enough for the threads to overlap
            return super().distinct_tenant_count(limit)

    store = _SlowStore(tenants=1)
    r = Recaller(retrievers=[_StaticArm("vector", [_hit("a")], store=store)])
    r.recall("q")
    monkeypatch.setattr(Recaller, "_PROBE_TTL_S", 3600.0)
    r._probe_at = -1e9  # force stale
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        r.recall("q")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert store.probes == 2  # initial + exactly one refresh


def test_synthesis_clone_probes_exactly_the_kept_arms():
    """A source filter keeping only an arm over a multi-tenant collection
    still refuses (BM25 loaded from Qdrant declares its collection) — and one
    keeping only a store-less arm is not vetoed by a collection it no longer
    reads."""
    from mnemostack.synthesis import synthesize

    s = _real_store([(1, "a"), (2, "b")])
    vec = _StaticArm("vector", [], store=s)
    qbm25 = BM25Retriever.from_qdrant(s.client, "probe")
    with pytest.raises(CrossTenantRecallError):
        synthesize("p1", recaller=Recaller(retrievers=[vec, qbm25]), sources=["bm25"])

    file_bm25 = BM25Retriever([BM25Doc(id="d1", text="alpha notes")])
    r = Recaller(retrievers=[vec, file_bm25])
    synthesize("alpha", recaller=r, sources=["bm25"])  # must not raise


def test_direct_retrievers_are_guarded_scoped_and_backstopped():
    from mnemostack.synthesis import _query_retrievers

    store = _FakeStore(tenants=2)
    capable = _StaticArm(
        "vector", [_hit("a", "t1"), _hit("b", "t2")], tenant_capable=True, store=store
    )
    incapable = _StaticArm("bm25", [_hit("c")])
    # unscoped over a multi-tenant collection: refused
    with pytest.raises(CrossTenantRecallError):
        _query_retrievers([capable, incapable], "q", 10, None, None)
    # scoped: the tenant reaches capable arms only, output is backstopped
    out = _query_retrievers([capable, incapable], "q", 10, None, None, tenant="t1")
    assert [x.id for x in out] == ["a"]
    assert capable.seen_tenants == ["t1"]
    assert incapable.seen_tenants == []


def test_server_config_normalizes_a_blank_tenant():
    pytest.importorskip("fastapi")
    from mnemostack.server import ServerConfig

    assert ServerConfig(provider_name="x", default_tenant="").default_tenant is None


def test_scoped_server_warns_about_unstamped_points(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from test_remote_ingest import _ingest_app

    import mnemostack.server as srv

    # Capture the call itself: other tests leave global logging state behind
    # (levels, propagation), which makes caplog order-dependent here.
    messages = []
    monkeypatch.setattr(srv.log, "warning", lambda msg, *a, **k: messages.append(msg % a))
    monkeypatch.setattr(srv.VectorStore, "count_unstamped", lambda self: 5, raising=False)
    _ingest_app(monkeypatch, tmp_path, auth=False, cfg_extra={"default_tenant": "alpha"})
    assert any("tenant-migrate" in m and "5 point" in m for m in messages)


def test_under_auth_the_key_decides_not_the_configured_scope(monkeypatch, tmp_path):
    """default_tenant is the auth-OFF scope only: with auth on, writes land
    in the key's tenant whatever the server config says."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from test_remote_ingest import _ingest_app

    app, store, _emb, keys = _ingest_app(
        monkeypatch, tmp_path, auth=True, cfg_extra={"default_tenant": "zzz"}
    )
    resp = TestClient(app).post(
        "/memories",
        json={"items": [{"text": "hello there"}]},
        headers={"Authorization": f"Bearer {keys['write']}"},
    )
    assert resp.status_code == 200, resp.text
    pts, _ = store.client.scroll(collection_name=store.collection, limit=10)
    assert pts and all(p.payload.get("tenant_id") == "alpha" for p in pts)


def test_cli_feedback_and_resolve_default_to_the_configured_tenant(monkeypatch):
    from mnemostack.cli import build_parser

    monkeypatch.setenv("MNEMOSTACK_TENANT", "t-env")
    parser = build_parser()
    fb = parser.parse_args(["feedback", "some-id", "--signal", "useful"])
    assert fb.tenant == "t-env"
    rs = parser.parse_args(["resolve", "some-id"])
    assert rs.tenant == "t-env"


# ============================================== review round 3 regressions


class _SampleStore:
    def __init__(self, tenants):
        self._tenants = tenants

    def tenant_sample(self):
        return None if self._tenants is None else set(self._tenants)


def test_every_probe_source_is_asked_and_samples_are_unioned():
    """One tenant in each of two collections is still two tenants."""
    a = _StaticArm("vec_a", [], store=_SampleStore({"x"}))
    b = _StaticArm("vec_b", [], store=_SampleStore({"y"}))
    with pytest.raises(CrossTenantRecallError):
        Recaller(retrievers=[a, b]).recall("q")


def test_same_single_tenant_across_sources_stays_open():
    a = _StaticArm("vec_a", [_hit("1")], store=_SampleStore({"x"}))
    b = _StaticArm("vec_b", [], store=_SampleStore({"x"}))
    assert [r.id for r in Recaller(retrievers=[a, b]).recall("q")] == ["1"]


def test_a_later_multi_tenant_source_is_not_missed():
    a = _StaticArm("vec_a", [], store=_SampleStore(set()))
    b = _StaticArm("vec_b", [], store=_SampleStore({"x", "y"}))
    with pytest.raises(CrossTenantRecallError):
        Recaller(retrievers=[a, b]).recall("q")


def test_bm25_from_qdrant_declares_its_collection_to_the_guard():
    """The in-memory corpus is the whole collection, every tenant included;
    a recaller holding only this arm must still be able to refuse."""
    s = _real_store([(1, "a"), (2, "b")])
    arm = BM25Retriever.from_qdrant(s.client, "probe")
    assert arm.tenant_probe_store.tenant_sample() == {"a", "b"}
    with pytest.raises(CrossTenantRecallError):
        Recaller(retrievers=[arm]).recall("p1")


def test_real_store_sample_returns_the_tenant_ids():
    assert _real_store([(1, "a"), (2, "a")]).tenant_sample() == {"a"}
    assert _real_store([(1, None)]).tenant_sample() == set()


def test_synthesize_tenant_scopes_the_supplied_recaller_too():
    """One report, one tenant: the explicit tenant reaches the recaller, and
    the merged results are backstopped."""
    from mnemostack.synthesis import synthesize

    rec_arm = _StaticArm("vector", [_hit("from_x", "x"), _hit("from_y", "y")], tenant_capable=True)
    direct = _StaticArm("vector2", [_hit("d_x", "x")], tenant_capable=True)
    r = Recaller(retrievers=[rec_arm], default_tenant="y")
    synthesize("q", recaller=r, retrievers=[direct], tenant="x")
    assert rec_arm.seen_tenants == ["x"]
    assert direct.seen_tenants == ["x"]


# ============================================== review round 5 regressions


def test_direct_retrievers_probe_only_the_selected_sources():
    """A multi-tenant collection behind an arm the source filter drops must
    not veto the arms it keeps."""
    from mnemostack.synthesis import _query_retrievers

    vec = _StaticArm("vector", [_hit("v", "t1")], store=_SampleStore({"t1", "t2"}))
    file_bm25 = BM25Retriever([BM25Doc(id="d1", text="alpha notes")])
    out = _query_retrievers([vec, file_bm25], "alpha", 10, {"bm25"}, None)
    assert [r.id for r in out] == ["d1"]
    with pytest.raises(CrossTenantRecallError):
        _query_retrievers([vec, file_bm25], "alpha", 10, {"vector"}, None)


def test_search_many_probes_only_the_store_it_searches():
    """search_many reads the recaller's own vector store only; a multi-tenant
    collection behind another arm must not veto it — and a multi-tenant own
    store still refuses."""

    class _Own(_SampleStore):
        def search(self, vector, limit, filters=None, hide_invalidated=None, **kw):
            return []

    other = _StaticArm("other", [], store=_SampleStore({"x", "y"}))
    ok = Recaller(vector_store=_Own({"a"}), retrievers=[other])
    ok._ensure_space_compat = lambda tenant=None: None
    assert ok.search_many([[0.1]], limit=3) == []

    bad = Recaller(vector_store=_Own({"a", "b"}), retrievers=[other])
    bad._ensure_space_compat = lambda tenant=None: None
    with pytest.raises(CrossTenantRecallError):
        bad.search_many([[0.1]], limit=3)


# ============================================ PR #195 bot round 1 regressions


def test_bm25_snapshot_probe_describes_the_loaded_corpus_not_the_live_collection():
    """Points deleted after the load stay searchable in the snapshot, so the
    guard must still see their tenant; and a filtered single-tenant load of
    a shared collection must not be refused."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue, PointIdsList

    s = _real_store([(1, "a"), (2, "b")])
    arm = BM25Retriever.from_qdrant(s.client, "probe")
    s.client.delete(collection_name="probe", points_selector=PointIdsList(points=[2]))
    with pytest.raises(CrossTenantRecallError):  # B still in the snapshot
        Recaller(retrievers=[arm]).recall("p2")

    s2 = _real_store([(1, "a"), (2, "b")])
    only_a = BM25Retriever.from_qdrant(
        s2.client,
        "probe",
        scroll_filter=Filter(must=[FieldCondition(key="tenant_id", match=MatchValue(value="a"))]),
    )
    assert only_a.tenant_probe_store.tenant_sample() == {"a"}
    Recaller(retrievers=[only_a]).recall("p1")  # must not raise


def test_narrowed_probe_warns_when_undetermined(monkeypatch):
    import mnemostack.recall.recaller as rec_mod

    messages = []
    monkeypatch.setattr(rec_mod.logger, "warning", lambda msg, *a, **k: messages.append(msg))
    monkeypatch.setattr(rec_mod, "_PROBE_UNKNOWN_WARNED", False)

    class _Broken(_SampleStore):
        def search(self, vector, limit, filters=None, hide_invalidated=None, **kw):
            return []

    r = Recaller(vector_store=_Broken(None))
    r._ensure_space_compat = lambda tenant=None: None
    r.search_many([[0.1]], limit=3)
    assert sum("could not determine" in m for m in messages) == 1


def test_synthesize_opt_out_governs_a_supplied_recaller_without_mutating_it():
    from mnemostack.synthesis import synthesize

    store = _FakeStore(tenants=2)
    r = Recaller(retrievers=[_StaticArm("vector", [_hit("a", "t1")], store=store)])
    synthesize("alpha", recaller=r, allow_cross_tenant=True)  # must not raise
    assert r.allow_cross_tenant is False


@pytest.mark.parametrize("blank", ["", "   "])
def test_whitespace_default_tenant_is_unscoped_and_guarded(blank):
    store = _FakeStore(tenants=2)
    r = Recaller(retrievers=[_StaticArm("vector", [], store=store)], default_tenant=blank)
    assert r.default_tenant is None
    with pytest.raises(CrossTenantRecallError):
        r.recall("q")


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_per_call_tenant_is_refused_never_defaulted(blank):
    """A per-call tenant is an identity (under auth, the key's principal): a
    blank one must never fall back to the configured default tenant."""
    arm = _StaticArm("vector", [_hit("m", "main")], tenant_capable=True)
    r = Recaller(retrievers=[arm], default_tenant="main")
    with pytest.raises(CrossTenantRecallError, match="blank tenant"):
        r.recall("q", tenant=blank)
    assert arm.seen_tenants == []


def test_per_call_tenant_is_never_rewritten():
    """ "acme " is its own identity, not "acme"."""
    arm = _StaticArm("vector", [_hit("a", "acme"), _hit("b", "acme ")], tenant_capable=True)
    r = Recaller(retrievers=[arm])
    assert [x.id for x in r.recall("q", tenant="acme ")] == ["b"]
    assert arm.seen_tenants == ["acme "]


def test_whitespace_tenant_from_env_is_no_tenant(monkeypatch):
    from mnemostack.config import Config

    monkeypatch.setenv("MNEMOSTACK_TENANT", "   ")
    assert Config.load(path=None).recall.tenant is None


def test_configured_tenant_keeps_nonblank_values_verbatim():
    """Reads and writes must agree: a configured " acme " stays " acme "."""
    from mnemostack.config import normalize_tenant

    assert normalize_tenant(" acme ") == " acme "
    assert normalize_tenant("  ") is None
    assert Recaller(retrievers=[], default_tenant=" acme ").default_tenant == " acme "


def test_synthesize_refuses_a_blank_explicit_tenant():
    from mnemostack.synthesis import synthesize

    arm = _StaticArm("vector", [_hit("m", "main")], tenant_capable=True)
    with pytest.raises(CrossTenantRecallError, match="blank tenant"):
        synthesize("q", recaller=Recaller(retrievers=[arm], default_tenant="main"), tenant=" ")
