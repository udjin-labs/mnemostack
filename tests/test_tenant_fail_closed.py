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


def test_unknown_probe_is_retried_and_warns_once(monkeypatch, caplog):
    store = _FakeStore(tenants=None)
    r = Recaller(retrievers=[_StaticArm("vector", [_hit("a")], store=store)])
    monkeypatch.setattr(Recaller, "_PROBE_TTL_S", 0.0)
    with caplog.at_level(logging.WARNING, logger="mnemostack.recall.recaller"):
        r.recall("q")
        r.recall("q")
    assert store.probes == 2  # "unknown" is not cached for the process lifetime
    assert r._probe_unknown_warned is True


def test_distinct_tenant_count_falls_back_to_scroll_when_facet_is_rejected():
    """A real server rejects faceting an unindexed tenant_id (collections
    created before tenancy); the probe must fall back to a bounded scroll
    instead of reporting "unknown" and leaving the guard open."""
    from mnemostack.vector.qdrant import VectorStore

    class _Pt:
        def __init__(self, tid):
            self.payload = {"tenant_id": tid} if tid else {}

    class _Client:
        def facet(self, **kw):
            raise RuntimeError("No appropriate index for faceting: tenant_id")

        def scroll(self, **kw):
            return [_Pt(None), _Pt("a"), _Pt("a"), _Pt("b")], None

    s = VectorStore.__new__(VectorStore)
    s.client = _Client()
    s.collection = "c"
    assert s.distinct_tenant_count() == 2


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
