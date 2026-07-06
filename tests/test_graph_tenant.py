"""Graph tenant-scoping — store writes/reads, retriever, and resurrection.

These pin the isolation contract with fakes (no live Memgraph): the store folds
``tenant`` into node MERGE keys and confines reads to it; the retriever and the
resurrection stage confine their walks to the tenant and stamp ``tenant_id`` on
what they surface so it survives the recall ``filter_by_tenant`` backstop. With
``tenant=None`` every path emits the byte-for-byte legacy Cypher, so an existing
single-tenant graph is untouched.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from mnemostack.cli import _graph_relabel_preflight
from mnemostack.graph.store import GraphStore
from mnemostack.recall.pipeline.base import PipelineContext
from mnemostack.recall.pipeline.resurrection import GraphResurrection
from mnemostack.recall.retrievers import MemgraphRetriever


class _RecordingSession:
    """Fake neo4j session that records every (cypher, params) run.

    ``rows`` answers node probes; ``rel_rows`` answers the retriever's
    relationship-expansion query (detected by ``startNode(r)`` in the Cypher).
    """

    def __init__(
        self,
        rows: list[dict[str, Any]] | None = None,
        rel_rows: list[dict[str, Any]] | None = None,
    ):
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._rows = rows or []
        self._rel_rows = rel_rows or []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, cypher, **params):
        self.calls.append((cypher, params))
        if "startNode(r)" in cypher:
            return _Result(self._rel_rows)
        return _Result(self._rows)

    def execute_write(self, fn, *args):
        return fn(self, *args)

    def single(self):  # for count-style single() callers
        return None


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows

    def single(self):
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


def _store_with(session):
    store = GraphStore.__new__(GraphStore)
    driver = MagicMock()
    driver.session.return_value = session
    store.driver = driver
    store.database = None
    return store


# ---------- store writes ----------


def test_add_triple_scoped_folds_tenant_into_key_and_stamps():
    session = _RecordingSession()
    store = _store_with(session)
    store.add_triple("alice", "KNOWS", "bob", tenant="acme")
    cypher, params = session.calls[0]
    assert "{name: $subject, tenant: $tenant}" in cypher
    assert "{name: $obj, tenant: $tenant}" in cypher
    assert "s.tenant = $tenant, o.tenant = $tenant" in cypher
    assert params["tenant"] == "acme"
    assert params["props"]["tenant"] == "acme"  # edge stamped too


def test_add_triple_unscoped_does_not_tenant_key_or_clobber():
    session = _RecordingSession()
    store = _store_with(session)
    store.add_triple("alice", "KNOWS", "bob")
    cypher, params = session.calls[0]
    assert "tenant:" not in cypher  # no tenant folded into the node MERGE keys
    assert "tenant" not in params  # no tenant param bound
    assert "tenant" not in params["props"]  # edge not stamped
    # The edge SET is guarded so an unscoped write can't overwrite a tenant-owned
    # edge's props (on a single-tenant graph every edge is tenant-less, so it runs).
    assert "CASE WHEN r.tenant IS NULL" in cypher


def test_query_triples_scoped_confines_both_endpoints():
    session = _RecordingSession()
    store = _store_with(session)
    store.query_triples(subject="alice", tenant="acme")
    cypher, params = session.calls[0]
    assert "s.tenant = $tenant AND o.tenant = $tenant" in cypher
    assert "r.tenant = $tenant" in cypher  # the edge is confined too
    assert params["tenant"] == "acme"


def test_query_triples_unscoped_has_no_tenant_predicate():
    session = _RecordingSession()
    store = _store_with(session)
    store.query_triples(subject="alice")
    cypher, params = session.calls[0]
    assert "tenant" not in cypher
    assert "tenant" not in params


def test_invalidate_scoped_pins_both_endpoints():
    session = _RecordingSession([{"n": 1}])
    store = _store_with(session)
    store.invalidate("alice", "KNOWS", "bob", "2026-01-01", tenant="acme")
    cypher, params = session.calls[0]
    assert "{name: $subject, tenant: $tenant}" in cypher
    assert "{name: $obj, tenant: $tenant}" in cypher
    # The closed edge is confined to the tenant too, with the validity OR parenthesized.
    assert "(r.valid_until = 'current' OR r.valid_until IS NULL) AND r.tenant = $tenant" in cypher
    assert params["tenant"] == "acme"


def test_invalidate_unscoped_only_closes_tenant_less_edges():
    session = _RecordingSession([{"n": 0}])
    store = _store_with(session)
    store.invalidate("alice", "KNOWS", "bob", "2026-01-01")  # unscoped
    cypher, params = session.calls[0]
    # Guarded so an unscoped invalidate can't close a tenant-owned edge.
    assert "r.tenant IS NULL" in cypher
    assert "tenant" not in params


def test_sync_file_links_scoped_keys_file_nodes_by_tenant():
    session = _RecordingSession()
    store = _store_with(session)
    store.sync_file_links("a.md", ["b.md"], index_root="/root", tenant="acme")
    cyphers = [c for c, _ in session.calls]
    joined = " ".join(cyphers)
    assert "index_root: $root, tenant: $tenant" in joined
    assert "s.tenant = $tenant, o.tenant = $tenant" in joined
    # The LINKS_TO edge itself is stamped, so scoped recall (which requires
    # r.tenant = $tenant) can traverse links written through this scoped API.
    merge_cypher = next(c for c in cyphers if "MERGE (s)-[r:LINKS_TO]->(o)" in c)
    assert "r.tenant = $tenant" in merge_cypher
    # The edge-clearing DELETE pins the far end AND the edge to the tenant.
    del_cypher = next(c for c in cyphers if "DELETE r" in c)
    assert "-[r:LINKS_TO]->({tenant: $tenant})" in del_cypher
    assert "WHERE r.tenant = $tenant" in del_cypher


def test_referrers_of_dangling_scoped_binds_edge_tenant():
    session = _RecordingSession([])
    store = _store_with(session)
    store.referrers_of_dangling(["b"], index_root="/root", tenant="acme")
    cypher, params = session.calls[0]
    assert "[r:LINKS_TO]" in cypher and "r.tenant = $tenant" in cypher
    assert params["tenant"] == "acme"


def test_referrers_of_dangling_unscoped_leaves_edge_anonymous():
    session = _RecordingSession([])
    store = _store_with(session)
    store.referrers_of_dangling(["b"], index_root="/root")
    cypher, _ = session.calls[0]
    assert "[:LINKS_TO]" in cypher and "tenant" not in cypher


def test_stamp_tenant_backfills_nodes_and_edges():
    session = _RecordingSession([{"n": 5}])
    store = _store_with(session)
    counts = store.stamp_tenant("acme")
    cyphers = [c for c, _ in session.calls]
    assert any("SET n.tenant = $tenant" in c and "n.tenant IS NULL" in c for c in cyphers)
    assert any("SET r.tenant = $tenant" in c and "r.tenant IS NULL" in c for c in cyphers)
    assert counts == {"nodes": 5, "relationships": 5}


def _fake_gf(monkeypatch, node_total, node_missing, rel_total, rel_missing):
    import mnemostack.graph.factory as gf

    calls: list[dict[str, Any]] = []

    class _FakeGS:
        def stamp_tenant(self, tenant, *, only_missing=True, dry_run=False):
            calls.append({"only_missing": only_missing, "dry_run": dry_run})
            return {
                "nodes": node_missing if only_missing else node_total,
                "relationships": rel_missing if only_missing else rel_total,
            }

        def close(self):
            pass

    monkeypatch.setattr(gf, "make_graph_store", lambda *a, **k: _FakeGS())
    return calls


def test_graph_relabel_preflight_refuses_without_yes_when_nodes_tenanted(monkeypatch):
    # --all against a graph whose NODES already carry tenants needs --yes.
    import argparse

    _fake_gf(monkeypatch, node_total=10, node_missing=3, rel_total=0, rel_missing=0)
    args = argparse.Namespace(memgraph_uri="bolt://x", tenant="acme", yes=False, graph_timeout=5.0)
    assert _graph_relabel_preflight(args) == 2  # 7 nodes already tenanted → refused
    args.yes = True
    assert _graph_relabel_preflight(args) == 0  # --yes proceeds


def test_graph_relabel_preflight_counts_relationship_tenants(monkeypatch):
    # Tenants on RELATIONSHIPS but not nodes must still trip the guard.
    import argparse

    _fake_gf(monkeypatch, node_total=5, node_missing=5, rel_total=8, rel_missing=2)
    args = argparse.Namespace(memgraph_uri="bolt://x", tenant="acme", yes=False, graph_timeout=5.0)
    assert _graph_relabel_preflight(args) == 2  # 6 edges already tenanted → refused


def test_graph_relabel_preflight_noop_without_memgraph(monkeypatch):
    import argparse

    args = argparse.Namespace(memgraph_uri=None, tenant="acme", yes=False, graph_timeout=5.0)
    assert _graph_relabel_preflight(args) == 0  # no graph configured → no gate


def test_sync_file_links_unscoped_delete_skips_tenant_edges():
    # An unscoped re-index (e.g. index-markdown) must not delete a tenant's edges
    # after the graph has been migrated — the DELETE clears only tenant-less edges.
    session = _RecordingSession()
    store = _store_with(session)
    store.sync_file_links("a.md", ["b.md"], index_root="/root")  # tenant=None
    del_cypher = next(c for c, _ in session.calls if "DELETE r" in c)
    # Gated on the SOURCE NODE's tenant (not the edge's), so a migrated file's
    # edges are preserved while a genuine single-tenant file's stale links —
    # even ones carrying a legacy `tenant` edge property — are still cleared.
    assert "WHERE s.tenant IS NULL" in del_cypher


def test_file_link_sources_scoped_requires_owned_edge():
    session = _RecordingSession([])
    store = _store_with(session)
    store.file_link_sources(index_root="/root", tenant="acme")
    cypher, params = session.calls[0]
    assert "[r:LINKS_TO]" in cypher and "WHERE r.tenant = $tenant" in cypher
    assert params["tenant"] == "acme"


def test_graph_result_id_escapes_tenant_delimiter():
    from mnemostack.recall.retrievers import graph_result_id

    # tenant 'a:b' + node 'c'  vs  tenant 'a' + node 'b:c' must NOT collide.
    assert graph_result_id("c", "a:b") != graph_result_id("b:c", "a")
    assert graph_result_id("n", None) == "graph:n"  # unscoped unchanged


def test_add_file_tags_adapter_receives_tenant():
    from mnemostack.ingest import IngestItem, _sync_wrapper_graph

    got: dict[str, Any] = {}

    class _Adapter:  # no .driver → add_file_tags path; accepts tenant
        def add_file_tags(self, name, path, indexed_date, tags, tenant=None):
            got["tenant"] = tenant

    item = IngestItem(source="a.txt", text="hi", metadata={"tags": ["x"]})
    _sync_wrapper_graph(_Adapter(), item, "pid", tenant="acme")
    assert got["tenant"] == "acme"


def test_add_file_tags_legacy_adapter_falls_back_to_tenant_add_triple():
    # A legacy add_file_tags (no tenant kwarg) can't scope, so under a tenant we
    # skip it and use the tenant-threading add_triple path instead.
    from mnemostack.ingest import IngestItem, _sync_wrapper_graph

    seen: list[tuple[str, str | None]] = []

    class _LegacyAdapter:
        def add_file_tags(self, name, path, indexed_date, tags):  # no tenant
            seen.append(("add_file_tags", None))

        def add_triple(self, subject, predicate, obj, subject_label="Entity",
                       obj_label="Entity", properties=None, tenant=None):
            seen.append(("add_triple", tenant))

    item = IngestItem(source="a.txt", text="hi", metadata={"tags": ["x"]})
    _sync_wrapper_graph(_LegacyAdapter(), item, "pid", tenant="acme")
    assert ("add_triple", "acme") in seen and ("add_file_tags", None) not in seen


def test_wrapper_graph_unscoped_guards_file_metadata():
    # An unscoped wrapper ingest must not overwrite a tenant-owned :File node's
    # metadata (path key subset-matches it) — the SET is FOREACH-guarded.
    from mnemostack.ingest import IngestItem, _sync_wrapper_graph

    session = _RecordingSession()
    driver = MagicMock()
    driver.session.return_value = session

    class _G:
        def __init__(self, d):
            self.driver = d
            self.database = None

    item = IngestItem(source="a.txt", text="hi", metadata={"tags": ["x"]})
    _sync_wrapper_graph(_G(driver), item, "pid", tenant=None)
    cypher = session.calls[0][0]
    assert "FOREACH (_ IN CASE WHEN f.tenant IS NULL" in cypher
    assert "f.tenant" not in cypher.split("FOREACH")[0]  # no unconditional tenant SET


def test_wrapper_graph_scoped_folds_tenant_into_keys():
    from mnemostack.ingest import IngestItem, _sync_wrapper_graph

    session = _RecordingSession()
    driver = MagicMock()
    driver.session.return_value = session

    class _G:
        def __init__(self, d):
            self.driver = d
            self.database = None

    item = IngestItem(source="a.txt", text="hi", metadata={"tags": ["x"]})
    _sync_wrapper_graph(_G(driver), item, "pid", tenant="acme")
    cypher, params = session.calls[0]
    assert "path: $path, tenant: $tenant" in cypher  # File key folds tenant
    assert "name: tag, tenant: $tenant" in cypher  # Tag key folds tenant
    assert "SET r.tenant = $tenant" in cypher  # TAGGED edge stamped
    assert params["tenant"] == "acme"


def test_ingest_fallback_adapter_without_tenant_kwarg():
    # A custom graph adapter with the legacy add_triple signature (no tenant kwarg)
    # must still sync tags in a single-tenant ingest — tenant=None must NOT be
    # passed, or it raises TypeError and _write_wrappers silently drops the tags.
    from mnemostack.ingest import IngestItem, _sync_wrapper_graph

    calls: list[dict[str, Any]] = []

    class _LegacyAdapter:
        # No .driver, no .add_file_tags → falls through to the add_triple path.
        def add_triple(self, subject, predicate, obj, subject_label="Entity",
                       obj_label="Entity", properties=None):  # NO tenant kwarg
            calls.append({"subject": subject, "obj": obj})

    item = IngestItem(source="a.txt", text="hi", metadata={"tags": ["x"]})
    _sync_wrapper_graph(_LegacyAdapter(), item, "pid-1", tenant=None)  # must not raise
    assert calls and calls[0]["obj"] == "x"
    session = _RecordingSession([{"n": 9}])
    store = _store_with(session)
    store.stamp_tenant("acme", only_missing=False)
    cyphers = [c for c, _ in session.calls]
    assert not any("IS NULL" in c for c in cyphers)  # force-relabel touches every record


# ---------- retriever ----------


def test_retriever_advertises_accepts_tenant():
    assert MemgraphRetriever.accepts_tenant is True


def test_retriever_scopes_probes_and_stamps_tenant_id():
    # A node probe matches; assert the tenant predicate rode along and the result
    # carries tenant_id (so filter_by_tenant keeps it).
    rows = [{"name": "alice", "type": "Entity", "mc": "", "index_root": None}]
    rel_rows = [{"from_n": "alice", "rel": "KNOWS", "to_n": "bob"}]
    session = _RecordingSession(rows, rel_rows)
    driver = MagicMock()
    driver.session.return_value = session
    retr = MemgraphRetriever(driver=driver)
    results = retr.search("alice", tenant="acme")
    probe_cyphers = [c for c, _ in session.calls if "labels(n)[0]" in c]
    assert probe_cyphers and all("n.tenant = $tenant" in c for c in probe_cyphers)
    rel_cyphers = [c for c, _ in session.calls if "startNode(r)" in c]
    assert rel_cyphers and all("r.tenant = $tenant" in c for c in rel_cyphers)
    assert all(p.get("tenant") == "acme" for c, p in session.calls if "tenant" in c)
    assert results and results[0].payload["tenant_id"] == "acme"
    # id namespaced by tenant so it can't collide with another tenant's node in
    # the stateful pipeline's IoR/feedback state.
    assert results[0].id.startswith("graph:acme:")


def test_retriever_unscoped_adds_no_tenant_predicate():
    rows = [{"name": "alice", "type": "Entity", "mc": "", "index_root": None}]
    rel_rows = [{"from_n": "alice", "rel": "KNOWS", "to_n": "bob"}]
    session = _RecordingSession(rows, rel_rows)
    driver = MagicMock()
    driver.session.return_value = session
    retr = MemgraphRetriever(driver=driver)
    results = retr.search("alice")
    assert all("tenant" not in c for c, _ in session.calls)
    assert results and "tenant_id" not in results[0].payload
    assert results[0].id == "graph:alice"  # unscoped id unchanged (no tenant prefix)


# ---------- resurrection ----------


def test_resurrection_scopes_walk_and_stamps_tenant():
    rows = [{"name": "charlie", "type": "Entity", "mc": "", "rel": "KNOWS"}]
    session = _RecordingSession(rows)
    driver = MagicMock()
    driver.session.return_value = session
    stage = GraphResurrection(driver=driver, min_seed_len=3)
    ctx = PipelineContext(query="alice bob")
    ctx.extras["tenant"] = "acme"
    out = stage.apply(ctx, [])
    walk_cyphers = [c for c, _ in session.calls]
    assert walk_cyphers and all(
        "n.tenant = $tenant AND m.tenant = $tenant AND r1.tenant = $tenant" in c
        for c in walk_cyphers
    )
    assert all(p.get("tenant") == "acme" for _, p in session.calls)
    injected = [r for r in out if r.payload.get("resurrected")]
    assert injected and injected[0].payload["tenant_id"] == "acme"
    assert injected[0].id.startswith("graph:acme:")  # tenant-namespaced id


def test_resurrection_unscoped_has_no_tenant_predicate():
    rows = [{"name": "charlie", "type": "Entity", "mc": "", "rel": "KNOWS"}]
    session = _RecordingSession(rows)
    driver = MagicMock()
    driver.session.return_value = session
    stage = GraphResurrection(driver=driver, min_seed_len=3)
    out = stage.apply(PipelineContext(query="alice bob"), [])
    assert all("tenant" not in c for c, _ in session.calls)
    injected = [r for r in out if r.payload.get("resurrected")]
    assert injected and "tenant_id" not in injected[0].payload
