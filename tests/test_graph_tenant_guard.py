"""Fail-closed tenant guard over the graph (#194).

A tenantless recall over a graph holding several tenants is refused like one
over a multi-tenant collection: the graph arm declares a tenant probe the
recaller asks, and the resurrection stage (which walks the graph on its own)
skips itself instead of resurrecting any tenant's nodes. Scoped recall, a
single-tenant graph, and deliberate cross-tenant tooling are unaffected.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from mnemostack.graph.extractor import TripleExtractor
from mnemostack.graph.store import Triple
from mnemostack.recall import CrossTenantRecallError, Recaller, RecallResult
from mnemostack.recall.pipeline.base import Pipeline, PipelineContext
from mnemostack.recall.pipeline.resurrection import GraphResurrection
from mnemostack.recall.retrievers import MemgraphRetriever, graph_tenant_sample

PROBE = "n.tenant IS NOT NULL"


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows


class _GraphSession:
    """Answers the tenant probe from ``tenants`` (node tenant values), and
    every other query with ``rows``."""

    def __init__(self, tenants: list[Any], rows: list[dict] | None = None, calls=None):
        self.tenants = tenants
        self.rows = rows or []
        self.calls = calls if calls is not None else []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, cypher, **params):
        self.calls.append(cypher)
        if PROBE in cypher:
            pool = [t for t in self.tenants if "t1" not in params or t != params["t1"]]
            return _Result([{"t": pool[0]}] if pool else [])
        return _Result(self.rows)


class _Driver:
    def __init__(self, tenants: list[Any], rows: list[dict] | None = None):
        self.tenants = tenants
        self.rows = rows
        self.calls: list[str] = []

    def session(self, **_):
        return _GraphSession(self.tenants, self.rows, self.calls)


class _Boom:
    def session(self, **_):
        raise RuntimeError("bad query")


def _probe_runs(driver: _Driver) -> int:
    return sum(PROBE in c for c in driver.calls)


# ---------- the probe ----------


@pytest.mark.parametrize(
    "tenants, expected",
    [([], []), (["a"], ["a"]), (["a", "a"], ["a"]), (["a", "b"], ["a", "b"])],
)
def test_graph_tenant_sample(tenants, expected):
    retr = MemgraphRetriever(uri="bolt://x", driver=_Driver(tenants))
    assert graph_tenant_sample(retr) == expected


def test_graph_tenant_sample_undetermined_on_failure():
    assert graph_tenant_sample(MemgraphRetriever(uri="bolt://x", driver=_Boom())) is None


# ---------- the recaller guard ----------


def test_tenantless_recall_over_multi_tenant_graph_is_refused():
    retr = MemgraphRetriever(uri="bolt://x", driver=_Driver(["a", "b"]))
    with pytest.raises(CrossTenantRecallError):
        Recaller(retrievers=[retr]).recall("alice project")


def test_scoped_recall_over_multi_tenant_graph_runs():
    driver = _Driver(["a", "b"])
    retr = MemgraphRetriever(uri="bolt://x", driver=driver)
    Recaller(retrievers=[retr]).recall("alice project", tenant="a")
    assert _probe_runs(driver) == 0  # a scoped recall never needs the probe


def test_single_tenant_graph_and_cross_tenant_tooling_run():
    Recaller(retrievers=[MemgraphRetriever(uri="bolt://x", driver=_Driver(["a"]))]).recall(
        "alice project"
    )
    Recaller(
        retrievers=[MemgraphRetriever(uri="bolt://x", driver=_Driver(["a", "b"]))],
        allow_cross_tenant=True,
    ).recall("alice project")


def test_one_tenant_in_graph_and_another_in_the_collection_is_refused():
    class _Store:
        def tenant_sample(self):
            return ["b"]

    class _VectorArm(MemgraphRetriever):
        tenant_probe_store = _Store()

        def search(self, *a, **k):
            return []

    graph = MemgraphRetriever(uri="bolt://x", driver=_Driver(["a"]))
    arm = _VectorArm(uri="bolt://y", driver=_Driver([]))
    arm.name = "other"
    with pytest.raises(CrossTenantRecallError):
        Recaller(retrievers=[graph, arm]).recall("alice project")


# ---------- the resurrection stage ----------

_SEED_ROWS = [{"name": "charlie", "type": "Entity", "mc": "", "rel": "KNOWS"}]


def _walked(driver: _Driver) -> bool:
    return any(PROBE not in c for c in driver.calls)


def test_resurrection_skips_tenantless_walk_over_multi_tenant_graph(caplog):
    driver = _Driver(["a", "b"], _SEED_ROWS)
    stage = GraphResurrection(driver=driver, min_seed_len=3)
    seed = [RecallResult(id="1", text="x", score=1.0, payload={}, sources=["vector"])]
    with caplog.at_level(logging.WARNING):
        out = stage.apply(PipelineContext(query="alice bob"), seed)
    assert out == seed and not _walked(driver)
    assert "graph resurrection skipped" in caplog.text
    # The verdict is cached: a second recall inside the TTL does not re-probe.
    probes = _probe_runs(driver)
    stage.apply(PipelineContext(query="alice bob"), seed)
    assert _probe_runs(driver) == probes


class _Recalled:
    """A recaller as the stage sees it: the stores it read, its opt-out."""

    def __init__(self, tenants: list[Any], allow_cross_tenant: bool = False):
        self.allow_cross_tenant = allow_cross_tenant
        self._tenants = tenants

    def tenant_probe_sources(self):
        tenants = self._tenants

        class _Store:
            def tenant_sample(self):
                return list(tenants)

        return [_Store()]


@pytest.mark.parametrize(
    "tenants, extras",
    [
        (["a", "b"], {"tenant": "a"}),
        (["a", "b"], {"recaller": _Recalled([], allow_cross_tenant=True)}),
        (["a"], {}),
        (["a"], {"recaller": _Recalled(["a"])}),
    ],
)
def test_resurrection_walks_when_safe(tenants, extras):
    driver = _Driver(tenants, _SEED_ROWS)
    stage = GraphResurrection(driver=driver, min_seed_len=3)
    ctx = PipelineContext(query="alice bob")
    ctx.extras.update(extras)
    stage.apply(ctx, [])
    assert _walked(driver)


def test_resurrection_unions_the_graph_with_the_recalled_stores():
    # One tenant in the graph, another in the collection the recall read:
    # merging a tenantless walk into those results would mix tenants.
    driver = _Driver(["a"], _SEED_ROWS)
    stage = GraphResurrection(driver=driver, min_seed_len=3)
    ctx = PipelineContext(query="alice bob")
    ctx.extras["recaller"] = _Recalled(["b"])
    stage.apply(ctx, [])
    assert not _walked(driver)


def test_direct_pipeline_use_checks_the_results_tenants():
    # raw = recaller.recall(...); pipeline.apply(query, raw): no recaller in
    # context, but the results themselves carry tenant b.
    driver = _Driver(["a"], _SEED_ROWS)
    pipeline = Pipeline([GraphResurrection(driver=driver, min_seed_len=3)])
    raw = [RecallResult(id="1", text="x", score=1.0, payload={"tenant_id": "b"}, sources=["v"])]
    assert pipeline.apply("alice bob", raw) == raw
    assert not _walked(driver)


class _FlakyDriver(_Driver):
    """Fails the first ``fail`` sessions (graph not answering), then serves."""

    def __init__(self, tenants, rows=None, fail=1, exc=None):
        super().__init__(tenants, rows)
        self.fail = fail
        self.exc = exc or RuntimeError("graph not answering")

    def session(self, **kw):
        if self.fail:
            self.fail -= 1
            raise self.exc
        return super().session(**kw)


def test_undetermined_graph_verdict_is_not_cached_by_the_stage():
    driver = _FlakyDriver(["a", "b"], _SEED_ROWS)
    stage = GraphResurrection(driver=driver, min_seed_len=3)
    stage.apply(PipelineContext(query="alice bob"), [])  # probe fails: undetermined
    driver.calls.clear()
    stage.apply(PipelineContext(query="alice bob"), [])  # graph back: re-probed
    assert _probe_runs(driver) and not _walked(driver)


def test_a_graph_back_from_an_outage_is_not_read_before_its_own_probe(monkeypatch):
    # The verdict taken while the graph was down is undetermined (and cached,
    # like any verdict). The graph comes back inside that window: it must not
    # be searched tenantless until a probe of its own has concluded.
    import time

    import mnemostack.recall.retrievers as rmod

    monkeypatch.setattr(rmod, "GRAPH_UNAVAILABLE_COOLDOWN_S", 0.05)
    driver = _FlakyDriver(["a", "b"], fail=1, exc=OSError("connection refused"))
    recaller = Recaller(retrievers=[MemgraphRetriever(uri="bolt://x", driver=driver)])
    recaller.recall("alice project")  # graph unreachable: verdict undetermined
    time.sleep(0.1)  # cooldown over, graph back
    driver.calls.clear()
    recaller.recall("alice project")
    assert not driver.calls  # skipped, not read across tenants
    recaller._probe_at = -1e9  # verdict refresh: the graph's own probe runs
    with pytest.raises(CrossTenantRecallError):
        recaller.recall("alice project")


def test_another_probe_of_the_graph_does_not_open_it_under_an_undetermined_verdict():
    # Collection holds a, graph holds b; the cached verdict is undetermined
    # (the graph was down at refresh). Something else (the resurrection
    # stage, synthesize) then probes the graph conclusively: the arm must
    # still stay out, since only the verdict proves graph + collection.
    class _Collection:
        def tenant_sample(self):
            return ["a"]

    class _VectorArm(MemgraphRetriever):
        tenant_probe_store = _Collection()

        def search(self, *a, **k):
            return []

    driver = _Driver(["b"])
    graph = MemgraphRetriever(uri="bolt://x", driver=driver)
    arm = _VectorArm(uri="bolt://y", driver=_Driver([]))
    arm.name = "other"
    recaller = Recaller(retrievers=[graph, arm])
    recaller._multi_tenant_probe = "unknown"
    recaller._graph_gate = ("unknown", frozenset())
    recaller._probe_at = float("inf")  # fresh
    graph_tenant_sample(graph)  # a conclusive probe from elsewhere
    driver.calls.clear()
    recaller.recall("alice project")
    assert not driver.calls


def test_stage_does_not_walk_a_graph_it_could_not_check():
    # E.g. another recall holds the half-open claim: this one's probe gets
    # no answer, so it must not walk the graph that may be back by now.
    stage = GraphResurrection(driver=_Boom(), min_seed_len=3)
    ctx = PipelineContext(query="alice bob")
    ctx.extras["recaller"] = _Recalled(["a"])
    seed = [RecallResult(id="1", text="x", score=1.0, payload={}, sources=["v"])]
    assert stage.apply(ctx, seed) == seed


def test_synthesize_direct_arms_skip_an_unchecked_graph():
    from mnemostack.synthesis import _query_retrievers

    class _ProbeFails(_Driver):
        def session(self, **kw):
            session = super().session(**kw)
            run = session.run

            def _run(cypher, **params):
                if PROBE in cypher:
                    raise RuntimeError("probe rejected")
                return run(cypher, **params)

            session.run = _run
            return session

    driver = _ProbeFails(["a", "b"], _SEED_ROWS)
    retr = MemgraphRetriever(uri="bolt://x", driver=driver)
    assert _query_retrievers([retr], "alice", 5, None, None) == []
    assert not _walked(driver)  # the graph was never searched


def test_graph_that_cannot_answer_does_not_reprobe_the_collections_per_recall():
    asked: list[int] = []

    class _Collection:
        def tenant_sample(self):
            asked.append(1)
            return ["a"]

    class _Arm(MemgraphRetriever):
        tenant_probe_store = _Collection()

        def search(self, *a, **k):
            return []

    arm = _Arm(uri="bolt://y", driver=_Driver([]))
    arm.name = "other"
    graph = MemgraphRetriever(uri="bolt://x", driver=_Boom())  # answers nothing
    recaller = Recaller(retrievers=[graph, arm])
    for _ in range(3):
        recaller.recall("alice project")
    assert len(asked) == 1


def test_cross_tenant_tooling_reads_the_graph():
    from mnemostack.synthesis import _query_retrievers

    driver = _Driver(["a", "b"], _SEED_ROWS)
    retr = MemgraphRetriever(uri="bolt://x", driver=driver)
    Recaller(retrievers=[retr], allow_cross_tenant=True).recall("alice project")
    assert _walked(driver)
    driver.calls.clear()
    _query_retrievers([retr], "alice", 5, None, None, allow_cross_tenant=True)
    assert _walked(driver)


def test_a_healthy_single_tenant_graph_is_read():
    driver = _Driver(["a"], _SEED_ROWS)
    Recaller(retrievers=[MemgraphRetriever(uri="bolt://x", driver=driver)]).recall("alice project")
    assert _walked(driver)


def test_recall_flow_carries_the_wrapped_recallers_opt_out():
    from mnemostack.recall.expansion import QueryExpander
    from mnemostack.recall.flow import recall_flow

    graph = _Driver(["a", "b"], _SEED_ROWS)
    inner = Recaller(
        retrievers=[MemgraphRetriever(uri="bolt://x", driver=_Driver(["a", "b"]))],
        allow_cross_tenant=True,
    )
    pipeline = Pipeline([GraphResurrection(driver=graph, min_seed_len=3)], stop_on_empty=False)
    recall_flow(QueryExpander(inner, llm=None), "alice bob", pipeline=pipeline)
    assert _walked(graph)


# ---------- SDK writer ----------


def test_extract_and_store_writes_into_the_tenant_subgraph():
    calls: list[dict] = []

    class _Graph:
        def add_triple(self, **kw):
            calls.append(kw)

    ex = TripleExtractor.__new__(TripleExtractor)
    ex.extract = lambda _text: [Triple(subject="a", predicate="uses", obj="b")]
    ex.extract_and_store("text", _Graph(), tenant="t1")
    ex.extract_and_store("text", _Graph())
    assert calls[0]["tenant"] == "t1" and "tenant" not in calls[1]


def test_recall_flow_accepts_a_spec_mocked_expander():
    from unittest.mock import MagicMock

    from mnemostack.recall.expansion import QueryExpander
    from mnemostack.recall.flow import recall_flow

    expander = MagicMock(spec=QueryExpander)  # isinstance holds, no .recaller
    expander.recall.return_value = []
    assert recall_flow(expander, "q", pipeline=Pipeline([])) == []


def test_a_missing_collection_holds_no_tenant():
    from qdrant_client import QdrantClient

    from mnemostack.vector.qdrant import tenant_sample

    assert tenant_sample(QdrantClient(":memory:"), "not_ingested_yet") == []


def test_a_store_that_cannot_answer_keeps_the_graph_out():
    class _Unknown:
        def tenant_sample(self):
            return None

    class _Arm(MemgraphRetriever):
        tenant_probe_store = _Unknown()

        def search(self, *a, **k):
            return []

    driver = _Driver(["a"], _SEED_ROWS)
    arm = _Arm(uri="bolt://y", driver=_Driver([]))
    arm.name = "other"
    Recaller(retrievers=[MemgraphRetriever(uri="bolt://x", driver=driver), arm]).recall(
        "alice project"
    )
    assert not _walked(driver)


def test_synthesize_direct_arms_read_a_single_tenant_graph():
    from mnemostack.synthesis import _query_retrievers

    driver = _Driver(["a"], _SEED_ROWS)
    _query_retrievers([MemgraphRetriever(uri="bolt://x", driver=driver)], "alice", 5, None, None)
    assert _walked(driver)
