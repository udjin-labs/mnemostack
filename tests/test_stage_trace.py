"""Per-stage recall trace and loss report (#200).

The trace records the order after every ranking-pipeline stage (ids and
scores only), never changes a result, records only results inside the
caller's tenant / filters scope, and `loss_report` tells where each expected id sat
and at which checkpoint it crossed a top-k boundary.
"""

from __future__ import annotations

from mnemostack.recall import RecallResult
from mnemostack.recall.flow import recall_flow
from mnemostack.recall.pipeline.base import Pipeline, PipelineContext, Stage
from mnemostack.recall.trace import RecallTrace, RetrieverTrace, StageTrace


def _r(rid: str, score: float, tenant: str | None = None) -> RecallResult:
    payload = {"tenant_id": tenant} if tenant is not None else {}
    return RecallResult(id=rid, text=rid, score=score, payload=payload, sources=["vector"])


class _Reverse(Stage):
    def apply(self, context: PipelineContext, results):
        return list(reversed(results))


class _DropFirst(Stage):
    def apply(self, context: PipelineContext, results):
        return results[1:]


class _Inject(Stage):
    """Appends a result, as graph resurrection does."""

    def __init__(self, result: RecallResult):
        self.result = result

    def apply(self, context: PipelineContext, results):
        return [*results, self.result]


def _results():
    return [_r("a", 0.9), _r("b", 0.8), _r("c", 0.7)]


# ---------- recording ----------


def test_pipeline_records_the_order_after_every_stage():
    trace = RecallTrace()
    out = Pipeline([_Reverse(), _DropFirst()]).apply("q", _results(), trace=trace)
    assert [s.name for s in trace.stages] == ["_Reverse", "_DropFirst"]
    assert [rid for rid, _ in trace.stages[0].ranked] == ["c", "b", "a"]
    assert trace.stages[1].ranked == [("b", 0.8), ("a", 0.9)]
    assert [r.id for r in out] == ["b", "a"]


def test_results_are_identical_with_and_without_a_trace():
    pipeline = Pipeline([_Reverse(), _DropFirst()])
    plain = pipeline.apply("q", _results())
    traced = pipeline.apply("q", _results(), trace=RecallTrace())
    assert [(r.id, r.score) for r in plain] == [(r.id, r.score) for r in traced]


def test_a_stage_skipped_on_empty_results_leaves_no_snapshot():
    trace = RecallTrace()
    Pipeline([_DropFirst(), _DropFirst(), _DropFirst(), _Reverse()]).apply(
        "q", _results(), trace=trace
    )
    assert [s.name for s in trace.stages] == ["_DropFirst"] * 3


def test_to_dict_carries_stages_only_when_a_pipeline_ran():
    assert "stages" not in RecallTrace().to_dict()
    trace = RecallTrace(stages=[StageTrace("s", [("a", 0.1234567)])])
    assert trace.to_dict()["stages"] == [{"name": "s", "ranked": [["a", 0.123457]]}]


def test_recall_flow_records_stages_and_scrubs_them_by_tenant():
    class _Recaller:
        def recall(self, query, **kw):
            return [_r("a", 0.9, "t1"), _r("b", 0.8, "t1")]

    trace = RecallTrace()
    foreign = _r("x", 0.5, "t2")
    out = recall_flow(
        _Recaller(),
        "q",
        pipeline=Pipeline([_Reverse(), _Inject(foreign)]),
        trace=trace,
        tenant="t1",
    )
    assert [r.id for r in out] == ["b", "a"]
    assert [s.name for s in trace.stages] == ["_Reverse", "_Inject"]
    # the injected foreign-tenant id never reaches the trace
    assert all(rid != "x" for s in trace.stages for rid, _ in s.ranked)
    assert [rid for rid, _ in trace.stages[1].ranked] == ["b", "a"]


# ---------- loss report ----------


def _trace() -> RecallTrace:
    return RecallTrace(
        retrievers=[
            RetrieverTrace("vector", [("a", 0.9), ("b", 0.8)]),
            RetrieverTrace("bm25", [("c", 3.0)]),
        ],
        fused=[("a", 0.3), ("b", 0.2), ("c", 0.1)],
        stages=[
            StageTrace("freshness", [("c", 0.5), ("a", 0.4), ("b", 0.3)]),
            StageTrace("dampen", [("c", 0.5), ("a", 0.4)]),
        ],
        post_rerank=[("a", 0.9), ("c", 0.8)],
    )


def test_checkpoints_are_in_pipeline_order_and_disambiguated():
    trace = _trace()
    trace.stages.append(StageTrace("dampen", []))
    labels = [label for label, _ in trace.checkpoints()]
    assert labels == [
        "fused",
        "stage:freshness",
        "stage:dampen",
        "stage:dampen#2",
        "post_rerank",
    ]


def test_loss_report_positions_per_retriever_and_checkpoint():
    rep = _trace().loss_report(["b", "c"], cutoffs=(1, 2))
    assert rep["positions"]["b"] == {
        "retriever:vector": 2,
        "retriever:bm25": None,
        "fused": 2,
        "stage:freshness": 3,
        "stage:dampen": None,
        "post_rerank": None,
    }
    assert rep["positions"]["c"]["retriever:bm25"] == 1
    assert rep["positions"]["c"]["post_rerank"] == 2


def test_loss_report_gains_and_losses_at_each_boundary():
    rep = _trace().loss_report(["b", "c", "a"], cutoffs=(1, 2))
    losses = {(e["id"], e["cutoff"], e["from"], e["to"]) for e in rep["losses"]}
    gains = {(e["id"], e["cutoff"], e["from"], e["to"]) for e in rep["gains"]}
    # b: 2 in fused, 3 after freshness -> out of top-2 there
    assert ("b", 2, "fused", "stage:freshness") in losses
    # c: 3 in fused, 1 after freshness -> into top-1 and top-2
    assert ("c", 1, "fused", "stage:freshness") in gains
    assert ("c", 2, "fused", "stage:freshness") in gains
    # the reranker moves a back to 1 and c to 2
    assert ("a", 1, "stage:dampen", "post_rerank") in gains
    assert ("c", 1, "stage:dampen", "post_rerank") in losses
    entry = next(e for e in rep["losses"] if e["id"] == "b" and e["cutoff"] == 2)
    assert (entry["from_pos"], entry["to_pos"]) == (2, 3)


def test_loss_report_ids_are_compared_as_strings():
    trace = RecallTrace(fused=[("7", 0.1)])
    assert trace.loss_report([7])["positions"]["7"]["fused"] == 1


def test_recall_flow_scrubs_stage_snapshots_by_filter_without_a_tenant():
    class _Recaller:
        def recall(self, query, **kw):
            return [
                RecallResult(id="u1", text="t", score=0.9, payload={"user": "alice"}, sources=["v"])
            ]

    trace = RecallTrace()
    injected = RecallResult(
        id="graph:Other Project", text="t", score=0.2, payload={}, sources=["g"]
    )
    out = recall_flow(
        _Recaller(),
        "q",
        pipeline=Pipeline([_Inject(injected)]),
        trace=trace,
        filters={"user": "alice"},
    )
    assert [r.id for r in out] == ["u1"]
    assert [rid for rid, _ in trace.stages[0].ranked] == ["u1"]


def test_checkpoints_after_a_weak_retry_keep_the_first_pass_in_order():
    from mnemostack.recall.retry import _retrace

    trace = _trace()
    _retrace(trace, [_r("c", 1.0), _r("z", 0.5)])
    labels = [label for label, _ in trace.checkpoints()]
    assert labels[0] == "fused" and labels[-1] == "weak_retry_merge"
    assert trace.checkpoints()[0][1] == ["a", "b", "c"]  # the first pass
    assert trace.checkpoints()[-1][1] == ["c", "z"]  # what was returned
    assert trace.to_dict()["first_pass_fused"][0][0] == "a"


def test_stage_trace_is_exported():
    from mnemostack.recall import StageTrace as Exported

    assert Exported is StageTrace


class _DropId(Stage):
    def __init__(self, rid: str):
        self.rid = rid

    def apply(self, context: PipelineContext, results):
        return [r for r in results if r.id != self.rid]


def test_an_in_scope_record_a_later_stage_drops_still_shows_as_a_loss():
    class _Recaller:
        def recall(self, query, **kw):
            return [_r("a", 0.9, "t1")]

    trace = RecallTrace()
    recall_flow(
        _Recaller(),
        "q",
        pipeline=Pipeline([_Inject(_r("g", 0.5, "t1")), _DropId("g")]),
        trace=trace,
        tenant="t1",
    )
    assert [rid for rid, _ in trace.stages[0].ranked] == ["a", "g"]
    rep = trace.loss_report(["g"], cutoffs=(5,))
    assert rep["losses"][0]["to"] == "stage:_DropId"


def test_a_filter_scope_leaves_the_pre_pipeline_trace_untouched():
    class _Recaller:
        def recall(self, query, trace=None, **kw):
            # a retriever found "far", fusion cut it: a fusion-stage loss
            trace.retrievers.append(RetrieverTrace("vector", [("u1", 0.9), ("far", 0.1)]))
            trace.fused = [("u1", 0.9)]
            return [
                RecallResult(id="u1", text="t", score=0.9, payload={"user": "alice"}, sources=["v"])
            ]

    trace = RecallTrace()
    recall_flow(
        _Recaller(), "q", pipeline=Pipeline([_Reverse()]), trace=trace, filters={"user": "alice"}
    )
    assert [rid for rid, _ in trace.retrievers[0].ranked] == ["u1", "far"]


def test_an_untraced_flow_keeps_an_older_pipeline_override_working():
    class _OldPipeline(Pipeline):
        def apply(
            self,
            query,
            results,
            *,
            as_of=None,
            include_invalidated=False,
            tenant=None,
            recaller=None,
        ):
            return results

    class _Recaller:
        def recall(self, query, **kw):
            return [_r("a", 0.9)]

    out = recall_flow(_Recaller(), "q", pipeline=_OldPipeline([]))
    assert [r.id for r in out] == ["a"]
