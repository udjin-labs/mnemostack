"""Budget diagnostics must describe the cut without changing recall or scope."""

from types import SimpleNamespace

import pytest

from mnemostack.recall import Recaller, RecallResult, RecallTrace, recall_flow, recall_flow_async
from mnemostack.recall.pipeline.base import Pipeline, Stage
from mnemostack.recall.retrievers import Retriever
from mnemostack.recall.retry import retry_weak_recall


def hit(rid, tenant="one", **payload):
    return RecallResult(
        id=rid,
        text=rid,
        score=1.0,
        sources=["fixture"],
        payload={"tenant_id": tenant, **payload},
    )


class Fixture(Retriever):
    name = "fixture"
    accepts_tenant = True

    def search(self, query, limit=20, filters=None, tenant=None):
        return [hit("a"), hit("b"), hit("foreign", tenant="two")][:limit]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["recall", "recall_async", "flow", "flow_async"])
@pytest.mark.parametrize("budget,expected", [(None, ["a", "b"]), (2, ["a", "b"]), (1, ["a"])])
async def test_real_recall_budget_trace(mode, budget, expected):
    recaller = Recaller(retrievers=[Fixture()])
    trace = RecallTrace()
    calls = []

    def count(text):
        calls.append(text)
        return 1

    kw = dict(limit=3, tenant="one", trace=trace, token_budget=budget, token_counter=count)
    if mode == "recall":
        out = recaller.recall("q", **kw)
    elif mode == "recall_async":
        out = await recaller.recall_async("q", **kw)
    elif mode == "flow":
        out = recall_flow(recaller, "q", **kw)
    else:
        out = await recall_flow_async(recaller, "q", **kw)
    assert [r.id for r in out] == expected
    assert "foreign" not in str(trace.to_dict())
    assert calls == ([] if budget is None else ["a", "b"])
    if budget is None:
        assert "budget" not in trace.to_dict()
    else:
        assert trace.budget is not None
        assert [rid for rid, _ in trace.budget.before] == ["a", "b"]
        assert [rid for rid, _ in trace.budget.after] == expected
        assert trace.budget.used == len(expected)
        losses = trace.loss_report(["b"], cutoffs=(2,))["losses"]
        assert bool(losses) == (budget == 1)
        if losses:
            assert (losses[0]["from"], losses[0]["to"]) == ("pre_budget", "post_budget")


class Identity(Stage):
    def apply(self, context, results):
        return results


def test_top_k_loss_is_not_reported_as_budget_loss():
    recaller = SimpleNamespace(recall=lambda *a, **k: [hit("a"), hit("b"), hit("c")])
    trace = RecallTrace()
    out = recall_flow(
        recaller,
        "q",
        2,
        pipeline=Pipeline([Identity()]),
        trace=trace,
        token_budget=1,
        token_counter=lambda _: 1,
    )
    assert [r.id for r in out] == ["a"]
    losses = trace.loss_report(["b", "c"], cutoffs=(3,))["losses"]
    assert [(x["id"], x["from"], x["to"]) for x in losses] == [
        ("b", "pre_budget", "post_budget"),
        ("c", "stage:Identity", "pre_budget"),
    ]


@pytest.mark.parametrize(
    "foreign",
    [hit("x", tenant="two"), hit("x", tag="wrong"), SimpleNamespace(id="x", score=1.0, text="x")],
)
def test_budget_snapshots_do_not_expose_injected_out_of_scope_records(foreign):
    a = hit("a", tag="ok")
    recaller = SimpleNamespace(default_tenant="one", recall=lambda *a, **k: [hit("a", tag="ok")])
    reranker = SimpleNamespace(rerank=lambda *args: [a, foreign])
    trace = RecallTrace()
    out = recall_flow(
        recaller,
        "q",
        trace=trace,
        reranker=reranker,
        filters={"tag": "ok"},
        token_budget=1,
        token_counter=lambda _: 1,
    )
    assert [r.id for r in out] == ["a"]
    assert trace.budget.before == [("a", 1.0)]
    assert trace.budget.after == [("a", 1.0)]
    assert trace.budget.used is None


def test_scrub_covers_both_budget_snapshots():
    trace = RecallTrace()
    for retry in (False, True):
        trace.record_budget([hit("a"), hit("b")], [hit("a")], 1, 1, recaller=None, retry=retry)
    trace.restrict_to_ids(["b"])
    for key in ("budget", "retry_budget"):
        assert trace.to_dict()[key] == {
            "limit": 1,
            "used": None,
            "before": [["b", 1.0]],
            "after": [],
        }


class LLM:
    def generate(self, *args, **kwargs):
        return SimpleNamespace(ok=True, text="variant one\nvariant two", tokens_used=0)


@pytest.mark.parametrize("reject", [False, True])
def test_retry_keeps_first_budget_and_records_only_accepted_merge(reject):
    a, b = hit("a"), hit("b")
    trace = RecallTrace(fused=[("a", 1.0), ("b", 1.0)])
    trace.record_budget([a, b], [a], 3, 1, recaller=None)
    original = trace.budget.to_dict()
    # Corroborated b wins the merge. It fits each variant but not the merge
    # when this deliberately stateful counter makes its third count larger.
    counts = {"b": 0}

    def count(text):
        if text == "b":
            counts["b"] += 1
            return 4 if reject and counts["b"] >= 3 else 2
        return 1

    recaller = SimpleNamespace(recall=lambda *a, **k: [hit("b")])
    out, retried = retry_weak_recall(
        recaller,
        "q",
        3,
        llm=LLM(),
        results=[a],
        below=2,
        trace=trace,
        token_budget=3,
        token_counter=count,
    )
    assert retried
    assert trace.budget.to_dict() == original
    if reject:
        assert out == [a]
        assert trace.retry_budget is None
        assert trace.first_pass_fused is None
    else:
        assert [r.id for r in out] == ["b", "a"]
        assert trace.retry_budget.used == 3
        assert trace.checkpoints()[-1] == ("weak_retry_merge", ["b", "a"])
        report = trace.loss_report(["b"], cutoffs=(2,))
        assert [(x["from"], x["to"]) for x in report["losses"]] == [("pre_budget", "post_budget")]
        assert [(x["from"], x["to"]) for x in report["gains"]] == [
            ("post_budget", "retry_pre_budget")
        ]
