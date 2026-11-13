"""Query selection and provider failure must not widen or abort recall."""
from types import SimpleNamespace

import pytest

from mnemostack.recall import QueryExpander, RecallResult


@pytest.mark.parametrize("include_original", [False, True])
@pytest.mark.parametrize("failure", ["raise", "status"])
def test_expansion_failure_falls_back_once_with_same_scope(include_original, failure):
    calls = []
    generated = []

    def generate(*args, **kwargs):
        generated.append(1)
        if failure == "raise":
            raise RuntimeError("synthetic provider fault")
        return SimpleNamespace(ok=False, text="", error="synthetic fault")

    def recall(query, **kwargs):
        calls.append((query, kwargs))
        return [RecallResult(id="item", text="synthetic evidence", score=0.5)]

    expander = QueryExpander(SimpleNamespace(recall=recall), SimpleNamespace(generate=generate), include_original=include_original, apply_to=lambda _: True)
    rows = expander.recall("original", limit=3, filters={"project": "p"}, tenant="alpha", include_invalidated=True, as_of="2026-01-01")
    assert [r.id for r in rows] == ["item"]
    assert len(generated) == len(calls) == 1
    assert calls[0][0] == "original"
    assert calls[0][1]["tenant"] == "alpha"
    assert calls[0][1]["filters"] == {"project": "p"}
    assert calls[0][1]["include_invalidated"] is True
    assert calls[0][1]["as_of"] == "2026-01-01"


def test_single_paraphrase_does_not_silently_restore_excluded_original():
    calls = []

    def recall(query, **kwargs):
        calls.append(query)
        return [RecallResult(id="item", text="synthetic evidence", score=0.5)]

    expander = QueryExpander(SimpleNamespace(recall=recall), SimpleNamespace(generate=lambda *a, **k: SimpleNamespace(ok=True, text="only variant")), include_original=False, apply_to=lambda _: True)
    assert len(expander.recall("original")) == 1
    assert calls == ["only variant"]


def test_single_paraphrase_keeps_expansion_trace_identity():
    class Trace:
        def __init__(self):
            self.retrievers, self.degraded, self.notes = [], [], []
            self.fused = []

    def recall(query, *, trace, **kwargs):
        trace.retrievers.append(SimpleNamespace(name="fake", query=query))
        return [RecallResult(id="item", text="synthetic evidence", score=0.5)]

    trace = Trace()
    expander = QueryExpander(SimpleNamespace(recall=recall), SimpleNamespace(generate=lambda *a, **k: SimpleNamespace(ok=True, text="only variant")), include_original=False, apply_to=lambda _: True)
    expander.recall("original", trace=trace)
    assert [(r.name, r.query) for r in trace.retrievers] == [("fake:expansion", "only variant")]


def test_non_matching_query_does_not_invoke_expansion():
    def generate(*args, **kwargs):
        pytest.fail("expansion must remain optional")

    calls = []
    expander = QueryExpander(SimpleNamespace(recall=lambda query, **kw: calls.append(query) or []), SimpleNamespace(generate=generate), apply_to=lambda _: False)
    assert expander.recall("original") == []
    assert calls == ["original"]
