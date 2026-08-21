"""Tests for query expansion helpers and Recaller integration."""

from __future__ import annotations

from unittest.mock import MagicMock

from mnemostack.llm.base import LLMProvider, LLMResponse
from mnemostack.recall import QueryExpander, Recaller, RecallResult, expand_query


class FakeLLM(LLMProvider):
    def __init__(self, text: str = "variant one\nvariant two\nvariant three\nvariant four"):
        self._text = text
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake"

    def generate(self, prompt, max_tokens=200, temperature=0.0):
        self.calls += 1
        return LLMResponse(text=self._text, tokens_used=10)


def _rr(id_, text, score):
    return RecallResult(id=id_, text=text, score=score, payload={"text": text}, sources=["vector"])


def test_query_expansion_generates_variants():
    llm = FakeLLM("what books did Tim read?\nwhich novels has Tim finished?\nTim reading history")
    recaller = MagicMock()
    recaller.recall.return_value = [_rr(1, "a", 0.9)]
    qe = QueryExpander(recaller=recaller, llm=llm, n_variants=3)
    variants = qe.generate_variants("what books has Tim read?")
    assert len(variants) == 3
    assert "which novels has Tim finished?" in variants


def test_query_expansion_skips_non_list_queries():
    llm = FakeLLM("ignored")
    recaller = MagicMock()
    recaller.recall.return_value = [_rr(1, "one", 0.5)]
    qe = QueryExpander(recaller=recaller, llm=llm)
    # Not a list-like question — should fall back to plain recall
    qe.recall("When did Tim leave?", limit=5)
    recaller.recall.assert_called_once()
    # LLM should NOT be called for paraphrases


def test_query_expansion_merges_with_rrf():
    llm = FakeLLM("what X has Y done?\nwhich things did Y like?")

    # Build recaller that returns different results per query
    def _recall(query, **kwargs):
        if "original" in query.lower():
            return [_rr(1, "alpha", 0.9), _rr(2, "beta", 0.8)]
        if "what x" in query.lower():
            return [_rr(2, "beta", 0.7), _rr(3, "gamma", 0.6)]
        if "which things" in query.lower():
            return [_rr(3, "gamma", 0.7), _rr(1, "alpha", 0.5)]
        return []

    recaller = MagicMock()
    recaller.recall.side_effect = _recall

    qe = QueryExpander(recaller=recaller, llm=llm, n_variants=2)
    results = qe.recall("What kinds of things did X original collect?", limit=10)
    assert len(results) == 3
    # All of alpha/beta/gamma appear across the fused lists
    ids = [r.id for r in results]
    assert set(ids) == {1, 2, 3}


def test_query_expansion_falls_back_when_llm_fails():
    class FailingLLM(LLMProvider):
        @property
        def name(self):
            return "fail"

        def generate(self, prompt, max_tokens=200, temperature=0.0):
            return LLMResponse(text="", error="boom")

    recaller = MagicMock()
    recaller.recall.return_value = [_rr(1, "x", 0.5)]
    qe = QueryExpander(recaller=recaller, llm=FailingLLM())
    res = qe.recall("what are X hobbies?", limit=5)
    # With no variants generated, it should fall back to plain recall
    assert len(res) == 1
    assert res[0].id == 1


def test_query_expansion_deduplicates_against_original():
    # LLM echoes original — should be filtered out
    llm = FakeLLM("what books has Tim read?\ndifferent phrasing")
    recaller = MagicMock()
    qe = QueryExpander(recaller=recaller, llm=llm, n_variants=3)
    variants = qe.generate_variants("What books has Tim read?")
    lowered = [v.lower() for v in variants]
    assert "what books has tim read?" not in lowered
    assert any("different" in v for v in variants)


class QueryAwareRetriever:
    name = "fake"

    def __init__(self):
        self.calls = []

    def search(self, query, limit=20, filters=None):
        self.calls.append(query)
        if query == "Where did Caroline move from?":
            return [RecallResult(id="a", text="Caroline lived abroad", score=1.0, sources=["fake"])]
        if "Sweden" in query:
            return [
                RecallResult(id="b", text="Caroline is from Sweden", score=1.0, sources=["fake"]),
                RecallResult(id="a", text="Caroline lived abroad", score=0.9, sources=["fake"]),
            ]
        return []


def test_expand_query_returns_valid_variants():
    llm = FakeLLM(
        "Where was Caroline previously based?\nWhat country did Caroline come from?\nCaroline Sweden origin"
    )

    variants = expand_query("Where did Caroline move from?", llm, n_variants=3)

    assert variants == [
        "Where was Caroline previously based?",
        "What country did Caroline come from?",
        "Caroline Sweden origin",
    ]
    assert llm.calls == 1


def test_recaller_with_expansion_produces_superset_and_uses_cache():
    retriever = QueryAwareRetriever()
    llm = FakeLLM(
        "Caroline Sweden origin\nWhere was Caroline previously based?\nWhat country did Caroline come from?"
    )
    expanded = Recaller(retrievers=[retriever], query_expansion=True, expansion_llm=llm)
    baseline = Recaller(retrievers=[QueryAwareRetriever()])

    baseline_ids = {r.id for r in baseline.recall("Where did Caroline move from?", limit=10)}
    expanded_ids = {r.id for r in expanded.recall("Where did Caroline move from?", limit=10)}
    expanded.recall("Where did Caroline move from?", limit=10)

    assert baseline_ids <= expanded_ids
    assert "b" in expanded_ids
    assert llm.calls == 1


def test_recaller_query_expansion_deduplicates_by_chunk_id():
    retriever = QueryAwareRetriever()
    llm = FakeLLM("Caroline Sweden origin")
    recaller = Recaller(retrievers=[retriever], query_expansion=True, expansion_llm=llm)

    results = recaller.recall("Where did Caroline move from?", limit=10)

    ids = [r.id for r in results]
    assert ids.count("a") == 1
    assert ids.count("b") == 1


class _ScopeRecordingRecaller:
    """A recaller that answers, and remembers the scope it was asked with."""

    def __init__(self):
        self.calls: list[dict] = []

    def recall(self, query, limit=10, vector_limit=20, bm25_limit=20, filters=None, **kw):
        self.calls.append({"query": query, **kw})
        # Tenant-scoped stores return only that tenant's rows; an unscoped
        # read is what leaks, so answer accordingly.
        if kw.get("tenant") == "acme":
            return [_rr("acme-1", "acme memory", 0.9)]
        return [_rr("acme-1", "acme memory", 0.9), _rr("other-1", "another tenant", 0.8)]


def test_expansion_carries_the_callers_scope_into_every_variant():
    """#166: `QueryExpander` is a public export whose `recall()` dropped
    `tenant`, `include_invalidated` and `as_of` on the floor. The failure
    was silent — a multi-tenant caller got an unscoped read with no error —
    and it applied to the paraphrases as much as to the original query. A
    paraphrase is the same question asked differently, not a wider one."""
    recaller = _ScopeRecordingRecaller()
    expander = QueryExpander(recaller=recaller, llm=FakeLLM("a\nb"), n_variants=2)

    # A query the default `apply_to` actually expands — otherwise this
    # falls back to a single plain recall and proves nothing about the
    # paraphrases, which is exactly what it did on the first attempt.
    out = expander.recall("what are the options", tenant="acme", as_of="2026-01-01")

    assert len(recaller.calls) == 3, [c["query"] for c in recaller.calls]
    for call in recaller.calls:
        assert call["tenant"] == "acme", call
        assert call["as_of"] == "2026-01-01", call
        assert call["include_invalidated"] is False, call
    assert [r.id for r in out] == ["acme-1"]  # nothing from another tenant


def test_expansion_folds_each_paraphrases_trace_into_the_callers():
    """#166 also left the expansion invisible: a variant recall that
    degraded said nothing on the caller's trace. Each paraphrase records
    into its own trace — sharing one would leave `fused` describing
    whichever pass ran last — and is folded back labelled, so the extra
    retrieval an operator paid for is the extra retrieval they can see."""
    from mnemostack.recall.trace import RecallTrace, RetrieverTrace

    class _DegradingRecaller:
        def recall(self, query, limit=10, vector_limit=20, bm25_limit=20, filters=None, **kw):
            trace = kw.get("trace")
            if trace is not None:
                trace.retrievers.append(RetrieverTrace(name="vector"))
                if query != "what are the options":
                    trace.mark("bm25:down")
            return [_rr("m1", "a memory", 0.9)]

    trace = RecallTrace()
    expander = QueryExpander(recaller=_DegradingRecaller(), llm=FakeLLM("a\nb"), n_variants=2)
    expander.recall("what are the options", trace=trace)

    names = [e.name for e in trace.retrievers]
    assert names.count("vector") == 1  # the caller's own query, unlabelled
    assert names.count("vector:expansion") == 2  # ...and one per paraphrase
    assert [e.query for e in trace.retrievers if e.name == "vector:expansion"] == ["a", "b"]
    assert "bm25:down" in trace.degraded  # a fault in a paraphrase is a fault


def test_the_scope_survives_both_ways_out_of_the_expansion():
    """The two fallbacks are the COMMON paths — `apply_to` rejects anything
    that is not a list-like question, and a model that returns no usable
    paraphrase falls back too. Scoping them only on the expansion branch
    would leave the leak open for ordinary queries, which is most of them."""
    not_a_list_question = _ScopeRecordingRecaller()
    QueryExpander(recaller=not_a_list_question, llm=FakeLLM("a\nb")).recall(
        "what did we decide about auth", tenant="acme", as_of="2026-01-01"
    )
    assert len(not_a_list_question.calls) == 1  # `apply_to` said no
    assert not_a_list_question.calls[0]["tenant"] == "acme"
    assert not_a_list_question.calls[0]["as_of"] == "2026-01-01"

    no_variants = _ScopeRecordingRecaller()
    QueryExpander(recaller=no_variants, llm=FakeLLM("   \n  ")).recall(
        "what are the options", tenant="acme", include_invalidated=True
    )
    assert len(no_variants.calls) == 1  # the model offered nothing to ask
    assert no_variants.calls[0]["tenant"] == "acme"
    assert no_variants.calls[0]["include_invalidated"] is True


def test_the_trace_describes_the_merge_the_caller_receives():
    """codex P2 on the #166 fix: passing the caller's trace to the original
    query's pass leaves `fused` describing THAT pass — or empty, with
    `include_original=False` — while the caller receives a cross-query
    merge no single pass produced. `fused` is documented as the order
    recall returned."""
    from mnemostack.recall.trace import RecallTrace

    class _PerQueryRecaller:
        def recall(self, query, limit=10, vector_limit=20, bm25_limit=20, filters=None, **kw):
            trace = kw.get("trace")
            hit = {"what are the options": "orig", "a": "va", "b": "vb"}[query]
            if trace is not None:
                trace.fused = [(hit, 0.5)]
            return [_rr(hit, f"{hit} memory", 0.9)]

    for include_original in (True, False):
        trace = RecallTrace()
        expander = QueryExpander(
            recaller=_PerQueryRecaller(),
            llm=FakeLLM("a\nb"),
            n_variants=2,
            include_original=include_original,
        )
        out = expander.recall("what are the options", trace=trace)
        assert out, include_original
        assert trace.fused == [(str(r.id), r.score) for r in out], include_original
