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
        def name(self): return "fail"
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
    llm = FakeLLM("Where was Caroline previously based?\nWhat country did Caroline come from?\nCaroline Sweden origin")

    variants = expand_query("Where did Caroline move from?", llm, n_variants=3)

    assert variants == [
        "Where was Caroline previously based?",
        "What country did Caroline come from?",
        "Caroline Sweden origin",
    ]
    assert llm.calls == 1


def test_recaller_with_expansion_produces_superset_and_uses_cache():
    retriever = QueryAwareRetriever()
    llm = FakeLLM("Caroline Sweden origin\nWhere was Caroline previously based?\nWhat country did Caroline come from?")
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
