"""Tests for score-based reranking."""

import math

import pytest

from mnemostack.recall import RecallResult, ScoringReranker


class FakeScorer:
    def __init__(self, scores=None, error: Exception | None = None):
        self.scores = scores if scores is not None else []
        self.error = error
        self.calls = []

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, documents))
        if self.error is not None:
            raise self.error
        return list(self.scores)


class RawScorer:
    def __init__(self, scores):
        self.scores = scores

    def score(self, query: str, documents: list[str]):
        return self.scores


@pytest.fixture
def sample_results():
    return [
        RecallResult(id="a", text="first", score=0.9, payload={}),
        RecallResult(id="b", text="second", score=0.8, payload={}),
        RecallResult(id="c", text="third", score=0.7, payload={}),
        RecallResult(id="d", text="fourth", score=0.6, payload={}),
    ]


def test_scoring_reranker_orders_by_descending_score(sample_results):
    reranker = ScoringReranker(FakeScorer([0.1, 0.9, 0.4, 0.2]))

    out = reranker.rerank("query", sample_results)

    assert [r.id for r in out] == ["b", "c", "d", "a"]


def test_scoring_reranker_respects_max_items_and_keeps_tail(sample_results):
    scorer = FakeScorer([0.1, 0.8])
    reranker = ScoringReranker(scorer, max_items=2)

    out = reranker.rerank("query", sample_results)

    assert [r.id for r in out] == ["b", "a", "c", "d"]
    assert scorer.calls == [("query", ["first", "second"])]


def test_scoring_reranker_ties_keep_input_order(sample_results):
    reranker = ScoringReranker(FakeScorer([0.5, 0.5, 0.4, 0.4]))

    out = reranker.rerank("query", sample_results)

    assert [r.id for r in out] == ["a", "b", "c", "d"]


def test_scoring_reranker_fail_open_on_scorer_error(sample_results):
    reranker = ScoringReranker(FakeScorer(error=RuntimeError("down")))

    out = reranker.rerank("query", sample_results)

    assert out is sample_results


def test_scoring_reranker_fail_open_on_wrong_score_count(sample_results):
    reranker = ScoringReranker(FakeScorer([0.1]))

    out = reranker.rerank("query", sample_results)

    assert out is sample_results


def test_scoring_reranker_fail_open_on_non_numeric_scores(sample_results):
    reranker = ScoringReranker(FakeScorer(["not-a-number", 0.2, 0.3, 0.4]))

    out = reranker.rerank("query", sample_results)

    assert out is sample_results


def test_scoring_reranker_fail_open_on_none_scores(sample_results):
    reranker = ScoringReranker(RawScorer(None))

    out = reranker.rerank("query", sample_results)

    assert out is sample_results


def test_scoring_reranker_accepts_generator_scores(sample_results):
    reranker = ScoringReranker(RawScorer(score for score in [0.1, 0.9, 0.4, 0.2]))

    out = reranker.rerank("query", sample_results)

    assert [r.id for r in out] == ["b", "c", "d", "a"]
    assert out is not sample_results  # success returns a new list, not the input


def test_scoring_reranker_treats_non_finite_scores_as_lowest(sample_results):
    reranker = ScoringReranker(FakeScorer([math.nan, 0.2, math.inf, -math.inf]))

    out = reranker.rerank("query", sample_results)

    assert [r.id for r in out] == ["b", "a", "c", "d"]


def test_scoring_reranker_fallback_returns_input_object(sample_results):
    # The fallback signal is identity, not state: a kept-order fallback returns
    # the exact input list, which apply_rerank_safe detects as the degradation.
    reranker = ScoringReranker(FakeScorer([0.1]))

    out = reranker.rerank("query", sample_results)

    assert out is sample_results


def test_scoring_reranker_empty_input_does_not_call_scorer():
    scorer = FakeScorer([1.0])
    reranker = ScoringReranker(scorer)

    assert reranker.rerank("query", []) == []
    assert scorer.calls == []


def test_scoring_reranker_validates_max_items():
    with pytest.raises(ValueError, match="max_items"):
        ScoringReranker(FakeScorer(), max_items=0)
