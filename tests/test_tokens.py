"""Token estimation + token-budget trimming, and their wiring into recall.

No Qdrant here — fake retrievers/recallers keep the tests hermetic. The
budget contract under test: hard cap, ranked-prefix trim, never overshoot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from mnemostack.recall import recall_flow
from mnemostack.recall.recaller import Recaller, RecallResult
from mnemostack.recall.tokens import apply_token_budget, estimate_tokens, sum_tokens


def _result(id: str, text: str, score: float = 0.5) -> RecallResult:
    return RecallResult(id=id, text=text, score=score, payload={}, sources=["fake"])


# ---------- estimate_tokens ----------


def test_estimate_tokens_empty_is_zero():
    assert estimate_tokens("") == 0


def test_estimate_tokens_ascii_about_four_chars_per_token():
    assert estimate_tokens("a" * 400) == 100


def test_estimate_tokens_non_ascii_counts_denser():
    # Non-ASCII scripts encode to more tokens per char in BPE tokenizers;
    # equal-length Cyrillic must not be under-budgeted vs ASCII.
    latin = estimate_tokens("a" * 100)
    cyrillic = estimate_tokens("ж" * 100)
    assert cyrillic > latin


def test_estimate_tokens_nonempty_is_at_least_one():
    assert estimate_tokens("i") == 1
    assert estimate_tokens("я") == 1


# ---------- sum_tokens ----------


def test_sum_tokens_uses_custom_counter():
    results = [_result("a", "one"), _result("b", "two")]
    assert sum_tokens(results, counter=lambda s: 7) == 14


def test_sum_tokens_default_counter_matches_estimator():
    results = [_result("a", "x" * 40), _result("b", "y" * 40)]
    assert sum_tokens(results) == estimate_tokens("x" * 40) + estimate_tokens("y" * 40)


# ---------- apply_token_budget ----------


def test_budget_keeps_ranked_prefix_and_reports_tokens():
    results = [_result("a", "t"), _result("b", "t"), _result("c", "t")]
    kept, total = apply_token_budget(results, 10, counter=lambda s: 4)
    assert [r.id for r in kept] == ["a", "b"]
    assert total == 8


def test_budget_exact_fit_keeps_everything():
    results = [_result("a", "t"), _result("b", "t")]
    kept, total = apply_token_budget(results, 8, counter=lambda s: 4)
    assert [r.id for r in kept] == ["a", "b"]
    assert total == 8


def test_budget_is_hard_cap_stops_at_first_overflow():
    # Prefix semantics: a big second result blocks everything after it even
    # if a later result would fit — ranking order is never reshuffled.
    results = [_result("a", "s"), _result("b", "XXXL"), _result("c", "s")]
    counts = {"s": 2, "XXXL": 100}
    kept, total = apply_token_budget(results, 10, counter=lambda s: counts[s])
    assert [r.id for r in kept] == ["a"]
    assert total == 2


def test_budget_oversized_top_result_yields_empty_not_overshoot():
    results = [_result("a", "huge")]
    kept, total = apply_token_budget(results, 5, counter=lambda s: 50)
    assert kept == []
    assert total == 0


def test_budget_must_be_positive():
    with pytest.raises(ValueError):
        apply_token_budget([_result("a", "t")], 0)
    with pytest.raises(ValueError):
        apply_token_budget([_result("a", "t")], -5)


def test_budget_empty_input_is_empty_output():
    kept, total = apply_token_budget([], 100)
    assert kept == []
    assert total == 0


# ---------- Recaller wiring ----------


@dataclass
class _FakeRetriever:
    name: str = "fake"
    texts: tuple[str, ...] = ("aaaa" * 10, "bbbb" * 10, "cccc" * 10)

    def search(self, query: str, limit: int, filters: dict[str, Any] | None = None):
        return [
            _result(str(i), text) for i, text in enumerate(self.texts[:limit])
        ]


def test_recaller_recall_applies_token_budget():
    recaller = Recaller(retrievers=[_FakeRetriever()])
    unbudgeted = recaller.recall("q", limit=3)
    assert len(unbudgeted) == 3
    budgeted = recaller.recall("q", limit=3, token_budget=10, token_counter=lambda s: 4)
    assert [r.id for r in budgeted] == [r.id for r in unbudgeted[:2]]


@pytest.mark.asyncio
async def test_recall_async_passes_token_budget_through():
    recaller = Recaller(retrievers=[_FakeRetriever()])
    budgeted = await recaller.recall_async("q", limit=3, token_budget=10, token_counter=lambda s: 4)
    assert len(budgeted) == 2


# ---------- recall_flow wiring ----------


class _FakeRecaller:
    def __init__(self, results):
        self._results = results

    def recall(self, query, limit=10, filters=None, **_):
        return self._results[:limit]

    def apply_vector_floor_after_rerank(self, results, recalled_results):
        return results


def test_recall_flow_trims_final_order_to_budget():
    results = [_result(str(i), "tttt") for i in range(5)]
    out = recall_flow(
        _FakeRecaller(results),
        "q",
        limit=5,
        token_budget=12,
        token_counter=lambda s: 4,
    )
    assert [r.id for r in out] == ["0", "1", "2"]


def test_recall_flow_without_budget_is_unchanged():
    results = [_result(str(i), "tttt") for i in range(5)]
    out = recall_flow(_FakeRecaller(results), "q", limit=5)
    assert len(out) == 5
