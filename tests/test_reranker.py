"""Tests for Reranker — uses FakeLLM for deterministic behavior."""
import pytest

from mnemostack.llm.base import LLMProvider, LLMResponse
from mnemostack.recall import RecallResult, Reranker


class FakeLLM(LLMProvider):
    def __init__(self, response: str = "", error: str | None = None):
        self.response = response
        self.error = error
        self.last_prompt = ""

    @property
    def name(self):
        return "fake"

    def generate(self, prompt, max_tokens=200, temperature=0.0):
        self.last_prompt = prompt
        if self.error:
            return LLMResponse(text="", error=self.error)
        return LLMResponse(text=self.response)


@pytest.fixture
def sample_results():
    return [
        RecallResult(id="1", text="Paris is the capital of France", score=0.9, payload={}),
        RecallResult(id="2", text="London is a big city in UK", score=0.85, payload={}),
        RecallResult(id="3", text="Berlin is the capital of Germany", score=0.8, payload={}),
        RecallResult(id="4", text="Tokyo is the capital of Japan", score=0.75, payload={}),
    ]


def test_rerank_reorders_by_llm_output(sample_results):
    llm = FakeLLM(response="1 3")  # put 1 first, then 3
    reranker = Reranker(llm=llm)
    reranked = reranker.rerank("which city is capital of France", sample_results)
    assert reranked[0].id == "1"
    assert reranked[1].id == "3"
    # Unranked (2, 4) should still be in the list
    ids = [r.id for r in reranked]
    assert set(ids) == {"1", "2", "3", "4"}


def test_rerank_none_response_keeps_original(sample_results):
    llm = FakeLLM(response="NONE")
    reranker = Reranker(llm=llm)
    reranked = reranker.rerank("unrelated query", sample_results)
    # LLM said nothing relevant → original order preserved
    assert [r.id for r in reranked] == ["1", "2", "3", "4"]


def test_rerank_llm_error_falls_back_to_original(sample_results):
    llm = FakeLLM(error="timeout")
    reranker = Reranker(llm=llm)
    reranked = reranker.rerank("q", sample_results)
    assert [r.id for r in reranked] == ["1", "2", "3", "4"]


def test_rerank_empty_input():
    llm = FakeLLM(response="should not be called")
    reranker = Reranker(llm=llm)
    assert reranker.rerank("q", []) == []
    assert llm.last_prompt == ""


def test_rerank_respects_max_items(sample_results):
    llm = FakeLLM(response="1 2")
    reranker = Reranker(llm=llm, max_items=2)  # only rerank top 2
    reranked = reranker.rerank("q", sample_results)
    # First 2 should be reranked, 3 and 4 should keep positions
    assert reranked[0].id == "1"
    assert reranked[1].id == "2"
    assert reranked[2].id == "3"
    assert reranked[3].id == "4"


def test_rerank_parses_ids_with_prose():
    llm = FakeLLM(response="The most relevant is 3\nthen 1")  # first line only
    reranker = Reranker(llm=llm)
    results = [
        RecallResult(id="1", text="foo", score=0.5, payload={}),
        RecallResult(id="3", text="bar", score=0.5, payload={}),
    ]
    reranked = reranker.rerank("q", results)
    # Should pick "3" from first line
    assert reranked[0].id == "3"


def test_rerank_ignores_unknown_ids(sample_results):
    llm = FakeLLM(response="999 1 777")
    reranker = Reranker(llm=llm)
    reranked = reranker.rerank("q", sample_results)
    # Unknown IDs ignored, "1" comes first, rest in original order
    assert reranked[0].id == "1"


def test_rerank_includes_query_and_text_in_prompt(sample_results):
    llm = FakeLLM(response="1")
    reranker = Reranker(llm=llm)
    reranker.rerank("capital of France", sample_results)
    prompt = llm.last_prompt
    assert "capital of France" in prompt
    assert "Paris" in prompt
