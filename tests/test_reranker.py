"""Tests for Reranker — uses FakeLLM for deterministic behavior."""
import logging
import uuid

import pytest

from mnemostack.llm.base import LLMProvider, LLMResponse
from mnemostack.recall import RecallResult, Reranker


class FakeLLM(LLMProvider):
    def __init__(self, response: str = "", error: str | None = None):
        self.response = response
        self.error = error
        self.last_prompt = ""
        self.last_max_tokens = None
        self.calls = 0

    @property
    def name(self):
        return "fake"

    def generate(self, prompt, max_tokens=200, temperature=0.0):
        self.calls += 1
        self.last_prompt = prompt
        self.last_max_tokens = max_tokens
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
    llm = FakeLLM(response="0 2")  # put result 0 first, then result 2
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
    llm = FakeLLM(response="0 1")
    reranker = Reranker(llm=llm, max_items=2)  # only rerank top 2
    reranked = reranker.rerank("q", sample_results)
    # First 2 should be reranked, 3 and 4 should keep positions
    assert reranked[0].id == "1"
    assert reranked[1].id == "2"
    assert reranked[2].id == "3"
    assert reranked[3].id == "4"


def test_rerank_cache_key_includes_tail_ids():
    llm = FakeLLM(response="1 0")
    reranker = Reranker(llm=llm, max_items=2)
    first = [
        RecallResult(id="1", text="one", score=0.9, payload={}),
        RecallResult(id="2", text="two", score=0.8, payload={}),
        RecallResult(id="3", text="three", score=0.7, payload={}),
    ]
    second = [
        RecallResult(id="1", text="one", score=0.9, payload={}),
        RecallResult(id="2", text="two", score=0.8, payload={}),
        RecallResult(id="4", text="four", score=0.7, payload={}),
    ]

    assert [r.id for r in reranker.rerank("q", first)] == ["2", "1", "3"]
    assert [r.id for r in reranker.rerank("q", second)] == ["2", "1", "4"]
    assert llm.calls == 2


def test_rerank_cache_key_preserves_input_order():
    llm = FakeLLM(response="0")
    reranker = Reranker(llm=llm, max_items=3)
    first = [
        RecallResult(id="1", text="one", score=0.9, payload={}),
        RecallResult(id="2", text="two", score=0.8, payload={}),
        RecallResult(id="3", text="three", score=0.7, payload={}),
    ]
    second = [
        RecallResult(id="1", text="one", score=0.9, payload={}),
        RecallResult(id="3", text="three", score=0.7, payload={}),
        RecallResult(id="2", text="two", score=0.8, payload={}),
    ]

    assert [r.id for r in reranker.rerank("q", first)] == ["1", "2", "3"]
    assert [r.id for r in reranker.rerank("q", second)] == ["1", "3", "2"]
    assert llm.calls == 2


def test_rerank_parses_ids_with_prose():
    llm = FakeLLM(response="The most relevant is 1\nthen 0")  # first line only
    reranker = Reranker(llm=llm)
    results = [
        RecallResult(id="1", text="foo", score=0.5, payload={}),
        RecallResult(id="3", text="bar", score=0.5, payload={}),
    ]
    reranked = reranker.rerank("q", results)
    # Should pick "3" from first line
    assert reranked[0].id == "3"


def test_rerank_ignores_unknown_ids(sample_results):
    llm = FakeLLM(response="999 0 777")
    reranker = Reranker(llm=llm)
    reranked = reranker.rerank("q", sample_results)
    # Unknown IDs ignored, "1" comes first, rest in original order
    assert reranked[0].id == "1"


def test_rerank_includes_query_and_text_in_prompt(sample_results):
    llm = FakeLLM(response="0")
    reranker = Reranker(llm=llm)
    reranker.rerank("capital of France", sample_results)
    prompt = llm.last_prompt
    assert "capital of France" in prompt
    assert "Paris" in prompt
    assert "ID=0:" in prompt
    assert "ID=1:" in prompt
    assert "ID=1: Paris" not in prompt


def test_rerank_reorders_uuid_ids_from_ordinal_response():
    ids = [str(uuid.uuid4()) for _ in range(3)]
    results = [
        RecallResult(id=ids[0], text="irrelevant city note", score=0.9, payload={}),
        RecallResult(id=ids[1], text="target fact about Qdrant UUIDs", score=0.8, payload={}),
        RecallResult(id=ids[2], text="secondary fact", score=0.7, payload={}),
    ]
    llm = FakeLLM(response="1 2")
    reranker = Reranker(llm=llm)

    reranked = reranker.rerank("Qdrant UUID target", results)

    assert [r.id for r in reranked] == [ids[1], ids[2], ids[0]]
    for original_id in ids:
        assert original_id not in llm.last_prompt


def test_rerank_keeps_composite_id_fuzzy_fallback():
    results = [
        RecallResult(id="notes/MEMORY.md:45", text="target", score=0.9, payload={}),
        RecallResult(id="notes/OTHER.md:10", text="other", score=0.8, payload={}),
    ]
    llm = FakeLLM(response="notes/MEMORY.md")
    reranker = Reranker(llm=llm)

    reranked = reranker.rerank("target", results)

    assert [r.id for r in reranked] == ["notes/MEMORY.md:45", "notes/OTHER.md:10"]


def test_rerank_logs_partial_parse_warning(sample_results, caplog):
    llm = FakeLLM(response="999 0")
    reranker = Reranker(llm=llm)
    reranker_logger = logging.getLogger("mnemostack.recall.reranker")
    mnemostack_logger = logging.getLogger("mnemostack")
    original_reranker = reranker_logger.propagate
    original_mnemostack = mnemostack_logger.propagate
    reranker_logger.propagate = True
    mnemostack_logger.propagate = True

    try:
        with caplog.at_level("WARNING", logger="mnemostack.recall.reranker"):
            reranker.rerank("q", sample_results)
    finally:
        reranker_logger.propagate = original_reranker
        mnemostack_logger.propagate = original_mnemostack

    assert "rerank parsed 1 of 4 candidate ids" in caplog.text


def test_rerank_logs_no_usable_ids_warning(sample_results, caplog):
    llm = FakeLLM(response="999 777")
    reranker = Reranker(llm=llm)
    reranker_logger = logging.getLogger("mnemostack.recall.reranker")
    mnemostack_logger = logging.getLogger("mnemostack")
    original_reranker = reranker_logger.propagate
    original_mnemostack = mnemostack_logger.propagate
    reranker_logger.propagate = True
    mnemostack_logger.propagate = True

    try:
        with caplog.at_level("WARNING", logger="mnemostack.recall.reranker"):
            reranked = reranker.rerank("q", sample_results)
    finally:
        reranker_logger.propagate = original_reranker
        mnemostack_logger.propagate = original_mnemostack

    assert [r.id for r in reranked] == ["1", "2", "3", "4"]
    assert "rerank produced no usable ids, keeping original order" in caplog.text


def test_rerank_scales_max_tokens_with_candidate_count():
    results = [
        RecallResult(id=str(i), text=f"memory {i}", score=1.0 - i * 0.01, payload={})
        for i in range(30)
    ]
    llm = FakeLLM(response="0")
    reranker = Reranker(llm=llm, max_items=30, max_tokens=20)

    reranker.rerank("q", results)

    assert llm.last_max_tokens == 240
