"""Tests for cat_3 inference retry with query decomposition."""

from __future__ import annotations

import pytest

from mnemostack.llm.base import LLMProvider, LLMResponse
from mnemostack.recall import AnswerGenerator, RecallResult
from mnemostack.recall.inference_retry import decompose_query, merge_results, should_retry


class FakeLLM(LLMProvider):
    def __init__(self, responses: list[str] | None = None, error: Exception | None = None):
        self.responses = responses or []
        self.error = error
        self.prompts: list[str] = []

    @property
    def name(self) -> str:
        return "fake"

    def generate(self, prompt, max_tokens=200, temperature=0.0):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        text = self.responses.pop(0) if self.responses else "Not in memory.\nCONFIDENCE: 0.1"
        return LLMResponse(text=text, tokens_used=10)


class FakeRecaller:
    def __init__(self, by_query: dict[str, list[RecallResult]] | None = None):
        self.by_query = by_query or {}
        self.calls: list[tuple[str, int, dict[str, object] | None]] = []

    def recall(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, object] | None = None,
    ):
        self.calls.append((query, limit, filters))
        return self.by_query.get(query, [])


@pytest.fixture
def memories():
    return [
        RecallResult(
            id="orig",
            text="Caroline volunteers at a community center.",
            score=0.9,
            payload={"source": "orig.md"},
            sources=["vector"],
        )
    ]


def result(id_: str, text: str, score: float = 0.8) -> RecallResult:
    return RecallResult(
        id=id_,
        text=text,
        score=score,
        payload={"source": f"{id_}.md"},
        sources=["bm25"],
    )


def test_decompose_query_parses_valid_json():
    llm = FakeLLM(['{"queries": ["Caroline activism", "Caroline voting", "Caroline values"]}'])

    assert decompose_query("What would Caroline's political leaning be?", llm) == [
        "Caroline activism",
        "Caroline voting",
        "Caroline values",
    ]


def test_decompose_query_handles_code_fences():
    llm = FakeLLM(['```json\n{"queries": ["Y religion", "Y church"]}\n```'])

    assert decompose_query("Would Y be religious?", llm) == ["Y religion", "Y church"]


def test_decompose_query_returns_empty_on_invalid_json():
    llm = FakeLLM(["not-json"])

    assert decompose_query("Would Y be religious?", llm) == []


def test_decompose_query_returns_empty_when_queries_missing():
    llm = FakeLLM(['{"items": ["x"]}'])

    assert decompose_query("Would Y be religious?", llm) == []


def test_decompose_query_handles_llm_failure():
    llm = FakeLLM(error=RuntimeError("boom"))

    assert decompose_query("Would Y be religious?", llm) == []


def test_should_retry_on_not_in_memory():
    assert should_retry("Not in memory.", 0.9) is True


def test_should_retry_on_low_confidence():
    assert should_retry("liberal", 0.3) is True


def test_should_retry_on_normal_answer():
    assert should_retry("liberal", 0.8) is False


def test_merge_results_uses_rrf():
    a = result("a", "original")
    b = result("b", "sub one")
    c = result("c", "sub two")

    merged = merge_results([a], [b], [c])

    assert {r.id for r in merged} == {"a", "b", "c"}
    assert all(isinstance(r.score, float) for r in merged)


def test_merge_results_handles_empty_sub_results():
    a = result("a", "original")

    merged = merge_results([a], [], [])

    assert [r.id for r in merged] == ["a"]


def test_inference_retry_executes_full_flow(memories):
    llm = FakeLLM(
        [
            "Not in memory.\nCONFIDENCE: 0.2",
            '{"queries": ["Caroline activism", "Caroline values"]}',
            "likely progressive\nCONFIDENCE: 0.7",
        ]
    )
    recaller = FakeRecaller(
        {
            "Caroline activism": [result("a", "Caroline supports LGBTQ rights.")],
            "Caroline values": [result("b", "Caroline campaigns for climate action.")],
        }
    )
    gen = AnswerGenerator(
        llm=llm,
        category_aware_prompts=True,
        inference_retry=True,
        recaller=recaller,
    )

    answer = gen.generate("What would Caroline's political leaning be?", memories)

    assert answer.text == "likely progressive"
    assert answer.confidence == 0.7
    assert recaller.calls == [
        ("Caroline activism", 10, None),
        ("Caroline values", 10, None),
    ]
    assert len(llm.prompts) == 3


def test_inference_retry_forwards_recall_filters(memories):
    llm = FakeLLM(
        [
            "Not in memory.\nCONFIDENCE: 0.2",
            '{"queries": ["Caroline activism", "Caroline values"]}',
            "likely progressive\nCONFIDENCE: 0.7",
        ]
    )
    recaller = FakeRecaller(
        {
            "Caroline activism": [result("a", "Caroline supports LGBTQ rights.")],
            "Caroline values": [result("b", "Caroline campaigns for climate action.")],
        }
    )
    gen = AnswerGenerator(
        llm=llm,
        category_aware_prompts=True,
        inference_retry=True,
        recaller=recaller,
    )

    answer = gen.generate(
        "What would Caroline's political leaning be?",
        memories,
        recall_filters={"workspace": "team-a", "session": "conv-26"},
    )

    assert answer.text == "likely progressive"
    assert recaller.calls == [
        ("Caroline activism", 10, {"workspace": "team-a", "session": "conv-26"}),
        ("Caroline values", 10, {"workspace": "team-a", "session": "conv-26"}),
    ]


@pytest.mark.parametrize(
    "query",
    [
        "What are Caroline's hobbies?",
        "When did Caroline move?",
        "Who is Caroline?",
        "What fictional award did Caroline win?",
    ],
)
def test_inference_retry_skipped_for_non_inference_categories(query, memories):
    llm = FakeLLM(["Not in memory.\nCONFIDENCE: 0.1"])
    recaller = FakeRecaller({"should not": [result("x", "x")]})
    gen = AnswerGenerator(
        llm=llm,
        category_aware_prompts=True,
        inference_retry=True,
        recaller=recaller,
    )

    answer = gen.generate(query, memories)

    assert answer.text == "Not in memory."
    assert recaller.calls == []
    assert len(llm.prompts) == 1


def test_inference_retry_returns_draft_when_decompose_fails(memories):
    llm = FakeLLM(["Not in memory.\nCONFIDENCE: 0.1", "malformed-json"])
    recaller = FakeRecaller()
    gen = AnswerGenerator(
        llm=llm,
        category_aware_prompts=True,
        inference_retry=True,
        recaller=recaller,
    )

    answer = gen.generate("Would Caroline support the policy?", memories)

    assert answer.text == "Not in memory."
    assert recaller.calls == []


def test_inference_retry_returns_draft_when_subqueries_empty(memories):
    llm = FakeLLM(["Not in memory.\nCONFIDENCE: 0.1", '{"queries": []}'])
    recaller = FakeRecaller()
    gen = AnswerGenerator(
        llm=llm,
        category_aware_prompts=True,
        inference_retry=True,
        recaller=recaller,
    )

    answer = gen.generate("Would Caroline support the policy?", memories)

    assert answer.text == "Not in memory."
    assert recaller.calls == []


def test_inference_retry_returns_draft_when_new_answer_no_better(memories):
    llm = FakeLLM(
        [
            "Not in memory.\nCONFIDENCE: 0.2",
            '{"queries": ["Caroline activism"]}',
            "Not in memory.\nCONFIDENCE: 0.8",
        ]
    )
    recaller = FakeRecaller({"Caroline activism": [result("a", "Caroline volunteers.")]})
    gen = AnswerGenerator(
        llm=llm,
        category_aware_prompts=True,
        inference_retry=True,
        recaller=recaller,
    )

    answer = gen.generate("Would Caroline support the policy?", memories)

    assert answer.text == "Not in memory."
    assert recaller.calls == [("Caroline activism", 10, None)]


def test_inference_retry_disabled_when_flag_false(memories):
    llm = FakeLLM(["Not in memory.\nCONFIDENCE: 0.1"])
    recaller = FakeRecaller()
    gen = AnswerGenerator(
        llm=llm,
        category_aware_prompts=True,
        inference_retry=False,
        recaller=recaller,
    )

    answer = gen.generate("Would Caroline support the policy?", memories)

    assert answer.text == "Not in memory."
    assert recaller.calls == []


def test_inference_retry_requires_recaller(memories):
    llm = FakeLLM(["Not in memory.\nCONFIDENCE: 0.1"])
    gen = AnswerGenerator(llm=llm, category_aware_prompts=True, inference_retry=True)

    answer = gen.generate("Would Caroline support the policy?", memories)

    assert answer.text == "Not in memory."
    assert len(llm.prompts) == 1
