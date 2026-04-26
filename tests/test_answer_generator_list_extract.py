"""Tests for list/count retrieval-then-extract answer flow."""

import pytest

from mnemostack.llm.base import LLMProvider, LLMResponse
from mnemostack.recall import AnswerGenerator, RecallResult
from mnemostack.recall.answer import _DEFAULT_PROMPT, _LIST_EXTRACT_PROMPT, _LIST_PROMPT


class SequenceLLM(LLMProvider):
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    @property
    def name(self) -> str:
        return "sequence"

    def generate(self, prompt, max_tokens=200, temperature=0.0):
        self.prompts.append(prompt)
        return LLMResponse(text=self.responses.pop(0), tokens_used=10)


class SequenceMaybeFailLLM(LLMProvider):
    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    @property
    def name(self) -> str:
        return "sequence"

    def generate(self, prompt, max_tokens=200, temperature=0.0):
        self.prompts.append(prompt)
        return self.responses.pop(0)


@pytest.fixture
def memories():
    return [
        RecallResult(
            id=str(i),
            text=f"Memory {i}: Melanie detail {i}.",
            score=1.0 - (i * 0.01),
            payload={"source": f"chat-{i}", "timestamp": "2023-05-08T00:00:00Z"},
            sources=["vector"],
        )
        for i in range(1, 46)
    ]


def test_list_extract_pass_returns_items_from_extracted_json(memories):
    llm = SequenceLLM([
        '{"items": ["Luna", "Oliver"]}',
        "Luna, Oliver",
    ])
    gen = AnswerGenerator(llm=llm, category_aware_prompts=True, list_extract_mode=True)

    answer = gen.generate("What are Melanie's pets?", memories)

    assert answer.text == "Luna, Oliver"
    assert answer.confidence == 0.8
    assert len(llm.prompts) == 2
    assert "extract ALL items matching the question" in llm.prompts[0]
    assert 'EXTRACTED ITEMS: ["Luna", "Oliver"]' in llm.prompts[1]


def test_list_extract_falls_back_on_empty_items(memories):
    llm = SequenceLLM([
        '{"items": []}',
        "Not in memory.\nCONFIDENCE: 0.9",
    ])
    gen = AnswerGenerator(llm=llm, category_aware_prompts=True, list_extract_mode=True)

    answer = gen.generate("What are Melanie's pets?", memories)

    assert answer.text == "Not in memory."
    assert answer.confidence == 0.3
    assert len(llm.prompts) == 2
    assert _LIST_PROMPT.split("\n", 1)[0] in llm.prompts[1]
    assert "Memory 16" not in llm.prompts[1]


@pytest.mark.parametrize("extract_output", ["not json", "[]", "{}"])
def test_list_extract_falls_back_on_malformed_json(memories, extract_output):
    llm = SequenceLLM([
        extract_output,
        "Luna, Oliver\nCONFIDENCE: 0.8",
    ])
    gen = AnswerGenerator(llm=llm, category_aware_prompts=True, list_extract_mode=True)

    answer = gen.generate("What are Melanie's pets?", memories)

    assert answer.text == "Luna, Oliver"
    assert answer.confidence == 0.3
    assert len(llm.prompts) == 2
    assert _LIST_PROMPT.split("\n", 1)[0] in llm.prompts[1]


def test_list_extract_falls_back_when_finalize_call_fails(memories):
    llm = SequenceMaybeFailLLM([
        LLMResponse(text='{"items": ["Luna", "Oliver"]}', tokens_used=10),
        LLMResponse(text="", error="rate limited"),
        LLMResponse(text="Luna, Oliver\nCONFIDENCE: 0.8", tokens_used=10),
    ])
    gen = AnswerGenerator(llm=llm, category_aware_prompts=True, list_extract_mode=True)

    answer = gen.generate("What are Melanie's pets?", memories)

    assert answer.ok is True
    assert answer.text == "Luna, Oliver"
    assert answer.confidence == 0.3
    assert len(llm.prompts) == 3
    assert _LIST_PROMPT.split("\n", 1)[0] in llm.prompts[2]


def test_parse_extracted_items_deduplicates_preserving_order():
    items = AnswerGenerator._parse_extracted_items(
        '{"items": ["Luna", "Oliver", "Luna", "", 42, "Bailey", "Oliver"]}'
    )

    assert items == ["Luna", "Oliver", "Bailey"]


def test_list_extract_passes_more_memories_than_max_memories(memories):
    llm = SequenceLLM([
        '{"items": ["item"]}',
        "item",
    ])
    gen = AnswerGenerator(
        llm=llm,
        max_memories=15,
        category_aware_prompts=True,
        list_extract_mode=True,
    )

    gen.generate("What are Melanie's pets?", memories)

    assert "scan 40 memories" in llm.prompts[0]
    assert "Memory 40" in llm.prompts[0]
    assert "Memory 41" not in llm.prompts[0]


def test_list_extract_only_for_list_count_categories(memories):
    queries = [
        "Would Melanie enjoy a pet meetup?",
        "When did Melanie adopt Luna?",
        "Who is Melanie?",
        "What motivated Melanie to adopt Luna?",
    ]

    for query in queries:
        llm = SequenceLLM(["ok\nCONFIDENCE: 0.7"])
        gen = AnswerGenerator(llm=llm, category_aware_prompts=True, list_extract_mode=True)
        answer = gen.generate(query, memories)

        assert answer.text == "ok"
        assert len(llm.prompts) == 1
        assert _LIST_EXTRACT_PROMPT.split("\n", 1)[0] not in llm.prompts[0]

    count_llm = SequenceLLM(['{"items": ["Luna", "Oliver"]}', "2"])
    count_gen = AnswerGenerator(
        llm=count_llm,
        category_aware_prompts=True,
        list_extract_mode=True,
    )
    count_gen.generate("How many pets does Melanie have?", memories)
    assert len(count_llm.prompts) == 2
    assert "extract ALL items matching the question" in count_llm.prompts[0]


def test_list_extract_disabled_when_flag_false(memories):
    llm = SequenceLLM(["Luna, Oliver\nCONFIDENCE: 0.9"])
    gen = AnswerGenerator(llm=llm, category_aware_prompts=True, list_extract_mode=False)

    answer = gen.generate("What are Melanie's pets?", memories)

    assert answer.text == "Luna, Oliver"
    assert answer.confidence == 0.9
    assert len(llm.prompts) == 1
    assert _LIST_PROMPT.split("\n", 1)[0] in llm.prompts[0]
    assert _LIST_EXTRACT_PROMPT.split("\n", 1)[0] not in llm.prompts[0]


def test_list_extract_ignored_when_category_aware_disabled(memories):
    llm = SequenceLLM(["Luna, Oliver\nCONFIDENCE: 0.9"])
    gen = AnswerGenerator(llm=llm, category_aware_prompts=False, list_extract_mode=True)

    gen.generate("What are Melanie's pets?", memories)

    assert len(llm.prompts) == 1
    assert _DEFAULT_PROMPT.split("\n", 1)[0] in llm.prompts[0]
    assert _LIST_EXTRACT_PROMPT.split("\n", 1)[0] not in llm.prompts[0]


def test_list_extract_handles_specificity(memories):
    llm = SequenceLLM([
        '{"items": ["Oliver", "Luna", "Bailey"]}',
        "Oliver, Luna, Bailey",
    ])
    gen = AnswerGenerator(llm=llm, category_aware_prompts=True, list_extract_mode=True)

    answer = gen.generate("What are Melanie's pets?", memories)

    assert answer.text == "Oliver, Luna, Bailey"
    items_line = next(line for line in llm.prompts[1].splitlines() if line.startswith("EXTRACTED ITEMS:"))
    assert items_line == 'EXTRACTED ITEMS: ["Oliver", "Luna", "Bailey"]'
    assert "her dog" not in items_line
