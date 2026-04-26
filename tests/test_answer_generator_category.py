"""Tests for category-aware AnswerGenerator prompt routing."""

import pytest

from mnemostack.llm.base import LLMProvider, LLMResponse
from mnemostack.recall import AnswerGenerator, RecallResult, classify_question
from mnemostack.recall.answer import _DEFAULT_PROMPT, _INFERENCE_PROMPT, _LIST_PROMPT


class FakeLLM(LLMProvider):
    def __init__(self, response_text: str = "ok\nCONFIDENCE: 0.9"):
        self.response_text = response_text
        self.last_prompt = ""

    @property
    def name(self) -> str:
        return "fake"

    def generate(self, prompt, max_tokens=200, temperature=0.0):
        self.last_prompt = prompt
        return LLMResponse(text=self.response_text, tokens_used=10)


@pytest.fixture
def sample_memories():
    return [
        RecallResult(
            id="1",
            text="Melanie has pets named Luna, Oliver, and Bailey.",
            score=0.9,
            payload={"source": "chat", "timestamp": "2023-05-08T00:00:00Z"},
            sources=["vector"],
        )
    ]


def test_classify_question_list_patterns():
    assert classify_question("What are X's pets?") == "list"
    assert classify_question("Which cities did Y visit?") == "list"
    assert classify_question("Name all members") == "list"


def test_classify_question_count_patterns():
    assert classify_question("How many pets does Melanie have?") == "count"
    assert classify_question("Count the events Maria planned") == "count"


def test_classify_question_temporal_patterns():
    assert classify_question("When did Maria get in a car accident?") == "temporal"
    assert classify_question("In which year did John move?") == "temporal"
    assert classify_question("What date was the first call-out?") == "temporal"


def test_classify_question_inference_patterns():
    assert classify_question("Would X be open to moving abroad?") == "inference"
    assert classify_question("Might Y enjoy House of MinaLima?") == "inference"
    assert classify_question("What's their political leaning?") == "inference"


def test_classify_question_general_default():
    assert classify_question("Who is John?") == "general"


def test_classify_question_ambiguous_returns_general():
    assert classify_question("") == "general"
    assert classify_question("This might've been discussed already") == "general"
    assert classify_question("Tell me about John's move") == "general"
    assert classify_question("Color of Melanie's dog") == "general"


def test_generate_uses_list_prompt_for_list_question(sample_memories):
    llm = FakeLLM()
    gen = AnswerGenerator(llm=llm, category_aware_prompts=True)

    gen.generate("What are Melanie's pets?", sample_memories)

    assert _LIST_PROMPT.split("\n", 1)[0] in llm.last_prompt
    assert "COMPLETE list of ALL items" in llm.last_prompt
    assert "SHORTEST factual answer" not in llm.last_prompt


def test_generate_uses_inference_prompt_for_would_question(sample_memories):
    llm = FakeLLM()
    gen = AnswerGenerator(llm=llm, category_aware_prompts=True)

    gen.generate("Would Melanie enjoy a pet meetup?", sample_memories)

    assert _INFERENCE_PROMPT.split("\n", 1)[0] in llm.last_prompt
    assert "DO NOT answer 'Not in memory' if even partial evidence exists" in llm.last_prompt


def test_generate_uses_default_when_category_aware_disabled(sample_memories):
    llm = FakeLLM()
    gen = AnswerGenerator(llm=llm)

    gen.generate("What are Melanie's pets?", sample_memories)

    assert _DEFAULT_PROMPT.split("\n", 1)[0] in llm.last_prompt
    assert "SHORTEST factual answer" in llm.last_prompt
    assert "COMPLETE list of ALL items" not in llm.last_prompt
