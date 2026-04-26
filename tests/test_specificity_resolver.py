"""Tests for generic-placeholder specificity resolver."""

import pytest

from mnemostack.llm.base import LLMProvider, LLMResponse
from mnemostack.recall import (
    AnswerGenerator,
    RecallResult,
    detect_placeholders,
    resolve_specificity,
)


class FakeLLM(LLMProvider):
    def __init__(self, response_text: str = "Sweden", error: str | None = None):
        self.response_text = response_text
        self.error = error
        self.calls = 0
        self.prompts: list[str] = []

    @property
    def name(self) -> str:
        return "fake"

    def generate(self, prompt, max_tokens=200, temperature=0.0):
        self.calls += 1
        self.prompts.append(prompt)
        return LLMResponse(text=self.response_text, tokens_used=10, error=self.error)


class SequenceLLM(FakeLLM):
    def __init__(self, responses: list[str]):
        super().__init__(responses[0])
        self.responses = responses

    def generate(self, prompt, max_tokens=200, temperature=0.0):
        self.calls += 1
        self.prompts.append(prompt)
        text = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        return LLMResponse(text=text, tokens_used=10)


class RaisingLLM(FakeLLM):
    def generate(self, prompt, max_tokens=200, temperature=0.0):
        self.calls += 1
        raise RuntimeError("boom")


@pytest.fixture
def sample_memories():
    return [
        RecallResult(
            id="1",
            text="Caroline moved from Sweden four years ago.",
            score=0.9,
            payload={"source": "chat", "timestamp": "2023-05-08T00:00:00Z"},
            sources=["vector"],
        ),
        RecallResult(
            id="2",
            text="Melanie read Becoming Nicole after Caroline recommended it.",
            score=0.8,
            payload={"source": "chat", "timestamp": "2023-05-09T00:00:00Z"},
            sources=["vector"],
        ),
    ]


def test_detect_placeholders_finds_possessive_generic():
    placeholders = detect_placeholders("her home country")

    assert placeholders
    assert placeholders[0][0] == "her home country"


def test_detect_placeholders_ignores_specific():
    assert detect_placeholders("her brother John") == []


def test_detect_placeholders_finds_article_generic():
    placeholders = detect_placeholders("that book she liked")

    assert placeholders
    assert placeholders[0][0] == "that book"


def test_detect_placeholders_finds_category_generic():
    placeholders = detect_placeholders("video games")

    assert placeholders == [("video games", "specific video game titles or platforms")]


def test_detect_placeholders_returns_empty_for_clean_answer():
    assert detect_placeholders("pottery, camping, painting") == []


def test_resolve_specificity_replaces_placeholder_using_memories(sample_memories):
    llm = FakeLLM("Sweden")

    rewritten = resolve_specificity(
        "Where did Caroline move from 4 years ago?",
        "her home country",
        sample_memories,
        llm,
    )

    assert rewritten == "Sweden"
    assert llm.calls == 1
    assert "her home" in llm.prompts[0]
    assert "Caroline moved from Sweden" in llm.prompts[0]


def test_resolve_specificity_preserves_when_no_match(sample_memories):
    llm = FakeLLM("her home country")

    rewritten = resolve_specificity(
        "Where did Caroline move from 4 years ago?",
        "her home country",
        sample_memories,
        llm,
    )

    assert rewritten == "her home country"
    assert llm.calls == 1


def test_resolve_specificity_handles_llm_failure(sample_memories):
    llm = RaisingLLM()

    rewritten = resolve_specificity(
        "Where did Caroline move from 4 years ago?",
        "her home country",
        sample_memories,
        llm,
    )

    assert rewritten == "her home country"
    assert llm.calls == 1


def test_resolve_specificity_skips_adversarial_category(sample_memories):
    llm = FakeLLM("Sweden\nCONFIDENCE: 0.9")
    gen = AnswerGenerator(llm=llm, category_aware_prompts=True, specificity_resolver=True)

    answer = gen.generate("What fictional country is never mentioned?", sample_memories)

    assert answer.text == "Sweden"
    assert llm.calls == 1


def test_resolve_specificity_disabled_when_flag_false(sample_memories):
    llm = FakeLLM("her home country\nCONFIDENCE: 0.9")
    gen = AnswerGenerator(llm=llm, category_aware_prompts=True, specificity_resolver=False)

    answer = gen.generate("Where did Caroline move from 4 years ago?", sample_memories)

    assert answer.text == "her home country"
    assert llm.calls == 1


def test_specificity_resolver_only_runs_with_category_aware(sample_memories):
    llm = FakeLLM("her home country\nCONFIDENCE: 0.9")
    gen = AnswerGenerator(llm=llm, category_aware_prompts=False, specificity_resolver=True)

    answer = gen.generate("Where did Caroline move from 4 years ago?", sample_memories)

    assert answer.text == "her home country"
    assert llm.calls == 1


def test_specificity_resolver_rewrites_answer_when_enabled(sample_memories):
    llm = SequenceLLM(["her home country\nCONFIDENCE: 0.9", "Sweden"])
    gen = AnswerGenerator(llm=llm, category_aware_prompts=True, specificity_resolver=True)

    answer = gen.generate("Where did Caroline move from 4 years ago?", sample_memories)

    assert answer.text == "Sweden"
    assert answer.confidence == 0.85
    assert llm.calls == 2
