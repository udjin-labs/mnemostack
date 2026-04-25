"""Tests for AnswerGenerator — uses FakeLLM for deterministic behavior."""
import json

import pytest

from mnemostack.llm.base import LLMProvider, LLMResponse
from mnemostack.recall import AnswerGenerator, RecallResult


class FakeLLM(LLMProvider):
    """Deterministic LLM — returns pre-configured response."""

    def __init__(self, response_text: str = "", error: str | None = None):
        self.response_text = response_text
        self.error = error
        self.last_prompt: str = ""

    @property
    def name(self) -> str:
        return "fake"

    def generate(self, prompt, max_tokens=200, temperature=0.0):
        self.last_prompt = prompt
        if self.error:
            return LLMResponse(text="", error=self.error)
        return LLMResponse(text=self.response_text, tokens_used=50)


@pytest.fixture
def sample_memories():
    return [
        RecallResult(
            id=1,
            text="On 2024-01-15 we decided to migrate to Postgres",
            score=0.9,
            payload={"source": "notes/2024-01-15.md", "timestamp": "2024-01-15T10:00:00Z"},
            sources=["vector"],
        ),
        RecallResult(
            id=2,
            text="Migration completed by end of Q1 2024",
            score=0.85,
            payload={"source": "notes/2024-03-30.md"},
            sources=["bm25"],
        ),
    ]


def test_answer_basic(sample_memories):
    llm = FakeLLM(response_text="Postgres\nCONFIDENCE: 0.95")
    gen = AnswerGenerator(llm=llm)
    answer = gen.generate("what database did we migrate to", sample_memories)
    assert answer.ok
    assert answer.text == "Postgres"
    assert answer.confidence == 0.95
    assert answer.sources == ["notes/2024-01-15.md", "notes/2024-03-30.md"]


def test_answer_multiline_answer(sample_memories):
    llm = FakeLLM(response_text="Postgres, MySQL\nBoth are relational\nCONFIDENCE: 0.7")
    gen = AnswerGenerator(llm=llm)
    answer = gen.generate("databases", sample_memories)
    assert "Postgres" in answer.text
    assert "Both are relational" in answer.text
    assert answer.confidence == 0.7


def test_answer_no_confidence_line_defaults_to_half(sample_memories):
    llm = FakeLLM(response_text="Some answer without confidence")
    gen = AnswerGenerator(llm=llm)
    answer = gen.generate("q", sample_memories)
    assert answer.confidence == 0.5  # default


def test_answer_confidence_clamped(sample_memories):
    llm = FakeLLM(response_text="foo\nCONFIDENCE: 2.5")
    gen = AnswerGenerator(llm=llm)
    answer = gen.generate("q", sample_memories)
    assert answer.confidence == 1.0

    llm2 = FakeLLM(response_text="foo\nCONFIDENCE: -0.5")
    gen2 = AnswerGenerator(llm=llm2)
    answer2 = gen2.generate("q", sample_memories)
    assert answer2.confidence == 0.0


def test_answer_empty_memories():
    llm = FakeLLM(response_text="should not be called")
    gen = AnswerGenerator(llm=llm)
    answer = gen.generate("q", [])
    assert answer.text == "Not in memory."
    assert answer.confidence == 0.0
    assert llm.last_prompt == ""  # LLM never called


def test_answer_llm_error(sample_memories):
    llm = FakeLLM(error="rate limit")
    gen = AnswerGenerator(llm=llm)
    answer = gen.generate("q", sample_memories)
    assert not answer.ok
    assert answer.error == "rate limit"
    assert answer.confidence == 0.0


def test_should_fallback_low_confidence(sample_memories):
    llm = FakeLLM(response_text="uncertain answer\nCONFIDENCE: 0.3")
    gen = AnswerGenerator(llm=llm, confidence_threshold=0.5)
    answer = gen.generate("q", sample_memories)
    assert gen.should_fallback(answer)


def test_should_fallback_high_confidence(sample_memories):
    llm = FakeLLM(response_text="clear answer\nCONFIDENCE: 0.95")
    gen = AnswerGenerator(llm=llm, confidence_threshold=0.5)
    answer = gen.generate("q", sample_memories)
    assert not gen.should_fallback(answer)


def test_should_fallback_on_error(sample_memories):
    llm = FakeLLM(error="timeout")
    gen = AnswerGenerator(llm=llm)
    answer = gen.generate("q", sample_memories)
    assert gen.should_fallback(answer)


def test_context_formatting_includes_timestamps_and_sources(sample_memories):
    llm = FakeLLM(response_text="x\nCONFIDENCE: 0.5")
    gen = AnswerGenerator(llm=llm)
    gen.generate("q", sample_memories)
    prompt = llm.last_prompt
    assert "2024-01-15" in prompt
    assert "notes/2024-01-15.md" in prompt
    assert "Postgres" in prompt


def test_llm_registry():
    from mnemostack.llm import get_llm, list_llms

    assert "gemini" in list_llms()
    assert "ollama" in list_llms()

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm("nonexistent")


def test_custom_llm_registration():
    from mnemostack.llm import get_llm, register_llm

    class MyLLM(LLMProvider):
        @property
        def name(self):
            return "my-llm"

        def generate(self, prompt, max_tokens=200, temperature=0.0):
            return LLMResponse(text="hello", tokens_used=2)

    register_llm("my-llm", MyLLM)
    llm = get_llm("my-llm")
    resp = llm.generate("test")
    assert resp.text == "hello"


def test_gemini_llm_uses_api_key_header_not_query_string(monkeypatch):
    from mnemostack.llm.gemini import GeminiLLM

    seen = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
            ).encode()

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["headers"] = dict(req.header_items())
        seen["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    llm = GeminiLLM(api_key="secret-key", timeout=9)
    response = llm.generate("hello")

    assert response.text == "ok"
    assert "key=" not in seen["url"]
    assert seen["headers"]["X-goog-api-key"] == "secret-key"
    assert seen["timeout"] == 9


def test_prompt_has_temporal_reasoning_rules(sample_memories):
    """Regression: prompt must teach relative-time subtraction."""
    llm = FakeLLM(response_text="x\nCONFIDENCE: 0.9")
    gen = AnswerGenerator(llm=llm)
    gen.generate("when", sample_memories)
    prompt = llm.last_prompt
    assert "yesterday" in prompt.lower()
    assert "last week" in prompt.lower()
    # At least one concrete few-shot example
    assert "session date MINUS" in prompt or "MINUS 1" in prompt


def test_prompt_allows_hypothetical_inference(sample_memories):
    """Regression: prompt must allow 'might be'/'would be' reasoning, not just Not in memory."""
    llm = FakeLLM(response_text="x\nCONFIDENCE: 0.6")
    gen = AnswerGenerator(llm=llm)
    gen.generate("would X happen", sample_memories)
    prompt = llm.last_prompt
    p = prompt.lower()
    # Must instruct to attempt inference for hypothetical questions
    assert "might be" in p and "would be" in p
    assert "reasonable inference" in p
