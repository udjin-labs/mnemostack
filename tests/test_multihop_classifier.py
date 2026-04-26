"""Tests for multi-hop prompt routing and cat_4 regression protection."""

from mnemostack.llm.base import LLMProvider, LLMResponse
from mnemostack.recall import AnswerGenerator, RecallResult, classify_question
from mnemostack.recall.answer import _MULTIHOP_PROMPT


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


def sample_memories():
    return [
        RecallResult(
            id="1",
            text="Caroline pursued counseling after her own journey and support helped her.",
            score=0.9,
            payload={"source": "chat", "timestamp": "2023-05-08T00:00:00Z"},
            sources=["vector"],
        )
    ]


def test_classify_question_multihop_patterns():
    assert classify_question("What motivated Caroline to pursue counseling?") == "multihop"
    assert classify_question('What did Caroline take away from the book "Becoming Nicole"?') == "multihop"
    assert classify_question("What did Caroline and Melanie both learn?") == "multihop"
    assert classify_question("What do Caroline and Melanie have in common?") == "multihop"
    assert classify_question("What pattern appears across sessions?") == "multihop"


def test_across_single_hop_phrases_are_not_multihop():
    assert classify_question("Where did she travel across Europe?") == "general"
    assert classify_question("Who moved across town?") == "general"


def test_multihop_takes_precedence_over_inference_markers():
    assert classify_question("What did Caroline and Melanie both say would help?") == "multihop"


def test_cat4_setback_regression_stays_general_not_temporal():
    assert classify_question("What setback did Melanie face in October 2023?") == "general"


def test_generate_uses_multihop_prompt_for_multihop_question():
    llm = FakeLLM()
    gen = AnswerGenerator(llm=llm, category_aware_prompts=True)

    gen.generate("What motivated Caroline to pursue counseling?", sample_memories())

    assert _MULTIHOP_PROMPT.split("\n", 1)[0] in llm.last_prompt
    assert "requiring evidence from MULTIPLE memories" in llm.last_prompt


def test_adversarial_not_in_memory_question_stays_general():
    assert classify_question("How did Caroline's children handle the accident?") == "general"
