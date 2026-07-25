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
            return json.dumps({"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}).encode()

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


def test_display_ts_keeps_meaningful_time():
    from mnemostack.recall.answer import _display_ts

    assert _display_ts("2023-05-08T13:41:00") == "2023-05-08 13:41"
    assert _display_ts("2023-05-08T13:41:00Z") == "2023-05-08 13:41"


def test_display_ts_date_only_at_midnight():
    from mnemostack.recall.answer import _display_ts

    assert _display_ts("2026-06-01T00:00:00") == "2026-06-01"
    assert _display_ts("2026-06-01") == "2026-06-01"


def test_display_ts_garbage_falls_back_without_raising():
    from mnemostack.recall.answer import _display_ts

    assert _display_ts("May 2023, around noon") == "May 2023, "


def test_format_context_renders_time_of_day():
    from mnemostack.recall.answer import AnswerGenerator
    from mnemostack.recall.recaller import RecallResult

    ctx = AnswerGenerator._format_context(
        [
            RecallResult(
                id=1,
                text="met Jean at the cafe",
                score=1.0,
                payload={"timestamp": "2023-05-08T13:41:00", "source": "conv-1"},
            )
        ]
    )
    assert "[2023-05-08 13:41]" in ctx


def test_format_context_projects_context_fields():
    from mnemostack.recall.answer import AnswerGenerator
    from mnemostack.recall.recaller import RecallResult

    memories = [
        RecallResult(
            id=1,
            text="the payment was approved",
            score=1.0,
            payload={
                "timestamp": "2026-03-02T10:15:00",
                "source": "thread-1",
                "author": "participant A",
                "amount": 1250,
                "tags": ["finance", "approved"],
            },
        ),
        RecallResult(
            id=2,
            text="a note without extra fields",
            score=0.9,
            payload={"source": "thread-2"},
        ),
    ]

    ctx = AnswerGenerator._format_context(memories, context_fields=("author", "amount", "tags"))

    line1, line2 = ctx.splitlines()
    assert "author=participant A" in line1
    assert "amount=1250" in line1
    assert "tags=finance, approved" in line1  # lists render comma-joined
    # missing fields are silently skipped, not rendered empty
    assert "author=" not in line2
    assert "amount=" not in line2


def test_format_context_truncates_long_field_values():
    from mnemostack.recall.answer import AnswerGenerator
    from mnemostack.recall.recaller import RecallResult

    memories = [
        RecallResult(
            id=1,
            text="short text",
            score=1.0,
            payload={"source": "s", "blob": "x" * 500},
        )
    ]

    ctx = AnswerGenerator._format_context(memories, context_fields=("blob",))

    assert "x" * 100 + "…" in ctx
    assert "x" * 101 not in ctx


def test_generator_threads_context_fields_into_prompt():
    from mnemostack.recall import AnswerGenerator
    from mnemostack.recall.recaller import RecallResult

    llm = FakeLLM("ok\nCONFIDENCE: 0.9")
    gen = AnswerGenerator(
        llm=llm,
        specificity_resolver=False,
        inference_retry=False,
        context_fields=["author"],
    )
    memory = RecallResult(
        id=1,
        text="the meeting was moved",
        score=1.0,
        payload={"source": "thread-3", "author": "Teilnehmer B"},
    )

    gen.generate("who moved the meeting?", [memory])

    assert "author=Teilnehmer B" in llm.last_prompt


def test_answer_carries_provider_token_usage(sample_memories):
    llm = FakeLLM(response_text="Postgres\nCONFIDENCE: 0.95")
    gen = AnswerGenerator(llm=llm)
    answer = gen.generate("what database did we migrate to", sample_memories)
    assert answer.ok
    # FakeLLM reports tokens_used=50; the Answer must not drop it.
    assert answer.tokens_used == 50


def test_answer_token_usage_none_when_provider_silent(sample_memories):
    class SilentLLM(FakeLLM):
        def generate(self, prompt, max_tokens=200, temperature=0.0):
            return LLMResponse(text=self.response_text)  # no tokens_used

    gen = AnswerGenerator(llm=SilentLLM(response_text="Postgres\nCONFIDENCE: 0.9"))
    answer = gen.generate("what database did we migrate to", sample_memories)
    assert answer.ok
    assert answer.tokens_used is None


def test_answer_empty_memories_has_no_token_usage():
    gen = AnswerGenerator(llm=FakeLLM(response_text="unused"))
    answer = gen.generate("anything", [])
    # abstention is assembled without an LLM call
    assert answer.tokens_used is None


class _PromptCapturingLLM(FakeLLM):
    """FakeLLM returning queued responses and recording every prompt."""

    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt, max_tokens=200, temperature=0.0):
        self.prompts.append(prompt)
        text = self.responses.pop(0) if self.responses else ""
        return LLMResponse(text=text, tokens_used=50)


def test_generate_applies_token_budget_to_prompt(sample_memories):
    llm = _PromptCapturingLLM(["Postgres\nCONFIDENCE: 0.95"])
    gen = AnswerGenerator(llm=llm, specificity_resolver=False, inference_retry=False)

    gen.generate(
        "what database did we migrate to",
        sample_memories,
        token_budget=1,
        token_counter=lambda s: 1,  # each memory costs 1 token -> only the first fits
    )

    prompt = llm.prompts[0]
    assert "migrate to Postgres" in prompt
    assert "Migration completed" not in prompt


def test_inference_retry_recall_honors_token_budget(monkeypatch, sample_memories):
    import mnemostack.recall.answer as answer_mod

    big = RecallResult(
        id=99,
        text="fresh sub-recall memory that must not blow the budget",
        score=0.99,
        payload={"source": "sub.md"},
    )

    class _StubRecaller:
        def recall(self, query, limit=10, filters=None, **_):
            return [big]

    # draft is low-confidence -> retry runs; retry answer is confident
    llm = _PromptCapturingLLM(
        ["draft\nCONFIDENCE: 0.1", "final\nCONFIDENCE: 0.9"]
    )
    gen = AnswerGenerator(
        llm=llm,
        recaller=_StubRecaller(),
        specificity_resolver=False,
        inference_retry=True,
        retry_with_expansion=False,
    )
    monkeypatch.setattr(answer_mod, "decompose_query", lambda q, llm: ["sub question"])

    # merge_results puts the sub-recall hit ahead of the primary memories when
    # it scores higher; a 1-token budget must keep only the first merged hit.
    gen.generate(
        "why did the migration happen",
        sample_memories,
        category="inference",
        token_budget=1,
        token_counter=lambda s: 1,
    )

    retry_prompt = llm.prompts[-1]
    memory_texts = [m.text for m in sample_memories] + [big.text]
    included = [t for t in memory_texts if t[:30] in retry_prompt]
    assert len(included) == 1, f"budget of 1 token must keep exactly one memory: {included}"


def test_expansion_retry_recall_honors_token_budget(sample_memories):
    big = RecallResult(
        id=77,
        text="expansion sub-recall memory",
        score=0.99,
        payload={"source": "exp.md"},
    )

    class _StubEmbedding:
        def embed_batch(self, texts):
            return [[0.1, 0.2]] * len(texts)

    class _StubRecaller:
        embedding = _StubEmbedding()

        def search_many(self, vectors, limit, filters=None, **_):
            return [big, *sample_memories]

    # call order: draft answer -> expansion rephrase -> retry answer
    llm = _PromptCapturingLLM(
        [
            "draft\nCONFIDENCE: 0.1",
            "rephrase one\nrephrase two\nhypothetical answer",
            "final\nCONFIDENCE: 0.9",
        ]
    )
    gen = AnswerGenerator(
        llm=llm,
        recaller=_StubRecaller(),
        expansion_llm=llm,
        retry_with_expansion=True,
        specificity_resolver=False,
        inference_retry=False,
    )

    gen.generate(
        "what happened",
        sample_memories,
        token_budget=1,
        token_counter=lambda s: 1,
    )

    retry_prompt = llm.prompts[-1]
    assert "expansion sub-recall memory" in retry_prompt
    assert "migrate to Postgres" not in retry_prompt
    assert "Migration completed" not in retry_prompt


def test_duck_typed_llm_response_without_usage_field(sample_memories):
    from types import SimpleNamespace

    class _DuckLLM(FakeLLM):
        def generate(self, prompt, max_tokens=200, temperature=0.0):
            # LLMResponse-like object: ok/text/error but no tokens_used
            return SimpleNamespace(ok=True, text="Postgres\nCONFIDENCE: 0.9", error=None)

    gen = AnswerGenerator(
        llm=_DuckLLM(), specificity_resolver=False, inference_retry=False
    )
    answer = gen.generate("what database did we migrate to", sample_memories)
    assert answer.ok
    assert answer.tokens_used is None


def test_specificity_rewrite_adds_resolver_usage(monkeypatch, sample_memories):
    import mnemostack.recall.answer as answer_mod

    def _fake_resolver(query, draft_answer, candidate_memories, llm, **kwargs):
        llm.generate("rewrite it")  # tracked call, reports tokens_used=50
        return "rewritten answer"

    monkeypatch.setattr(answer_mod, "detect_placeholders", lambda text: True)
    monkeypatch.setattr(answer_mod, "resolve_specificity", _fake_resolver)

    llm = FakeLLM(response_text="draft with placeholder\nCONFIDENCE: 0.9")
    gen = AnswerGenerator(llm=llm, inference_retry=False)
    answer = gen.generate("who did it", sample_memories)

    assert answer.text == "rewritten answer"
    # draft call (50) + resolver call (50)
    assert answer.tokens_used == 100


def test_answer_reports_context_tokens_estimate(sample_memories):
    llm = FakeLLM(response_text="Postgres\nCONFIDENCE: 0.95")
    gen = AnswerGenerator(llm=llm, specificity_resolver=False, inference_retry=False)
    answer = gen.generate(
        "what database did we migrate to",
        sample_memories,
        token_counter=lambda s: 1,
    )
    # two memories in the prompt, one "token" each under the custom counter
    assert answer.context_tokens_estimate == 2


def test_retry_answer_reports_context_of_merged_pool(monkeypatch, sample_memories):
    import mnemostack.recall.answer as answer_mod

    big = RecallResult(
        id=99,
        text="fresh sub-recall memory",
        score=0.99,
        payload={"source": "sub.md"},
    )

    class _StubRecaller:
        def recall(self, query, limit=10, filters=None, **_):
            return [big]

    llm = _PromptCapturingLLM(["draft\nCONFIDENCE: 0.1", "final\nCONFIDENCE: 0.9"])
    gen = AnswerGenerator(
        llm=llm,
        recaller=_StubRecaller(),
        specificity_resolver=False,
        inference_retry=True,
        retry_with_expansion=False,
    )
    monkeypatch.setattr(answer_mod, "decompose_query", lambda q, llm: ["sub question"])

    answer = gen.generate(
        "why did the migration happen",
        sample_memories,
        category="inference",
        token_budget=1,
        token_counter=lambda s: 1,
    )

    # the accepted retry prompted over the budget-capped merged pool
    # (1 memory x 1 token), not the caller's 2 primary memories
    assert answer.context_tokens_estimate == 1


def test_inference_retry_threads_validity_into_sub_recall(monkeypatch, sample_memories):
    import mnemostack.recall.answer as answer_mod

    captured = {}

    class _StubRecaller:
        def recall(self, query, limit=10, filters=None, include_invalidated=False,
                   as_of=None, **_):
            captured["include_invalidated"] = include_invalidated
            captured["as_of"] = as_of
            return [
                RecallResult(id=5, text="fresh evidence", score=0.9,
                             payload={"source": "s.md"}, sources=["vector"]),
            ]

    llm = _PromptCapturingLLM(["draft\nCONFIDENCE: 0.1", "final\nCONFIDENCE: 0.9"])
    gen = AnswerGenerator(
        llm=llm,
        recaller=_StubRecaller(),
        specificity_resolver=False,
        inference_retry=True,
        retry_with_expansion=False,
    )
    monkeypatch.setattr(answer_mod, "decompose_query", lambda q, llm: ["sub question"])

    gen.generate(
        "why did it change",
        sample_memories,
        category="inference",
        as_of="2026-03-01",
    )

    # the retry sub-recall must run inside the same point-in-time view, not
    # default-current
    assert captured["as_of"] == "2026-03-01"
    assert captured["include_invalidated"] is False


def test_expansion_retry_filters_sub_recall_by_validity(sample_memories):
    stale = RecallResult(id=9, text="stale", score=0.99,
                         payload={"source": "x", "invalidated_at": "2026-07-04"})
    fresh = RecallResult(id=8, text="fresh", score=0.5, payload={"source": "y"})

    class _StubEmbedding:
        def embed_batch(self, texts):
            return [[0.1, 0.2]] * len(texts)

    class _StubRecaller:
        embedding = _StubEmbedding()

        def search_many(self, vectors, limit, filters=None, *,
                        include_invalidated=False, as_of=None):
            # search_many now filters for validity itself (per vector, before
            # RRF); mirror that so the retry pool matches the real behavior.
            from mnemostack.recall import filter_by_validity

            return filter_by_validity(
                [stale, fresh], include_invalidated=include_invalidated, as_of=as_of
            )

    llm = _PromptCapturingLLM(
        ["draft\nCONFIDENCE: 0.1", "r1\nr2\nhypothetical", "final\nCONFIDENCE: 0.9"]
    )
    gen = AnswerGenerator(
        llm=llm,
        recaller=_StubRecaller(),
        expansion_llm=llm,
        retry_with_expansion=True,
        specificity_resolver=False,
        inference_retry=False,
    )

    # default view: the invalidated hit must be filtered out of the retry pool
    gen.generate("what happened", sample_memories)
    retry_prompt = llm.prompts[-1]
    assert "fresh" in retry_prompt
    assert "stale" not in retry_prompt
