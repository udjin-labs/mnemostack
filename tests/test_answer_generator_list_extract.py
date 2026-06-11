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
    llm = SequenceLLM(
        [
            '{"items": ["Luna", "Oliver"]}',
            '{"items": ["Oliver"]}',  # second batch (45 memories, batch=40)
            "Luna, Oliver",
        ]
    )
    gen = AnswerGenerator(llm=llm, category_aware_prompts=True, list_extract_mode=True)

    answer = gen.generate("What are Melanie's pets?", memories)

    assert answer.text == "Luna, Oliver"
    assert answer.confidence == 0.8
    assert len(llm.prompts) == 3
    assert "extract ALL items matching the question" in llm.prompts[0]
    assert "extract ALL items matching the question" in llm.prompts[1]
    # items merged across batches, deduplicated, first-seen order
    assert 'EXTRACTED ITEMS: ["Luna", "Oliver"]' in llm.prompts[2]


def test_list_extract_falls_back_on_empty_items(memories):
    llm = SequenceLLM(
        [
            '{"items": []}',
            '{"items": []}',  # both batches of pass 1 empty
            '{"items": []}',
            '{"items": []}',  # retry pass also empty
            "Not in memory.\nCONFIDENCE: 0.9",
        ]
    )
    gen = AnswerGenerator(llm=llm, category_aware_prompts=True, list_extract_mode=True)

    answer = gen.generate("What are Melanie's pets?", memories)

    assert answer.text == "Not in memory."
    assert answer.confidence == 0.3
    assert len(llm.prompts) == 5  # 2 batches + 2 retry batches + fallback
    assert _LIST_PROMPT.split("\n", 1)[0] in llm.prompts[4]
    assert "Memory 16" not in llm.prompts[4]


@pytest.mark.parametrize("extract_output", ["not json", "[]", "{}"])
def test_list_extract_falls_back_on_malformed_json(memories, extract_output):
    llm = SequenceLLM(
        [
            extract_output,
            extract_output,  # second batch equally malformed
            "Luna, Oliver\nCONFIDENCE: 0.8",
        ]
    )
    gen = AnswerGenerator(llm=llm, category_aware_prompts=True, list_extract_mode=True)

    answer = gen.generate("What are Melanie's pets?", memories)

    assert answer.text == "Luna, Oliver"
    assert answer.confidence == 0.3
    # every batch failed -> direct fallback, NO empty-retry pass
    assert len(llm.prompts) == 3
    assert _LIST_PROMPT.split("\n", 1)[0] in llm.prompts[2]


def test_list_extract_finalize_failure_returns_extracted_items(memories):
    # Extract succeeded — a finalize hiccup must not discard verified items
    # by re-generating (which can also leak default-language output when
    # prompts are localized). Deterministic join instead, no extra LLM call.
    llm = SequenceMaybeFailLLM(
        [
            LLMResponse(text='{"items": ["Luna", "Oliver"]}', tokens_used=10),
            LLMResponse(text='{"items": []}', tokens_used=10),
            LLMResponse(text="", error="rate limited"),
        ]
    )
    gen = AnswerGenerator(llm=llm, category_aware_prompts=True, list_extract_mode=True)

    answer = gen.generate("What are Melanie's pets?", memories)

    assert answer.ok is True
    assert answer.text == "Luna, Oliver"
    assert answer.confidence == 0.6
    assert len(llm.prompts) == 3  # 2 extract batches + failed finalize, NO re-generation


def test_list_extract_finalize_failure_count_returns_number(memories):
    llm = SequenceMaybeFailLLM(
        [
            LLMResponse(text='{"items": ["a", "b", "c"]}', tokens_used=10),
            LLMResponse(text='{"items": []}', tokens_used=10),
            LLMResponse(text="", error="rate limited"),
        ]
    )
    gen = AnswerGenerator(llm=llm, category_aware_prompts=True, list_extract_mode=True)

    answer = gen.generate("How many pets does Melanie have?", memories)

    assert answer.text == "3"
    assert answer.confidence == 0.6


def test_abstention_text_is_localizable():
    llm = SequenceMaybeFailLLM([])
    gen = AnswerGenerator(llm=llm, abstention_text="Нет в памяти.")

    answer = gen.generate("anything", [])

    assert answer.text == "Нет в памяти."
    assert answer.confidence == 0.0


def test_parse_extracted_items_deduplicates_preserving_order():
    items = AnswerGenerator._parse_extracted_items(
        '{"items": ["Luna", "Oliver", "Luna", "", 42, "Bailey", "Oliver"]}'
    )

    assert items == ["Luna", "Oliver", "Bailey"]


def test_list_extract_passes_more_memories_than_max_memories(memories):
    llm = SequenceLLM(
        [
            '{"items": ["item"]}',
            '{"items": []}',
            "item",
        ]
    )
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
    # the tail of the pool is no longer dropped — it lands in the next batch
    assert "scan 5 memories" in llm.prompts[1]
    assert "Memory 41" in llm.prompts[1]


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

    count_llm = SequenceLLM(['{"items": ["Luna", "Oliver"]}', '{"items": []}', "2"])
    count_gen = AnswerGenerator(
        llm=count_llm,
        category_aware_prompts=True,
        list_extract_mode=True,
    )
    count_gen.generate("How many pets does Melanie have?", memories)
    assert len(count_llm.prompts) == 3
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
    llm = SequenceLLM(
        [
            '{"items": ["Oliver", "Luna", "Bailey"]}',
            '{"items": []}',
            "Oliver, Luna, Bailey",
        ]
    )
    gen = AnswerGenerator(llm=llm, category_aware_prompts=True, list_extract_mode=True)

    answer = gen.generate("What are Melanie's pets?", memories)

    assert answer.text == "Oliver, Luna, Bailey"
    items_line = next(
        line for line in llm.prompts[2].splitlines() if line.startswith("EXTRACTED ITEMS:")
    )
    assert items_line == 'EXTRACTED ITEMS: ["Oliver", "Luna", "Bailey"]'
    assert "her dog" not in items_line


def test_localized_abstention_reaches_builtin_prompts(memories):
    """The built-in prompts instruct the literal English marker — with a
    configured abstention text the instruction must use that text instead,
    or localized deployments still get English abstentions back."""
    llm = SequenceLLM(["irgendeine Antwort"])
    gen = AnswerGenerator(
        llm=llm,
        abstention_text="Nicht im Speicher.",
        specificity_resolver=False,
        inference_retry=False,
    )

    gen.generate("wann ist es passiert?", memories[:3])

    prompt = llm.prompts[0]
    assert "Nicht im Speicher." in prompt
    assert "Not in memory" not in prompt


def test_default_abstention_prompts_unchanged(memories):
    llm = SequenceLLM(["some answer"])
    gen = AnswerGenerator(llm=llm, specificity_resolver=False, inference_retry=False)

    gen.generate("when did it happen?", memories[:3])

    assert 'reply: "Not in memory."' in llm.prompts[0]


def test_prompt_overrides_left_verbatim_under_localized_abstention(memories):
    """Override authors write their own abstention line — localization must
    not rewrite their template."""
    override = "Réponds: {query}\n{context}\nSi rien trouvé: \"Pas en mémoire.\""
    llm = SequenceLLM(["réponse"])
    gen = AnswerGenerator(
        llm=llm,
        abstention_text="Nicht im Speicher.",
        prompt_overrides={"general": override},
        specificity_resolver=False,
        inference_retry=False,
    )

    gen.generate("une question quelconque", memories[:3])

    assert "Pas en mémoire." in llm.prompts[0]
    assert "Nicht im Speicher." not in llm.prompts[0]


def test_should_retry_detects_localized_abstention():
    from mnemostack.recall.inference_retry import should_retry

    assert should_retry("Нет в памяти.", 0.9, abstention_text="Нет в памяти.") is True
    assert should_retry("eine echte Antwort", 0.9, abstention_text="Nicht im Speicher.") is False
    # the English literal is still recognized regardless of configuration
    assert should_retry("Not in memory.", 0.9, abstention_text="Nicht im Speicher.") is True


def test_extract_finds_items_beyond_first_window(memories):
    """Order-insensitivity: an item that only appears in the SECOND batch
    (below the old [:40] cut) must still be extracted — pool position no
    longer decides whether a memory is seen."""
    llm = SequenceLLM(
        [
            '{"items": []}',  # nothing relevant in the first 40
            '{"items": ["Bailey"]}',  # the relevant memory sits at position 41+
            "Bailey",
        ]
    )
    gen = AnswerGenerator(llm=llm, category_aware_prompts=True, list_extract_mode=True)

    answer = gen.generate("What are Melanie's pets?", memories)

    assert answer.text == "Bailey"
    assert answer.confidence == 0.8
    assert "Memory 41" in llm.prompts[1]


def test_extract_retries_once_on_empty_then_succeeds(memories):
    """A non-empty pool with an empty extract gets ONE retry pass before
    abstaining — extraction is the non-deterministic step, not retrieval."""
    llm = SequenceLLM(
        [
            '{"items": []}',
            '{"items": []}',  # pass 1: both batches empty
            '{"items": ["Luna"]}',
            '{"items": []}',  # retry pass recovers the item
            "Luna",
        ]
    )
    gen = AnswerGenerator(llm=llm, category_aware_prompts=True, list_extract_mode=True)

    answer = gen.generate("What are Melanie's pets?", memories)

    assert answer.text == "Luna"
    assert answer.confidence == 0.8
    assert len(llm.prompts) == 5  # 2 + 2 retry + finalize


def test_verbatim_finalize_skips_llm_pass(memories):
    """list_finalize="verbatim" assembles the answer deterministically from
    the extracted items — no second LLM pass to paraphrase or distort them."""
    llm = SequenceLLM(
        [
            '{"items": ["Luna", "Oliver"]}',
            '{"items": []}',
            # NO finalize response — it must not be requested
        ]
    )
    gen = AnswerGenerator(
        llm=llm,
        category_aware_prompts=True,
        list_extract_mode=True,
        list_finalize="verbatim",
    )

    answer = gen.generate("What are Melanie's pets?", memories)

    assert answer.text == "Luna, Oliver"
    assert answer.confidence == 0.8
    assert len(llm.prompts) == 2  # extract batches only


def test_verbatim_finalize_count_returns_number(memories):
    llm = SequenceLLM(['{"items": ["a", "b", "c"]}', '{"items": []}'])
    gen = AnswerGenerator(
        llm=llm,
        category_aware_prompts=True,
        list_extract_mode=True,
        list_finalize="verbatim",
    )

    answer = gen.generate("How many pets does Melanie have?", memories)

    assert answer.text == "3"
    assert answer.confidence == 0.8


def test_partial_batch_failure_keeps_other_batches(memories):
    """One failed batch must not discard items the other batches produced."""
    llm = SequenceMaybeFailLLM(
        [
            LLMResponse(text="", error="rate limited"),  # batch 1 fails
            LLMResponse(text='{"items": ["Luna"]}', tokens_used=10),
            LLMResponse(text="Luna", tokens_used=10),
        ]
    )
    gen = AnswerGenerator(llm=llm, category_aware_prompts=True, list_extract_mode=True)

    answer = gen.generate("What are Melanie's pets?", memories)

    assert answer.text == "Luna"
    assert answer.confidence == 0.8


def test_invalid_list_finalize_rejected():
    llm = SequenceLLM([])
    with pytest.raises(ValueError, match="list_finalize"):
        AnswerGenerator(llm=llm, list_finalize="loose")
    with pytest.raises(ValueError, match="list_extract_batch_size"):
        AnswerGenerator(llm=llm, list_extract_batch_size=0)


def test_sources_come_from_the_batch_that_produced_items(memories):
    """An item found deep in the pool must cite its own batch's sources —
    _extract_sources truncates to 5, so attributing from the pool prefix
    would cite unrelated early memories and omit the real one."""
    llm = SequenceLLM(
        [
            '{"items": []}',  # first 40 memories: nothing
            '{"items": ["Bailey"]}',  # found in memories 41-45
        ]
    )
    gen = AnswerGenerator(
        llm=llm,
        category_aware_prompts=True,
        list_extract_mode=True,
        list_finalize="verbatim",
    )

    answer = gen.generate("What are Melanie's pets?", memories)

    assert answer.text == "Bailey"
    # sources from the second batch (chat-41..45), not the pool prefix
    assert answer.sources
    assert all(src in {f"chat-{i}" for i in range(41, 46)} for src in answer.sources)


def test_sources_prefer_memories_containing_the_items(memories):
    """Within a 40-memory batch the 5-source truncation could drop the memory
    that actually holds the answer — memories literally containing an
    extracted item are cited first."""
    llm = SequenceLLM(
        [
            '{"items": ["Melanie detail 30"]}',  # matches memory 30, deep in batch 1
            '{"items": []}',
        ]
    )
    gen = AnswerGenerator(
        llm=llm,
        category_aware_prompts=True,
        list_extract_mode=True,
        list_finalize="verbatim",
    )

    answer = gen.generate("What are Melanie's pets?", memories)

    assert answer.sources
    assert answer.sources[0] == "chat-30"  # the supporting memory leads


def test_item_provenance_requires_word_boundaries():
    """A short item must not claim a memory via substring inside a longer
    word ("Ann" in "annual") — boundary-anchored match only."""
    from mnemostack.recall import RecallResult

    def _mem(i, text):
        return RecallResult(
            id=str(i),
            text=text,
            score=1.0 - i * 0.01,
            payload={"source": f"chat-{i}"},
            sources=["vector"],
        )

    pool = [
        _mem(1, "the annual report was published in May, maybe earlier"),
        _mem(2, "participant Ann joined the call"),
    ]
    llm = SequenceLLM(['{"items": ["Ann"]}'])
    gen = AnswerGenerator(
        llm=llm,
        category_aware_prompts=True,
        list_extract_mode=True,
        list_finalize="verbatim",
    )

    answer = gen.generate("List the participants of the call", pool)

    assert answer.text == "Ann"
    assert answer.sources[0] == "chat-2"  # not chat-1 via "annual"
