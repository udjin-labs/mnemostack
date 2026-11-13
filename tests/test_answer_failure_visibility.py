"""Incomplete extraction and failed optional retries must remain observable."""
from types import SimpleNamespace

import pytest

import mnemostack.recall.answer as answer_module
from mnemostack.llm.base import LLMResponse
from mnemostack.recall import AnswerGenerator, RecallResult
from mnemostack.recall.specificity import resolve_specificity


def memories():
    return [RecallResult(id=str(i), text="Alpha is recorded here.", score=1.0, payload={"source": f"synthetic-{i}"}) for i in range(2)]


@pytest.mark.parametrize("category", ["list", "count"])
@pytest.mark.parametrize("finalize", ["llm", "verbatim"])
@pytest.mark.parametrize("failed_index", [0, 1])
@pytest.mark.parametrize("failure", ["status", "malformed"])
def test_partial_extract_is_not_a_complete_list_or_count(category, finalize, failed_index, failure):
    calls = []
    bad = LLMResponse(text="", error="synthetic provider fault") if failure == "status" else LLMResponse(text="not valid JSON")
    replies = [LLMResponse(text='{"items": ["Alpha"]}'), LLMResponse(text='{"items": ["Alpha"]}')]
    replies[failed_index] = bad

    def generate(*args, **kwargs):
        index = len(calls)
        calls.append(1)
        return replies[index] if index < 2 else LLMResponse(text="Alpha")

    recaller = SimpleNamespace(embedding=object())
    gen = AnswerGenerator(llm=SimpleNamespace(generate=generate), list_extract_mode=True, list_extract_batch_size=1, list_finalize=finalize, retry_with_expansion=True, expansion_llm=SimpleNamespace(generate=lambda *a, **k: pytest.fail("incomplete extraction must not trigger extra synthesis")), recaller=recaller)
    result = gen.generate("List the recorded items", memories(), category=category)
    assert not result.ok
    assert result.error == "list extraction incomplete"
    assert result.confidence == 0.0
    assert result.text == ("Alpha" if category == "list" else "")
    assert len(calls) == 2
    assert len(result.context_memories) == 2


@pytest.mark.parametrize("category,expected", [("list", "Alpha"), ("count", "1")])
def test_complete_extraction_keeps_existing_success_contract(category, expected):
    llm = SimpleNamespace(generate=lambda *a, **k: LLMResponse(text='{"items": ["Alpha"]}'))
    result = AnswerGenerator(llm, list_extract_mode=True, list_extract_batch_size=1, list_finalize="verbatim").generate("List the recorded items", memories(), category=category)
    assert result.ok and result.error is None
    assert result.text == expected and result.confidence == 0.8


def test_retry_embedding_failure_keeps_draft_and_logs_only_error_type(monkeypatch, caplog):
    def fail(*args, **kwargs):
        raise ValueError("PRIVATE_SYNTHETIC_MARKER")

    monkeypatch.setattr(answer_module, "embed_queries_via", fail)
    gen = AnswerGenerator(SimpleNamespace(), expansion_llm=SimpleNamespace(generate=lambda *a, **k: LLMResponse(text="first\nsecond\nhypothetical")), recaller=SimpleNamespace(embedding=object()))
    draft = answer_module.Answer(text="draft", confidence=0.1)
    pool = memories()
    result, context = gen._retry_with_expansion_answer("question", pool, draft, "unused")
    assert result is draft and context is pool
    assert caplog.records and "ValueError" in caplog.text
    assert "PRIVATE_SYNTHETIC_MARKER" not in caplog.text


def test_inference_subrecall_failures_are_visible_without_leaking_query_or_error(monkeypatch, caplog):
    monkeypatch.setattr(answer_module, "decompose_query", lambda *a: ["first", "second"])
    calls = []

    def recall(query, **kwargs):
        calls.append((query, kwargs))
        raise RuntimeError("PRIVATE_SYNTHETIC_MARKER")

    gen = AnswerGenerator(SimpleNamespace(), recaller=SimpleNamespace(recall=recall))
    draft = answer_module.Answer(text="draft", confidence=0.1)
    gen._generate_single_prompt = lambda **kwargs: draft
    result, _ = gen._retry_inference_answer("question", memories(), draft, "unused", {"project": "p"}, tenant="alpha")
    assert result is draft
    assert len(calls) == 2 and all(k["tenant"] == "alpha" and k["filters"] == {"project": "p"} for _, k in calls)
    assert len(caplog.records) == 2 and "RuntimeError" in caplog.text
    assert "PRIVATE_SYNTHETIC_MARKER" not in caplog.text


@pytest.mark.parametrize("failure", ["raise", "status"])
def test_specificity_failure_is_visible_and_keeps_draft(failure, caplog):
    calls = []

    def generate(*args, **kwargs):
        calls.append(1)
        if failure == "raise":
            raise RuntimeError("PRIVATE_SYNTHETIC_MARKER")
        return LLMResponse(text="", error="PRIVATE_SYNTHETIC_MARKER")

    assert resolve_specificity("question", "Their friend.", memories(), SimpleNamespace(generate=generate)) == "Their friend."
    assert len(calls) == 1 and len(caplog.records) == 1
    assert "PRIVATE_SYNTHETIC_MARKER" not in caplog.text
