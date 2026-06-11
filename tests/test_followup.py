"""Tests for follow-up question rewriting."""

from __future__ import annotations

import pytest

from mnemostack.llm.base import LLMProvider, LLMResponse
from mnemostack.recall import rewrite_followup


class FakeLLM(LLMProvider):
    def __init__(self, response: str = "", error: str | None = None):
        self.response = response
        self.error = error
        self.calls = 0
        self.last_prompt = ""

    @property
    def name(self) -> str:
        return "fake"

    def generate(self, prompt, max_tokens=200, temperature=0.0):
        self.calls += 1
        self.last_prompt = prompt
        if self.error:
            return LLMResponse(text="", error=self.error)
        return LLMResponse(text=self.response)


HISTORY = [
    ("Wer hat das Deployment freigegeben?", "Teilnehmer B hat es freigegeben."),
    ("когда это случилось?", "Это случилось 3 июня."),
]


def test_rewrites_dependent_question():
    llm = FakeLLM("Wer hat das Deployment am 3. Juni freigegeben?")

    out = rewrite_followup("und wer genau?", HISTORY, llm)

    assert out == "Wer hat das Deployment am 3. Juni freigegeben?"
    assert "Q: Wer hat das Deployment freigegeben?" in llm.last_prompt
    assert "A: Это случилось 3 июня." in llm.last_prompt
    assert "FOLLOW-UP QUESTION: und wer genau?" in llm.last_prompt


def test_no_history_returns_query_without_llm_call():
    llm = FakeLLM("should not be used")

    assert rewrite_followup("a standalone question", [], llm) == "a standalone question"
    assert llm.calls == 0


def test_needs_rewrite_trigger_skips_llm():
    llm = FakeLLM("should not be used")

    out = rewrite_followup(
        "Wer hat das Deployment freigegeben?",
        HISTORY,
        llm,
        needs_rewrite=lambda q: False,
    )

    assert out == "Wer hat das Deployment freigegeben?"
    assert llm.calls == 0


def test_llm_error_is_fail_open():
    llm = FakeLLM(error="rate limited")

    assert rewrite_followup("und wer?", HISTORY, llm) == "und wer?"


def test_empty_response_is_fail_open():
    llm = FakeLLM("   ")

    assert rewrite_followup("und wer?", HISTORY, llm) == "und wer?"


def test_echoed_label_is_stripped():
    llm = FakeLLM("STANDALONE_QUESTION: кто запустил третий опрос?")

    out = rewrite_followup("а кто запустил?", HISTORY, llm)

    assert out == "кто запустил третий опрос?"


def test_max_history_trims_old_entries():
    llm = FakeLLM("ok")
    history = [(f"question {i}?", f"answer {i}") for i in range(10)]

    rewrite_followup("and then?", history, llm, max_history=2)

    assert "question 9?" in llm.last_prompt
    assert "question 8?" in llm.last_prompt
    assert "question 7?" not in llm.last_prompt


def test_string_history_entries_supported():
    llm = FakeLLM("ok")

    rewrite_followup("and then?", ["participant A: the report is ready"], llm)

    assert "participant A: the report is ready" in llm.last_prompt


def test_custom_prompt_template_validated():
    llm = FakeLLM("ok")

    with pytest.raises(ValueError, match="prompt_template"):
        rewrite_followup("q?", HISTORY, llm, prompt_template="no placeholders here")

    out = rewrite_followup(
        "q?", HISTORY, llm, prompt_template="H:{history}\nQ:{query}\nOUT:"
    )
    assert out == "ok"
    assert "H:Q: Wer hat das Deployment freigegeben?" in llm.last_prompt
