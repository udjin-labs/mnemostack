"""Tests for AnswerGenerator(prompt_overrides=...) — per-prompt localization."""

from __future__ import annotations

import pytest

from mnemostack.llm.base import LLMProvider, LLMResponse
from mnemostack.recall.answer import AnswerGenerator
from mnemostack.recall.recaller import RecallResult


class SequenceLLM(LLMProvider):
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts: list[str] = []

    @property
    def name(self) -> str:
        return "seq"

    def generate(self, prompt, max_tokens=200, temperature=0.0):
        self.prompts.append(prompt)
        return LLMResponse(text=self.responses.pop(0), tokens_used=1)


def _memories(n=3):
    return [
        RecallResult(id=str(i), text=f"Nachricht {i}: Teilnehmer A erstellte Eintrag {i}.", score=1.0, payload={})
        for i in range(1, n + 1)
    ]


DE_EXTRACT = """Lies den Speicher und extrahiere ALLE passenden Einträge.

FRAGE: {query}

SPEICHER:
{context}

Gib JSON zurück: {{"items": ["..."]}}"""

DE_FINALIZE = """Formatiere die endgültige Antwort.

FRAGE: {query}
EINTRÄGE: {items}

Bei Zählfragen nur die Zahl, sonst eine kommagetrennte Liste."""


def test_list_extract_and_finalize_overrides_are_used():
    llm = SequenceLLM(['{"items": ["Eintrag 1", "Eintrag 2"]}', "Eintrag 1, Eintrag 2"])
    gen = AnswerGenerator(
        llm=llm,
        category_aware_prompts=True,
        list_extract_mode=True,
        prompt_overrides={"list_extract": DE_EXTRACT, "list_finalize": DE_FINALIZE},
    )

    answer = gen.generate("Liste alle Einträge von Teilnehmer A auf", _memories(), category="list")

    assert answer.text == "Eintrag 1, Eintrag 2"
    assert llm.prompts[0].startswith("Lies den Speicher")
    assert "EINTRÄGE:" in llm.prompts[1]


def test_category_prompt_override_used():
    llm = SequenceLLM(["el 7 de mayo\nCONFIDENCE: 0.9"])
    es_temporal = "Responde la pregunta temporal.\nMEMORIA:\n{context}\nPREGUNTA: {query}\nRESPUESTA:"
    gen = AnswerGenerator(
        llm=llm,
        category_aware_prompts=True,
        prompt_overrides={"temporal": es_temporal},
    )

    answer = gen.generate("¿Cuándo ocurrió el evento?", _memories(), category="temporal")

    assert answer.text.startswith("el 7 de mayo")
    assert llm.prompts[0].startswith("Responde la pregunta temporal.")


def test_non_overridden_categories_keep_defaults():
    llm = SequenceLLM(["ok\nCONFIDENCE: 0.9"])
    gen = AnswerGenerator(
        llm=llm,
        category_aware_prompts=True,
        prompt_overrides={"temporal": "T: {context} {query}"},
    )

    gen.generate("Who is participant A?", _memories())

    assert "Answer a question based on retrieved memories" in llm.prompts[0]


def test_unknown_override_name_raises():
    with pytest.raises(ValueError, match="unknown prompt override"):
        AnswerGenerator(llm=SequenceLLM([]), prompt_overrides={"nonexistent": "x {context} {query}"})


def test_missing_placeholder_raises():
    with pytest.raises(ValueError, match="must contain"):
        AnswerGenerator(llm=SequenceLLM([]), prompt_overrides={"list_extract": "keine Platzhalter"})


def test_explicit_category_validates():
    with pytest.raises(ValueError, match="unknown category"):
        AnswerGenerator(llm=SequenceLLM([])).generate("q", _memories(), category="nope")


def test_explicit_category_routes_list_extract_for_non_english():
    # The built-in classifier is English-regex; explicit category is the
    # supported path for non-English pipelines that route classes themselves.
    llm = SequenceLLM(['{"items": ["a", "b"]}', "2"])
    gen = AnswerGenerator(llm=llm, category_aware_prompts=True, list_extract_mode=True)

    answer = gen.generate("Сколько записей создал участник A?", _memories(), category="count")

    assert answer.text == "2"
    assert "extract ALL items" in llm.prompts[0]


def test_custom_question_classifier_routes_categories():
    llm = SequenceLLM(['{"items": ["x", "y"]}', "x, y"])

    def my_classifier(query: str) -> str:
        return "list" if query.startswith("Liste") else "general"

    gen = AnswerGenerator(
        llm=llm,
        category_aware_prompts=True,
        list_extract_mode=True,
        question_classifier=my_classifier,
    )

    answer = gen.generate("Liste alle Einträge auf", _memories())

    assert answer.text == "x, y"
    assert "extract ALL items" in llm.prompts[0]


def test_misbehaving_classifier_falls_back_to_general():
    llm = SequenceLLM(["ok\nCONFIDENCE: 0.9"])
    gen = AnswerGenerator(
        llm=llm,
        category_aware_prompts=True,
        question_classifier=lambda q: "garbage-category",
    )

    answer = gen.generate("anything", _memories())

    assert answer.text == "ok"
    assert "Answer a question based on retrieved memories" in llm.prompts[0]
