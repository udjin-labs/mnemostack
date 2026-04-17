"""Tests for TripleExtractor with FakeLLM."""
import pytest

from mnemostack.graph import TripleExtractor
from mnemostack.llm.base import LLMProvider, LLMResponse


class FakeLLM(LLMProvider):
    def __init__(self, response="", error=None):
        self.response = response
        self.error = error
        self.last_prompt = ""

    @property
    def name(self):
        return "fake"

    def generate(self, prompt, max_tokens=200, temperature=0.0):
        self.last_prompt = prompt
        if self.error:
            return LLMResponse(text="", error=self.error)
        return LLMResponse(text=self.response)


def test_extract_basic():
    llm = FakeLLM(response='[{"subject": "alice", "predicate": "works_on", "object": "project-x", "valid_from": "2024-01-01"}]')
    ext = TripleExtractor(llm=llm)
    triples = ext.extract("Alice started project-x on Jan 1, 2024")
    assert len(triples) == 1
    t = triples[0]
    assert t.subject == "alice"
    assert t.predicate == "works_on"
    assert t.obj == "project-x"
    assert t.valid_from == "2024-01-01"


def test_extract_multiple():
    response = '''[
        {"subject": "alice", "predicate": "works_on", "object": "project-x", "valid_from": null},
        {"subject": "bob", "predicate": "manages", "object": "team-a", "valid_from": "2023-06-01"}
    ]'''
    llm = FakeLLM(response=response)
    ext = TripleExtractor(llm=llm)
    triples = ext.extract("Alice works on project-x. Bob manages team-a since June 2023.")
    assert len(triples) == 2
    assert triples[0].valid_from is None
    assert triples[1].valid_from == "2023-06-01"


def test_extract_strips_markdown_fences():
    response = '```json\n[{"subject": "s", "predicate": "p", "object": "o", "valid_from": null}]\n```'
    llm = FakeLLM(response=response)
    ext = TripleExtractor(llm=llm)
    triples = ext.extract("s p o")
    assert len(triples) == 1


def test_extract_handles_leading_prose():
    response = 'Sure, here are the triples:\n[{"subject": "s", "predicate": "p", "object": "o"}]\nDone!'
    llm = FakeLLM(response=response)
    ext = TripleExtractor(llm=llm)
    triples = ext.extract("x")
    assert len(triples) == 1
    assert triples[0].subject == "s"


def test_extract_empty_array():
    llm = FakeLLM(response="[]")
    ext = TripleExtractor(llm=llm)
    assert ext.extract("irrelevant text") == []


def test_extract_malformed_json():
    llm = FakeLLM(response="not json at all")
    ext = TripleExtractor(llm=llm)
    assert ext.extract("x") == []


def test_extract_llm_error():
    llm = FakeLLM(error="timeout")
    ext = TripleExtractor(llm=llm)
    assert ext.extract("x") == []


def test_extract_empty_text_skips_llm():
    llm = FakeLLM(response='[{"subject":"s","predicate":"p","object":"o"}]')
    ext = TripleExtractor(llm=llm)
    assert ext.extract("   ") == []
    assert llm.last_prompt == ""


def test_extract_filters_incomplete_triples():
    response = '''[
        {"subject": "", "predicate": "p", "object": "o"},
        {"subject": "s", "predicate": "p", "object": ""},
        {"subject": "s", "predicate": "p", "object": "o"}
    ]'''
    llm = FakeLLM(response=response)
    ext = TripleExtractor(llm=llm)
    triples = ext.extract("x")
    assert len(triples) == 1  # only the complete one


def test_extract_respects_max_triples():
    items = ",".join(
        f'{{"subject":"s{i}","predicate":"p","object":"o{i}"}}'
        for i in range(20)
    )
    llm = FakeLLM(response=f"[{items}]")
    ext = TripleExtractor(llm=llm, max_triples=5)
    triples = ext.extract("x")
    assert len(triples) == 5


def test_prompt_includes_text_and_max_triples():
    llm = FakeLLM(response="[]")
    ext = TripleExtractor(llm=llm, max_triples=7)
    ext.extract("Sample text here")
    assert "Sample text here" in llm.last_prompt
    assert "7" in llm.last_prompt  # max_triples
