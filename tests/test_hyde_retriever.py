"""Unit tests for HyDERetriever.

We fake the LLM, embedding, and vector store so the test is pure and fast.
Real-prod verification against Qdrant + Gemini is done separately (see the
accompanying smoke script) — remember the mocks-can-lie lesson.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mnemostack.recall import HyDERetriever
from mnemostack.recall.recaller import RecallResult


class _FakeLLM:
    def __init__(self, text: str = "A short hypothetical answer."):
        self._text = text
        self.calls = 0

    def generate(self, prompt, max_tokens=120, temperature=0.0):
        self.calls += 1
        from mnemostack.llm.base import LLMResponse
        return LLMResponse(text=self._text, tokens_used=len(self._text.split()))


class _EmptyLLM:
    def generate(self, *a, **kw):
        from mnemostack.llm.base import LLMResponse
        return LLMResponse(text="", tokens_used=0)


class _RaisingLLM:
    def generate(self, *a, **kw):
        raise RuntimeError("LLM is down")


class _FakeEmbedding:
    dimension = 3

    def __init__(self, returns=None):
        self.last_text = None
        self._returns = returns if returns is not None else [0.1, 0.2, 0.3]

    def embed(self, text: str):
        self.last_text = text
        return list(self._returns) if text else []


@dataclass
class _Hit:
    id: str
    score: float
    payload: dict[str, Any]


class _FakeVectorStore:
    def __init__(self, results=None):
        self._results = results or []
        self.last_query = None
        self.last_limit = None
        self.last_filters = None

    def search(self, vector, limit=20, filters=None):
        self.last_query = list(vector)
        self.last_limit = limit
        self.last_filters = filters
        return list(self._results)


def _hits(n: int):
    return [
        _Hit(id=f"mem-{i}", score=1.0 - 0.1 * i, payload={"text": f"memory {i}"})
        for i in range(n)
    ]


def test_hyde_calls_llm_then_embedding_then_vector_search():
    llm = _FakeLLM("Caroline plans to study psychology and counselling.")
    emb = _FakeEmbedding(returns=[0.9, 0.8, 0.7])
    store = _FakeVectorStore(results=_hits(3))
    r = HyDERetriever(llm=llm, embedding=emb, vector_store=store)
    results = r.search("what fields would she likely pursue?", limit=3)

    assert llm.calls == 1
    # The embedding should have been given the LLM output, NOT the query
    assert emb.last_text == "Caroline plans to study psychology and counselling."
    assert store.last_query == [0.9, 0.8, 0.7]
    assert store.last_limit == 3
    assert len(results) == 3
    assert all(isinstance(r, RecallResult) for r in results)
    assert all("hyde" in r.sources for r in results)


def test_hyde_empty_on_empty_llm_output():
    r = HyDERetriever(
        llm=_EmptyLLM(), embedding=_FakeEmbedding(), vector_store=_FakeVectorStore()
    )
    assert r.search("anything") == []


def test_hyde_empty_on_raising_llm():
    r = HyDERetriever(
        llm=_RaisingLLM(), embedding=_FakeEmbedding(), vector_store=_FakeVectorStore()
    )
    assert r.search("anything") == []


def test_hyde_empty_on_empty_embedding():
    class _NoEmbed(_FakeEmbedding):
        def embed(self, text):
            return []

    r = HyDERetriever(
        llm=_FakeLLM("some text"),
        embedding=_NoEmbed(),
        vector_store=_FakeVectorStore(results=_hits(2)),
    )
    assert r.search("q") == []


def test_hyde_forwards_filters():
    store = _FakeVectorStore(results=_hits(1))
    r = HyDERetriever(
        llm=_FakeLLM("note"),
        embedding=_FakeEmbedding(),
        vector_store=store,
    )
    r.search("q", limit=5, filters={"chunk_type": "transcript"})
    assert store.last_filters == {"chunk_type": "transcript"}
    assert store.last_limit == 5


def test_hyde_generates_hypothetical_answer():
    llm = _FakeLLM("A plausible memory-shaped answer.")

    hypo = HyDERetriever.generate_hypothetical("what happened?", llm)

    assert hypo == "A plausible memory-shaped answer."
    assert llm.calls == 1
