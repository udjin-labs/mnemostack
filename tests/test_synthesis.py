from __future__ import annotations

import json

import pytest

from mnemostack.cli import build_parser
from mnemostack.llm.base import LLMResponse
from mnemostack.recall.recaller import RecallResult
from mnemostack.synthesis import SynthesisFact, SynthesisResult, dumps, synthesize


class FakeRecaller:
    def __init__(self, results):
        self.results = results

    def recall(self, query, limit=10, filters=None):
        return self.results[:limit]


class FakeRetriever:
    name = "bm25"

    def search(self, query, limit=20, filters=None):
        return [
            RecallResult(
                id="r1",
                text=f"{query} founded Project Atlas.",
                score=0.7,
                payload={"timestamp": "2026-01-02T03:04:05Z"},
                sources=["bm25"],
            )
        ]


class FakeLLM:
    name = "fake"

    def generate(self, prompt, max_tokens=200, temperature=0.0):
        return LLMResponse(text="Alice is connected to Project Atlas.")


def test_synthesize_builds_structured_result_and_deduplicates():
    results = [
        RecallResult(
            id="1",
            text="Alice founded Project Atlas.",
            score=0.9,
            payload={"timestamp": "2026-02-01T00:00:00Z", "source": "notes.md"},
            sources=["vector"],
        ),
        RecallResult(
            id="2",
            text="Alice founded Project Atlas.",
            score=0.8,
            payload={"timestamp": "2026-02-01T00:00:00Z"},
            sources=["bm25"],
        ),
        RecallResult(
            id="3",
            text="Alice-[WORKS_WITH]->Bob; Alice-[OWNS]->Project Atlas",
            score=0.5,
            payload={"source": "memgraph", "name": "Alice"},
            sources=["memgraph"],
        ),
    ]

    out = synthesize("Alice", recaller=FakeRecaller(results), max_results=10)

    assert out.entity == "Alice"
    assert [f.text for f in out.facts] == [
        "Alice founded Project Atlas.",
        "Alice-[WORKS_WITH]->Bob; Alice-[OWNS]->Project Atlas",
    ]
    assert [f.timestamp for f in out.timeline] == ["2026-02-01T00:00:00Z"]
    assert out.related_entities == ["Bob", "Project Atlas"]


def test_synthesize_filters_sources():
    results = [
        RecallResult(id="v", text="vector fact", score=1.0, sources=["vector"]),
        RecallResult(id="g", text="graph fact", score=1.0, sources=["memgraph"]),
    ]

    out = synthesize("Alice", sources=["graph"], recaller=FakeRecaller(results))

    assert [f.text for f in out.facts] == ["graph fact"]


def test_synthesize_gracefully_degrades_without_backends():
    out = synthesize("Missing Entity")

    assert out.facts == []
    assert out.related_entities == []
    assert out.timeline == []
    assert "(no facts found)" in out.markdown()


def test_synthesize_uses_direct_retrievers_and_llm_summary():
    out = synthesize(
        "Alice",
        retrievers=[FakeRetriever()],
        llm_summarize=True,
        llm=FakeLLM(),
    )

    assert out.summary == "Alice is connected to Project Atlas."
    assert out.facts[0].source == "bm25"


def test_result_renderers():
    result = SynthesisResult(
        entity="Alice",
        facts=[SynthesisFact(text="Alice founded Atlas", source="vector", relevance_score=0.25)],
        related_entities=["Atlas"],
    )

    payload = result.to_json()
    assert payload["entity"] == "Alice"
    assert payload["facts"][0]["source"] == "vector"
    assert "# Alice" in result.markdown()
    assert json.loads(dumps(result, "json"))["related_entities"] == ["Atlas"]


def test_cli_has_synthesize_command():
    parser = build_parser()
    args = parser.parse_args(["synthesize", "Alice", "--format", "json", "--source", "graph"])

    assert args.entity == "Alice"
    assert args.format == "json"
    assert args.source == ["graph"]


def test_synthesize_validates_inputs():
    with pytest.raises(ValueError):
        synthesize("   ")
    with pytest.raises(ValueError):
        synthesize("Alice", format="xml")
    with pytest.raises(ValueError):
        synthesize("Alice", max_results=0)
