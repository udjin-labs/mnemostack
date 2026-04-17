"""Tests for GraphResurrection stage.

Uses a fake neo4j driver injected via `driver=` so no live Memgraph needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from mnemostack.recall import RecallResult
from mnemostack.recall.pipeline import GraphResurrection, PipelineContext


def _make_driver(rows_per_seed: dict[str, list[dict]]):
    def _run(cypher, seed=None, lim=None, **kw):
        fake = MagicMock()
        fake.data.return_value = rows_per_seed.get(seed, [])
        return fake

    session = MagicMock()
    session.run.side_effect = _run
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    driver = MagicMock()
    driver.session.return_value = session
    return driver


def _ctx(query: str = "q") -> PipelineContext:
    return PipelineContext(query=query)


def test_resurrection_adds_neighbors():
    rows = {
        "mnemostack": [
            {"name": "LoCoMo", "type": "Benchmark", "mc": "world", "rel": "tested_on"},
            {"name": "Mem0", "type": "System", "mc": "world", "rel": "compared_with"},
        ],
    }
    stage = GraphResurrection(driver=_make_driver(rows))
    results: list[RecallResult] = []
    out = stage.apply(_ctx("mnemostack results"), results)
    names = {r.id for r in out}
    assert "graph:LoCoMo" in names
    assert "graph:Mem0" in names


def test_resurrection_skips_short_seeds():
    stage = GraphResurrection(driver=_make_driver({}), min_seed_len=4)
    out = stage.apply(_ctx("a b c"), [])
    assert out == []


def test_resurrection_dedupes_against_existing():
    rows = {"mnemostack": [{"name": "LoCoMo", "type": "B", "mc": "w", "rel": "x"}]}
    stage = GraphResurrection(driver=_make_driver(rows))
    existing = [
        RecallResult(id=1, text="LoCoMo is a benchmark", score=0.9, payload={"text": "LoCoMo is a benchmark"}, sources=["vector"]),
    ]
    out = stage.apply(_ctx("mnemostack results"), existing)
    # No graph result added because LoCoMo already present in text
    assert all(r.id != "graph:LoCoMo" for r in out)


def test_resurrection_multi_seed_higher_score():
    rows = {
        "mnemostack": [{"name": "shared", "type": "X", "mc": "w", "rel": "r1"}],
        "memgraph": [{"name": "shared", "type": "X", "mc": "w", "rel": "r2"}],
        "unique": [{"name": "only_one", "type": "X", "mc": "w", "rel": "r3"}],
    }
    stage = GraphResurrection(driver=_make_driver(rows), limit=5)
    out = stage.apply(_ctx("mnemostack memgraph unique"), [])
    scores = {r.id: r.score for r in out}
    assert scores["graph:shared"] > scores["graph:only_one"]


def test_resurrection_no_driver_is_noop():
    # Force driver resolution to None
    stage = GraphResurrection(driver=None, uri="bolt://invalid:0")
    # Monkey-patch internal to return None without trying real connection
    stage._get_driver = lambda: None  # type: ignore
    existing = [RecallResult(id=1, text="x", score=0.5, payload={"text": "x"}, sources=["vector"])]
    out = stage.apply(_ctx("mnemostack query"), existing)
    assert out == existing


def test_resurrection_cypher_failure_is_noop():
    session = MagicMock()
    session.run.side_effect = RuntimeError("boom")
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    driver = MagicMock()
    driver.session.return_value = session
    stage = GraphResurrection(driver=driver)
    existing = [RecallResult(id=1, text="x", score=0.5, payload={"text": "x"}, sources=["vector"])]
    out = stage.apply(_ctx("mnemostack seed"), existing)
    # Should silently return untouched results
    assert out == existing


def test_resurrection_respects_limit():
    rows = {
        "project": [
            {"name": f"n{i}", "type": "T", "mc": "w", "rel": "r"}
            for i in range(10)
        ],
    }
    stage = GraphResurrection(driver=_make_driver(rows), limit=2, max_per_seed=10)
    out = stage.apply(_ctx("project search"), [])
    graph_results = [r for r in out if r.id.startswith("graph:")]
    assert len(graph_results) == 2
