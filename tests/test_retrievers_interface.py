from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from mnemostack.recall import (
    BM25Doc,
    BM25Retriever,
    Hit,
    MemgraphRetriever,
    RecallResult,
    TemporalRetriever,
    VectorRetriever,
)


class _FakeEmbedding:
    dimension = 3

    def embed(self, text: str):
        return [0.1, 0.2, 0.3] if text else []


@dataclass
class _StoreHit:
    id: str
    score: float
    payload: dict[str, Any]


class _FakeVectorStore:
    def __init__(self, hits: list[_StoreHit]):
        self.hits = hits

    def search(self, vector, limit=20, filters=None):
        return self.hits[:limit]


def _assert_hit_list(results: list[Hit], source: str) -> None:
    assert isinstance(results, list)
    assert results
    assert all(isinstance(item, RecallResult) for item in results)
    assert all(source in item.sources for item in results)


def test_vector_retriever_implements_common_retrieve_interface():
    retriever = VectorRetriever(
        embedding=_FakeEmbedding(),
        vector_store=_FakeVectorStore([
            _StoreHit(id="v1", score=0.9, payload={"text": "vector memory"}),
        ]),
    )

    _assert_hit_list(retriever.retrieve("vector", limit=1), "vector")


def test_bm25_retriever_implements_common_retrieve_interface():
    retriever = BM25Retriever(docs=[BM25Doc(id="b1", text="exact token memory")])

    _assert_hit_list(retriever.retrieve("exact token", limit=1), "bm25")


def test_memgraph_retriever_implements_common_retrieve_interface():
    session = MagicMock()

    def _run(cypher, **kwargs):
        result = MagicMock()
        if "RETURN n.name AS name" in cypher:
            result.data.return_value = [{"name": "Alice", "type": "Person", "mc": "semantic"}]
        elif "MATCH (n {name: $name})-[r]->(m)" in cypher:
            result.data.return_value = [
                {"from_n": "Alice", "rel": "KNOWS", "to_n": "Bob"},
            ]
        else:
            result.data.return_value = []
        return result

    session.run.side_effect = _run
    session.__enter__.return_value = session
    session.__exit__.return_value = False
    driver = MagicMock()
    driver.session.return_value = session
    retriever = MemgraphRetriever(driver=driver)

    _assert_hit_list(retriever.retrieve("Alice", limit=1), "memgraph")


def test_temporal_retriever_implements_common_retrieve_interface():
    retriever = TemporalRetriever(
        embedding=_FakeEmbedding(),
        vector_store=_FakeVectorStore([
            _StoreHit(
                id="t1",
                score=0.8,
                payload={"text": "april memory", "timestamp": "2026-04-15T00:00:00Z"},
            ),
        ]),
    )

    _assert_hit_list(retriever.retrieve("what happened in april 2026?", limit=1), "temporal")


def test_search_alias_delegates_to_retrieve_for_new_interface():
    retriever = BM25Retriever(docs=[BM25Doc(id="b1", text="alias token memory")])

    assert [hit.id for hit in retriever.search("alias", limit=1)] == [
        hit.id for hit in retriever.retrieve("alias", limit=1)
    ]
