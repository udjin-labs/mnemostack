"""Tests for BM25 and RRF fusion."""
import pytest

from mnemostack.recall import BM25, reciprocal_rank_fusion, tokenize
from mnemostack.recall.bm25 import BM25Doc


def test_tokenize_basic():
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_tokenize_russian():
    assert tokenize("Привет мир") == ["привет", "мир"]


def test_tokenize_mixed_punctuation():
    assert tokenize("foo-bar_baz 42") == ["foo", "bar_baz", "42"]


@pytest.fixture
def sample_corpus():
    return [
        BM25Doc(id=1, text="The quick brown fox jumps over the lazy dog"),
        BM25Doc(id=2, text="Never jump over the fence"),
        BM25Doc(id=3, text="A lazy cat naps all day"),
        BM25Doc(id=4, text="Python programming is fun and productive"),
        BM25Doc(id=5, text="The quick fox and the lazy cat"),
    ]


def test_bm25_finds_relevant(sample_corpus):
    bm25 = BM25(sample_corpus)
    results = bm25.search("quick fox", limit=3)
    ids = [d.id for d, _ in results]
    assert 1 in ids  # "quick brown fox"
    assert 5 in ids  # "quick fox"


def test_bm25_scores_sorted(sample_corpus):
    bm25 = BM25(sample_corpus)
    results = bm25.search("lazy", limit=5)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)


def test_bm25_empty_query(sample_corpus):
    bm25 = BM25(sample_corpus)
    assert bm25.search("", limit=5) == []


def test_bm25_unknown_term(sample_corpus):
    bm25 = BM25(sample_corpus)
    assert bm25.search("quantumphysics", limit=5) == []


def test_bm25_payload_preserved(sample_corpus):
    docs = [
        BM25Doc(id=1, text="foo bar baz", payload={"source": "a.md", "line": 5}),
    ]
    bm25 = BM25(docs)
    (doc, score), = bm25.search("foo", limit=1)
    assert doc.payload == {"source": "a.md", "line": 5}


def test_rrf_single_list():
    ranked = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    fused = reciprocal_rank_fusion([ranked])
    # Rank order preserved
    assert [item for item, _ in fused] == ["a", "b", "c"]


def test_rrf_merges_two_lists():
    list1 = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    list2 = [("b", 0.95), ("d", 0.85), ("a", 0.75)]
    fused = reciprocal_rank_fusion([list1, list2])
    ids = [item for item, _ in fused]
    # "b" appears in both at rank 2 and rank 1 → should be at or near top
    # "a" appears at rank 1 and rank 3 → also top
    # Both should beat items present in only one list
    assert set(ids[:2]) == {"a", "b"}


def test_rrf_limit():
    ranked = [(chr(ord("a") + i), 1.0 - i * 0.1) for i in range(10)]
    fused = reciprocal_rank_fusion([ranked], limit=3)
    assert len(fused) == 3


def test_rrf_with_objects():
    class Obj:
        def __init__(self, id, text):
            self.id = id
            self.text = text

    obj_a = Obj("A", "first")
    obj_b = Obj("B", "second")
    fused = reciprocal_rank_fusion([[(obj_a, 0.9), (obj_b, 0.8)]])
    assert fused[0][0].id == "A"
    assert fused[1][0].id == "B"
