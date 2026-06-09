from mnemostack.recall import BM25Doc, Recaller, RecallResult
from mnemostack.vector.qdrant import Hit


class FakeEmbedding:
    def embed(self, query):
        return [1.0]


class FakeVectorStore:
    def __init__(self, hits):
        self.hits = hits

    def search(self, vector, limit=10, filters=None):
        return self.hits[:limit]


def _bm25_docs():
    return [
        BM25Doc(id=f"b{i}", text="lexical match", payload={"text": f"lexical {i}"})
        for i in range(1, 6)
    ]


def _vector_hits():
    return [
        Hit("v-anchor", 0.95, {"text": "top vector match"}),
        Hit("v-buried", 0.721, {"text": "strong vector-only match"}),
        Hit("v-extra", 0.60, {"text": "lower vector match"}),
    ]


def test_vector_floor_surfaces_buried_vector_strong_result():
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore(_vector_hits()),
        bm25_docs=_bm25_docs(),
        vector_floor=2,
    )

    results = recaller.recall("lexical", limit=2, vector_limit=3)

    ids = [result.id for result in results]
    assert ids[:2] == ["v-anchor", "b1"]
    assert "v-buried" in ids
    assert ids.index("v-buried") >= 2


def test_vector_floor_default_is_noop():
    kwargs = {
        "embedding_provider": FakeEmbedding(),
        "vector_store": FakeVectorStore(_vector_hits()),
        "bm25_docs": _bm25_docs(),
    }
    baseline = Recaller(**kwargs)
    disabled = Recaller(**kwargs, vector_floor=0)

    baseline_results = baseline.recall("lexical", limit=2, vector_limit=3)
    disabled_results = disabled.recall("lexical", limit=2, vector_limit=3)

    assert [result.id for result in disabled_results] == [
        result.id for result in baseline_results
    ]
    assert len(disabled_results) == len(baseline_results)


def test_vector_floor_deduplicates_existing_results():
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore(_vector_hits()),
        bm25_docs=_bm25_docs(),
        vector_floor=2,
    )

    results = recaller.recall("lexical", limit=2, vector_limit=3)

    ids = [result.id for result in results]
    assert ids.count("v-anchor") == 1
    assert ids.count("v-buried") == 1
    assert len(ids) == len(set(ids))


def test_vector_floor_with_no_vector_hits_does_not_add_results():
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore([]),
        bm25_docs=_bm25_docs(),
        vector_floor=5,
    )

    results = recaller.recall("lexical", limit=2, vector_limit=3)

    assert [result.id for result in results] == ["b1", "b2"]


class FixedRetriever:
    def __init__(self, name, results):
        self.name = name
        self.results = results

    def search(self, query, limit=20, filters=None):
        return self.results[:limit]


def test_vector_floor_works_with_retriever_mode():
    vector_results = [
        RecallResult(
            id="v-anchor",
            text="top vector match",
            score=0.95,
            payload={"raw_vector_score": 0.95},
            sources=["vector"],
        ),
        RecallResult(
            id="v-buried",
            text="strong vector-only match",
            score=0.721,
            payload={"raw_vector_score": 0.721},
            sources=["vector"],
        ),
    ]
    bm25_results = [
        RecallResult(id=f"b{i}", text="lexical match", score=1.0, sources=["bm25"])
        for i in range(1, 4)
    ]
    recaller = Recaller(
        retrievers=[
            FixedRetriever("vector", vector_results),
            FixedRetriever("bm25", bm25_results),
        ],
        vector_floor=2,
    )

    results = recaller.recall("lexical", limit=2, vector_limit=3)

    ids = [result.id for result in results]
    assert "v-buried" in ids
    assert len(ids) == len(set(ids))
