from mnemostack.llm.base import LLMResponse
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


class FakeExpansionLLM:
    def generate(self, prompt, max_tokens=120, temperature=0.0):
        return LLMResponse(text="lexical variant\nsemantic variant")


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


def _many_vector_hits():
    return [
        Hit("v-anchor", 0.95, {"text": "top vector match"}),
        Hit("v-second", 0.90, {"text": "second vector match"}),
        Hit("v-third", 0.85, {"text": "third vector match"}),
        Hit("v-tail", 0.80, {"text": "tail vector match"}),
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


def test_vector_floor_appended_scores_stay_below_fused_results():
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore(_vector_hits()),
        bm25_docs=_bm25_docs(),
        vector_floor=2,
    )

    results = recaller.recall("lexical", limit=2, vector_limit=3)
    by_id = {result.id: result for result in results}

    assert by_id["v-buried"].payload["raw_vector_score"] == 0.721
    assert by_id["v-buried"].score < min(result.score for result in results[:2])
    assert sorted(results, key=lambda result: result.score, reverse=True)[:2] == results[:2]


def test_vector_floor_with_no_vector_hits_does_not_add_results():
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore([]),
        bm25_docs=_bm25_docs(),
        vector_floor=5,
    )

    results = recaller.recall("lexical", limit=2, vector_limit=3)

    assert [result.id for result in results] == ["b1", "b2"]


def test_vector_floor_applies_once_after_query_expansion_fusion():
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore(_vector_hits()),
        bm25_docs=_bm25_docs(),
        query_expansion=True,
        expansion_llm=FakeExpansionLLM(),
        vector_floor=2,
    )
    calls = []
    original_apply_vector_floor = recaller._apply_vector_floor

    def spy_apply_vector_floor(results, vector_candidates):
        calls.append([result.id for result in results])
        return original_apply_vector_floor(results, vector_candidates)

    recaller._apply_vector_floor = spy_apply_vector_floor

    recaller.recall("lexical", limit=2, vector_limit=3)

    assert len(calls) == 1


def test_vector_floor_query_expansion_uses_raw_vector_candidates():
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore(_many_vector_hits()),
        bm25_docs=_bm25_docs(),
        query_expansion=True,
        expansion_llm=FakeExpansionLLM(),
        vector_floor=4,
    )

    results = recaller.recall("lexical", limit=2, vector_limit=4, bm25_limit=4)

    ids = [result.id for result in results]
    assert "v-tail" in ids
    assert len(ids) == len(set(ids))


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


def test_vector_floor_retriever_mode_preserves_raw_score_before_rrf_overwrite():
    vector_results = [
        RecallResult(
            id="v-anchor",
            text="top vector match",
            score=0.95,
            sources=["vector"],
        ),
        RecallResult(
            id="v-buried",
            text="lower vector match",
            score=0.721,
            sources=["vector"],
        ),
    ]
    bm25_results = [
        RecallResult(id="b1", text="lexical match", score=1.0, sources=["bm25"]),
    ]
    recaller = Recaller(
        retrievers=[
            FixedRetriever("vector", vector_results),
            FixedRetriever("bm25", bm25_results),
        ],
        vector_floor=1,
    )

    results = recaller.recall("lexical", limit=2, vector_limit=3)

    ids = [result.id for result in results]
    assert ids == ["v-anchor", "b1"]
    assert results[0].payload["raw_vector_score"] == 0.95


def test_vector_floor_applies_after_rerank_and_top_k_slice():
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore(_vector_hits()),
        bm25_docs=_bm25_docs(),
        vector_floor=2,
    )

    recalled = recaller.recall("lexical", limit=6, vector_limit=3)
    reranked_and_sliced = [
        result for result in recalled
        if result.id not in {"v-anchor", "v-buried"}
    ][:2]

    final = recaller.apply_vector_floor_after_rerank(reranked_and_sliced, recalled)

    ids = [result.id for result in final]
    assert ids[:2] == [result.id for result in reranked_and_sliced]
    assert "v-anchor" in ids
    assert "v-buried" in ids
    assert len(ids) > 2


def test_vector_floor_after_rerank_default_is_noop():
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore(_vector_hits()),
        bm25_docs=_bm25_docs(),
        vector_floor=0,
    )

    recalled = recaller.recall("lexical", limit=6, vector_limit=3)
    sliced = recalled[:2]

    assert recaller.apply_vector_floor_after_rerank(sliced, recalled) is sliced
    assert [result.id for result in sliced] == [result.id for result in recalled[:2]]


def test_vector_floor_after_rerank_deduplicates_ids():
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore(_vector_hits()),
        bm25_docs=_bm25_docs(),
        vector_floor=2,
    )

    recalled = recaller.recall("lexical", limit=6, vector_limit=3)
    reranked_and_sliced = [recalled[0], recalled[0], recalled[1]]

    final = recaller.apply_vector_floor_after_rerank(reranked_and_sliced, recalled)

    ids = [result.id for result in final]
    assert ids.count("v-anchor") == 1
    assert ids.count("v-buried") == 1
    assert len(ids) == len(set(ids))


def test_vector_floor_after_rerank_extends_without_dropping_winners():
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore(_many_vector_hits()),
        bm25_docs=_bm25_docs(),
        vector_floor=4,
    )

    recalled = recaller.recall("lexical", limit=8, vector_limit=4)
    reranked_winners = [
        result for result in recalled
        if result.id not in {"v-anchor", "v-second", "v-third", "v-tail"}
    ][:2]

    final = recaller.apply_vector_floor_after_rerank(reranked_winners, recalled)

    ids = [result.id for result in final]
    assert ids[:2] == [result.id for result in reranked_winners]
    assert {"v-anchor", "v-second", "v-third", "v-tail"} <= set(ids)
    assert len(ids) == len(reranked_winners) + 4
