import pytest

from mnemostack.observability import get_recorder, set_recorder
from mnemostack.observability.recorder import InMemoryRecorder
from mnemostack.recall import BM25Doc, Recaller
from mnemostack.vector.qdrant import Hit


class FakeEmbedding:
    def embed(self, query):
        return [1.0]


class FakeVectorStore:
    def __init__(self, hits):
        self.hits = hits
        self.calls = 0

    def search(self, vector, limit=10, filters=None):
        self.calls += 1
        return self.hits[:limit]


class FakeVectorStoreSequence:
    def __init__(self, hit_sequences):
        self.hit_sequences = hit_sequences
        self.calls = 0

    def search(self, vector, limit=10, filters=None):
        sequence_index = min(self.calls, len(self.hit_sequences) - 1)
        self.calls += 1
        return self.hit_sequences[sequence_index][:limit]


def test_default_fallback_threshold_is_035_and_triggers_when_top_score_is_low():
    original = get_recorder()
    recorder = InMemoryRecorder()
    set_recorder(recorder)
    try:
        store = FakeVectorStore([Hit("v1", 0.80, {"text": "vector match"})])
        recaller = Recaller(
            embedding_provider=FakeEmbedding(),
            vector_store=store,
            bm25_docs=[BM25Doc("b1", "lexical match")],
        )

        assert recaller.fallback_threshold == 0.35

        results = recaller.recall("lexical", limit=5, vector_limit=5)

        assert store.calls == 2  # normal vector retrieval + fallback retrieval
        assert results[0].id == "v1"
        assert results[0].score == 0.80
        assert recorder.counter_value("mnemostack.recall.fallback_triggered") == 1
    finally:
        set_recorder(original)


def test_fallback_does_not_trigger_when_top_score_is_high():
    original = get_recorder()
    recorder = InMemoryRecorder()
    set_recorder(recorder)
    try:
        store = FakeVectorStore([Hit("v1", 0.80, {"text": "vector match"})])
        recaller = Recaller(
            embedding_provider=FakeEmbedding(),
            vector_store=store,
            fallback_threshold=0.01,
        )

        results = recaller.recall("query", limit=5, vector_limit=5)

        assert store.calls == 1
        assert results[0].score < 0.45  # RRF score from normal path, not raw vector score
        assert recorder.counter_value("mnemostack.recall.fallback_triggered") == 0
    finally:
        set_recorder(original)


def test_fallback_only_results_rank_below_pipeline_results():
    store = FakeVectorStoreSequence([
        [],
        [Hit("fallback", 0.99, {"text": "fallback vector match"})],
    ])
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=store,
        bm25_docs=[BM25Doc("bm25", "lexical match")],
    )

    results = recaller.recall("lexical", limit=5, vector_limit=5)

    assert store.calls == 2
    assert [result.id for result in results] == ["bm25", "fallback"]
    assert results[1].score == pytest.approx(results[0].score - 0.01)


def test_fallback_dedupes_and_keeps_higher_score():
    store = FakeVectorStore([Hit("same", 0.80, {"text": "better vector text"})])
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=store,
        bm25_docs=[BM25Doc("same", "bm25 text")],
        fallback_threshold=0.45,
    )

    results = recaller.recall("bm25", limit=5, vector_limit=5)

    assert [result.id for result in results].count("same") == 1
    assert results[0].score == 0.80
    assert results[0].text == "better vector text"
    assert results[0].sources == ["vector", "bm25"]
