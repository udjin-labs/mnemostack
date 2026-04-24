"""Tests for observability — counters, histograms, recorder swap."""
import time

import pytest

from mnemostack.observability import (
    counter,
    get_recorder,
    histogram,
    set_recorder,
    timed,
)
from mnemostack.observability.recorder import (
    InMemoryRecorder,
    LoggingRecorder,
    NullRecorder,
)


@pytest.fixture
def in_memory():
    """Swap global recorder to InMemoryRecorder and reset after test."""
    original = get_recorder()
    recorder = InMemoryRecorder()
    set_recorder(recorder)
    yield recorder
    set_recorder(original)


def test_counter_records(in_memory):
    counter("my.test", 1)
    counter("my.test", 2)
    assert in_memory.counter_value("my.test") == 3.0


def test_counter_with_labels(in_memory):
    counter("calls", 1, labels={"provider": "gemini"})
    counter("calls", 1, labels={"provider": "ollama"})
    counter("calls", 3, labels={"provider": "gemini"})
    assert in_memory.counter_value("calls", {"provider": "gemini"}) == 4.0
    assert in_memory.counter_value("calls", {"provider": "ollama"}) == 1.0


def test_histogram_context_manager_records_elapsed(in_memory):
    with histogram("op.latency_ms"):
        time.sleep(0.02)  # 20ms
    values = in_memory.histogram_values("op.latency_ms")
    assert len(values) == 1
    assert values[0] >= 10  # at least ~20ms
    assert values[0] < 500  # sanity


def test_histogram_with_labels(in_memory):
    with histogram("op.latency_ms", {"kind": "search"}):
        pass
    with histogram("op.latency_ms", {"kind": "index"}):
        pass
    assert len(in_memory.histogram_values("op.latency_ms", {"kind": "search"})) == 1
    assert len(in_memory.histogram_values("op.latency_ms", {"kind": "index"})) == 1


def test_timed_decorator(in_memory):
    @timed("decorated.op_ms")
    def slow():
        time.sleep(0.01)
        return 42

    assert slow() == 42
    values = in_memory.histogram_values("decorated.op_ms")
    assert len(values) == 1


def test_null_recorder_zero_overhead():
    set_recorder(NullRecorder())
    counter("x", 1)  # should not raise, produces nothing
    with histogram("x"):
        pass


def test_faulty_recorder_does_not_break_callers():
    class BrokenRecorder:
        def record_counter(self, *a, **kw):
            raise RuntimeError("boom")

        def record_histogram(self, *a, **kw):
            raise RuntimeError("boom")

    set_recorder(BrokenRecorder())
    # Business logic should NOT crash even if recorder is broken
    counter("x", 1)
    with histogram("x"):
        pass
    set_recorder(NullRecorder())


def test_in_memory_reset(in_memory):
    counter("c", 5)
    with histogram("h"):
        pass
    in_memory.reset()
    assert in_memory.counter_value("c") == 0.0
    assert in_memory.histogram_values("h") == []


def test_logging_recorder_does_not_raise(caplog):
    """LoggingRecorder emits debug lines, no exceptions."""
    import logging
    set_recorder(LoggingRecorder())
    with caplog.at_level(logging.DEBUG, logger="mnemostack.observability.recorder"):
        counter("my.counter", 7, labels={"x": "y"})
        with histogram("my.hist"):
            pass
    set_recorder(NullRecorder())


def test_recaller_emits_metrics_via_observability():
    """Integration: Recaller should emit counter/histogram when recall() runs."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance

    from mnemostack.embeddings import EmbeddingProvider
    from mnemostack.recall import BM25Doc, Recaller
    from mnemostack.vector import VectorStore

    class FakeEmb(EmbeddingProvider):
        @property
        def dimension(self):
            return 4
        @property
        def name(self):
            return "fake"
        def embed(self, text):
            return [1.0, 0.0, 0.0, 0.0]
        def embed_batch(self, texts):
            return [self.embed(t) for t in texts]

    store = VectorStore.__new__(VectorStore)
    store.collection = "mtest"
    store.dimension = 4
    store.distance = Distance.COSINE
    store.client = QdrantClient(":memory:")
    store.ensure_collection()
    store.upsert(1, [1.0, 0.0, 0.0, 0.0], {"text": "hello"})

    rec = InMemoryRecorder()
    set_recorder(rec)
    try:
        recaller = Recaller(
            embedding_provider=FakeEmb(),
            vector_store=store,
            bm25_docs=[BM25Doc(id=1, text="hello")],
        )
        results = recaller.recall("hello", limit=5)
        assert len(results) > 0
        assert rec.counter_value("mnemostack.recall.calls") == 1
        assert rec.counter_value("mnemostack.recall.results") >= 1
        # Latency histograms should have observations
        assert len(rec.histogram_values("mnemostack.recall.latency_ms")) == 1
    finally:
        set_recorder(NullRecorder())
