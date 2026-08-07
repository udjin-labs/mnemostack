"""Tests for RecallTrace — observability on top of fail-open recall."""

from __future__ import annotations

import pytest

from mnemostack.recall import Recaller, RecallResult, RecallTrace, apply_rerank_safe
from mnemostack.recall.retrievers import Retriever


class _ListRetriever(Retriever):
    """Returns a fixed ranked list."""

    def __init__(self, name: str, results: list[RecallResult]):
        self.name = name
        self._results = results

    def search(self, query, limit=10, filters=None):
        return [
            RecallResult(id=r.id, text=r.text, score=r.score, payload=dict(r.payload), sources=[self.name])
            for r in self._results[:limit]
        ]


class _RaisingRetriever(Retriever):
    name = "broken"

    def search(self, query, limit=10, filters=None):
        raise RuntimeError("backend down")


class _EmptyWithReasonRetriever(Retriever):
    """Mimics TemporalRetriever's no-parse behavior."""

    name = "temporal"

    def search(self, query, limit=10, filters=None):
        return []

    def explain_empty(self, query):
        return "temporal:no_parse"


def _results(name: str, ids: list[str]) -> list[RecallResult]:
    return [
        RecallResult(id=i, text=f"text {i}", score=1.0 - n * 0.1, payload={})
        for n, i in enumerate(ids)
    ]


def test_trace_per_retriever_ranked_lists():
    recaller = Recaller(
        retrievers=[
            _ListRetriever("vector", _results("vector", ["a", "b"])),
            _ListRetriever("bm25", _results("bm25", ["b", "c"])),
        ]
    )
    trace = RecallTrace()
    recaller.recall("q", limit=5, trace=trace)

    by_name = {rt.name: rt for rt in trace.retrievers}
    assert set(by_name) == {"vector", "bm25"}
    assert [rid for rid, _ in by_name["vector"].ranked] == ["a", "b"]
    assert [rid for rid, _ in by_name["bm25"].ranked] == ["b", "c"]
    assert by_name["vector"].error is None
    assert by_name["vector"].latency_ms >= 0.0


def test_trace_fused_matches_returned_order():
    recaller = Recaller(
        retrievers=[
            _ListRetriever("vector", _results("vector", ["a", "b"])),
            _ListRetriever("bm25", _results("bm25", ["b", "c"])),
        ]
    )
    trace = RecallTrace()
    results = recaller.recall("q", limit=5, trace=trace)

    assert [rid for rid, _ in trace.fused] == [str(r.id) for r in results]
    assert trace.degraded == []


def test_failed_retriever_marks_degraded_and_keeps_others():
    recaller = Recaller(
        retrievers=[
            _ListRetriever("vector", _results("vector", ["a"])),
            _RaisingRetriever(),
        ]
    )
    trace = RecallTrace()
    results = recaller.recall("q", limit=5, trace=trace)

    assert [r.id for r in results] == ["a"]
    assert "retriever:broken:failed" in trace.degraded
    broken = next(rt for rt in trace.retrievers if rt.name == "broken")
    assert broken.error is not None and "backend down" in broken.error


def test_empty_retriever_with_reason_marks_note():
    recaller = Recaller(
        retrievers=[
            _ListRetriever("vector", _results("vector", ["a"])),
            _EmptyWithReasonRetriever(),
        ]
    )
    trace = RecallTrace()
    recaller.recall("q", limit=5, trace=trace)

    # Routine stage-did-not-apply signal: authoritative in notes, and kept as
    # a DEPRECATED duplicate in degraded until the next major so clients
    # matching the stable tag strings keep working across minor upgrades.
    assert "temporal:no_parse" in trace.notes
    assert "temporal:no_parse" in trace.degraded


def test_recall_without_trace_unchanged():
    retrievers = [
        _ListRetriever("vector", _results("vector", ["a", "b"])),
        _ListRetriever("bm25", _results("bm25", ["b", "c"])),
    ]
    traced = Recaller(retrievers=retrievers)
    plain = Recaller(retrievers=retrievers)

    with_trace = traced.recall("q", limit=5, trace=RecallTrace())
    without_trace = plain.recall("q", limit=5)

    assert [(r.id, r.score) for r in with_trace] == [(r.id, r.score) for r in without_trace]


def test_trace_mark_deduplicates():
    trace = RecallTrace()
    trace.mark("reranker:fallback")
    trace.mark("reranker:fallback")
    assert trace.degraded == ["reranker:fallback"]


def test_trace_to_dict_shape():
    recaller = Recaller(retrievers=[_ListRetriever("vector", _results("vector", ["a"]))])
    trace = RecallTrace()
    recaller.recall("q", limit=5, trace=trace)

    d = trace.to_dict()
    assert d["degraded"] == []
    assert d["retrievers"][0]["name"] == "vector"
    assert d["fused"] and d["fused"][0][0] == "a"
    assert "post_rerank" not in d  # no reranker ran


# ---------- apply_rerank_safe ----------


class _FakeReranker:
    def __init__(self, *, fail: bool = False, reverse: bool = True):
        self.fail = fail
        self.reverse = reverse

    def rerank(self, query, results):
        if self.fail:
            raise RuntimeError("llm down")
        return list(reversed(results)) if self.reverse else list(results)


def test_apply_rerank_safe_success_records_order():
    results = _results("vector", ["a", "b", "c"])
    trace = RecallTrace()
    out = apply_rerank_safe(_FakeReranker(), "q", results, trace)

    assert [r.id for r in out] == ["c", "b", "a"]
    assert trace.post_rerank == [("c", out[0].score), ("b", out[1].score), ("a", out[2].score)]
    assert trace.degraded == []


def test_apply_rerank_safe_fallback_marks_degraded():
    results = _results("vector", ["a", "b"])
    trace = RecallTrace()
    out = apply_rerank_safe(_FakeReranker(fail=True), "q", results, trace)

    assert out == results  # original order preserved
    assert trace.degraded == ["reranker:fallback"]
    assert trace.post_rerank is None


def test_apply_rerank_safe_self_fail_open_marks_degraded():
    from mnemostack.recall import ScoringReranker

    class _BadScorer:
        def score(self, query, documents):
            return [0.1]

    results = _results("vector", ["a", "b"])
    trace = RecallTrace()
    out = apply_rerank_safe(ScoringReranker(_BadScorer()), "q", results, trace)

    assert out == results
    assert trace.degraded == ["reranker:fallback"]
    assert trace.post_rerank is None


def test_apply_rerank_safe_inplace_reranker_not_misread_as_fallback():
    # A custom reranker that sorts the supplied list in place and returns the
    # same object is a SUCCESSFUL reorder, not a fallback. Without the opt-in
    # `fallback_keeps_input_object` marker, identity must not flag it.
    class _InPlaceReranker:
        def rerank(self, query, results):
            results.sort(key=lambda r: r.id, reverse=True)
            return results  # same object, but order changed → success

    results = _results("vector", ["a", "b", "c"])
    trace = RecallTrace()
    out = apply_rerank_safe(_InPlaceReranker(), "q", results, trace)

    assert [r.id for r in out] == ["c", "b", "a"]
    assert trace.degraded == []  # not a fallback
    assert trace.post_rerank == [("c", out[0].score), ("b", out[1].score), ("a", out[2].score)]


def test_apply_rerank_safe_none_reranker_noop():
    results = _results("vector", ["a"])
    trace = RecallTrace()
    assert apply_rerank_safe(None, "q", results, trace) == results
    assert trace.degraded == ["reranker:unavailable"]


def test_apply_rerank_safe_without_trace():
    results = _results("vector", ["a", "b"])
    assert [r.id for r in apply_rerank_safe(_FakeReranker(fail=True), "q", results)] == ["a", "b"]


def test_recall_flow_without_reranker_leaves_trace_clean():
    """reranker=None means 'not requested', not a degradation — the raw and
    no-pipeline paths must not report reranker:unavailable."""
    from mnemostack.recall import recall_flow

    recaller = Recaller(
        retrievers=[_ListRetriever("vector", _results("vector", ["a", "b"]))],
    )
    trace = RecallTrace()
    results = recall_flow(recaller, "q", limit=5, reranker=None, trace=trace)

    assert results
    assert trace.degraded == []


# ---------- query expansion path ----------


def test_trace_query_expansion_tags_queries():
    class _ExpansionLLM:
        name = "fake"

        def generate(self, prompt, max_tokens=200, temperature=0.0):
            from mnemostack.llm.base import LLMResponse

            return LLMResponse(text="variant one\nvariant two")

    recaller = Recaller(
        retrievers=[_ListRetriever("vector", _results("vector", ["a", "b"]))],
        query_expansion=True,
        expansion_llm=_ExpansionLLM(),
    )
    trace = RecallTrace()
    results = recaller.recall("original", limit=5, trace=trace)

    assert results
    queries = {rt.query for rt in trace.retrievers}
    assert "original" in queries  # original always recalled
    assert all(q is not None for q in queries)
    assert [rid for rid, _ in trace.fused] == [str(r.id) for r in results]


def test_trace_legacy_path_records_vector_and_fused():
    pytest.importorskip("qdrant_client")
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance

    from mnemostack.recall import BM25Doc
    from mnemostack.vector import VectorStore

    class _FakeEmbedder:
        dimension = 4

        def embed(self, text):
            return [0.1, 0.2, 0.3, 0.4]

    store = VectorStore.__new__(VectorStore)
    store.collection = "trace_legacy"
    store.dimension = 4
    store.distance = Distance.COSINE
    store.client = QdrantClient(":memory:")
    store.ensure_collection()
    store.upsert(1, [0.1, 0.2, 0.3, 0.4], {"text": "hello world"})

    recaller = Recaller(
        embedding_provider=_FakeEmbedder(),
        vector_store=store,
        bm25_docs=[BM25Doc(id=2, text="hello there")],
    )
    trace = RecallTrace()
    results = recaller.recall("hello", limit=5, trace=trace)

    names = [rt.name for rt in trace.retrievers]
    assert "vector" in names and "bm25" in names
    assert [rid for rid, _ in trace.fused] == [str(r.id) for r in results]


# ------------------------------------------------- notes vs degraded routing


def test_mark_routes_routine_tags_to_notes_and_skips_the_counter(monkeypatch):
    """A stage that did not apply is a note: notes is authoritative, the
    degraded list keeps a DEPRECATED back-compat duplicate (until the next
    major), and the process-wide counter must never fire for it."""
    import mnemostack.recall.trace as trace_mod

    calls: list[tuple] = []
    monkeypatch.setattr(trace_mod, "counter", lambda *a, **k: calls.append(a))
    trace = RecallTrace()
    trace.mark("temporal:no_parse")
    trace.mark("temporal:no_parse")  # deduped

    assert trace.notes == ["temporal:no_parse"]
    assert trace.degraded == ["temporal:no_parse"]
    assert calls == []


def test_mark_degradations_still_count_and_dedupe(monkeypatch):
    import mnemostack.recall.trace as trace_mod

    calls: list[tuple] = []
    monkeypatch.setattr(trace_mod, "counter", lambda *a, **k: calls.append(a))
    trace = RecallTrace()
    trace.mark("reranker:fallback")
    trace.mark("reranker:fallback")

    assert trace.degraded == ["reranker:fallback"]
    assert trace.notes == []
    assert len(calls) == 1  # counted once per call, not per repeat


def test_mark_routes_dynamic_no_tokens_tags_to_notes(monkeypatch):
    """Per-arm gate verdicts carry the arm's own name, so the routine
    classification is by suffix — a lexical arm with no usable query tokens
    is a note, not a degradation, and must not count."""
    import mnemostack.recall.trace as trace_mod

    calls: list[tuple] = []
    monkeypatch.setattr(trace_mod, "counter", lambda *a, **k: calls.append(a))
    trace = RecallTrace()
    trace.mark("qdrant_text:title:no_tokens")
    trace.mark("qdrant_text:no_tokens")

    assert trace.notes == ["qdrant_text:title:no_tokens", "qdrant_text:no_tokens"]
    assert trace.degraded == ["qdrant_text:title:no_tokens", "qdrant_text:no_tokens"]
    assert calls == []


def test_to_dict_emits_both_lists():
    trace = RecallTrace()
    d = trace.to_dict()
    assert d["degraded"] == [] and d["notes"] == []

    trace.mark("temporal:no_parse")
    trace.mark("reranker:fallback")
    d = trace.to_dict()
    assert d["notes"] == ["temporal:no_parse"]
    # Routine tags are duplicated into degraded (DEPRECATED, until next major)
    # so pre-notes clients matching the stable strings keep working; real
    # degradations are exactly the degraded entries absent from notes.
    assert d["degraded"] == ["temporal:no_parse", "reranker:fallback"]
