"""Regression tests for TemporalRetriever.

The temporal path was silently returning empty results because
``TemporalRetriever.search`` emitted a nested ``{"must": [...]}``
filter that ``VectorStore._build_filter`` did not handle. The
failure was swallowed by a broad ``except Exception: return []``, so
the retriever looked green end-to-end but never contributed any
hits.

These tests lock in the fix: the temporal filter is emitted in the
flat ``{field: {gte, lte}}`` shape that ``_build_filter`` actually
handles, ``_build_filter`` converts it to a valid
``qdrant_client.models.Filter``, and the retriever returns
``RecallResult`` objects with ``sources=['temporal']`` when hits
are returned by the store.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest

from mnemostack.recall.recaller import RecallResult
from mnemostack.recall.retrievers import TemporalRetriever, extract_temporal

# ----- test doubles (match the pattern in test_hyde_retriever.py) -----


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


class _RaisingVectorStore:
    def search(self, vector, limit=20, filters=None):
        raise RuntimeError("simulated qdrant 500")


def _hits(n: int):
    return [
        _Hit(
            id=f"mem-{i}",
            score=1.0 - 0.1 * i,
            payload={"text": f"memory {i}", "timestamp": "2026-04-15T00:00:00+00:00"},
        )
        for i in range(n)
    ]


# ----- filter shape is the flat shape _build_filter handles -----


def test_filter_uses_flat_shape_not_nested_must():
    """Regression: reject nested `{"must": [...]}` shape in favor of
    the flat `{field: {gte, lte}}` shape that VectorStore._build_filter
    actually supports."""
    store = _FakeVectorStore(results=_hits(2))
    r = TemporalRetriever(embedding=_FakeEmbedding(), vector_store=store)

    r.search("what happened in april 2026?", limit=5)

    flt = store.last_filters
    assert flt is not None, "retriever did not pass a filter"
    assert "must" not in flt, (
        "filter must not use the nested qdrant-raw shape — "
        "VectorStore._build_filter expects a flat {field: value/range} map"
    )
    assert "timestamp" in flt, "filter must target the timestamp payload field"
    ts_range = flt["timestamp"]
    assert isinstance(ts_range, dict)
    assert "gte" in ts_range, "range must set gte bound"
    assert "lte" in ts_range, (
        "range must use lte (inclusive upper bound) — _build_filter does "
        "not map `lt` into qdrant Range today; fix both sides together "
        "when lt semantics are wanted"
    )


def test_filter_bounds_match_extract_temporal_window():
    """The gte/lte the retriever emits exactly match the
    (start_iso, end_iso) tuple returned by extract_temporal."""
    window = extract_temporal("what happened in april 2026?")
    assert window is not None, "sanity: extractor should parse 'april 2026'"
    start_iso, end_iso = window

    store = _FakeVectorStore(results=_hits(1))
    r = TemporalRetriever(embedding=_FakeEmbedding(), vector_store=store)
    r.search("what happened in april 2026?", limit=1)

    ts = store.last_filters["timestamp"]
    assert ts["gte"] == start_iso
    assert ts["lte"] == end_iso


# ----- the flat shape round-trips through the real VectorStore._build_filter -----


def test_build_filter_accepts_the_emitted_shape():
    """The filter shape TemporalRetriever emits must produce a valid
    qdrant_client.models.Filter via VectorStore._build_filter — this
    is the contract that regressed. Validates against the real method,
    not a mock, so a future _build_filter refactor can't silently
    break the temporal path again.

    String-valued ISO timestamps must land in a qdrant
    ``DatetimeRange`` (qdrant-client normalizes the ISO string to a
    ``datetime``); numeric-valued ranges must land in ``Range``. The
    previous implementation only instantiated ``Range``, which
    raised a pydantic ValidationError on the string path — the real
    production bug.
    """
    pytest.importorskip("qdrant_client")
    from qdrant_client.models import DatetimeRange

    from mnemostack.vector.qdrant import VectorStore

    store = VectorStore.__new__(VectorStore)  # bypass __init__

    start_dt = datetime(2026, 4, 1, tzinfo=timezone.utc)
    end_dt = datetime(2026, 5, 1, tzinfo=timezone.utc)
    start = start_dt.isoformat()
    end = end_dt.isoformat()
    filters = {"timestamp": {"gte": start, "lte": end}}

    qfilter = store._build_filter(filters)
    assert qfilter is not None
    assert len(qfilter.must) == 1
    cond = qfilter.must[0]
    assert cond.key == "timestamp"
    # String inputs dispatch to DatetimeRange.
    assert isinstance(cond.range, DatetimeRange)
    # qdrant-client normalizes ISO strings to datetime objects.
    assert cond.range.gte == start_dt
    assert cond.range.lte == end_dt


def test_build_filter_numeric_range_still_uses_Range():
    """Numeric ranges must stay on qdrant Range — ensures the
    string-dispatch didn't accidentally convert every numeric
    range into DatetimeRange."""
    pytest.importorskip("qdrant_client")
    from qdrant_client.models import Range as QRange

    from mnemostack.vector.qdrant import VectorStore

    store = VectorStore.__new__(VectorStore)  # bypass __init__
    filters = {"score": {"gte": 0.5, "lte": 1.0}}

    qfilter = store._build_filter(filters)
    cond = qfilter.must[0]
    assert isinstance(cond.range, QRange)
    assert cond.range.gte == 0.5
    assert cond.range.lte == 1.0


# ----- happy path end-to-end -----


def test_returns_recallresults_with_temporal_source_tag():
    store = _FakeVectorStore(results=_hits(3))
    r = TemporalRetriever(embedding=_FakeEmbedding(), vector_store=store)

    results = r.search("anything from 2026?", limit=3)

    assert len(results) == 3
    for res in results:
        assert isinstance(res, RecallResult)
        assert res.sources == ["temporal"]
        assert res.payload.get("temporal_match") is True
        assert res.id.startswith("temporal:")


# ----- empty extractor result short-circuits -----


def test_no_window_returns_empty_without_calling_store():
    store = _FakeVectorStore(results=_hits(3))
    r = TemporalRetriever(embedding=_FakeEmbedding(), vector_store=store)

    # Query with no extractable date window.
    results = r.search("how does the embedding pipeline work?", limit=3)

    assert results == []
    assert store.last_query is None, (
        "no date window should short-circuit before the store is touched"
    )


# ----- defensive exception handling stays defensive but is now observable -----


def test_store_exception_returns_empty_and_logs(caplog):
    """The bare `except Exception: return []` that was hiding the
    filter-shape bug is still defensive (we never raise upward), but
    now emits a WARNING so the next time it happens, operators can
    see it in logs instead of chasing empty retrievals.
    """
    import logging

    # ``logging_config.configure_logging`` sets ``mnemostack.propagate
    # = False`` when any other test runs it — which blocks caplog's
    # root-handler-based capture. Re-enable propagation for the
    # retriever's logger specifically so this test is ordering-
    # independent. Other tests that care about propagation state
    # re-call configure_logging in their own setup.
    retriever_logger = logging.getLogger("mnemostack.recall.retrievers")
    mnemostack_logger = logging.getLogger("mnemostack")
    original_retr = retriever_logger.propagate
    original_mnem = mnemostack_logger.propagate
    retriever_logger.propagate = True
    mnemostack_logger.propagate = True
    try:
        r = TemporalRetriever(
            embedding=_FakeEmbedding(), vector_store=_RaisingVectorStore(),
        )
        with caplog.at_level("WARNING", logger="mnemostack.recall.retrievers"):
            results = r.search("april 2026 logs?", limit=5)

        assert results == []
        assert any(
            "TemporalRetriever" in rec.getMessage()
            and "simulated qdrant 500" in rec.getMessage()
            for rec in caplog.records
        ), "expected a WARNING log entry describing the store failure"
    finally:
        retriever_logger.propagate = original_retr
        mnemostack_logger.propagate = original_mnem
