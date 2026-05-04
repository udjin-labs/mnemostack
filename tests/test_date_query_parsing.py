from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from mnemostack.recall.retrievers import TemporalRetriever, extract_temporal_query

NOW = datetime(2026, 5, 4, 18, 16, tzinfo=timezone.utc)


def test_vague_russian_absolute_date_is_date_focused():
    parsed = extract_temporal_query("что делали 30 апреля", now=NOW)

    assert parsed is not None
    assert parsed.target_date == date(2026, 4, 30)
    assert parsed.date_focused is True
    assert parsed.start_iso.startswith("2026-04-29T00:00:00")
    assert parsed.end_iso.startswith("2026-05-01T23:59:59")


def test_english_relative_weeks_ago():
    parsed = extract_temporal_query("what happened 3 weeks ago", now=NOW)

    assert parsed is not None
    assert parsed.target_date == date(2026, 4, 13)
    assert parsed.date_focused is True


def test_english_common_relative_dates():
    cases = {
        "what happened yesterday": date(2026, 5, 3),
        "what happened today": date(2026, 5, 4),
        "what happened day before yesterday": date(2026, 5, 2),
        "what happened a day ago": date(2026, 5, 3),
        "what happened one day ago": date(2026, 5, 3),
        "what happened a week ago": date(2026, 4, 27),
        "what happened one week ago": date(2026, 4, 27),
        "what happened last week": date(2026, 4, 27),
    }

    for query, expected in cases.items():
        parsed = extract_temporal_query(query, now=NOW)
        assert parsed is not None
        assert parsed.target_date == expected
        assert parsed.date_focused is True


def test_russian_one_week_ago():
    parsed = extract_temporal_query("что было неделю назад", now=NOW)

    assert parsed is not None
    assert parsed.target_date == date(2026, 4, 27)
    assert parsed.date_focused is True


def test_russian_singular_relative_dates():
    day = extract_temporal_query("что было 1 день назад", now=NOW)
    week = extract_temporal_query("что было 1 неделя назад", now=NOW)

    assert day is not None
    assert day.target_date == date(2026, 5, 3)
    assert day.date_focused is True
    assert week is not None
    assert week.target_date == date(2026, 4, 27)
    assert week.date_focused is True


def test_specific_non_date_query_is_not_date_focused():
    parsed = extract_temporal_query("graph-sync stale openEMS", now=NOW)

    assert parsed is None or parsed.date_focused is False


def test_russian_month_day_parsing():
    cases = {
        "13 апреля": date(2026, 4, 13),
        "1 мая": date(2026, 5, 1),
        "28 февраля": date(2026, 2, 28),
    }

    for query, expected in cases.items():
        parsed = extract_temporal_query(query, now=NOW)
        assert parsed is not None
        assert parsed.target_date == expected


def test_specific_query_with_date_stays_semantic_filtered():
    parsed = extract_temporal_query("openEMS 30 апреля", now=NOW)

    assert parsed is not None
    assert parsed.target_date == date(2026, 4, 30)
    assert parsed.date_focused is False


def test_modal_may_is_not_parsed_as_month():
    parsed = extract_temporal_query("What personality traits may Melanie say Caroline has?", now=NOW)

    assert parsed is None


@dataclass
class _Hit:
    id: str
    score: float
    payload: dict[str, Any]


class _NoopEmbedding:
    def __init__(self):
        self.called = False

    def embed(self, text: str):
        self.called = True
        return [0.1, 0.2, 0.3]


class _ScrollableStore:
    def __init__(self, scroll_results=None):
        self.scroll_results = scroll_results or [
            _Hit("diary-1", 1.0, {"text": "diary entry", "timestamp": "2026-04-30T10:00:00Z"}),
        ]
        self.last_filters = None
        self.filters_seen = []
        self.last_batch_size = None
        self.search_called = False

    def scroll(self, batch_size=256, filters=None, with_vectors=False):
        self.last_batch_size = batch_size
        self.last_filters = filters
        self.filters_seen.append(filters)
        gte = (filters or {}).get("timestamp", {}).get("gte", "")
        lte = (filters or {}).get("timestamp", {}).get("lte", "")

        def in_window(hit):
            ts = (hit.payload or {}).get("timestamp", "")
            return (not gte or ts >= gte) and (not lte or ts <= lte)

        return iter([hit for hit in self.scroll_results if in_window(hit)])

    def search(self, vector, limit=10, filters=None):
        self.search_called = True
        return []


class _CountingScrollableStore(_ScrollableStore):
    def __init__(self, scroll_results=None):
        super().__init__(scroll_results)
        self.yielded = 0

    def scroll(self, batch_size=256, filters=None, with_vectors=False):
        self.last_batch_size = batch_size
        self.last_filters = filters
        self.filters_seen.append(filters)
        gte = (filters or {}).get("timestamp", {}).get("gte", "")
        lte = (filters or {}).get("timestamp", {}).get("lte", "")

        def generate():
            for hit in self.scroll_results:
                ts = (hit.payload or {}).get("timestamp", "")
                if (not gte or ts >= gte) and (not lte or ts <= lte):
                    self.yielded += 1
                    yield hit

        return generate()


def test_date_focused_query_uses_timestamp_scroll_instead_of_semantic_search():
    embedding = _NoopEmbedding()
    store = _ScrollableStore()
    retriever = TemporalRetriever(
        embedding=embedding,
        vector_store=store,
        extractor=lambda q: extract_temporal_query(q, now=NOW),
    )

    results = retriever.search("что делали 30 апреля", limit=1)

    assert len(results) == 1
    assert results[0].id == "diary-1"
    assert results[0].payload["temporal_match"] is True
    assert embedding.called is False
    assert store.search_called is False
    assert store.last_filters["timestamp"]["gte"].startswith("2026-04-30")
    assert store.last_filters["timestamp"]["lte"].startswith("2026-04-30")


def test_date_focused_query_prefers_same_day_over_previous_day():
    hits = [
        _Hit("prev", 1.0, {"text": "previous", "timestamp": "2026-04-12T12:00:00+00:00"}),
        _Hit("same", 1.0, {"text": "same", "timestamp": "2026-04-13T12:00:00+00:00"}),
    ]
    store = _ScrollableStore(hits)
    retriever = TemporalRetriever(
        embedding=_NoopEmbedding(),
        vector_store=store,
        extractor=lambda q: extract_temporal_query(q, now=NOW),
    )

    results = retriever.search("что делали 13 апреля", limit=2)

    assert [result.id for result in results] == ["same", "prev"]
    assert store.filters_seen[0]["timestamp"]["gte"].startswith("2026-04-13")


def test_date_focused_query_prefers_same_day_diary_over_transcripts():
    hits = [
        _Hit("transcript", 1.0, {
            "text": "call transcript",
            "timestamp": "2026-04-13T09:00:00+00:00",
            "source_file": "transcripts/meeting.md",
        }),
        _Hit("diary", 1.0, {
            "text": "diary",
            "timestamp": "2026-04-13T22:00:00+00:00",
            "source_file": "memory/2026-04-13.md",
        }),
    ]
    store = _ScrollableStore(hits)
    retriever = TemporalRetriever(
        embedding=_NoopEmbedding(),
        vector_store=store,
        extractor=lambda q: extract_temporal_query(q, now=NOW),
    )

    results = retriever.search("что делали 13 апреля", limit=2)

    assert [result.id for result in results] == ["diary", "transcript"]


def test_date_focused_neighbor_days_only_after_exact_day_hits():
    hits = [
        _Hit("prev", 1.0, {"text": "previous", "timestamp": "2026-04-12T12:00:00+00:00"}),
        _Hit("next", 1.0, {"text": "next", "timestamp": "2026-04-14T12:00:00+00:00"}),
        _Hit("exact-1", 1.0, {"text": "exact 1", "timestamp": "2026-04-13T09:00:00+00:00"}),
        _Hit("exact-2", 1.0, {"text": "exact 2", "timestamp": "2026-04-13T18:00:00+00:00"}),
    ]
    store = _ScrollableStore(hits)
    retriever = TemporalRetriever(
        embedding=_NoopEmbedding(),
        vector_store=store,
        extractor=lambda q: extract_temporal_query(q, now=NOW),
    )

    results = retriever.search("что делали 13 апреля", limit=4)

    assert [result.id for result in results][:2] == ["exact-1", "exact-2"]
    assert {result.id for result in results[2:]} == {"prev", "next"}


def test_date_focused_scroll_collection_is_bounded_before_sorting():
    hits = [
        _Hit(
            f"exact-{i}",
            1.0,
            {"text": f"exact {i}", "timestamp": "2026-04-13T12:00:00+00:00"},
        )
        for i in range(200)
    ]
    store = _CountingScrollableStore(hits)
    retriever = TemporalRetriever(
        embedding=_NoopEmbedding(),
        vector_store=store,
        extractor=lambda q: extract_temporal_query(q, now=NOW),
    )

    results = retriever.search("что делали 13 апреля", limit=1)

    assert len(results) == 1
    assert store.yielded == retriever.DATE_FOCUSED_SCROLL_BUFFER_MIN
    assert store.yielded < len(hits)


def test_non_date_focused_temporal_query_keeps_semantic_search_path():
    embedding = _NoopEmbedding()
    store = _ScrollableStore()
    retriever = TemporalRetriever(
        embedding=embedding,
        vector_store=store,
        extractor=lambda q: extract_temporal_query(q, now=NOW),
    )

    results = retriever.search("openEMS 30 апреля", limit=5)

    assert results == []
    assert embedding.called is True
    assert store.search_called is True
    assert store.filters_seen == []
