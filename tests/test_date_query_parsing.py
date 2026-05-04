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


def test_russian_one_week_ago():
    parsed = extract_temporal_query("что было неделю назад", now=NOW)

    assert parsed is not None
    assert parsed.target_date == date(2026, 4, 27)
    assert parsed.date_focused is True


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
    def __init__(self):
        self.last_filters = None
        self.last_batch_size = None
        self.search_called = False

    def scroll(self, batch_size=256, filters=None, with_vectors=False):
        self.last_batch_size = batch_size
        self.last_filters = filters
        return iter([
            _Hit("diary-1", 1.0, {"text": "diary entry", "timestamp": "2026-04-30T10:00:00Z"}),
        ])

    def search(self, vector, limit=10, filters=None):
        self.search_called = True
        return []


def test_date_focused_query_uses_timestamp_scroll_instead_of_semantic_search():
    embedding = _NoopEmbedding()
    store = _ScrollableStore()
    retriever = TemporalRetriever(
        embedding=embedding,
        vector_store=store,
        extractor=lambda q: extract_temporal_query(q, now=NOW),
    )

    results = retriever.search("что делали 30 апреля", limit=5)

    assert len(results) == 1
    assert results[0].id == "diary-1"
    assert results[0].payload["temporal_match"] is True
    assert embedding.called is False
    assert store.search_called is False
    assert store.last_filters["timestamp"]["gte"].startswith("2026-04-29")
    assert store.last_filters["timestamp"]["lte"].startswith("2026-05-01")
