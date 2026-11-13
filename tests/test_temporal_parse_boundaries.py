"""Absolute years, inclusive period edges, and fail-closed relative overflow."""
from datetime import datetime, timezone

import pytest

from mnemostack.recall.retrievers import extract_temporal_query

NOW = datetime(2026, 7, 15, tzinfo=timezone.utc)


@pytest.mark.parametrize("query", ["April 30, 2025", "April 30th,2025", "April 30 , 2025", "30 апреля, 2025", "30 April,2025"])
def test_explicit_year_after_comma_is_not_replaced_by_current_year(query):
    parsed = extract_temporal_query(query, now=NOW)
    assert parsed is not None
    assert parsed.target_date.isoformat() == "2025-04-30"


@pytest.mark.parametrize("query", ["April 30 2025", "30 апреля 2025", "2025-04-30"])
def test_existing_explicit_year_forms_remain(query):
    assert extract_temporal_query(query, now=NOW).target_date.isoformat() == "2025-04-30"


@pytest.mark.parametrize("query", ["April 30", "30 апреля"])
def test_no_year_keeps_current_year_default(query):
    assert extract_temporal_query(query, now=NOW).target_date.isoformat() == "2026-04-30"


@pytest.mark.parametrize("query,first,next_period", [("April 2025", "2025-04-01", "2025-05-01"), ("December 2025", "2025-12-01", "2026-01-01"), ("February 2024", "2024-02-01", "2024-03-01"), ("2025", "2025-01-01", "2026-01-01")])
def test_inclusive_range_stops_before_next_period(query, first, next_period):
    parsed = extract_temporal_query(query, now=NOW)
    start, end = map(datetime.fromisoformat, parsed.window)
    next_start = datetime.fromisoformat(next_period).replace(tzinfo=timezone.utc)
    assert start == datetime.fromisoformat(first).replace(tzinfo=timezone.utc)
    assert end < next_start
    assert (next_start - end).total_seconds() == pytest.approx(0.000001)


@pytest.mark.parametrize("query", ["999999999999 days ago", "999999999999 weeks ago", "999999999999 дней назад", "999999999999 недель назад", "9" * 5000 + " days ago"])
def test_unrepresentable_relative_window_returns_no_parse(query):
    assert extract_temporal_query(query, now=NOW) is None


@pytest.mark.parametrize("query,expected", [("2 days ago", "2026-07-13"), ("2 weeks ago", "2026-07-01"), ("2 дня назад", "2026-07-13"), ("2 недели назад", "2026-07-01")])
def test_normal_relative_dates_keep_existing_window(query, expected):
    parsed = extract_temporal_query(query, now=NOW)
    assert parsed.target_date.isoformat() == expected
    assert datetime.fromisoformat(parsed.start_iso).date() < parsed.target_date
    assert datetime.fromisoformat(parsed.end_iso).date() > parsed.target_date
