"""The inclusive Qdrant filter must exclude the next calendar period."""
from datetime import datetime, timedelta, timezone

import pytest
from qdrant_client import QdrantClient, models

from mnemostack.recall.retrievers import extract_temporal_query


@pytest.mark.parametrize("query,start,next_start", [("April 2025", "2025-04-01", "2025-05-01"), ("2025", "2025-01-01", "2026-01-01")])
def test_qdrant_inclusive_datetime_range_excludes_next_period(query, start, next_start):
    parsed = extract_temporal_query(query, now=datetime(2026, 7, 15, tzinfo=timezone.utc))
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    next_dt = datetime.fromisoformat(next_start).replace(tzinfo=timezone.utc)
    client = QdrantClient(":memory:")
    try:
        client.create_collection("synthetic-period", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT))
        client.upsert("synthetic-period", points=[models.PointStruct(id=1, vector=[0.1, 0.2], payload={"timestamp": (next_dt - timedelta(microseconds=1)).isoformat()}), models.PointStruct(id=2, vector=[0.1, 0.2], payload={"timestamp": next_dt.isoformat()}), models.PointStruct(id=3, vector=[0.1, 0.2], payload={"timestamp": start_dt.isoformat()})])
        hits, _ = client.scroll("synthetic-period", scroll_filter=models.Filter(must=[models.FieldCondition(key="timestamp", range=models.DatetimeRange(gte=parsed.start_iso, lte=parsed.end_iso))]))
        assert {hit.id for hit in hits} == {1, 3}
    finally:
        client.close()
