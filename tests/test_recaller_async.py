"""Async wrapper tests for Recaller.

We don't spin up Qdrant here — use a FakeRetriever so the test is hermetic.
The point is to prove `recall_async` returns the same shape as `recall` and
does not block the event loop (caller can `await` other coroutines alongside).
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import pytest

from mnemostack.recall.recaller import Recaller, RecallResult


@dataclass
class _FakeRetriever:
    name: str = "fake"
    delay_ms: int = 50
    ids: tuple[str, ...] = ("a", "b", "c")

    def search(self, query: str, limit: int, filters: dict[str, Any] | None = None):
        time.sleep(self.delay_ms / 1000)  # simulate blocking I/O
        return [
            RecallResult(
                id=self.ids[i],
                text=f"{self.name}:{self.ids[i]}",
                score=1.0 / (i + 1),
                payload={"source": f"{self.name}/{self.ids[i]}"},
                sources=[self.name],
            )
            for i in range(min(limit, len(self.ids)))
        ]


@pytest.mark.asyncio
async def test_recall_async_returns_same_shape_as_sync():
    r = _FakeRetriever(name="r1", delay_ms=10)
    recaller = Recaller(retrievers=[r])
    sync_results = recaller.recall("q", limit=3)
    async_results = await recaller.recall_async("q", limit=3)
    assert [x.id for x in sync_results] == [x.id for x in async_results]


@pytest.mark.asyncio
async def test_recall_async_does_not_block_event_loop():
    """While recall_async waits on its worker thread, the loop must keep ticking."""
    r = _FakeRetriever(name="slow", delay_ms=120)
    recaller = Recaller(retrievers=[r])

    tick_counter = {"n": 0}

    async def ticker():
        while True:
            tick_counter["n"] += 1
            await asyncio.sleep(0.01)

    t = asyncio.create_task(ticker())
    try:
        await recaller.recall_async("q", limit=3)
    finally:
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass

    # With a 120 ms blocking worker and a 10 ms ticker, we expect *many* ticks.
    # Anything above ~5 proves the loop was not frozen.
    assert tick_counter["n"] >= 5, f"event loop appeared blocked ({tick_counter['n']} ticks)"


@pytest.mark.asyncio
async def test_multiple_retrievers_run_in_parallel():
    """Two retrievers with 100 ms latency should finish in <~180 ms, not ~200 ms."""
    a = _FakeRetriever(name="a", delay_ms=100, ids=("a1", "a2"))
    b = _FakeRetriever(name="b", delay_ms=100, ids=("b1", "b2"))
    recaller = Recaller(retrievers=[a, b])

    start = time.monotonic()
    results = await recaller.recall_async("q", limit=4)
    elapsed_ms = (time.monotonic() - start) * 1000

    assert len(results) == 4
    assert elapsed_ms < 180, f"retrievers appear to have run serially ({elapsed_ms:.0f} ms)"
