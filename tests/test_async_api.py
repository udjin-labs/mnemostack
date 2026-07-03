"""Async mirrors of the sync surface: recall_flow / Ingestor / answer / synthesis.

All wrappers are asyncio.to_thread offloads with signature-stable contracts
(see Recaller.recall_async, the original of the pattern, covered in
test_recaller_async.py). The tests are hermetic — fakes, no backends — and
verify two things per wrapper: same results as the sync call, and the event
loop keeps ticking while the blocking work runs.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from mnemostack import synthesize, synthesize_async
from mnemostack.ingest import IngestItem, Ingestor
from mnemostack.llm.base import LLMProvider, LLMResponse
from mnemostack.recall import (
    AnswerGenerator,
    RecallResult,
    recall_flow,
    recall_flow_async,
)


class _FakeEmbedding:
    dimension = 3

    def embed(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        time.sleep(0.05)  # simulate blocking HTTP
        return [self.embed(t) for t in texts]


class _FakeStore:
    def __init__(self):
        self.upserts = []

    def upsert(self, id, vector, payload):
        self.upserts.append((id, vector, payload))

    def upsert_batch(self, points):
        for p in points:
            self.upsert(p[0], p[1], p[2])
        return len(points)


class _FakeRecaller:
    def __init__(self, results):
        self._results = results

    def recall(self, query, limit=10, filters=None, **_):
        time.sleep(0.05)  # simulate blocking retrieval
        return self._results[:limit]

    def apply_vector_floor_after_rerank(self, results, recalled_results):
        return results


class _FakeLLM(LLMProvider):
    @property
    def name(self) -> str:
        return "fake"

    def generate(self, prompt, max_tokens=200, temperature=0.0):
        time.sleep(0.05)  # simulate blocking HTTP
        return LLMResponse(text="Postgres\nCONFIDENCE: 0.9", tokens_used=10)


def _results(n=3):
    return [
        RecallResult(id=str(i), text=f"memory {i}", score=1.0 - i * 0.1, payload={})
        for i in range(n)
    ]


async def _assert_loop_alive(coro):
    """Await `coro` while proving the event loop keeps ticking."""
    ticks = {"n": 0}

    async def ticker():
        while True:
            ticks["n"] += 1
            await asyncio.sleep(0.005)

    t = asyncio.create_task(ticker())
    try:
        result = await coro
    finally:
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
    assert ticks["n"] >= 3, f"event loop appeared blocked ({ticks['n']} ticks)"
    return result


@pytest.mark.asyncio
async def test_recall_flow_async_matches_sync():
    results = _results()
    sync_out = recall_flow(_FakeRecaller(results), "q", limit=3)
    async_out = await _assert_loop_alive(
        recall_flow_async(_FakeRecaller(results), "q", limit=3)
    )
    assert [r.id for r in async_out] == [r.id for r in sync_out]


@pytest.mark.asyncio
async def test_recall_flow_async_passes_token_budget():
    out = await recall_flow_async(
        _FakeRecaller(_results()),
        "q",
        limit=3,
        token_budget=2,
        token_counter=lambda s: 1,
    )
    assert len(out) == 2


@pytest.mark.asyncio
async def test_ingest_async_matches_sync():
    items = [IngestItem(text=f"text {i}", source="notes.md", offset=i) for i in range(3)]

    sync_store = _FakeStore()
    sync_stats = Ingestor(_FakeEmbedding(), sync_store).ingest(items)

    async_store = _FakeStore()
    async_stats = await _assert_loop_alive(
        Ingestor(_FakeEmbedding(), async_store).ingest_async(items)
    )

    assert async_stats.upserted == sync_stats.upserted == 3
    assert [u[0] for u in async_store.upserts] == [u[0] for u in sync_store.upserts]


@pytest.mark.asyncio
async def test_ingest_one_async_ingests_single_item():
    store = _FakeStore()
    stats = await Ingestor(_FakeEmbedding(), store).ingest_one_async(
        IngestItem(text="solo", source="notes.md", offset=0)
    )
    assert stats.upserted == 1
    assert len(store.upserts) == 1


@pytest.mark.asyncio
async def test_generate_async_matches_sync():
    memories = _results()
    gen = AnswerGenerator(llm=_FakeLLM(), specificity_resolver=False, inference_retry=False)

    sync_answer = gen.generate("what db", memories)
    async_answer = await _assert_loop_alive(gen.generate_async("what db", memories))

    assert async_answer.text == sync_answer.text == "Postgres"
    assert async_answer.confidence == sync_answer.confidence
    assert async_answer.tokens_used == sync_answer.tokens_used


@pytest.mark.asyncio
async def test_generate_async_passes_token_budget():
    gen = AnswerGenerator(llm=_FakeLLM(), specificity_resolver=False, inference_retry=False)
    answer = await gen.generate_async(
        "what db",
        _results(),
        token_budget=1,
        token_counter=lambda s: 1,
    )
    assert answer.context_tokens_estimate == 1


@pytest.mark.asyncio
async def test_synthesize_async_matches_sync():
    recaller = _FakeRecaller(
        [
            RecallResult(
                id="f1",
                text="Entity Alpha launched the beacon project.",
                score=0.9,
                payload={"timestamp": "2026-01-02T03:04:05Z"},
                sources=["vector"],
            )
        ]
    )

    sync_result = synthesize("Entity Alpha", recaller=recaller)
    async_result = await _assert_loop_alive(
        synthesize_async("Entity Alpha", recaller=recaller)
    )

    assert async_result.entity == sync_result.entity
    assert [f.text for f in async_result.facts] == [f.text for f in sync_result.facts]


@pytest.mark.asyncio
async def test_async_wrappers_run_concurrently():
    """Two 50 ms blocking calls gathered together must not take ~100 ms."""
    gen = AnswerGenerator(llm=_FakeLLM(), specificity_resolver=False, inference_retry=False)
    memories = _results()

    start = time.monotonic()
    await asyncio.gather(
        gen.generate_async("q1", memories),
        gen.generate_async("q2", memories),
    )
    elapsed = time.monotonic() - start

    assert elapsed < 0.095, f"expected concurrent execution, took {elapsed:.3f}s"
