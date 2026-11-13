"""Offline ranking regressions; LLM responses are fixed fixtures."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from mnemostack.recall.pipeline.base import PipelineContext
from mnemostack.recall.pipeline.presets import (
    build_full_pipeline,
    build_stateless_pipeline,
)
from mnemostack.recall.pipeline.stages import (
    CuriosityBoost,
    ExactTokenRescue,
    FreshnessBlend,
)
from mnemostack.recall.pipeline.state import InMemoryStateStore
from mnemostack.recall.recaller import Recaller, RecallResult
from mnemostack.recall.reranker import Reranker


def hit(id, payload=None, text="ordinary text"):
    return RecallResult(
        id=id, text=text, score=0.5, payload=payload or {}, sources=["vector"]
    )


@pytest.mark.parametrize(
    "reply,ids,expected",
    [
        ("2", ["a", "2abcdef"], ["a", "2abcdef"]),
        ("ID=2:", ["a", "2abcdef"], ["a", "2abcdef"]),
        ("R1", ["a", "2abcdef"], ["2abcdef", "a"]),
        ("ID=R1:", ["a", "2abcdef"], ["2abcdef", "a"]),
        ("2", ["a", 2], [2, "a"]),
        ("2abcdef", ["a", "2abcdef"], ["2abcdef", "a"]),
        ("MEMORY.md", ["a", "MEMORY.md:12"], ["MEMORY.md:12", "a"]),
        ("NONE", ["a", "b"], ["a", "b"]),
    ],
)
def test_reranker_resolves_only_supported_ids(reply, ids, expected):
    calls = []

    def generate(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return SimpleNamespace(ok=True, text=reply)

    results = [hit(id) for id in ids]
    ranked = Reranker(SimpleNamespace(generate=generate)).rerank("lookup", results)
    assert [r.id for r in ranked] == expected
    assert len(calls) == 1
    assert sorted(str(r.id) for r in ranked) == sorted(str(r.id) for r in results)


@pytest.mark.parametrize(
    "minutes,penalized",
    [(-1440, False), (-1, False), (1, True), (9, True), (11, False)],
)
def test_echo_window_excludes_future_events(minutes, penalized):
    timestamp = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
    result = hit("event", {"timestamp": timestamp})
    FreshnessBlend().apply(PipelineContext("event"), [result])
    assert bool(result.payload.get("echo_penalty")) == penalized
    assert result.payload["access_boost"] == 1.0
    assert result.payload["timestamp"] == timestamp


@pytest.mark.parametrize("flags", ["custom", None, 7, {"caller": True}, ["existing"]])
def test_rescue_does_not_crash_on_foreign_flags(flags):
    result = hit("source", {"source": "MEMORY.md", "flags": flags}, "Port 5432")
    ExactTokenRescue().apply(PipelineContext("port 5432"), [result])
    assert result.score == 1.0
    assert result.payload["flags"] == flags
    assert result.annotations["flags"] == ["exact_rescue"]


@pytest.mark.parametrize("entry", [None, "bad", 3, [], {"id": "other"}])
def test_curiosity_ignores_nonrecord_log_entries(entry):
    store = InMemoryStateStore()
    store.set("ior_log", [entry, {"id": "other"}])
    result = hit("memory")
    CuriosityBoost(store).apply(PipelineContext("memory"), [result])
    assert result.score == pytest.approx(0.5125)


@pytest.mark.parametrize("builder", [build_full_pipeline, build_stateless_pipeline])
def test_calibration_precedes_utility_modifiers(builder):
    pipeline = builder()
    names = [s.name for s in pipeline.stages]
    assert names.count("CalibrateRelevance") == 1
    assert names.index("CalibrateRelevance") < names.index("ExactTokenRescue")
    assert names.index("CalibrateRelevance") < names.index("GravityDampen")


@pytest.mark.parametrize("scale", [1, 100, 0.01])
def test_minimal_pipeline_has_consistent_utility_scale(scale):
    pipeline = build_stateless_pipeline(freshness_weight=0)
    weak = hit("weak", {"source": "MEMORY.md"}, "Orchard port 5432")
    strong = hit("strong", text="Orchard port reference")
    unrelated = hit("unrelated", text="Unrelated fruit")
    strong.score, weak.score, unrelated.score = 0.04 * scale, 0.01 * scale, 0.03 * scale
    ranked = pipeline.apply("Orchard port 5432", [strong, unrelated, weak])
    by_id = {r.id: r for r in ranked}
    assert by_id["strong"].score == pytest.approx(1)
    assert by_id["weak"].score == pytest.approx(1)
    assert by_id["unrelated"].score == pytest.approx((0.03 / 0.04) ** 0.5 * 0.5)
    assert by_id["unrelated"].payload["dampened"] == "gravity"


@pytest.mark.parametrize("primary_score", [None, 0.001, 0.03, 0.0, -0.02])
def test_fallback_uses_rank_units_without_replacing_fused_evidence(primary_score):
    original = [] if primary_score is None else [hit("existing")]
    if original:
        original[0].score = primary_score
        original[0].sources = ["bm25"]
    candidates = [hit("existing"), hit("new")]
    candidates[0].score = 0.99
    candidates[1].score = 0.90
    receiver = SimpleNamespace(
        rrf_k=60, _vector_fallback_hits=lambda *a, **k: candidates
    )
    results = Recaller._maybe_apply_fallback(
        receiver, "lookup", original, limit=10, vector_limit=10, filters=None
    )
    if primary_score is None or primary_score >= 0:
        assert all(r.score >= 0 for r in results)
    if original:
        assert results[0] is original[0]
        assert results[0].score == primary_score
        assert results[0].sources == ["bm25", "vector"]
        ceiling = primary_score - abs(primary_score) * 0.5
        assert results[1].score == pytest.approx(min(1 / 62, ceiling))
        assert results[1].score <= primary_score
    else:
        assert [r.score for r in results] == pytest.approx([1 / 61, 1 / 62])


def test_fallback_retains_raw_vector_evidence_for_an_existing_id():
    existing = hit("existing", {"source": "fixture"})
    existing.score = 0.01
    existing.sources = ["bm25"]
    vector = hit("existing", {"raw_vector_score": 0.9})
    vector.score = 0.9
    receiver = SimpleNamespace(rrf_k=60, _vector_fallback_hits=lambda *a, **k: [vector])
    results = Recaller._maybe_apply_fallback(receiver, "lookup", [existing], limit=10, vector_limit=10, filters=None)
    assert results[0].payload["raw_vector_score"] == 0.9
