"""Scale-independent ranking without discarding recency or source evidence."""

import copy
import math
from datetime import datetime, timedelta, timezone

import pytest

from mnemostack.recall import InMemoryStateStore, RecallResult, build_full_pipeline
from mnemostack.recall.fusion import reciprocal_rank_fusion
from mnemostack.recall.pipeline import stages
from mnemostack.recall.pipeline.base import PipelineContext
from mnemostack.recall.pipeline.stages import CalibrateRelevance, CuriosityBoost

NOW = datetime(2026, 5, 20, 12, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)

    monkeypatch.setattr(stages, "datetime", Clock)


def result(pid, score, days=30, **metadata):
    return RecallResult(
        id=pid, text="stored information", score=score,
        payload={"timestamp": (NOW - timedelta(days=days)).isoformat(),
                 "source": "notes.txt", "offset": 12, **metadata},
        sources=["vector"],
    )


@pytest.mark.parametrize("scores", [[], [0, 0], [-4, -1, 0, 1, 4], [-4, -1], [1e-300, 4e-300]])
def test_calibration_is_bounded_monotone_and_preserves_provenance(scores):
    rows = [result(str(i), score) for i, score in enumerate(scores)]
    original = copy.deepcopy(rows)
    out = CalibrateRelevance().apply(PipelineContext("history"), rows)
    assert [r.id for r in out] == [r.id for r in original]
    assert all(-1 <= r.score <= 1 and math.isfinite(r.score) for r in out)
    assert [r.score for r in out] == sorted(r.score for r in out)
    for before, after in zip(original, out, strict=True):
        assert after.payload == before.payload
        assert after.text == before.text and after.sources == before.sources
        assert (after.score > 0) == (before.score > 0)
        assert (after.score < 0) == (before.score < 0)


@pytest.mark.parametrize("scale", [1e-12, .01, 1, 1e9])
@pytest.mark.parametrize("scores", [[.03, .015, .012], [-.03, -.015, -.012], [-.03, 0, .015]])
def test_utility_ranking_is_scale_independent_with_matching_lexical_evidence(scale, scores):
    rows = [result(str(i), score, days=i * 20) for i, score in enumerate(scores)]
    # Gravity's existing absolute min_score gate is outside this calibration.
    expected = build_full_pipeline().apply("stored", copy.deepcopy(rows))
    for row in rows:
        row.score *= scale
    actual = build_full_pipeline().apply("stored", rows)
    assert [r.id for r in actual] == [r.id for r in expected]
    assert [r.score for r in actual] == pytest.approx([r.score for r in expected])


@pytest.mark.parametrize("arms", [1, 2, 4])
def test_duplicate_arm_score_mass_does_not_change_full_ranking(arms):
    rows = [result("old", .9, days=180), result("new", .8, days=2)]
    ranked = [(r, r.score) for r in rows]
    fused = reciprocal_rank_fusion([copy.deepcopy(ranked) for _ in range(arms)])
    inputs = []
    for row, score in fused:
        row.score = score
        inputs.append(row)
    actual = build_full_pipeline().apply("history", inputs)
    expected = build_full_pipeline().apply("history", [result("old", 1/61, days=180), result("new", 1/62, days=2)])
    assert [r.id for r in actual] == [r.id for r in expected]
    assert [r.score for r in actual] == pytest.approx([r.score for r in expected])


def test_curiosity_is_bounded_relative_utility_with_existing_age_eligibility():
    rows = [result("recent", .02, days=6), result("eligible", .02, days=7), result("ancient", .02, days=700)]
    stage = CuriosityBoost(InMemoryStateStore(), bonus=.05, min_age_days=7)
    scores = {r.id: r.score for r in stage.apply(PipelineContext("history"), rows)}
    assert scores["recent"] == .02
    assert scores["eligible"] == pytest.approx(.0205)
    assert .0205 < scores["ancient"] <= .021


def test_curiosity_keeps_missing_time_and_tenant_recall_count_contracts():
    state = InMemoryStateStore()
    # Use the same namespace helper as production rather than guessing its encoding.
    from mnemostack.recall.pipeline.state import tenant_state_key

    state.set(tenant_state_key("ior_log", "tenant-a"), [{"id": "seen"}] * 4)
    rows = [result("seen", .02), result("missing", .02)]
    rows[1].payload.pop("timestamp")
    context = PipelineContext("history", extras={"tenant": "tenant-a"})
    scores = {r.id: r.score for r in CuriosityBoost(state).apply(context, rows)}
    assert scores["seen"] == .02
    assert scores["missing"] == pytest.approx(.0205)
    other = CuriosityBoost(state).apply(PipelineContext("history", extras={"tenant": "tenant-b"}), [result("seen", .02)])
    assert other[0].score > .02


def test_future_missing_time_and_nonzero_access_remain_explicit_signals():
    rows = [result("future", .02, days=-1), result("missing", .02), result("used", .02, days=2, last_accessed=NOW.isoformat(), access_count=10)]
    rows[1].payload.pop("timestamp")
    out = {r.id: r for r in build_full_pipeline().apply("history", rows)}
    assert out["future"].payload["echo_penalty"] is True
    assert out["future"].payload["freshness"] == .5
    assert out["missing"].payload["freshness"] == .5
    assert out["used"].payload["access_boost"] == 1.25


def test_scale_defect_negative_control_loses_old_consensus_evidence():
    def inputs():
        return [result("evidence", 2/61, days=365)] + [result(f"noise-{i}", 1/(62+i), days=2) for i in range(21)]

    good = build_full_pipeline().apply("history", inputs())[:20]
    broken_pipeline = build_full_pipeline()
    broken_pipeline.stages = [s for s in broken_pipeline.stages if not isinstance(s, CalibrateRelevance)]
    broken = broken_pipeline.apply("history", inputs())[:20]
    assert "evidence" in [r.id for r in good]
    assert "evidence" not in [r.id for r in broken]


@pytest.mark.parametrize("invalid", [math.inf, -math.inf, math.nan])
@pytest.mark.parametrize("first", [True, False])
@pytest.mark.parametrize("missing_time", [True, False])
def test_curiosity_nonfinite_entry_does_not_contaminate_healthy_scores(invalid, first, missing_time):
    def healthy():
        rows = [result("positive", .03), result("negative", -.01)]
        if missing_time:
            for row in rows:
                row.payload.pop("timestamp")
        return rows

    stage = CuriosityBoost(InMemoryStateStore())
    expected = {r.id: r.score for r in stage.apply(PipelineContext("stored"), healthy())}
    bad = result("invalid", invalid)
    rows = [bad, *healthy()] if first else [*healthy(), bad]
    actual = {r.id: r.score for r in stage.apply(PipelineContext("stored"), rows)}
    assert all(math.isfinite(actual[pid]) for pid in expected)
    assert {pid: actual[pid] for pid in expected} == pytest.approx(expected)


@pytest.mark.parametrize("invalid", [math.inf, -math.inf, math.nan])
@pytest.mark.parametrize("first", [True, False])
def test_calibration_nonfinite_entry_cannot_disable_healthy_normalization(invalid, first):
    healthy = [result("positive", .03), result("negative", -.01)]
    stage = CalibrateRelevance()
    expected = {r.id: r.score for r in stage.apply(PipelineContext("stored"), copy.deepcopy(healthy))}
    bad = result("invalid", invalid)
    rows = [bad, *healthy] if first else [*healthy, bad]
    actual = {r.id: r.score for r in stage.apply(PipelineContext("stored"), rows)}
    assert all(math.isfinite(actual[pid]) for pid in expected)
    assert {pid: actual[pid] for pid in expected} == pytest.approx(expected)


@pytest.mark.parametrize("invalid", [math.inf, -math.inf, math.nan])
def test_full_pipeline_nonfinite_entry_preserves_healthy_utilities(invalid):
    healthy = [result("positive", .03), result("negative", -.01)]
    expected = {r.id: r.score for r in build_full_pipeline().apply("stored", copy.deepcopy(healthy))}
    actual = {r.id: r.score for r in build_full_pipeline().apply("stored", [result("invalid", invalid), *healthy])}
    assert all(math.isfinite(actual[pid]) for pid in expected)
    assert {pid: actual[pid] for pid in expected} == pytest.approx(expected)
