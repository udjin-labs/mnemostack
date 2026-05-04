"""Tests for Ebbinghaus-style confidence decay in FreshnessBlend."""
from datetime import datetime, timedelta, timezone

import pytest

from mnemostack.recall import RecallResult
from mnemostack.recall.pipeline import FreshnessBlend, PipelineContext
from mnemostack.recall.pipeline.stages import compute_decay


def _iso_days_ago(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _result(score: float = 1.0, **payload):
    return RecallResult(id="r1", text="memory", score=score, payload=dict(payload))


def test_recent_access_decay_near_one():
    decay = compute_decay(_iso_days_ago(1), access_count=0)
    assert decay == pytest.approx(0.98, abs=0.03)


def test_old_access_decay_to_quarter():
    decay = compute_decay(_iso_days_ago(60), access_count=0)
    assert decay == pytest.approx(0.25, abs=0.03)


def test_reinforcement_extends_effective_half_life():
    old = _iso_days_ago(60)
    unrepeated = compute_decay(old, access_count=0)
    reinforced = compute_decay(old, access_count=10)
    assert reinforced > unrepeated
    assert reinforced == pytest.approx(0.63, abs=0.05)


def test_no_last_accessed_is_neutral_half():
    assert compute_decay(None) == 0.5


def test_very_old_access_uses_floor():
    assert compute_decay(_iso_days_ago(365), access_count=0) == 0.1


def test_freshness_blend_multiplies_score_and_records_decay_factor():
    result = _result(score=1.0, last_accessed=_iso_days_ago(60), access_count=0)
    out = FreshnessBlend(weight=0.0).apply(PipelineContext(query="memory"), [result])
    assert out[0].score == pytest.approx(0.25, abs=0.03)
    assert out[0].payload["decay_factor"] == pytest.approx(0.25, abs=0.03)
