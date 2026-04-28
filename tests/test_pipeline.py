"""Tests for pipeline stages and orchestrator."""
from datetime import datetime, timedelta, timezone

from mnemostack.recall import RecallResult
from mnemostack.recall.pipeline import (
    ClassifyQuery,
    CuriosityBoost,
    ExactTokenRescue,
    FreshnessBlend,
    GravityDampen,
    HubDampen,
    InhibitionOfReturn,
    InMemoryStateStore,
    Pipeline,
    PipelineContext,
    QLearningReranker,
    Stage,
    is_exact_token_query,
)


class DoublingStage(Stage):
    def apply(self, context, results):
        for r in results:
            r.score *= 2
        return results


class DropAllStage(Stage):
    def apply(self, context, results):
        return []


def _mk_result(id, text, score=1.0, **payload):
    return RecallResult(id=id, text=text, score=score, payload=dict(payload))


def test_pipeline_applies_in_order():
    rs = [_mk_result(1, "x", score=1.0)]
    pipe = Pipeline([DoublingStage(), DoublingStage()])
    out = pipe.apply("q", rs)
    assert out[0].score == 4.0


def test_pipeline_stops_on_empty_by_default():
    pipe = Pipeline([DropAllStage(), DoublingStage()])
    out = pipe.apply("q", [_mk_result(1, "x")])
    assert out == []


def test_classify_query_person():
    ctx = PipelineContext(query="кто работал с Atlas")
    ClassifyQuery().apply(ctx, [])
    assert ctx.query_type == "person"


def test_classify_query_technical():
    ctx = PipelineContext(query="how to configure nginx")
    ClassifyQuery().apply(ctx, [])
    assert ctx.query_type == "technical"


def test_classify_query_default_general():
    ctx = PipelineContext(query="Paris capital")
    ClassifyQuery().apply(ctx, [])
    assert ctx.query_type == "general"


def test_classify_query_does_not_match_markers_inside_words():
    ctx = PipelineContext(query="show bridge status")
    ClassifyQuery().apply(ctx, [])
    assert ctx.query_type == "general"


def test_classify_query_tokens_filtered():
    ctx = PipelineContext(query="how do I know about Paris")
    ClassifyQuery().apply(ctx, [])
    assert "how" not in ctx.query_tokens
    assert "paris" in ctx.query_tokens


def test_is_exact_token_query():
    assert is_exact_token_query("какой IP у сервера")
    assert is_exact_token_query("какой порт 6333")
    assert is_exact_token_query("version 2026.4.14")
    assert not is_exact_token_query("candidate idea bridge")
    assert not is_exact_token_query("what is love")


def test_exact_token_rescue_boosts_matching_memory():
    rs = [
        _mk_result(1, "Qdrant runs on port 6333", score=0.4, source="MEMORY.md"),
        _mk_result(2, "some unrelated text", score=0.5, source="random.md"),
    ]
    ctx = PipelineContext(query="какой port у Qdrant")
    out = ExactTokenRescue(boost=0.5).apply(ctx, rs)
    assert out[0].id == 1
    assert "exact_rescue" in out[0].payload["flags"]


def test_exact_token_rescue_noop_for_non_exact_query():
    rs = [_mk_result(1, "text", score=0.5, source="MEMORY.md")]
    ctx = PipelineContext(query="what is love")
    out = ExactTokenRescue().apply(ctx, rs)
    assert out[0].score == 0.5


def test_gravity_dampen_penalizes_no_keyword_match():
    rs = [
        _mk_result(1, "unrelated text about cats", score=0.9),
        _mk_result(2, "authentication system uses oauth", score=0.85),
    ]
    ctx = PipelineContext(query="authentication", query_tokens=["authentication"])
    out = GravityDampen(penalty=0.5).apply(ctx, rs)
    assert out[0].id == 2
    assert out[1].payload.get("dampened") == "gravity"


def test_gravity_dampen_respects_min_score_guard():
    rs = [_mk_result(1, "cats", score=0.05)]
    ctx = PipelineContext(query="dogs", query_tokens=["dogs"])
    out = GravityDampen(penalty=0.5, min_score=0.1).apply(ctx, rs)
    assert out[0].score == 0.05
    assert "dampened" not in out[0].payload


def test_hub_dampen_penalizes_top_degree_nodes():
    rs = [
        _mk_result("hub", "connected to everything", score=0.9),
        _mk_result("leaf", "lonely node", score=0.85),
    ]
    hub_degrees = {
        "hub": 100, "leaf": 1,
        "n1": 5, "n2": 5, "n3": 5, "n4": 5, "n5": 5,
        "n6": 5, "n7": 5, "n8": 5, "n9": 5,
    }
    ctx = PipelineContext(query="x")
    out = HubDampen(hub_degrees=hub_degrees).apply(ctx, rs)
    hub_result = next(r for r in out if r.id == "hub")
    leaf_result = next(r for r in out if r.id == "leaf")
    assert hub_result.score < 0.9
    assert leaf_result.score == 0.85


def test_hub_dampen_noop_when_no_hub_degrees():
    rs = [_mk_result(1, "x", score=0.5)]
    ctx = PipelineContext(query="q")
    out = HubDampen(hub_degrees={}).apply(ctx, rs)
    assert out[0].score == 0.5


def test_freshness_blend_boosts_recent():
    recent = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    rs = [
        _mk_result(1, "old content", score=0.6, timestamp=old),
        _mk_result(2, "recent content", score=0.6, timestamp=recent),
    ]
    ctx = PipelineContext(query="x")
    out = FreshnessBlend(weight=0.5, halflife_days=14).apply(ctx, rs)
    assert out[0].id == 2


def test_freshness_blend_echo_penalty():
    very_recent = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
    rs = [_mk_result(1, "x", score=0.5, timestamp=very_recent)]
    ctx = PipelineContext(query="x")
    out = FreshnessBlend(weight=0.5, echo_window_minutes=10).apply(ctx, rs)
    assert out[0].payload.get("echo_penalty") is True


def test_freshness_blend_falls_back_to_filename_date():
    rs = [_mk_result(1, "x", score=0.5, source="memory/2025-01-01.md")]
    ctx = PipelineContext(query="x")
    out = FreshnessBlend(weight=0.5).apply(ctx, rs)
    assert out[0].payload["freshness"] < 0.5


def test_ior_penalizes_recent_recalls():
    store = InMemoryStateStore()
    stage = InhibitionOfReturn(state_store=store, penalty_per_recall=0.1)
    stage.record_recall("hot")
    stage.record_recall("hot")
    rs = [
        _mk_result("hot", "often recalled", score=1.0),
        _mk_result("cold", "never recalled", score=1.0),
    ]
    out = stage.apply(PipelineContext(query="x"), rs)
    assert out[0].id == "cold"
    hot_result = next(r for r in out if r.id == "hot")
    assert "ior_penalty" in hot_result.payload


def test_ior_trims_log_to_keep_last():
    store = InMemoryStateStore()
    stage = InhibitionOfReturn(state_store=store, keep_last=3)
    for i in range(10):
        stage.record_recall(i)
    log = store.get("ior_log")
    assert len(log) == 3


def test_curiosity_boosts_old_rarely_recalled():
    store = InMemoryStateStore()
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    rs = [
        _mk_result("rare", "old and rare", score=0.5, timestamp=old),
        _mk_result("common", "recent", score=0.5,
                   timestamp=datetime.now(timezone.utc).isoformat()),
    ]
    out = CuriosityBoost(state_store=store, bonus=0.1, min_age_days=7).apply(
        PipelineContext(query="x"), rs
    )
    rare = next(r for r in out if r.id == "rare")
    assert rare.payload.get("curiosity_boosted") is True
    assert rare.score > 0.5


def test_qlearning_reranker_boosts_high_q_source():
    # Use blend=False (additive mode) so we can reason about absolute ordering.
    store = InMemoryStateStore()
    qlr = QLearningReranker(
        state_store=store, default_q=0.5, boost_weight=1.0,
        use_blend=False, min_samples=3,
    )
    for _ in range(5):
        qlr.record_feedback("vector", "technical", reward=1.0)
    for _ in range(5):
        qlr.record_feedback("bm25", "technical", reward=0.0)
    ctx = PipelineContext(query="x", query_type="technical")
    rs = [
        _mk_result("from_bm25", "x", score=0.6),
        _mk_result("from_vector", "x", score=0.55),
    ]
    rs[0].sources = ["bm25"]
    rs[1].sources = ["vector"]
    out = qlr.apply(ctx, rs)
    assert out[0].id == "from_vector"


def test_qlearning_default_q_no_change():
    """When n >= min_samples and q == default, score stays the same.

    Populate q_table so UCB1 exploration bonus does not fire (n >= min_samples),
    and q == default_q so the blend / boost contributes nothing.
    """
    store = InMemoryStateStore()
    qlr = QLearningReranker(state_store=store, min_samples=3, use_blend=False)
    store.set("q_table", {"vector": {"general": {"q": 0.5, "n": 10}}})
    rs = [_mk_result(1, "x", score=0.5)]
    rs[0].sources = ["vector"]
    ctx = PipelineContext(query="x", query_type="general")
    out = qlr.apply(ctx, rs)
    assert abs(out[0].score - 0.5) < 1e-6


def test_full_pipeline_integration():
    recent = datetime.now(timezone.utc).isoformat()
    old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    rs = [
        _mk_result(1, "unrelated content", score=0.95, timestamp=old),
        _mk_result(2, "auth system docs", score=0.85, timestamp=recent),
        _mk_result(3, "authentication setup", score=0.75, timestamp=recent),
    ]
    pipe = Pipeline([
        ClassifyQuery(),
        GravityDampen(penalty=0.3),
        FreshnessBlend(weight=0.2),
    ])
    out = pipe.apply("how to do authentication", rs)
    assert out[-1].id == 1


def test_preset_full_pipeline_runs():
    from mnemostack.recall.pipeline import build_full_pipeline

    rs = [
        _mk_result(1, "auth docs", score=0.8, timestamp=datetime.now(timezone.utc).isoformat()),
        _mk_result(2, "unrelated", score=0.7),
    ]
    pipe = build_full_pipeline()
    # All 7+ stages applied, no crash
    out = pipe.apply("authentication", rs)
    assert len(out) == 2


def test_preset_stateless_pipeline():
    from mnemostack.recall.pipeline import build_stateless_pipeline

    pipe = build_stateless_pipeline()
    rs = [_mk_result(1, "content", score=0.5)]
    out = pipe.apply("q", rs)
    assert len(out) == 1


def test_preset_full_pipeline_disabled_stages():
    from mnemostack.recall.pipeline import build_full_pipeline

    pipe = build_full_pipeline(
        enable_q_learning=False,
        enable_ior=False,
        enable_curiosity=False,
    )
    # Should still have classify + rescue + gravity + freshness at minimum
    assert len(pipe) >= 4


def test_preset_full_pipeline_stateful_master_toggle():
    from mnemostack.recall.pipeline import (
        CuriosityBoost,
        InhibitionOfReturn,
        QLearningReranker,
        build_full_pipeline,
    )

    pipe = build_full_pipeline(enable_stateful_stages=False)

    assert not any(isinstance(stage, QLearningReranker) for stage in pipe)
    assert not any(isinstance(stage, CuriosityBoost) for stage in pipe)
    assert not any(isinstance(stage, InhibitionOfReturn) for stage in pipe)
