from copy import deepcopy

from mnemostack.recall import RecallResult
from mnemostack.recall.pipeline import (
    ExactTokenProtection,
    GravityDampen,
    HubDampen,
    InhibitionOfReturn,
    Pipeline,
    PipelineContext,
    build_full_pipeline,
)


def _result(id_, *, score=0.1, raw_vector_score=None, sources=None, text="unrelated"):
    payload = {}
    if raw_vector_score is not None:
        payload["raw_vector_score"] = raw_vector_score
    return RecallResult(
        id=id_,
        text=text,
        score=score,
        payload=payload,
        sources=sources or ["vector"],
    )


def test_non_technical_queries_are_completely_unchanged():
    results = [
        _result("a", score=0.1, raw_vector_score=0.99),
        _result("b", score=0.3, raw_vector_score=0.8, sources=["bm25"]),
    ]
    before = deepcopy(results)

    out = Pipeline([ExactTokenProtection()]).apply("what did we discuss yesterday", results)

    assert out == before


def test_exact_token_protection_requires_exact_token_vector_source_and_raw_gate():
    results = [
        _result("protected", score=0.1, raw_vector_score=0.75, sources=["vector"]),
        _result("not-vector", score=0.1, raw_vector_score=0.99, sources=["bm25"]),
        _result("low-raw", score=0.1, raw_vector_score=0.749, sources=["vector"]),
        _result("missing-raw", score=0.1, sources=["vector"]),
    ]

    out = ExactTokenProtection().apply(
        PipelineContext(query="find api_key_backup"),
        results,
    )

    by_id = {r.id: r for r in out}
    assert by_id["protected"].score == 0.25
    assert by_id["not-vector"].score == 0.1
    assert by_id["low-raw"].score == 0.1
    assert by_id["missing-raw"].score == 0.1


def test_exact_token_protection_applies_floor_when_score_is_below_floor():
    result = _result("strong", score=0.04, raw_vector_score=0.9, sources=["vector"])

    out = ExactTokenProtection().apply(
        PipelineContext(query="lookup /var/log/app.log"),
        [result],
    )

    assert out[0].score == 0.25


def test_exact_token_protection_does_not_change_scores_above_floor():
    result = _result("strong", score=0.3, raw_vector_score=0.9, sources=["vector"])

    out = ExactTokenProtection().apply(
        PipelineContext(query="lookup 192.168.1.10"),
        [result],
    )

    assert out[0].score == 0.3


def test_exact_token_protection_runs_after_dampening_stages_in_full_pipeline():
    pipeline = build_full_pipeline(
        hub_degrees={"strong": 100, "other": 1},
        enable_q_learning=False,
        enable_curiosity=False,
    )
    stage_types = [type(stage) for stage in pipeline.stages]

    protection_idx = stage_types.index(ExactTokenProtection)
    assert protection_idx > stage_types.index(GravityDampen)
    assert protection_idx > stage_types.index(HubDampen)
    assert protection_idx > stage_types.index(InhibitionOfReturn)

    result = _result(
        "strong",
        score=0.2,
        raw_vector_score=0.9,
        sources=["vector"],
        text="no matching terms here",
    )

    out = pipeline.apply("lookup api_key_backup", [result])

    assert out[0].payload["dampened"] == "gravity"
    assert out[0].score == 0.25
