from mnemostack.recall import RecallResult, VectorRetriever
from mnemostack.vector.qdrant import Hit
from mnemostack.recall.pipeline import (
    GravityDampen,
    HubDampen,
    Pipeline,
    PipelineContext,
    TechnicalScoreFloor,
)


def _result(id_, text="unrelated text", score=1.0, raw_vector_score=None):
    payload = {}
    if raw_vector_score is not None:
        payload["raw_vector_score"] = raw_vector_score
    return RecallResult(id=id_, text=text, score=score, payload=payload, sources=["vector"])


def test_technical_query_with_exact_tokens_gets_reduced_gravity_dampening():
    ctx = PipelineContext(query="where is /etc/nginx/nginx.conf", query_tokens=["missing"])
    rs = [_result("tech", score=1.0)]

    out = GravityDampen(penalty=0.5, technical_query_dampening_scale=0.4).apply(ctx, rs)

    assert out[0].score == 0.8
    assert out[0].payload["dampened"] == "gravity"


def test_non_technical_query_gets_normal_gravity_dampening():
    ctx = PipelineContext(query="where is the config", query_tokens=["config"])
    rs = [_result("general", score=1.0)]

    out = GravityDampen(penalty=0.5, technical_query_dampening_scale=0.4).apply(ctx, rs)

    assert out[0].score == 0.5
    assert out[0].payload["dampened"] == "gravity"


def test_technical_query_with_exact_tokens_gets_reduced_hub_dampening():
    ctx = PipelineContext(query="find api_key_backup", query_tokens=["api_key_backup"])
    hub_degrees = {"hub": 100, "leaf": 1, **{f"n{i}": 5 for i in range(9)}}
    rs = [_result("hub", score=1.0), _result("leaf", score=1.0)]

    out = HubDampen(
        hub_degrees=hub_degrees,
        p90_floor=0.4,
        technical_query_dampening_scale=0.4,
    ).apply(ctx, rs)

    hub = next(r for r in out if r.id == "hub")
    assert hub.score == 0.76
    assert hub.payload["dampened"] == "hub"


def test_score_floor_is_disabled_for_technical_query():
    pipe = Pipeline([
        GravityDampen(penalty=0.1),
        TechnicalScoreFloor(),
    ])
    rs = [_result("strong", score=0.2, raw_vector_score=0.8)]

    out = pipe.apply("lookup /var/log/app.log", rs)

    assert out[0].score == 0.092
    assert "score_floor" not in out[0].payload


def test_raw_vector_score_preserved_in_vector_retriever_payload():
    class Embedder:
        def embed(self, query):
            return [0.1]

    class Store:
        def search(self, vector, limit=20, filters=None):
            return [Hit(id="v1", score=0.81, payload={"text": "technical hit"})]

    out = VectorRetriever(Embedder(), Store()).search("lookup api_key")

    assert out[0].payload["raw_vector_score"] == 0.81
    assert out[0].score == 0.81


def test_score_floor_not_applied_for_non_technical_query():
    pipe = Pipeline([
        GravityDampen(penalty=0.1),
        TechnicalScoreFloor(),
    ])
    rs = [_result("strong", score=0.2, raw_vector_score=0.8)]

    out = pipe.apply("lookup application logs", rs)

    assert out[0].score == 0.020000000000000004
    assert "score_floor" not in out[0].payload
