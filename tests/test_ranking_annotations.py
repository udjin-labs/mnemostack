"""Ranking annotations preserve stored metadata and copied result ownership."""

import copy
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone

import pytest

from mnemostack.recall import RecallResult
from mnemostack.recall.fusion import reciprocal_rank_fusion
from mnemostack.recall.pipeline.annotations import annotate
from mnemostack.recall.pipeline.presets import build_full_pipeline, build_stateless_pipeline
from mnemostack.recall.pipeline.state import InMemoryStateStore
from mnemostack.recall.recaller import Recaller

KEYS = (
    "flags", "dampened", "echo_penalty", "access_boost", "freshness",
    "ior_penalty", "curiosity_boosted", "q_value",
)


EXPECTED = {
    False: [("old", 1.3600000000000003), ("new", 0.28093838859037246)],
    True: [("old", 1.2305934504970184), ("new", 0.28275036705520235)],
}


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 11, 13, tzinfo=timezone.utc)


def pipeline(stateful):
    if not stateful:
        return build_stateless_pipeline(hub_degrees={"old": 100})
    store = InMemoryStateStore()
    store.set("ior_log", [{"id": "old", "timestamp": "2026-11-12T23:59:00+00:00"}])
    return build_full_pipeline(state_store=store, hub_degrees={"old": 100})


def results(payload=None):
    return [
        RecallResult("old", "Port 5432", .03,
                     {"source": "MEMORY.md", "timestamp": "2020-01-01", **(payload or {})}, ["vector"]),
        RecallResult("new", "unrelated", .01,
                     {"timestamp": "2026-11-12T23:59:00+00:00", **(payload or {})}, ["bm25"]),
    ]


@pytest.mark.parametrize("stateful", [False, True])
def test_pipeline_payload_copies_and_fixed_scores(monkeypatch, stateful):
    import mnemostack.recall.pipeline.stages as stages

    monkeypatch.setattr(stages, "datetime", Clock)
    current = pipeline(stateful)
    actual = current.apply("port 5432", results())
    assert [(r.id, r.score) for r in actual] == EXPECTED[stateful]
    for run in range(2):
        for result in actual:
            assert result.annotations
            assert result.annotations == {key: result.payload[key] for key in KEYS if key in result.payload}
        if run == 0:
            actual = current.apply("port 5432", actual)


@pytest.mark.parametrize("flags", ["client", ["client"]])
def test_full_pipeline_keeps_all_client_keys(monkeypatch, flags):
    import mnemostack.recall.pipeline.stages as stages

    monkeypatch.setattr(stages, "datetime", Clock)
    client = dict.fromkeys(KEYS, "client")
    client["flags"] = flags
    expected = pipeline(True).apply("port 5432", results())
    actual = pipeline(True).apply("port 5432", results(copy.deepcopy(client)))
    assert [(r.id, r.score) for r in actual] == [(r.id, r.score) for r in expected]
    for result, reference in zip(actual, expected, strict=True):
        assert {key: result.payload[key] for key in KEYS} == client
        assert result.annotations == reference.annotations
    assert set().union(*(r.annotations for r in actual)) == set(KEYS)


def test_annotation_updates_owned_payload_only():
    owned, client = results(), results({"dampened": "client"})
    for result in [owned[0], client[0]]:
        annotate(result, "dampened", "gravity")
        annotate(result, "dampened", "hub")
        assert result.annotations["dampened"] == "hub"
    assert owned[0].payload["dampened"] == "hub"
    assert client[0].payload["dampened"] == "client"


@pytest.mark.parametrize("kind", ["replace", "fusion", "inference", "floor"])
def test_copies_keep_isolated_annotations_and_ownership(kind):
    original = results()[0]
    annotate(original, "flags", ["exact_rescue"])
    annotate(original, "dampened", "gravity")
    if kind == "replace":
        copied = replace(original, payload=dict(original.payload), sources=list(original.sources))
    elif kind == "inference":
        from mnemostack.recall.inference_retry import merge_results

        copied = merge_results([original])[0]
    elif kind == "fusion":
        other = replace(original, sources=["bm25"])
        copied = reciprocal_rank_fusion([[(original, 1)], [(other, 1)]])[0][0]
    else:
        recaller = object.__new__(Recaller)
        recaller.vector_floor = 1
        carrier = results()[1]
        recaller._attach_vector_floor_candidates([carrier], [original])
        copied = recaller._vector_floor_candidates_from_results([carrier])[0]
        assert "_annotation_payload_keys" not in carrier.payload["_vector_floor_candidates"][0]
    assert copied.annotations == original.annotations
    assert copied._annotation_payload_keys == original._annotation_payload_keys
    copied.annotations["flags"].append("copy")
    copied._annotation_payload_keys += ("copy",)
    annotate(copied, "dampened", "hub")
    assert original.annotations == {"flags": ["exact_rescue"], "dampened": "gravity"}
    assert original.payload["dampened"] == "gravity"
    assert "copy" not in original._annotation_payload_keys
    assert copied.payload["dampened"] == "hub"


def test_http_memory_includes_annotations():
    from mnemostack.server import _memory_of

    result = results({"freshness": "client"})[0]
    annotate(result, "freshness", .5)
    memory = _memory_of(result).model_dump()
    assert memory["annotations"] == {"freshness": .5}
    assert memory["metadata"] == result.payload
    assert memory["id"] == result.id


def test_gravity_then_hub_updates_same_annotation():
    from mnemostack.recall.pipeline import GravityDampen, HubDampen, PipelineContext

    result = RecallResult("hub", "unrelated", .9)
    context = PipelineContext(query="authentication", query_tokens=["authentication"])
    GravityDampen().apply(context, [result])
    assert result.annotations["dampened"] == "gravity"
    HubDampen({**dict.fromkeys(range(20), 1), "hub": 100}).apply(context, [result])
    assert result.annotations["dampened"] == result.payload["dampened"] == "hub"


def test_reranker_cache_copies_annotations_on_store_and_hit():
    from unittest.mock import Mock

    from mnemostack.llm.base import LLMResponse
    from mnemostack.recall.reranker import Reranker

    llm = Mock()
    llm.generate.return_value = LLMResponse(text="R0 R1")
    reranker = Reranker(llm)
    original = results()
    annotate(original[0], "flags", ["exact_rescue"])
    reranker.rerank("query", original)
    original[0].annotations["flags"].append("original")
    original[0]._annotation_payload_keys += ("original",)
    first = reranker.rerank("query", results())[0]
    assert first.annotations == {"flags": ["exact_rescue"]}
    assert first._annotation_payload_keys == ("flags",)
    first.annotations["flags"].append("first")
    first._annotation_payload_keys += ("first",)
    second = reranker.rerank("query", results())[0]
    assert second.annotations == {"flags": ["exact_rescue"]}
    assert second._annotation_payload_keys == ("flags",)
    assert llm.generate.call_count == 1


@pytest.mark.parametrize("legacy", [False, True])
def test_deadline_fallback_copies_and_writeback(monkeypatch, legacy):
    from types import SimpleNamespace

    original = results()[0]
    annotate(original, "flags", ["exact_rescue"])
    if legacy:
        recaller = Recaller(
            embedding_provider=SimpleNamespace(),
            vector_store=SimpleNamespace(search=lambda *a, **kw: []),
            overall_deadline=5,
        )
        recaller.bm25 = SimpleNamespace(search=lambda *a, **kw: [(original, .9)])
        monkeypatch.setattr(recaller, "_ensure_space_compat", lambda **kw: None)
        monkeypatch.setattr("mnemostack.recall.recaller.embed_query_via", lambda *a, **kw: [1.])
    else:
        recaller = Recaller(
            retrievers=[SimpleNamespace(name="bm25", search=lambda *a, **kw: [original])],
            overall_deadline=5,
        )
    captured = []

    def fallback(query, candidates, **kwargs):
        candidate = candidates[0]
        assert candidate is not original
        assert candidate.annotations == original.annotations
        assert candidate._annotation_payload_keys == original._annotation_payload_keys
        candidate.annotations["flags"].append("fallback")
        candidate._annotation_payload_keys += ("fallback",)
        assert original.annotations["flags"] == ["exact_rescue"]
        assert original._annotation_payload_keys == ("flags",)
        captured.append(candidate)
        return candidates

    monkeypatch.setattr(recaller, "_maybe_apply_fallback", fallback)
    output = recaller.recall("port 5432")
    assert output[0] is original
    assert original.annotations["flags"] == ["exact_rescue", "fallback"]
    captured[0].annotations["flags"].append("later")
    captured[0]._annotation_payload_keys += ("later",)
    assert original.annotations["flags"] == ["exact_rescue", "fallback"]
    assert "later" not in original._annotation_payload_keys


def test_deadline_flow_reranker_copy():
    from types import SimpleNamespace

    from mnemostack.recall.flow import recall_flow

    original = results()[0]
    annotate(original, "flags", ["exact_rescue"])
    recaller = Recaller(
        retrievers=[SimpleNamespace(name="vector", search=lambda *a, **kw: [original])],
        overall_deadline=5,
    )

    def rerank(query, candidates):
        candidate = candidates[0]
        assert candidate.annotations == original.annotations
        assert candidate._annotation_payload_keys == original._annotation_payload_keys
        candidate.annotations["flags"].append("rerank")
        candidate._annotation_payload_keys += ("rerank",)
        return candidates

    output = recall_flow(recaller, "port 5432", reranker=SimpleNamespace(rerank=rerank))
    assert output[0].annotations["flags"] == ["exact_rescue", "rerank"]
    assert original.annotations["flags"] == ["exact_rescue"]
    assert original._annotation_payload_keys == ("flags",)


def test_asdict_annotations_are_json_serializable():
    result = results()[0]
    annotate(result, "flags", ["exact_rescue"])
    assert json.loads(json.dumps(asdict(result)))["_annotation_payload_keys"] == ["flags"]


class SlottedAnnotations:
    __slots__ = ("payload", "annotations")

    def __init__(self):
        self.payload = {"flags": "client"}
        self.annotations = {}


class SlottedPayload:
    __slots__ = ("payload",)

    def __init__(self):
        self.payload = {"flags": "client"}


@pytest.mark.parametrize("kind", ["missing", "none", "list", "slots", "payload"])
def test_duck_typed_annotations(kind):
    from types import SimpleNamespace

    from mnemostack.recall.pipeline.annotations import get_annotation

    if kind == "slots":
        result = SlottedAnnotations()
    elif kind == "payload":
        result = SlottedPayload()
    else:
        result = SimpleNamespace(payload={"flags": "client"})
        if kind != "missing":
            result.annotations = None if kind == "none" else []
    assert get_annotation(result, "flags", []) == []
    annotate(result, "flags", ["exact_rescue"])
    annotate(result, "freshness", .5)
    annotate(result, "freshness", .7)
    assert result.payload["flags"] == "client"
    assert result.payload["freshness"] == (.5 if kind in {"slots", "payload"} else .7)
    if kind != "payload":
        assert result.annotations == {"flags": ["exact_rescue"], "freshness": .7}


def test_fusion_pools_sources_without_ownership_attribute():
    from types import SimpleNamespace

    first = SimpleNamespace(id="a", score=1., payload={}, sources=["vector"], annotations={})
    second = SimpleNamespace(id="a", score=1., payload={}, sources=["bm25"], annotations={})
    merged = reciprocal_rank_fusion([[(first, 1.)], [(second, 1.)]])[0][0]
    assert merged.sources == ["vector", "bm25"]
    assert first.sources == ["vector"]


@pytest.mark.parametrize("through_retry", [False, True])
def test_retry_merges_annotations_and_payload_ownership(monkeypatch, through_retry):
    from types import SimpleNamespace

    from mnemostack.recall.retry import _canonical, retry_weak_recall

    incumbent = results({"freshness": "client"})[0]
    annotate(incumbent, "dampened", "gravity")
    later = results()[0]
    annotate(later, "dampened", "hub")
    annotate(later, "freshness", .5)
    annotate(later, "flags", ["exact_rescue"])
    if through_retry:
        monkeypatch.setattr("mnemostack.recall.expansion.QueryExpander.generate_variants", lambda *a: ["variant"])
        monkeypatch.setattr("mnemostack.recall.retry.recall_flow", lambda *a, **kw: [later])
        merged, retried = retry_weak_recall(
            None, "query", 10, llm=SimpleNamespace(), results=[incumbent], below=2,
        )
        assert retried
        result = merged[0]
    else:
        result = _canonical({str(incumbent.id): incumbent}, later)
    assert result.annotations == {"dampened": "gravity", "freshness": .5, "flags": ["exact_rescue"]}
    assert result.payload["freshness"] == "client"
    assert "freshness" not in result._annotation_payload_keys
    annotate(result, "flags", ["updated"])
    assert result.payload["flags"] == ["updated"]
    assert result.payload["dampened"] == "gravity"


def test_retry_restore_puts_back_annotation_state(monkeypatch):
    from types import SimpleNamespace

    from mnemostack.recall.retry import retry_weak_recall

    originals = [
        RecallResult("a", "x", 0.5, {"source": "x"}, ["vector"]),
        RecallResult("b", "y", 0.4, {"source": "y"}, ["vector"]),
    ]
    big = RecallResult("c", "z" * 500, 0.9, {}, ["vector"])
    again = RecallResult("a", "x", 0.6, {"source": "x"}, ["bm25"])
    annotate(again, "flags", ["exact_rescue"])
    monkeypatch.setattr(
        "mnemostack.recall.expansion.QueryExpander.generate_variants", lambda *a: ["variant"]
    )
    monkeypatch.setattr("mnemostack.recall.retry.recall_flow", lambda *a, **kw: [big, again])
    merged, _ = retry_weak_recall(
        None, "query", 10, llm=SimpleNamespace(), results=originals, below=3,
        token_budget=20, token_counter=len,
    )
    # The budget makes the retry hand the originals back: nothing of the
    # other pass may stay on them.
    assert merged == originals
    assert originals[0].annotations == {}
    assert originals[0]._annotation_payload_keys == ()
    assert originals[0].payload == {"source": "x"}
    for result in merged:
        for key, value in result.annotations.items():
            if key in result._annotation_payload_keys:
                assert result.payload[key] == value


def test_pipeline_accepts_mock_and_attribute_refusing_results():
    from unittest.mock import Mock

    class Strict:
        def __init__(self):
            object.__setattr__(self, "id", "old")
            object.__setattr__(self, "text", "Port 5432")
            object.__setattr__(self, "score", 0.5)
            object.__setattr__(self, "payload", {})
            object.__setattr__(self, "sources", ["vector"])
            object.__setattr__(self, "floor_score", None)

        def __setattr__(self, name, value):
            if name not in ("score", "sources", "payload", "floor_score"):
                raise ValueError(f"no field {name}")
            object.__setattr__(self, name, value)

    mock = Mock(id="old", text="Port 5432", score=0.5, payload={}, sources=["vector"],
                floor_score=None)
    for result in (mock, Strict()):
        out = build_stateless_pipeline().apply("what is port 5432", [result])
        assert out == [result]
        assert "freshness" in result.payload


def test_canonical_tolerates_incumbent_without_annotations_slot():
    from mnemostack.recall.retry import _canonical

    class Slotted:
        __slots__ = ("id", "payload", "sources", "floor_score")

        def __init__(self):
            self.id, self.payload, self.sources, self.floor_score = "old", {}, ["vector"], None

    incumbent = Slotted()
    later = RecallResult("old", "t", 0.5, {}, ["bm25"])
    annotate(later, "dampened", "gravity")
    assert _canonical({"old": incumbent}, later) is incumbent
    assert incumbent.sources == ["vector", "bm25"]
    assert incumbent.payload == {"dampened": "gravity"}


def test_canonical_copies_all_annotations_when_ownership_is_refused():
    from mnemostack.recall.retry import _canonical

    class AnnotationsOnly:
        __slots__ = ("id", "payload", "sources", "floor_score", "annotations")

        def __init__(self):
            self.id, self.payload, self.sources, self.floor_score = "old", {}, ["vector"], None
            self.annotations = {}

    incumbent = AnnotationsOnly()
    later = RecallResult("old", "t", 0.5, {}, ["bm25"])
    annotate(later, "flags", ["exact_rescue"])
    annotate(later, "dampened", "gravity")
    _canonical({"old": incumbent}, later)
    assert incumbent.annotations == {"flags": ["exact_rescue"], "dampened": "gravity"}
    assert incumbent.payload == {"flags": ["exact_rescue"], "dampened": "gravity"}
