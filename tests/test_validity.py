"""Validity model: predicates, the recall-level filter, and store.invalidate."""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance

from mnemostack.recall import filter_by_validity, is_current, valid_at
from mnemostack.recall.recaller import RecallResult
from mnemostack.vector import VectorStore


def _r(id: str, payload: dict) -> RecallResult:
    return RecallResult(id=id, text=f"t{id}", score=0.5, payload=payload, sources=["v"])


# ---------- predicates ----------


def test_is_current_absent_key_is_current():
    assert is_current({}) is True
    assert is_current({"text": "x"}) is True
    assert is_current(None) is True


def test_is_current_present_key_is_stale():
    assert is_current({"invalidated_at": "2026-07-04T00:00:00Z"}) is False


def test_is_current_empty_string_marker_is_current():
    # a falsy value is treated as no marker
    assert is_current({"invalidated_at": ""}) is True


def test_valid_at_within_window():
    p = {"valid_from": "2026-01-01", "valid_until": "2026-06-01"}
    assert valid_at(p, "2026-03-01") is True


def test_valid_at_before_start_excluded():
    assert valid_at({"valid_from": "2026-05-01"}, "2026-01-01") is False


def test_valid_at_after_end_excluded():
    assert valid_at({"valid_until": "2026-06-01"}, "2026-07-01") is False


def test_valid_at_open_bounds_are_indefinite():
    assert valid_at({}, "2026-03-01") is True
    assert valid_at({"valid_from": "2020-01-01"}, "2026-03-01") is True


def test_valid_at_end_is_exclusive():
    # valid_until is the successor's start: at exactly valid_until it's no longer true
    assert valid_at({"valid_until": "2026-06-01"}, "2026-06-01") is False


# ---------- filter_by_validity ----------


def test_filter_default_drops_invalidated():
    results = [_r("a", {}), _r("b", {"invalidated_at": "2026-07-04"}), _r("c", {})]
    kept = filter_by_validity(results)
    assert [r.id for r in kept] == ["a", "c"]


def test_filter_include_invalidated_keeps_all():
    results = [_r("a", {}), _r("b", {"invalidated_at": "2026-07-04"})]
    kept = filter_by_validity(results, include_invalidated=True)
    assert [r.id for r in kept] == ["a", "b"]


def test_filter_as_of_uses_world_time_ignoring_invalidation():
    # invalidated_at present but as_of asks what was true then → still returned
    results = [
        _r("a", {"valid_from": "2026-01-01", "valid_until": "2026-06-01",
                 "invalidated_at": "2026-07-04"}),
        _r("b", {"valid_from": "2026-07-01"}),
    ]
    kept = filter_by_validity(results, as_of="2026-03-01")
    assert [r.id for r in kept] == ["a"]  # a was valid in March; b not yet


# ---------- VectorStore.invalidate ----------


@pytest.fixture
def store():
    s = VectorStore.__new__(VectorStore)
    s.collection = "test_inv"
    s.dimension = 4
    s.distance = Distance.COSINE
    s.client = QdrantClient(":memory:")
    s.ensure_collection()
    return s


def _seed(store, id, payload):
    # Qdrant point ids must be unsigned ints or UUIDs — use ints in tests.
    store.upsert(id, [0.1, 0.2, 0.3, 0.4], payload)


def _hit(store, id):
    return next(h for h in store.scroll() if str(h.id) == str(id))


def test_invalidate_sets_marker(store):
    _seed(store, 1, {"text": "fact"})
    n = store.invalidate(1, invalidated_at="2026-07-04T00:00:00Z")
    assert n == 1
    hit = _hit(store, 1)
    assert hit.payload["invalidated_at"] == "2026-07-04T00:00:00Z"
    assert hit.payload["text"] == "fact"  # merge, not overwrite


def test_invalidate_defaults_stamp_to_now(store):
    _seed(store, 1, {"text": "fact"})
    n = store.invalidate(1)
    assert n == 1
    assert _hit(store, 1).payload.get("invalidated_at")  # a stamp was written


def test_invalidate_sets_valid_until(store):
    _seed(store, 1, {"text": "fact"})
    store.invalidate(1, valid_until="2026-06-01")
    assert _hit(store, 1).payload["valid_until"] == "2026-06-01"


def test_invalidate_skips_missing_points(store):
    _seed(store, 1, {"text": "fact"})
    n = store.invalidate([1, 999])
    assert n == 1  # only the existing point counted


def test_invalidate_owner_guard_skips_foreign_root(store):
    _seed(store, 1, {"text": "a", "index_root": "/root/A"})
    _seed(store, 2, {"text": "b", "index_root": "/root/B"})
    n = store.invalidate([1, 2], index_root="/root/A")
    assert n == 1
    assert "invalidated_at" in _hit(store, 1).payload
    assert "invalidated_at" not in _hit(store, 2).payload


def test_invalidate_empty_ids_is_noop(store):
    assert store.invalidate([]) == 0


# ---------- recall_flow exclusion ----------


class _StubRecaller:
    def __init__(self, results):
        self._results = results

    def recall(self, query, limit=10, filters=None, include_invalidated=False,
               as_of=None, **_):
        from mnemostack.recall import filter_by_validity

        # A real Recaller applies the validity filter internally; mirror that
        # so recall_flow's threading is what's under test here.
        out = filter_by_validity(
            self._results, include_invalidated=include_invalidated, as_of=as_of
        )
        return out[:limit]

    def apply_vector_floor_after_rerank(self, results, recalled_results):
        return results


def test_recall_flow_hides_invalidated_by_default():
    from mnemostack.recall import recall_flow

    results = [_r("a", {}), _r("b", {"invalidated_at": "2026-07-04"}), _r("c", {})]
    out = recall_flow(_StubRecaller(results), "q", limit=10)
    assert [r.id for r in out] == ["a", "c"]


def test_recall_flow_include_invalidated_passthrough():
    from mnemostack.recall import recall_flow

    results = [_r("a", {}), _r("b", {"invalidated_at": "2026-07-04"})]
    out = recall_flow(_StubRecaller(results), "q", limit=10, include_invalidated=True)
    assert [r.id for r in out] == ["a", "b"]


def test_recall_flow_as_of_passthrough():
    from mnemostack.recall import recall_flow

    results = [
        _r("a", {"valid_until": "2026-06-01"}),
        _r("b", {"valid_from": "2026-07-01"}),
    ]
    out = recall_flow(_StubRecaller(results), "q", limit=10, as_of="2026-03-01")
    assert [r.id for r in out] == ["a"]


def test_recaller_recall_excludes_invalidated_end_to_end():
    # Real Recaller over a fake retriever: the exclusion happens inside recall().
    from mnemostack.recall.recaller import Recaller

    class _FakeRetriever:
        name = "fake"

        def search(self, query, limit, filters=None):
            return [
                RecallResult(id="a", text="a", score=0.9, payload={}, sources=["fake"]),
                RecallResult(id="b", text="b", score=0.8,
                             payload={"invalidated_at": "2026-07-04"}, sources=["fake"]),
            ]

    recaller = Recaller(retrievers=[_FakeRetriever()])
    default = recaller.recall("q", limit=10)
    assert [r.id for r in default] == ["a"]
    with_stale = recaller.recall("q", limit=10, include_invalidated=True)
    assert {r.id for r in with_stale} == {"a", "b"}


# ---------- async mirror ----------


@pytest.mark.asyncio
async def test_async_invalidate_mirrors_sync():
    from qdrant_client import AsyncQdrantClient

    from mnemostack.vector import AsyncVectorStore

    s = AsyncVectorStore.__new__(AsyncVectorStore)
    s.collection = "test_async_inv"
    s.dimension = 4
    s.distance = Distance.COSINE
    s.client = AsyncQdrantClient(":memory:")
    await s.ensure_collection()
    try:
        await s.upsert(1, [0.1, 0.2, 0.3, 0.4], {"text": "fact"})
        n = await s.invalidate(1, invalidated_at="2026-07-04T00:00:00Z")
        assert n == 1
        retrieved = await s.client.retrieve(collection_name=s.collection, ids=[1],
                                            with_payload=True)
        assert retrieved[0].payload["invalidated_at"] == "2026-07-04T00:00:00Z"
        assert retrieved[0].payload["text"] == "fact"
    finally:
        await s.close()


# ---------- review round: ISO instant comparison ----------


def test_valid_at_timezone_offset_compared_as_instant():
    # valid_from 2026-07-04T00:00+02:00 == 2026-07-03T22:00Z. A naive string
    # compare would reject as_of=2026-07-03T23:00Z (text starts with the prior
    # day); instant comparison correctly accepts it.
    p = {"valid_from": "2026-07-04T00:00:00+02:00"}
    assert valid_at(p, "2026-07-03T23:00:00Z") is True   # after real start
    assert valid_at(p, "2026-07-03T21:00:00Z") is False  # before real start


def test_valid_at_z_suffix_parsed():
    p = {"valid_until": "2026-06-01T00:00:00Z"}
    assert valid_at(p, "2026-05-01T00:00:00Z") is True
    assert valid_at(p, "2026-07-01T00:00:00Z") is False


def test_valid_at_bare_dates_fall_back_to_string_compare():
    p = {"valid_from": "2026-01-01", "valid_until": "2026-06-01"}
    assert valid_at(p, "2026-03-01") is True
    assert valid_at(p, "2026-12-01") is False


# ---------- review round: filter before top-K cut (crowd-out) ----------


def test_recall_stale_top_hit_does_not_starve_limit():
    # limit=1 with a stale top hit and a current second hit must return the
    # current one, not empty — the filter runs before fusion's top-K cut.
    from mnemostack.recall.recaller import Recaller

    class _FakeRetriever:
        name = "vector"

        def search(self, query, limit, filters=None):
            return [
                RecallResult(id="stale", text="s", score=0.9,
                             payload={"invalidated_at": "2026-07-04"}, sources=["vector"]),
                RecallResult(id="fresh", text="f", score=0.8, payload={}, sources=["vector"]),
            ]

    recaller = Recaller(retrievers=[_FakeRetriever()])
    out = recaller.recall("q", limit=1)
    assert [r.id for r in out] == ["fresh"]


def test_vector_floor_does_not_reinject_stale():
    # A stale hit with a high raw vector score would be re-appended by the
    # vector floor if it survived into the floor candidates; filtering before
    # candidate collection prevents that.
    from mnemostack.recall.recaller import Recaller

    class _FakeRetriever:
        name = "vector"

        def search(self, query, limit, filters=None):
            return [
                RecallResult(id="fresh", text="f", score=0.5,
                             payload={"raw_vector_score": 0.5}, sources=["vector"]),
                RecallResult(id="stale", text="s", score=0.99,
                             payload={"invalidated_at": "x", "raw_vector_score": 0.99},
                             sources=["vector"]),
            ]

    recaller = Recaller(retrievers=[_FakeRetriever()], vector_floor=1)
    out = recaller.recall("q", limit=1)
    assert "stale" not in {r.id for r in out}


# ---------- review round: graph as_of push-down ----------


def test_memgraph_valid_clause_current_and_as_of():
    from mnemostack.recall.retrievers import MemgraphRetriever

    current = MemgraphRetriever._valid_clause("n", None)
    assert current == "coalesce(n.valid_until, 'current') = 'current'"

    at = MemgraphRetriever._valid_clause("r", "2026-03-01")
    assert "$as_of" in at
    assert "r.valid_from" in at and "r.valid_until" in at


def test_recall_pushes_as_of_only_to_accepting_retriever():
    from mnemostack.recall.recaller import Recaller

    captured = {}

    class _GraphRetriever:
        name = "memgraph"
        accepts_as_of = True

        def search(self, query, limit, filters=None, as_of=None, include_invalidated=False):
            captured["graph_as_of"] = as_of
            captured["graph_include_invalidated"] = include_invalidated
            return []

    class _VectorRetriever:
        name = "vector"

        # no as_of param — must NOT be called with as_of (would TypeError)
        def search(self, query, limit, filters=None):
            captured["vector_called"] = True
            return []

    recaller = Recaller(retrievers=[_GraphRetriever(), _VectorRetriever()])
    recaller.recall("q", limit=5, as_of="2026-03-01")
    assert captured["graph_as_of"] == "2026-03-01"
    assert captured["vector_called"] is True


# ---------- review round 2: over-fetch so stale don't starve the window ----------


def test_over_fetch_recovers_valid_below_stale_window():
    # First 20 hits (the default per-source window) are stale; the current one
    # sits at rank 21. Over-fetching when a validity filter is active must
    # still surface it instead of returning empty.
    from mnemostack.recall.recaller import Recaller

    class _FakeRetriever:
        name = "vector"

        def __init__(self):
            self.last_limit = None

        def search(self, query, limit, filters=None):
            self.last_limit = limit
            out = []
            for i in range(min(limit, 25)):
                payload = {"invalidated_at": "x"} if i < 20 else {}
                out.append(RecallResult(id=str(i), text=f"t{i}", score=1.0 - i * 0.01,
                                        payload=payload, sources=["vector"]))
            return out

    retr = _FakeRetriever()
    recaller = Recaller(retrievers=[retr])
    out = recaller.recall("q", limit=5)
    assert retr.last_limit >= 60  # over-fetched past the 20-wide window
    assert len(out) == 5
    assert all(is_current(r.payload) for r in out)


def test_no_over_fetch_when_validity_inactive():
    from mnemostack.recall.recaller import Recaller

    class _FakeRetriever:
        name = "vector"

        def __init__(self):
            self.last_limit = None

        def search(self, query, limit, filters=None):
            self.last_limit = limit
            return []

    retr = _FakeRetriever()
    recaller = Recaller(retrievers=[retr])
    recaller.recall("q", limit=5, include_invalidated=True)
    assert retr.last_limit == 20  # per_source_limit, no over-fetch


# ---------- review round 2: pipeline as_of threading ----------


def test_pipeline_apply_threads_as_of_into_context():
    from mnemostack.recall.pipeline.base import Pipeline, Stage

    captured = {}

    class _Probe(Stage):
        def apply(self, context, results):
            captured["as_of"] = context.extras.get("as_of")
            return results

    Pipeline([_Probe()]).apply("q", [_r("a", {})], as_of="2026-03-01")
    assert captured["as_of"] == "2026-03-01"


def test_pipeline_apply_no_as_of_leaves_extras_empty():
    from mnemostack.recall.pipeline.base import Pipeline, Stage

    captured = {}

    class _Probe(Stage):
        def apply(self, context, results):
            captured["as_of"] = context.extras.get("as_of", "MISSING")
            return results

    Pipeline([_Probe()]).apply("q", [_r("a", {})])
    assert captured["as_of"] == "MISSING"


# ---------- review round 3 ----------


def test_to_utc_iso_normalizes_offsets():
    from mnemostack.recall.validity import to_utc_iso

    # offset-bearing instant -> UTC
    assert to_utc_iso("2026-07-04T00:00:00+02:00") == "2026-07-03T22:00:00+00:00"
    # Z suffix -> UTC
    assert to_utc_iso("2026-06-01T00:00:00Z") == "2026-06-01T00:00:00+00:00"
    # bare date and marker pass through unchanged
    assert to_utc_iso("2024-01-15") == "2024-01-15"
    assert to_utc_iso("current") == "current"
    assert to_utc_iso(None) is None


def test_graph_valid_clause_include_invalidated_is_permissive():
    from mnemostack.recall.retrievers import graph_valid_clause

    # include_invalidated + no as_of -> no filter
    assert graph_valid_clause("n", None, include_invalidated=True) == "true"
    # default -> current-only
    assert "= 'current'" in graph_valid_clause("n", None)
    # as_of wins over include_invalidated
    assert "$as_of" in graph_valid_clause("n", "2026-03-01", include_invalidated=True)


def test_recall_pushes_include_invalidated_to_graph():
    from mnemostack.recall.recaller import Recaller

    captured = {}

    class _GraphRetriever:
        name = "memgraph"
        accepts_as_of = True

        def search(self, query, limit, filters=None, as_of=None, include_invalidated=False):
            captured["include_invalidated"] = include_invalidated
            return []

    recaller = Recaller(retrievers=[_GraphRetriever()])
    recaller.recall("q", limit=5, include_invalidated=True)
    assert captured["include_invalidated"] is True


def test_mca_hits_filtered_by_validity_before_fusion():
    # A stale MCA exact-token hit must not survive to fusion and win limit=1.
    from mnemostack.recall.recaller import Recaller

    class _FakeRetriever:
        name = "vector"

        def search(self, query, limit, filters=None):
            return [RecallResult(id="fresh", text="f", score=0.5, payload={},
                                 sources=["vector"])]

    recaller = Recaller(retrievers=[_FakeRetriever()], mca_prefilter=True)
    recaller._mca_hits = lambda q, limit, filters: [
        RecallResult(id="stale", text="s", score=0.99,
                     payload={"invalidated_at": "2026-07-04"}, sources=["mca"])
    ]
    out = recaller.recall("q", limit=1)
    assert "stale" not in {r.id for r in out}


def test_graph_store_to_iso_normalizes_to_utc():
    from mnemostack.graph.store import _to_iso

    assert _to_iso("2026-07-04T00:00:00+02:00") == "2026-07-03T22:00:00+00:00"
    assert _to_iso("2024-01-15") == "2024-01-15"  # bare date unchanged
    assert _to_iso("current") == "current"
