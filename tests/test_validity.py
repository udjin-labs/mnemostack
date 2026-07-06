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
    assert to_utc_iso("2026-07-04T00:00:00+02:00") == "2026-07-03T22:00:00Z"
    # Z suffix -> UTC
    assert to_utc_iso("2026-06-01T00:00:00Z") == "2026-06-01T00:00:00Z"
    # a bare calendar date stays a date (not expanded to an instant), keeping
    # date-only graph bounds in their long-standing form; marker/None unchanged
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


def test_graph_as_of_predicate_parses_instants_and_guards_markers():
    # #87: point-in-time bounds are compared as PARSED instants, so mixed
    # sub-second precision orders correctly (a raw string compare misreads
    # `…00.5Z` as before `…00Z`). Structure is asserted here; the datetime()
    # semantics (incl. bare-date append and marker guards) were verified against
    # a live Memgraph — datetime(NULL) and datetime('current') both raise there,
    # so the markers must be guarded by CASE *before* any datetime() call.
    from mnemostack.recall.retrievers import graph_valid_clause
    from mnemostack.recall.validity import graph_as_of_predicate

    p = graph_as_of_predicate("r")
    assert "datetime(" in p and "datetime($as_of)" in p  # parsed, not raw-string
    # a bare-date bound gets midnight-Z appended (Memgraph rejects a date with
    # no timezone), an instant (already has 'T') is parsed as-is
    assert "CONTAINS 'T'" in p and "+ 'T00:00:00Z'" in p
    # markers guarded by CASE before datetime() is ever evaluated
    assert "CASE WHEN r.valid_from IS NULL THEN true" in p
    assert "r.valid_until = 'current' OR r.valid_until IS NULL THEN true" in p
    # a non-canonical bound (unvalidated LLM free text like "early 2024") is
    # regex-vetted and falls back to the old raw-string compare, so it never
    # reaches datetime() (which would raise and abort the WHOLE query)
    assert "=~ '" in p
    assert "ELSE r.valid_from <= $as_of END" in p
    assert "ELSE r.valid_until > $as_of END" in p
    # the shared predicate is what graph_valid_clause emits for an as_of query
    assert graph_valid_clause("r", "2026-03-01") == p


def test_graph_ts_regex_accepts_canonical_rejects_garbage():
    # The vetting regex decides which bounds reach Cypher datetime(). Python
    # re.fullmatch mirrors Cypher `=~` (full-string match) for this subset, so
    # this is a re-runnable check that impossible dates / junk fall back instead
    # of aborting the query. (Verified equivalent against a live Memgraph.)
    import re

    from mnemostack.recall.validity import _GRAPH_TS_RE

    rx = re.compile(_GRAPH_TS_RE)
    for good in (
        "2024-01-01",
        "2024-12-31",
        "2024-02-28",
        "2026-03-01T00:00:00Z",
        "2026-03-01T00:00:00.500Z",  # 3 frac digits (ms) — Memgraph accepts
        "2026-03-01T00:00:00.500000Z",  # 6 frac digits (µs) — Memgraph accepts
        "2026-03-01T00:00:00+02:00",
    ):
        assert rx.fullmatch(good), good
    for bad in (
        "early 2024",
        "recently",
        "current",
        "2024-13-01",  # month 13
        "2024-02-31",  # Feb has no 31st
        "2024-02-29",  # Feb capped at 28 (regex can't check leap) -> raw-string fallback
        "2023-02-29",  # non-leap Feb 29 (would abort datetime) -> fallback
        "2024-04-31",  # April has no 31st
        "2024-01-01TBD",  # junk time suffix
        "2024-01-01T99:99:99Z",  # impossible time
        "2026-03-01T00:00:00.5Z",  # 1 frac digit — Memgraph datetime() raises
        "2026-03-01T00:00:00.123456789Z",  # 9 frac digits — Memgraph raises
    ):
        assert not rx.fullmatch(bad), bad


def test_recall_pushes_include_invalidated_to_graph():
    from mnemostack.recall.recaller import Recaller

    captured = {}

    class _GraphRetriever:
        name = "memgraph"
        accepts_as_of = True
        accepts_include_invalidated = True

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

    assert _to_iso("2026-07-04T00:00:00+02:00") == "2026-07-03T22:00:00Z"
    assert _to_iso("2024-01-15") == "2024-01-15"  # bare date stays a date
    assert _to_iso("current") == "current"


# ---------- review round 4: over-fetch trims back to the window ----------


def test_over_fetch_trims_back_to_per_source_limit_on_clean_index():
    # A clean index (nothing invalidated) must keep exactly the original
    # per-source window, not the wider over-fetch — no ranking drift.
    from mnemostack.recall.recaller import Recaller

    class _FakeRetriever:
        name = "vector"

        def __init__(self):
            self.last_limit = None

        def search(self, query, limit, filters=None):
            self.last_limit = limit
            # return `limit` current hits (nothing stale)
            return [
                RecallResult(id=str(i), text=f"t{i}", score=1.0 - i * 0.001,
                             payload={}, sources=["vector"])
                for i in range(limit)
            ]

    retr = _FakeRetriever()
    recaller = Recaller(retrievers=[retr])
    # recall with a large limit so fusion doesn't cut below the window
    out = recaller.recall("q", limit=100, vector_limit=20)
    assert retr.last_limit >= 60          # over-fetched
    assert len(out) == 20                 # but trimmed back to the 20-wide window


def test_graph_valid_clause_used_for_target_node():
    # The rel query must filter the target node m, not just n and r.
    import inspect

    from mnemostack.recall.retrievers import MemgraphRetriever

    src = inspect.getsource(MemgraphRetriever.search)
    assert 'target_valid = self._valid_clause("m"' in src
    assert "AND {target_valid}" in src


# ---------- review round 5 ----------


def test_to_utc_iso_emits_z_for_cypher_string_compat():
    from mnemostack.recall.validity import to_utc_iso

    # Z form (not +00:00) so graph Cypher string comparison stays compatible
    # with existing UTC rows written with a Z suffix.
    out = to_utc_iso("2026-06-01T00:00:00+02:00")
    assert out.endswith("Z")
    assert "+00:00" not in out
    # exact-boundary lexical order holds against a bound written by the same
    # normalizer (the canonical fixed-microsecond Z form on both sides).
    stored = to_utc_iso("2026-05-31T22:00:00Z")
    assert stored <= out and out <= stored  # equal -> boundary compares correctly


def test_include_invalidated_only_sent_to_advertising_retriever():
    # A custom retriever that accepts only as_of (not include_invalidated) must
    # not be handed include_invalidated (which the broad except would swallow,
    # silently dropping the retriever). It also must still receive as_of.
    from mnemostack.recall.recaller import Recaller

    captured = {}

    class _AsOfOnlyRetriever:
        name = "custom"
        accepts_as_of = True  # but NOT accepts_include_invalidated

        def search(self, query, limit, filters=None, as_of=None):
            captured["as_of"] = as_of
            captured["called"] = True
            return [RecallResult(id="x", text="t", score=0.5, payload={}, sources=["custom"])]

    recaller = Recaller(retrievers=[_AsOfOnlyRetriever()])
    out = recaller.recall("q", limit=5, as_of="2026-03-01")
    assert captured["called"] is True          # not silently dropped
    assert captured["as_of"] == "2026-03-01"
    assert [r.id for r in out] == ["x"]


def test_neither_marker_retriever_gets_no_validity_kwargs():
    from mnemostack.recall.recaller import Recaller

    class _PlainRetriever:
        name = "plain"

        # no markers, no as_of/include_invalidated params
        def search(self, query, limit, filters=None):
            return [RecallResult(id="p", text="t", score=0.5, payload={}, sources=["plain"])]

    recaller = Recaller(retrievers=[_PlainRetriever()])
    out = recaller.recall("q", limit=5, as_of="2026-03-01", include_invalidated=True)
    assert [r.id for r in out] == ["p"]  # works, no TypeError


# ---------- review round 6 ----------


def test_to_utc_instant_expands_bare_date():
    from mnemostack.recall.validity import to_utc_instant

    # bare date -> full midnight-UTC instant with Z (so it compares against
    # full-instant graph bounds, not shorter than valid_from='...T00:00:00Z')
    assert to_utc_instant("2026-03-01") == "2026-03-01T00:00:00Z"
    assert to_utc_instant("2026-03-01T00:00:00+02:00") == "2026-02-28T22:00:00Z"
    assert to_utc_instant("current") == "current"
    assert to_utc_instant(None) is None


def test_bare_date_as_of_matches_full_instant_bound_lexically():
    # The exact-boundary case the fix targets: a fact starting at midnight is
    # valid at a bare-date as_of once as_of is expanded to the full instant.
    from mnemostack.recall.validity import to_utc_instant, to_utc_iso

    as_of = to_utc_instant("2026-03-01")               # 2026-03-01T00:00:00Z
    valid_from = to_utc_iso("2026-03-01T00:00:00Z")    # stored bound, same canonical form
    assert valid_from <= as_of                         # was False without expansion


def test_search_many_filters_per_vector_before_fusion():
    from mnemostack.recall.recaller import Recaller
    from mnemostack.vector.qdrant import Hit

    class _FakeVector:
        def search(self, vector, limit, filters=None, *, hide_invalidated=False):
            # ignores the push-down flag on purpose: the client-side per-vector
            # filter must still drop the stale hit (backstop), so the test holds.
            return [
                Hit(id="stale", score=0.99, payload={"invalidated_at": "2026-07-04", "text": "s"}),
                Hit(id="fresh", score=0.5, payload={"text": "f"}),
            ][:limit]

    recaller = Recaller.__new__(Recaller)
    recaller.vector = _FakeVector()
    recaller.rrf_k = 60
    out = recaller.search_many([[0.1], [0.2]], limit=5)
    assert {r.id for r in out} == {"fresh"}  # stale filtered per vector


# ---------- review round 7 ----------


def test_graph_bare_node_skipped_when_no_valid_edge():
    # Under a validity view (default current-only, or as_of), a node whose
    # rel query returns nothing must not surface as a bare entity.
    from mnemostack.recall.retrievers import MemgraphRetriever

    class _FakeSession:
        def run(self, cypher, **params):
            class _R:
                def data(self_inner):
                    if "labels(n)[0]" in cypher:      # node probe
                        return [{"name": "Alice", "type": "Person", "mc": ""}]
                    return []                          # rel query: no valid edges
            return _R()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _FakeDriver:
        # No-arg session() — an injected driver from before database= existed;
        # must still work when no non-default database is configured.
        def session(self):
            return _FakeSession()

    r = MemgraphRetriever.__new__(MemgraphRetriever)
    r.database = None
    r.min_word = 3
    r.contains_min = 5
    r.max_nodes = 10
    r.max_rels = 5
    r._driver = _FakeDriver()
    r._own_driver = False

    # default (current-only) view: bare node with no valid edge -> dropped
    assert r.search("alice person") == []
    # include_invalidated=True (no as_of): bare node allowed (legacy behavior)
    out = r.search("alice person", include_invalidated=True)
    assert any(res.id == "graph:Alice" for res in out)


def test_fallback_filters_validity_before_truncate():
    from mnemostack.recall.recaller import Recaller

    recaller = Recaller.__new__(Recaller)
    recaller.embedding = None
    recaller.vector = None

    # stub the fallback hit source: stale hit first, valid one below
    stale = RecallResult(id="stale", text="s", score=0.9,
                         payload={"invalidated_at": "2026-07-04"}, sources=["vector"])
    fresh = RecallResult(id="fresh", text="f", score=0.5, payload={}, sources=["vector"])
    recaller._vector_fallback_hits = (
        lambda query, limit, filters, hide_invalidated=False, tenant=None: [stale, fresh]
    )

    out = recaller._maybe_apply_fallback(
        "q", [], limit=1, vector_limit=1, filters=None
    )
    assert "stale" not in {r.id for r in out}
    assert "fresh" in {r.id for r in out}


def test_graph_rel_probe_is_undirected_and_overfetches_nodes():
    # #3: undirected rel match so target-only nodes (valid incoming edge, no
    # outgoing) aren't wrongly dropped. #2: node candidates over-fetched
    # before the bare-node skip so stale-only early nodes don't hide valid ones.
    import inspect

    from mnemostack.recall.retrievers import MemgraphRetriever

    src = inspect.getsource(MemgraphRetriever.search)
    assert "-[r]-(m)" in src and "-[r]->(m)" not in src   # undirected
    assert "startNode(r).name" in src and "endNode(r).name" in src
    assert "self.max_nodes * 3" in src                     # node over-fetch


def test_to_utc_iso_normalizes_non_T_separators():
    import sys

    from mnemostack.recall.validity import to_utc_iso

    # Every separator datetime.fromisoformat accepts marks a real datetime that
    # is canonicalized to one UTC form, or the graph string comparison
    # misclassifies exact boundaries.
    assert to_utc_iso("2026-07-04 00:00:00+02:00") == "2026-07-03T22:00:00Z"
    # an arbitrary separator (datetime.isoformat(sep="_")) is recognized too
    assert to_utc_iso("2024-01-15_10:00:00") == "2024-01-15T10:00:00Z"
    # a bare calendar date stays a date (see the dedicated bare-date tests)
    assert to_utc_iso("2024-01-15") == "2024-01-15"
    # fromisoformat parses date+separator forms as naive datetimes (the "+" from
    # isoformat(sep="+") is a separator, not an offset) — all canonicalized to
    # the same UTC form, so a value can't be preserved in one shape and compared
    # against another. This is the resolution of the ambiguous +HH:MM:SS forms.
    assert to_utc_iso("2024-01-15Z") == "2024-01-15T00:00:00Z"
    assert to_utc_iso("2024-01-15+02:00") == "2024-01-15T02:00:00Z"
    assert to_utc_iso("2024-01-15+02") == "2024-01-15T02:00:00Z"
    assert to_utc_iso("2024-01-15+10:20:00") == "2024-01-15T10:20:00Z"
    # comma-decimal fractions are 3.11+ in fromisoformat; on 3.10 the value is
    # unparseable and passes through untouched (consistent on both sides).
    if sys.version_info >= (3, 11):
        assert to_utc_iso("2024-01-15+02:30:15,5") == "2024-01-15T02:30:15.500000Z"
    else:
        assert to_utc_iso("2024-01-15+02:30:15,5") == "2024-01-15+02:30:15,5"
    # non-instant marker passes through
    assert to_utc_iso("current") == "current"


def test_to_utc_iso_preserves_precision_for_existing_bound_compat():
    from mnemostack.recall.validity import to_utc_instant, to_utc_iso

    # Precision is preserved (NOT widened to microseconds), so a whole-second
    # bound written by the previous normalizer still compares equal to a
    # normalized as_of at the exact boundary — widening would make '...00Z' and
    # '...00.000000Z' (the same instant) sort unequal and drop the fact.
    stored = "2026-03-01T00:00:00Z"          # existing graph bound
    as_of = to_utc_instant("2026-03-01")
    assert as_of == "2026-03-01T00:00:00Z"
    assert stored <= as_of and as_of <= stored   # equal -> boundary holds
    # a fractional bound keeps its precision (sub-second lexical ordering across
    # mixed precision is a known limitation of the raw-string graph compare)
    assert to_utc_iso("2026-01-01t00:00:00.500000+00:00") == "2026-01-01T00:00:00.500000Z"


def test_to_utc_iso_normalizes_basic_and_week_datetimes():
    import sys

    from mnemostack.recall.validity import to_utc_iso

    # datetime.fromisoformat only parses basic-format and ISO-week datetimes on
    # 3.11+. Where it parses, these carry a real time-of-day + offset and MUST
    # be UTC-normalized too. On 3.10 the value is unparseable, so to_utc_iso
    # safely leaves it untouched rather than inventing a wrong instant.
    if sys.version_info >= (3, 11):
        assert to_utc_iso("20260704T000000+0200") == "2026-07-03T22:00:00Z"
        assert to_utc_iso("2026-W27-6T00:00:00+02:00") == "2026-07-03T22:00:00Z"
        # separatorless basic datetime (no T/space) is still recognized
        assert to_utc_iso("20260704000000+0200") == "2026-07-03T22:00:00Z"
        # a basic-format bare date is canonicalized to the extended date form
        # (still a date, not an instant), so it compares against extended bounds
        assert to_utc_iso("20260704") == "2026-07-04"
    else:
        # 3.10's fromisoformat can't parse these, so they pass through untouched
        assert to_utc_iso("20260704T000000+0200") == "20260704T000000+0200"
        assert to_utc_iso("2026-W27-6T00:00:00+02:00") == "2026-W27-6T00:00:00+02:00"
        assert to_utc_iso("20260704000000+0200") == "20260704000000+0200"
        assert to_utc_iso("20260704") == "20260704"


def test_to_utc_iso_normalizes_naive_datetime_with_separator():
    from mnemostack.recall.validity import to_utc_iso

    # A naive datetime (no offset) is assumed UTC and emitted in the Z form so
    # it sorts against normalized bounds. A bare date stays a date.
    assert to_utc_iso("2024-01-15T10:00:00") == "2024-01-15T10:00:00Z"
    assert to_utc_iso("2024-01-15") == "2024-01-15"


def test_filter_by_tenant_scrubs_foreign_vector_floor_candidates():
    from mnemostack.recall import filter_by_tenant

    r = _r(
        "1",
        {
            "tenant_id": "alpha",
            # Raw vector-floor candidates are re-materialized AFTER the tenant
            # backstop, so a foreign one hidden here must be scrubbed.
            "_vector_floor_candidates": [
                {"id": "2", "payload": {"tenant_id": "alpha"}},
                {"id": "3", "payload": {"tenant_id": "beta"}},
            ],
        },
    )
    out = filter_by_tenant([r], "alpha")
    assert len(out) == 1
    assert [c["id"] for c in out[0].payload["_vector_floor_candidates"]] == ["2"]


def test_recall_trace_restrict_to_ids_scrubs_foreign():
    from mnemostack.recall import RecallTrace
    from mnemostack.recall.trace import RetrieverTrace

    tr = RecallTrace(
        retrievers=[RetrieverTrace(name="vector", ranked=[("1", 0.9), ("2", 0.8)])],
        fused=[("1", 0.9), ("2", 0.8)],
        post_rerank=[("2", 0.8), ("1", 0.9)],
    )
    tr.restrict_to_ids(["1"])
    assert tr.retrievers[0].ranked == [("1", 0.9)]
    assert tr.fused == [("1", 0.9)]
    assert tr.post_rerank == [("1", 0.9)]


def test_vector_fallback_hits_scopes_to_tenant():
    # A tenant-scoped fallback must query the store with tenant= (scoped at
    # source), not fetch the whole shared collection and rely on the backstop.
    from mnemostack.recall.recaller import Recaller

    seen = {}

    class _Hit:
        def __init__(self, id, score, payload):
            self.id, self.score, self.payload = id, score, payload

    class _FakeEmbed:
        def embed(self, q):
            return [1.0, 0.0, 0.0, 0.0]

    class _FakeVector:
        def search(self, vec, limit, filters=None, hide_invalidated=False, tenant=None, **kw):
            seen["tenant"] = tenant
            return [_Hit("x", 0.9, {"tenant_id": tenant, "text": "t"})]

    r = Recaller.__new__(Recaller)
    r.embedding = _FakeEmbed()
    r.vector = _FakeVector()
    r.retrievers = []
    out = r._vector_fallback_hits("q", limit=5, filters=None, tenant="alpha")
    assert seen["tenant"] == "alpha"
    assert out and out[0].payload["tenant_id"] == "alpha"
