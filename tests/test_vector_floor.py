from mnemostack.llm.base import LLMResponse
from mnemostack.recall import BM25Doc, Recaller, RecallResult
from mnemostack.vector.qdrant import Hit


class FakeEmbedding:
    def embed(self, query):
        return [1.0]


class FakeVectorStore:
    def __init__(self, hits):
        self.hits = hits

    def search(self, vector, limit=10, filters=None, **_):
        return self.hits[:limit]


class FakeExpansionLLM:
    def generate(self, prompt, max_tokens=120, temperature=0.0):
        return LLMResponse(text="lexical variant\nsemantic variant")


def _bm25_docs():
    return [
        BM25Doc(id=f"b{i}", text="lexical match", payload={"text": f"lexical {i}"})
        for i in range(1, 6)
    ]


def _vector_hits():
    return [
        Hit("v-anchor", 0.95, {"text": "top vector match"}),
        Hit("v-buried", 0.721, {"text": "strong vector-only match"}),
        Hit("v-extra", 0.60, {"text": "lower vector match"}),
    ]


def _many_vector_hits():
    return [
        Hit("v-anchor", 0.95, {"text": "top vector match"}),
        Hit("v-second", 0.90, {"text": "second vector match"}),
        Hit("v-third", 0.85, {"text": "third vector match"}),
        Hit("v-tail", 0.80, {"text": "tail vector match"}),
    ]


def test_vector_floor_surfaces_buried_vector_strong_result():
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore(_vector_hits()),
        bm25_docs=_bm25_docs(),
        vector_floor=2,
    )

    results = recaller.recall("lexical", limit=2, vector_limit=3)

    ids = [result.id for result in results]
    assert ids[:2] == ["v-anchor", "b1"]
    assert "v-buried" in ids
    assert ids.index("v-buried") >= 2


def test_vector_floor_default_is_noop():
    kwargs = {
        "embedding_provider": FakeEmbedding(),
        "vector_store": FakeVectorStore(_vector_hits()),
        "bm25_docs": _bm25_docs(),
    }
    baseline = Recaller(**kwargs)
    disabled = Recaller(**kwargs, vector_floor=0)

    baseline_results = baseline.recall("lexical", limit=2, vector_limit=3)
    disabled_results = disabled.recall("lexical", limit=2, vector_limit=3)

    assert [result.id for result in disabled_results] == [result.id for result in baseline_results]
    assert len(disabled_results) == len(baseline_results)


def test_vector_floor_deduplicates_existing_results():
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore(_vector_hits()),
        bm25_docs=_bm25_docs(),
        vector_floor=2,
    )

    results = recaller.recall("lexical", limit=2, vector_limit=3)

    ids = [result.id for result in results]
    assert ids.count("v-anchor") == 1
    assert ids.count("v-buried") == 1
    assert len(ids) == len(set(ids))


def test_vector_floor_appended_scores_stay_below_fused_results():
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore(_vector_hits()),
        bm25_docs=_bm25_docs(),
        vector_floor=2,
    )

    results = recaller.recall("lexical", limit=2, vector_limit=3)
    by_id = {result.id: result for result in results}

    assert by_id["v-buried"].payload["raw_vector_score"] == 0.721
    assert by_id["v-buried"].score < min(result.score for result in results[:2])
    assert sorted(results, key=lambda result: result.score, reverse=True)[:2] == results[:2]


def test_vector_floor_with_no_vector_hits_does_not_add_results():
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore([]),
        bm25_docs=_bm25_docs(),
        vector_floor=5,
    )

    results = recaller.recall("lexical", limit=2, vector_limit=3)

    assert [result.id for result in results] == ["b1", "b2"]


def test_vector_floor_applies_once_after_query_expansion_fusion():
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore(_vector_hits()),
        bm25_docs=_bm25_docs(),
        query_expansion=True,
        expansion_llm=FakeExpansionLLM(),
        vector_floor=2,
    )
    calls = []
    original_apply_vector_floor = recaller._apply_vector_floor

    def spy_apply_vector_floor(results, vector_candidates):
        calls.append([result.id for result in results])
        return original_apply_vector_floor(results, vector_candidates)

    recaller._apply_vector_floor = spy_apply_vector_floor

    recaller.recall("lexical", limit=2, vector_limit=3)

    assert len(calls) == 1


def test_vector_floor_query_expansion_uses_raw_vector_candidates():
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore(_many_vector_hits()),
        bm25_docs=_bm25_docs(),
        query_expansion=True,
        expansion_llm=FakeExpansionLLM(),
        vector_floor=4,
    )

    results = recaller.recall("lexical", limit=2, vector_limit=4, bm25_limit=4)

    ids = [result.id for result in results]
    assert "v-tail" in ids
    assert len(ids) == len(set(ids))


class FixedRetriever:
    def __init__(self, name, results):
        self.name = name
        self.results = results

    def search(self, query, limit=20, filters=None):
        return self.results[:limit]


def test_vector_floor_works_with_retriever_mode():
    vector_results = [
        RecallResult(
            id="v-anchor",
            text="top vector match",
            score=0.95,
            payload={"raw_vector_score": 0.95},
            sources=["vector"],
        ),
        RecallResult(
            id="v-buried",
            text="strong vector-only match",
            score=0.721,
            payload={"raw_vector_score": 0.721},
            sources=["vector"],
        ),
    ]
    bm25_results = [
        RecallResult(id=f"b{i}", text="lexical match", score=1.0, sources=["bm25"])
        for i in range(1, 4)
    ]
    recaller = Recaller(
        retrievers=[
            FixedRetriever("vector", vector_results),
            FixedRetriever("bm25", bm25_results),
        ],
        vector_floor=2,
    )

    results = recaller.recall("lexical", limit=2, vector_limit=3)

    ids = [result.id for result in results]
    assert "v-buried" in ids
    assert len(ids) == len(set(ids))


def test_vector_floor_retriever_mode_preserves_raw_score_before_rrf_overwrite():
    vector_results = [
        RecallResult(
            id="v-anchor",
            text="top vector match",
            score=0.95,
            sources=["vector"],
        ),
        RecallResult(
            id="v-buried",
            text="lower vector match",
            score=0.721,
            sources=["vector"],
        ),
    ]
    bm25_results = [
        RecallResult(id="b1", text="lexical match", score=1.0, sources=["bm25"]),
    ]
    recaller = Recaller(
        retrievers=[
            FixedRetriever("vector", vector_results),
            FixedRetriever("bm25", bm25_results),
        ],
        vector_floor=1,
    )

    results = recaller.recall("lexical", limit=2, vector_limit=3)

    ids = [result.id for result in results]
    assert ids == ["v-anchor", "b1"]
    assert results[0].payload["raw_vector_score"] == 0.95


def test_vector_floor_applies_after_rerank_and_top_k_slice():
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore(_vector_hits()),
        bm25_docs=_bm25_docs(),
        vector_floor=2,
    )

    recalled = recaller.recall("lexical", limit=6, vector_limit=3)
    reranked_and_sliced = [
        result for result in recalled if result.id not in {"v-anchor", "v-buried"}
    ][:2]

    final = recaller.apply_vector_floor_after_rerank(reranked_and_sliced, recalled)

    ids = [result.id for result in final]
    assert ids[:2] == [result.id for result in reranked_and_sliced]
    assert "v-anchor" in ids
    assert "v-buried" in ids
    assert len(ids) > 2


def test_vector_floor_after_rerank_default_is_noop():
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore(_vector_hits()),
        bm25_docs=_bm25_docs(),
        vector_floor=0,
    )

    recalled = recaller.recall("lexical", limit=6, vector_limit=3)
    sliced = recalled[:2]

    assert recaller.apply_vector_floor_after_rerank(sliced, recalled) is sliced
    assert [result.id for result in sliced] == [result.id for result in recalled[:2]]


def test_vector_floor_after_rerank_deduplicates_ids():
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore(_vector_hits()),
        bm25_docs=_bm25_docs(),
        vector_floor=2,
    )

    recalled = recaller.recall("lexical", limit=6, vector_limit=3)
    reranked_and_sliced = [recalled[0], recalled[0], recalled[1]]

    final = recaller.apply_vector_floor_after_rerank(reranked_and_sliced, recalled)

    ids = [result.id for result in final]
    assert ids.count("v-anchor") == 1
    assert ids.count("v-buried") == 1
    assert len(ids) == len(set(ids))


def test_vector_floor_after_rerank_extends_without_dropping_winners():
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore(_many_vector_hits()),
        bm25_docs=_bm25_docs(),
        vector_floor=4,
    )

    recalled = recaller.recall("lexical", limit=8, vector_limit=4)
    reranked_winners = [
        result
        for result in recalled
        if result.id not in {"v-anchor", "v-second", "v-third", "v-tail"}
    ][:2]

    final = recaller.apply_vector_floor_after_rerank(reranked_winners, recalled)

    ids = [result.id for result in final]
    assert ids[:2] == [result.id for result in reranked_winners]
    assert {"v-anchor", "v-second", "v-third", "v-tail"} <= set(ids)
    assert len(ids) == len(reranked_winners) + 4


def test_vector_floor_metadata_replaces_stale_partial_candidates():
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore(_many_vector_hits()),
        bm25_docs=_bm25_docs(),
        vector_floor=4,
    )
    result = RecallResult(
        id="winner",
        text="winner",
        score=1.0,
        payload={
            "_vector_floor_candidates": [
                {
                    "id": "v-anchor",
                    "text": "stale partial candidate",
                    "score": 0.95,
                    "payload": {"raw_vector_score": 0.95},
                    "sources": ["vector"],
                }
            ]
        },
        sources=["bm25"],
    )
    candidates = [
        RecallResult(
            id="v-anchor",
            text="top vector match",
            score=0.95,
            payload={"raw_vector_score": 0.95},
            sources=["vector"],
        ),
        RecallResult(
            id="v-tail",
            text="tail vector match",
            score=0.80,
            payload={"raw_vector_score": 0.80},
            sources=["vector"],
        ),
    ]

    recaller._attach_vector_floor_candidates([result], candidates)
    final = recaller.apply_vector_floor_after_rerank([result], [result])

    ids = [item.id for item in final]
    assert "v-tail" in ids


def _floor_recaller(n=1):
    """A recaller exposing the REAL floor logic and nothing else."""

    class _R:
        vector_floor = n
        _apply_vector_floor = Recaller._apply_vector_floor
        _vector_floor_candidates_from_results = staticmethod(
            Recaller._vector_floor_candidates_from_results
        )
        apply_vector_floor_after_rerank = Recaller.apply_vector_floor_after_rerank

    return _R()


def _hit(pid, score=0.9, raw=None, text="a memory"):
    payload = {"text": text}
    if raw is not None:
        payload["raw_vector_score"] = raw
    return RecallResult(id=pid, text=text, score=score, payload=payload, sources=["vector"])


def test_the_floor_does_not_seat_one_memory_twice(tmp_path=None):
    """#168: the floor's dedup against the page compared RAW ids, so a
    guaranteed candidate carrying `"1"` was invisible to it when the page
    already held `1` — it appended the same memory a second time, and the
    caller's page showed one memory twice. A point id is `str | int` in
    Qdrant, and one memory's representation can differ between the arm
    that ranked it and the arm that offered it as a floor candidate."""
    ranked = _hit(1, score=0.8)
    candidate = _hit("1", score=0.4, raw=0.99)  # same memory, other type

    out = _floor_recaller()._apply_vector_floor([ranked], [candidate])

    assert [str(r.id) for r in out] == ["1"], [r.id for r in out]
    assert out[0] is ranked  # the page keeps the object it already had


def test_duplicates_of_one_memory_do_not_crowd_the_floors_slots():
    """The pool's own keying, isolated. "It keeps the strongest view" reads
    like the property to test here, but it is not observable on its own:
    the dedup on the way out drops the second view whichever one it is, so
    a test asserting it passes with the pool keyed raw. What raw keying
    really costs is a SLOT — it spends the
    floor's slots on one memory twice, so a DIFFERENT memory that the
    guarantee was meant to seat never gets considered at all. That is the
    consequence the dedup on the way out cannot undo: by then the other
    candidate has already been sliced away."""
    view_a = _hit("3", score=0.5, raw=0.90, text="memory A")
    view_a_again = _hit(3, score=0.5, raw=0.85, text="memory A, other id type")
    memory_b = _hit("4", score=0.5, raw=0.80, text="memory B")

    out = _floor_recaller(n=2)._apply_vector_floor(
        [_hit("keeper")], [view_a, view_a_again, memory_b]
    )

    seated = sorted(str(r.id) for r in out if str(r.id) != "keeper")
    assert seated == ["3", "4"], [r.id for r in out]


def test_the_page_check_holds_whichever_way_the_types_run():
    """The dedup against the page has to normalise BOTH sides. Comparing a
    raw candidate id against normalised keys only misses when the types run
    this way round — an int candidate against a page that holds the string
    — which is exactly the case a test using the other order cannot see."""
    ranked = _hit("5", score=0.8)
    candidate = _hit(5, score=0.4, raw=0.99)  # same memory, int this time

    out = _floor_recaller()._apply_vector_floor([ranked], [candidate])

    assert [str(r.id) for r in out] == ["5"], [r.id for r in out]


def test_a_rescored_extra_stops_claiming_the_floor_put_it_there():
    """#169: the claim expires by itself. A flag would have to be cleared
    by whoever re-ranked the page, and nothing on the recall path can tell
    "a stage promoted this" from "nothing touched it" — a pipeline can be
    all non-scoring stages, and `apply_rerank_safe` is fail-open, so a
    reranker that raised leaves one configured and nothing reranked. Both
    proxies were tried and both were wrong. A score needs no clearing:
    rescoring the result IS the clearing."""
    from mnemostack.recall.recaller import is_floor_extra

    out = _floor_recaller()._apply_vector_floor([_hit("ranked", score=0.8)], [_hit("F", raw=0.9)])
    appended = out[-1]
    assert str(appended.id) == "F"
    assert is_floor_extra(appended)  # straight out of the floor

    appended.score = 0.42  # any stage that rescores it, by any route
    assert not is_floor_extra(appended)


def test_a_floor_stamp_from_an_earlier_page_does_not_carry_over():
    """The other half: the claim is about THIS page. An object the floor
    appended once, then a later ranking placed on its own merits, must not
    keep saying a floor put it there — and it does not, because that
    ranking gave it a score of its own."""
    from mnemostack.recall.recaller import is_floor_extra

    earlier = _floor_recaller()._apply_vector_floor([_hit("x", score=0.8)], [_hit("P", raw=0.9)])
    promoted = earlier[-1]
    assert is_floor_extra(promoted)

    promoted.score = 0.95  # a pipeline stage ranks it top on the next pass
    again = _floor_recaller()._apply_vector_floor([promoted], [_hit("Q", raw=0.7)])

    assert not is_floor_extra(again[0]), "a stale stamp survived a re-ranking"
    assert str(again[0].id) == "P"


def test_the_arms_merge_one_memory_into_one_result(tmp_path=None):
    """#172: the recaller's own merge dictionaries keyed the RAW id, and
    the lists they feed the fusion carry bare ids — so a memory the vector
    arm calls `7` and the lexical arm calls `"7"` was merged as two, took
    two of the caller's slots, and pushed a genuinely different memory off
    the page. The arms also stopped pooling their `sources` for it."""
    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore([Hit(7, 0.95, {"text": "one memory"})]),
        bm25_docs=[
            BM25Doc(id="7", text="one memory", payload={"text": "one memory"}),
            BM25Doc(id="other", text="one memory too", payload={"text": "another memory"}),
        ],
    )

    results = recaller.recall("one memory", limit=2, vector_limit=3)

    ids = [str(r.id) for r in results]
    assert sorted(ids) == ["7", "other"], ids  # not ["7", "7"]
    merged = next(r for r in results if str(r.id) == "7")
    assert sorted(merged.sources) == ["bm25", "vector"], merged.sources


def test_the_retriever_arms_merge_one_memory_into_one_result():
    """#172, on the retriever path: those merge dicts keyed the raw id and
    the lists they hand the fusion carry bare ids, so one memory reported
    under two id types became two entries — two of the caller's slots, and
    a different memory pushed off the page — with the arms' `sources`
    never pooled onto it."""
    from mnemostack.recall import RecallResult

    class _VectorArm:
        name = "vector"

        def search(self, query, limit=20, filters=None):
            return [
                RecallResult(id=7, text="one memory", score=0.9, payload={}, sources=["vector"])
            ]

    class _LexicalArm:
        name = "bm25"

        def search(self, query, limit=20, filters=None):
            return [
                RecallResult(id="7", text="one memory", score=0.8, payload={}, sources=["bm25"]),
                RecallResult(id="other", text="another", score=0.7, payload={}, sources=["bm25"]),
            ]

    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore([]),
        retrievers=[_VectorArm(), _LexicalArm()],
    )
    results = recaller.recall("one memory", limit=2)

    ids = [str(r.id) for r in results]
    assert sorted(ids) == ["7", "other"], ids
    merged = next(r for r in results if str(r.id) == "7")
    assert sorted(merged.sources) == ["bm25", "vector"], merged.sources


def test_the_vector_score_lookup_matches_across_id_types():
    """`raw_vector_score` is what the floor ranks candidates by, and it is
    found by matching a fused item against the vector arm's own hits.
    Raw comparison lost that match when the arms disagreed about the id's
    type, so a memory the vector arm HAD scored looked to the floor like
    one it never saw.

    Pinned at the helper rather than through a recall: on the legacy path
    the vector arm's list is fused first, so its own object wins and takes
    the score directly — the lookup only runs for an item another arm
    contributed, which needs the MCA prefilter to be the one that won."""
    hits = [Hit(9, 0.93, {"text": "shared memory"})]

    assert Recaller._raw_vector_score_for("9", hits) == 0.93  # str against int
    assert Recaller._raw_vector_score_for(9, hits) == 0.93  # and the way it came
    assert Recaller._raw_vector_score_for("absent", hits) is None


def test_query_expansion_merges_one_memory_across_its_variants():
    """#172 on the expansion path: each variant recalls separately and the
    results are merged by id before fusing. Keyed raw, a memory one variant
    reported as `5` and another as `"5"` merged as two — two slots, and the
    variants' `sources` never pooled onto it."""
    from mnemostack.llm.base import LLMResponse
    from mnemostack.recall import RecallResult

    class _TwoVariants:
        @property
        def name(self):
            return "fake"

        def generate(self, prompt, max_tokens=120, temperature=0.0):
            return LLMResponse(text="say it again\nask it differently", tokens_used=5)

    class _ByQuery:
        name = "vector"

        def search(self, query, limit=20, filters=None):
            # the same memory, reported with a different id type per phrasing
            pid = 5 if query == "what are the options" else "5"
            src = "vector" if isinstance(pid, int) else "bm25"
            return [RecallResult(id=pid, text="one memory", score=0.9, payload={}, sources=[src])]

    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore([]),
        retrievers=[_ByQuery()],
        query_expansion=True,
        expansion_llm=_TwoVariants(),
    )
    results = recaller.recall("what are the options", limit=5)

    ids = [str(r.id) for r in results]
    assert ids == ["5"], ids
    assert sorted(results[0].sources) == ["bm25", "vector"], results[0].sources


def test_the_low_confidence_fallback_merges_onto_the_memory_it_found():
    """#172, seventh site: the fallback is a SEPARATE vector call, so it is
    exactly the kind of second opinion that reports one memory under the
    other id representation. Keyed raw, it merged onto nothing — the same
    memory came back twice and the `vector` arm never joined the entry the
    caller already had."""
    from mnemostack.recall import RecallResult

    recaller = Recaller(
        embedding_provider=FakeEmbedding(),
        vector_store=FakeVectorStore([Hit(4, 0.88, {"text": "one memory"})]),
    )
    already_had = RecallResult(id="4", text="one memory", score=0.20, payload={}, sources=["bm25"])

    merged = recaller._maybe_apply_fallback(
        "one memory",
        [already_had],
        limit=5,
        vector_limit=5,
        filters=None,
    )

    assert [str(r.id) for r in merged] == ["4"], [r.id for r in merged]
    assert sorted(merged[0].sources) == ["bm25", "vector"], merged[0].sources


def test_search_many_merges_one_memory_across_its_vectors():
    """#172: `search_many` fuses one ranked list per vector and keys its
    own merge dict — a public method the answer path uses. The lists it
    hands the fusion carry BARE IDS, so the fusion's key falls through to
    the value: normalising the dict alone would leave the two disagreeing.
    A memory one vector reports as `3` and another as `"3"` was two
    entries, taking two of the caller's slots."""
    from mnemostack.recall.recaller import Recaller

    class _PerVectorStore:
        def search(self, vector, limit, filters=None, *, hide_invalidated=False):
            # the same memory, a different id type per vector
            pid = 3 if vector == [0.1] else "3"
            return [
                Hit(id=pid, score=0.9, payload={"text": "one memory"}),
                Hit(id="other", score=0.4, payload={"text": "another memory"}),
            ][:limit]

    recaller = Recaller.__new__(Recaller)
    recaller.vector = _PerVectorStore()
    recaller.rrf_k = 60
    recaller.text_key = "text"

    out = recaller.search_many([[0.1], [0.2]], limit=5)

    ids = sorted(str(r.id) for r in out)
    assert ids == ["3", "other"], ids
