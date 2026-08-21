"""Weighted RRF: retrievers should be able to express confidence per list."""

from mnemostack.recall.fusion import reciprocal_rank_fusion


def _doc(i):
    return (f"m{i}", 1.0)  # (id, irrelevant original_score)


def test_unweighted_matches_classic_rrf():
    a = [_doc(1), _doc(2), _doc(3)]
    b = [_doc(4), _doc(1), _doc(5)]
    out_default = reciprocal_rank_fusion([a, b])
    out_no_weights = reciprocal_rank_fusion([a, b], weights=None)
    assert out_default == out_no_weights


def test_weight_1_1_equivalent_to_default():
    a = [_doc(1), _doc(2)]
    b = [_doc(2), _doc(3)]
    out_default = reciprocal_rank_fusion([a, b])
    out_explicit = reciprocal_rank_fusion([a, b], weights=[1.0, 1.0])
    assert [x[0] for x in out_default] == [x[0] for x in out_explicit]
    for (_, s1), (_, s2) in zip(out_default, out_explicit, strict=False):
        assert abs(s1 - s2) < 1e-9


def test_higher_weight_promotes_source():
    # List B has id 'm9' at rank 1, list A has 'm1' at rank 1.
    # Without weights: tied. With b_weight=2: m9 wins.
    a = [_doc(1), _doc(2)]
    b = [_doc(9), _doc(3)]
    tied = reciprocal_rank_fusion([a, b])
    # Both are rank 1 in their own list -> same score
    score_map = dict(tied)
    assert abs(score_map["m1"] - score_map["m9"]) < 1e-9

    weighted = reciprocal_rank_fusion([a, b], weights=[1.0, 2.0])
    ranked_ids = [x[0] for x in weighted]
    assert ranked_ids[0] == "m9"  # B's top beats A's top when B is weighted 2x


def test_zero_weight_eliminates_list():
    a = [_doc(1), _doc(2)]
    b = [_doc(99), _doc(100)]
    # With zero weight on list b, only A's items should appear
    out = reciprocal_rank_fusion([a, b], weights=[1.0, 0.0])
    ids = [x[0] for x in out]
    assert ids == ["m1", "m2"]


def test_missing_weight_defaults_to_one():
    a = [_doc(1)]
    b = [_doc(2)]
    c = [_doc(3)]
    # Only two weights supplied; third should default to 1.0
    out = reciprocal_rank_fusion([a, b, c], weights=[1.0, 1.0])
    ids = {x[0] for x in out}
    assert ids == {"m1", "m2", "m3"}


def test_negative_weight_clamped_to_zero():
    a = [_doc(1)]
    b = [_doc(2)]
    out = reciprocal_rank_fusion([a, b], weights=[1.0, -5.0])
    ids = [x[0] for x in out]
    # Negative -> 0 -> list b is effectively removed
    assert ids == ["m1"]


def test_one_memory_is_one_entry_whatever_type_its_id_arrived_as():
    """#172: the fusion's own dedup keyed the RAW id, so a memory carrying
    `1` from one arm and `"1"` from another was two entries. That does not
    merely list it twice — the `limit` cut then spends two of the caller's
    slots on it and drops a genuinely different memory to make room."""

    class _Hit:
        def __init__(self, pid):
            self.id = pid

    same_a, same_b, other = _Hit(1), _Hit("1"), _Hit("other")
    fused = reciprocal_rank_fusion([[(same_a, 0.9)], [(same_b, 0.9)], [(other, 0.5)]], limit=2)

    ids = [str(item.id) for item, _ in fused]
    assert sorted(ids) == ["1", "other"], ids


def test_an_item_with_no_id_keeps_its_own_identity():
    """The rule applies to IDS. An item that carries none is the caller's
    business — stringifying arbitrary objects here would merge things this
    function has no business merging.

    Two DISTINCT objects that happen to print the same, on purpose: with
    plain strings the assertion cannot tell "returned as-is" from
    "normalised", and a test that cannot fail for the reason it names is
    not pinning anything."""

    class _Twin:
        def __str__(self):
            return "same-text"

    left, right = _Twin(), _Twin()
    fused = reciprocal_rank_fusion([[(left, 1.0)], [(right, 1.0)]])

    assert len(fused) == 2, "two distinct id-less items were merged by their text"
    assert {id(item) for item, _ in fused} == {id(left), id(right)}


def test_collapsing_a_memory_keeps_what_both_copies_knew():
    """Collapsing two objects for one memory must not discard the second
    one's arms. The generic callers — query expansion and inference retry
    — have no pooling step of their own, so an arm that found the memory
    under the other id representation would simply vanish from `sources`."""

    class _Hit:
        def __init__(self, pid, sources):
            self.id = pid
            self.sources = list(sources)

    first, second = _Hit(1, ["vector"]), _Hit("1", ["bm25", "graph"])
    fused = reciprocal_rank_fusion([[(first, 0.9)], [(second, 0.8)]])

    assert len(fused) == 1
    kept = fused[0][0]
    assert sorted(kept.sources) == ["bm25", "graph", "vector"]  # both arms kept
    # ...and the caller's own objects are untouched: these are the very
    # objects an upstream caller may still be holding — `merge_results`
    # fuses a caller's memories directly — and one of them is returned
    # unchanged when a retry is REJECTED, to represent what the served
    # draft was generated from.
    assert first.sources == ["vector"]
    assert second.sources == ["bm25", "graph"]


def test_pooling_leaves_an_unpoolable_shape_alone():
    """Duck-typed on BOTH sides, and the guards are what this pins — the
    first version of this test only checked which ids survived dedup,
    which was already true before any pooling existed, so it passed with
    the pooling deleted outright.

    A `sources` that is not a list must be ignored, not iterated: a string
    would be walked character by character and seed the survivor's arms
    with single letters."""

    class _Hit:
        def __init__(self, pid, sources):
            self.id = pid
            self.sources = sources

    keeper = _Hit(1, ["vector"])
    stringy = _Hit("1", "graph")  # not a list
    fused = reciprocal_rank_fusion([[(keeper, 0.9)], [(stringy, 0.8)]])

    assert len(fused) == 1
    assert fused[0][0].sources == ["vector"], fused[0][0].sources

    # ...and an item with no `sources` at all is fused, and returned, as-is
    plain = reciprocal_rank_fusion([[("m1", 1.0)], [("m1", 1.0)]])
    assert [item for item, _ in plain] == ["m1"]


def test_one_list_gets_one_vote_for_one_memory():
    """A list that names the same memory twice — `1` at one rank and `"1"`
    at another — must not vote twice. Normalising the key alone does not
    stop that: the loop scores whatever it is handed, so a single arm would
    out-vote genuine agreement between two and push a real result past the
    limit."""

    class _Hit:
        def __init__(self, pid, sources=()):
            self.id = pid
            self.sources = list(sources)

    twice = [(_Hit(1, ["vector"]), 0.9), (_Hit("1", ["graph"]), 0.8)]
    corroborated = [(_Hit("C"), 0.9)], [(_Hit("C"), 0.9)]
    fused = reciprocal_rank_fusion([twice, *corroborated])

    ranking = [(str(item.id), round(score, 6)) for item, score in fused]
    assert ranking[0][0] == "C", ranking  # two lists agreeing beats one list twice
    assert dict(ranking)["1"] == round(1 / 61, 6)  # exactly one vote, at rank 1
    kept = next(item for item, _ in fused if str(item.id) == "1")
    assert sorted(kept.sources) == ["graph", "vector"]  # ...and both arms kept


def test_a_replacement_does_not_share_the_payload_it_was_copied_from():
    """`copy.copy` is shallow, and the fuse loops downstream WRITE into
    `payload` — the legacy recall path stamps `raw_vector_score` on
    whatever fusion hands back. Left shared, that write lands on an object
    this function promised not to touch: the same lesson as `sources`, one
    field over."""
    from mnemostack.recall.recaller import RecallResult

    first = RecallResult(id=1, text="m", score=0.9, payload={"k": "orig"}, sources=["vector"])
    second = RecallResult(id="1", text="m", score=0.8, payload={"k": "orig"}, sources=["bm25"])

    kept = reciprocal_rank_fusion([[(first, 0.9)], [(second, 0.8)]])[0][0]
    kept.payload["raw_vector_score"] = 0.42

    assert first.payload == {"k": "orig"}, first.payload
    assert second.payload == {"k": "orig"}, second.payload


def test_an_immutable_item_is_fused_rather_than_refused():
    """A frozen dataclass copies happily and then raises on assignment, so
    guarding only the copy would break a PUBLIC function for input it used
    to accept. Pooling is an observability nicety; fusing is the job."""
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class _Frozen:
        id: object
        sources: list

    a, b = _Frozen(1, ["vector"]), _Frozen("1", ["bm25"])
    fused = reciprocal_rank_fusion([[(a, 0.9)], [(b, 0.8)]])

    assert len(fused) == 1  # still fused, still deduped by the rule
    assert fused[0][0] is a  # ...and handed back untouched


def test_dictionary_items_keep_both_copies_arms_too():
    """`_get_key` explicitly supports a mapping with an `id`, so the
    pooling has to support that shape as well — handling only objects meant
    two dicts collapsed under the identity rule and the second one's arms
    vanished, in a shape the key function documents as supported."""
    first = {"id": 1, "sources": ["vector"], "payload": {"k": "orig"}}
    second = {"id": "1", "sources": ["bm25"], "payload": {"k": "orig"}}

    kept = reciprocal_rank_fusion([[(first, 0.9)], [(second, 0.8)]])[0][0]

    assert sorted(kept["sources"]) == ["bm25", "vector"], kept["sources"]
    assert first["sources"] == ["vector"]  # the caller's own dicts, untouched
    assert second["sources"] == ["bm25"]
    kept["payload"]["written"] = True
    assert first["payload"] == {"k": "orig"}
