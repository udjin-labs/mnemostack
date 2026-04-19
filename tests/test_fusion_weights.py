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
    for (_, s1), (_, s2) in zip(out_default, out_explicit):
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
