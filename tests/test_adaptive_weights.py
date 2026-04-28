"""Tests for adaptive per-query-shape retriever weights in Recaller.

We don't need a live Qdrant — the logic under test is pure string detection
and weight lookup. The real-prod A/B check is run from a separate smoke script.
"""
from mnemostack.recall.recaller import Recaller


def test_detect_exact_token_shape_by_regex():
    assert Recaller._detect_query_shape("what is the IP 185.31.164.237") == "exact_token"
    assert Recaller._detect_query_shape("open port 7687") == "exact_token"
    assert Recaller._detect_query_shape("version 2026.4.12 is out") == "exact_token"
    assert Recaller._detect_query_shape("Ticker LKOH-123") == "exact_token"


def test_detect_exact_token_shape_by_marker():
    assert Recaller._detect_query_shape("какой UUID у записи") == "exact_token"
    assert Recaller._detect_query_shape("какая версия mnemostack") == "exact_token"
    assert Recaller._detect_query_shape("show me the API key usage") == "exact_token"


def test_detect_person_shape():
    assert Recaller._detect_query_shape("кто такой Greycheg") == "person"
    assert Recaller._detect_query_shape("who is John Smith") == "person"
    # Numeric telegram id still classified as exact_token (regex hits first),
    # which is the desired behaviour — graph AND bm25 both get boosted.
    shape = Recaller._detect_query_shape("telegram id 250339146")
    assert shape in ("exact_token", "person")


def test_detect_temporal_shape():
    assert Recaller._detect_query_shape("когда закрыли trading") == "temporal"
    assert Recaller._detect_query_shape("when did we launch") == "temporal"
    assert Recaller._detect_query_shape("what happened yesterday") == "temporal"


def test_detect_general_default():
    assert Recaller._detect_query_shape("how does the pipeline work") == "general"
    assert Recaller._detect_query_shape("summarise the architecture") == "general"


def test_weight_for_static_overrides_adaptive():
    r = Recaller(
        retrievers=[],
        retriever_weights={"bm25": 2.0},
        adaptive_weights=True,
    )
    # Even if adaptive would compute a different weight, static takes priority.
    assert r._weight_for("bm25", "когда это было") == 2.0


def test_weight_for_adaptive_off_returns_one():
    r = Recaller(retrievers=[], adaptive_weights=False)
    assert r._weight_for("bm25", "IP 1.2.3.4") == 1.0
    assert r._weight_for("vector", "anything") == 1.0


def test_weight_for_adaptive_on_exact_token():
    r = Recaller(retrievers=[], adaptive_weights=True)
    # BM25 and memgraph should both get a lift on an IP query
    assert r._weight_for("bm25", "IP 185.31.164.237") > 1.0
    assert r._weight_for("memgraph", "IP 185.31.164.237") > 1.0
    # Vector keeps 1.0, temporal is slightly lowered
    assert r._weight_for("vector", "IP 185.31.164.237") == 1.0
    assert r._weight_for("temporal", "IP 185.31.164.237") < 1.0


def test_weight_for_adaptive_on_temporal():
    r = Recaller(retrievers=[], adaptive_weights=True)
    assert r._weight_for("temporal", "когда был релиз a13") > 1.0
    assert r._weight_for("vector", "когда был релиз a13") == 1.0


def test_weight_for_adaptive_on_person():
    r = Recaller(retrievers=[], adaptive_weights=True)
    assert r._weight_for("memgraph", "кто такой Сергей") > 1.0
    # bm25 / vector stay at 1.0 for person queries — graph wins
    assert r._weight_for("bm25", "кто такой Сергей") == 1.0


def test_weight_for_general_is_all_ones():
    r = Recaller(retrievers=[], adaptive_weights=True)
    for name in ("vector", "bm25", "memgraph", "temporal"):
        assert r._weight_for(name, "how does the memory pipeline work") == 1.0
