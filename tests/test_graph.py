"""Integration tests for GraphStore — requires live Memgraph/Neo4j on bolt://localhost:7687.

Uses a test-specific node label to avoid polluting production data.
All tests clean up after themselves.
"""
import os
import uuid

import pytest

from mnemostack.graph import GraphStore, Triple

TEST_LABEL = "MnemoTestEntity"


def _skip_if_no_memgraph():
    """Skip graph tests if Memgraph is unreachable."""
    try:
        store = GraphStore()
        ok, _ = store.health_check()
        store.close()
        if not ok:
            pytest.skip("Memgraph not available")
    except Exception:
        pytest.skip("Memgraph not available")


@pytest.fixture
def store():
    _skip_if_no_memgraph()
    s = GraphStore()
    # Run each test in unique namespace to avoid conflicts
    ns = uuid.uuid4().hex[:8]
    yield s, ns
    # Cleanup: delete only nodes created by this test
    with s.driver.session() as session:
        session.run(
            f"MATCH (n:{TEST_LABEL}) WHERE n.name STARTS WITH $ns DETACH DELETE n",
            ns=ns,
        )
    s.close()


def test_health_check(store):
    s, _ns = store
    ok, msg = s.health_check()
    assert ok
    assert "ok" in msg.lower()


def test_add_and_query_triple(store):
    s, ns = store
    subj = f"{ns}-alice"
    obj = f"{ns}-project-x"
    s.add_triple(
        subj, "WORKS_ON", obj,
        valid_from="2024-01-01",
        subject_label=TEST_LABEL, obj_label=TEST_LABEL,
    )
    triples = s.query_triples(subject=subj)
    assert len(triples) == 1
    t = triples[0]
    assert t.subject == subj
    assert t.obj == obj
    assert t.predicate == "WORKS_ON"
    assert t.valid_from == "2024-01-01"
    assert t.valid_until is None


def test_invalidate(store):
    s, ns = store
    subj = f"{ns}-bob"
    obj = f"{ns}-project-y"
    s.add_triple(
        subj, "WORKS_ON", obj,
        valid_from="2024-01-01",
        subject_label=TEST_LABEL, obj_label=TEST_LABEL,
    )
    n = s.invalidate(subj, "WORKS_ON", obj, ended="2024-06-30")
    assert n == 1
    triples = s.query_triples(subject=subj)
    assert triples[0].valid_until == "2024-06-30"


def test_point_in_time_query(store):
    s, ns = store
    subj = f"{ns}-carol"
    o1 = f"{ns}-team-A"
    o2 = f"{ns}-team-B"
    # Carol was in team-A from Jan to June, then team-B from July
    s.add_triple(subj, "MEMBER_OF", o1,
                 valid_from="2024-01-01", valid_until="2024-06-30",
                 subject_label=TEST_LABEL, obj_label=TEST_LABEL)
    s.add_triple(subj, "MEMBER_OF", o2,
                 valid_from="2024-07-01",
                 subject_label=TEST_LABEL, obj_label=TEST_LABEL)

    # Query as of March 2024 → should only find team-A
    march = s.query_triples(subject=subj, as_of="2024-03-15")
    assert len(march) == 1
    assert march[0].obj == o1

    # Query as of August 2024 → should only find team-B
    august = s.query_triples(subject=subj, as_of="2024-08-15")
    assert len(august) == 1
    assert august[0].obj == o2


def test_filter_by_predicate(store):
    s, ns = store
    subj = f"{ns}-dave"
    s.add_triple(subj, "LIKES", f"{ns}-pizza",
                 subject_label=TEST_LABEL, obj_label=TEST_LABEL)
    s.add_triple(subj, "WORKS_ON", f"{ns}-project",
                 subject_label=TEST_LABEL, obj_label=TEST_LABEL)

    likes = s.query_triples(subject=subj, predicate="LIKES")
    assert len(likes) == 1
    assert likes[0].obj.endswith("pizza")

    works = s.query_triples(subject=subj, predicate="WORKS_ON")
    assert len(works) == 1
    assert works[0].obj.endswith("project")


def test_neighbors(store):
    s, ns = store
    subj = f"{ns}-eve"
    for i in range(5):
        s.add_triple(subj, "KNOWS", f"{ns}-person-{i}",
                     subject_label=TEST_LABEL, obj_label=TEST_LABEL)
    nbrs = s.neighbors(subj)
    assert len(nbrs) == 5


def test_safe_label_sanitization():
    """Unit test — no DB needed."""
    assert GraphStore._safe_label("Person") == "Person"
    assert GraphStore._safe_label("has space") == "has_space"
    assert GraphStore._safe_label("123name") == "Node"
    assert GraphStore._safe_label("") == "Node"


def test_safe_rel_sanitization():
    """Unit test — no DB needed."""
    assert GraphStore._safe_rel("works_on") == "WORKS_ON"
    assert GraphStore._safe_rel("has space") == "HAS_SPACE"
    assert GraphStore._safe_rel("knows-well") == "KNOWS_WELL"


def test_context_manager(store):
    s, _ns = store
    # Test that __enter__/__exit__ exist
    with GraphStore() as gs:
        ok, _ = gs.health_check()
        assert ok
