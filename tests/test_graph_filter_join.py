"""Graph filter attribution: file hits prove filter scope via their chunks."""

from __future__ import annotations

from unittest.mock import MagicMock

from qdrant_client import QdrantClient

from mnemostack.recall import MemgraphRetriever, chunk_filter_probe_via
from mnemostack.vector import VectorStore


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows


class _Session:
    """Fake neo4j session over an in-memory (name, index_root) file graph."""

    def __init__(self, files):
        self._files = files

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, cypher, **params):
        if "labels(n)[0]" in cypher:  # node probe
            w = params.get("w", "")
            rows = [
                {"name": name, "type": "File", "mc": "", "index_root": root}
                for (name, root) in self._files
                if name.lower() == w
            ]
            return _Result(rows[: params.get("probe_lim") or len(rows)])
        name, root_key = params["name"], params["root_key"]
        targets = self._files.get((name, root_key), [])
        return _Result(
            [{"from_n": name, "rel": "LINKS_TO", "to_n": t} for t in targets]
        )


class _Driver:
    def __init__(self, files):
        self._files = files

    def session(self, **_):
        return _Session(self._files)


def _retriever(files, probe=None) -> MemgraphRetriever:
    return MemgraphRetriever(
        uri="bolt://x", driver=_Driver(files), chunk_filter_probe=probe
    )


FILES = {("note.md", "/corpus/a"): ["other.md"]}


# ------------------------------------------------------------ no probe


def test_without_a_probe_filters_still_drop_everything():
    retr = _retriever(FILES)
    assert retr.search("note.md", filters={"project": "x"}, include_invalidated=True) == []


# ------------------------------------------------------------ attribution


def test_residual_keys_are_proven_through_the_chunks():
    calls: list[tuple[dict, str | None]] = []

    def probe(filters, tenant):
        calls.append((dict(filters), tenant))
        return True

    retr = _retriever(FILES, probe)
    out = retr.search("note.md", filters={"project": "x"}, include_invalidated=True)

    assert [r.payload["name"] for r in out] == ["note.md"]
    assert calls == [
        ({"project": "x", "source": "note.md", "index_root": "/corpus/a"}, None)
    ]


def test_unprovable_hits_are_dropped_not_leaked():
    retr = _retriever(FILES, lambda f, t: False)  # no chunk matches
    assert retr.search("note.md", filters={"project": "x"}, include_invalidated=True) == []


def test_own_payload_keys_short_circuit_without_a_probe_call():
    def probe(_f, _t):  # pragma: no cover - must not run
        raise AssertionError("fully self-attributed hits must not probe")

    retr = _retriever(FILES, probe)
    # index_root is IN the hit payload: a matching filter needs no probe...
    out = retr.search(
        "note.md", filters={"index_root": "/corpus/a"}, include_invalidated=True
    )
    assert [r.payload["name"] for r in out] == ["note.md"]
    # ...and a mismatching one drops the hit, also without probing.
    assert (
        retr.search(
            "note.md", filters={"index_root": "/elsewhere"}, include_invalidated=True
        )
        == []
    )


def test_source_filter_is_evaluated_against_the_node_name():
    """The graph payload's `source` key is the arm marker ("memgraph"), not a
    document — a caller's source condition must check the node NAME."""

    def probe(_f, _t):  # pragma: no cover - must not run
        raise AssertionError("source is provable from the name — no probe")

    retr = _retriever(FILES, probe)
    out = retr.search("note.md", filters={"source": "note.md"}, include_invalidated=True)
    assert [r.payload["name"] for r in out] == ["note.md"]
    assert (
        retr.search("note.md", filters={"source": "other.md"}, include_invalidated=True)
        == []
    )


def test_probe_failure_fails_closed():
    def probe(_f, _t):
        raise RuntimeError("store down")

    retr = _retriever(FILES, probe)
    assert retr.search("note.md", filters={"project": "x"}, include_invalidated=True) == []


def test_tenant_is_threaded_into_the_probe():
    calls: list[tuple[dict, str | None]] = []

    def probe(filters, tenant):
        calls.append((dict(filters), tenant))
        return True

    files = {("note.md", "/corpus/a"): ["other.md"]}

    class _TenantSession(_Session):
        def run(self, cypher, **params):
            # The tenant-scoped Cypher shape differs; reuse the fake matching.
            return super().run(cypher, **params)

    retr = MemgraphRetriever(
        uri="bolt://x", driver=_Driver(files), chunk_filter_probe=probe
    )
    out = retr.search(
        "note.md", filters={"project": "x"}, include_invalidated=True, tenant="acme"
    )
    assert out and calls[0][1] == "acme"


class _EntitySession(_Session):
    """Probe returns an :Entity-style node: no index_root, non-File type."""

    def run(self, cypher, **params):
        if "labels(n)[0]" in cypher:
            w = params.get("w", "")
            rows = (
                [{"name": "Alice", "type": "Person", "mc": "", "index_root": None}]
                if w == "alice"
                else []
            )
            return _Result(rows)
        return _Result([{"from_n": "Alice", "rel": "KNOWS", "to_n": "Bob"}])


def test_entity_hits_are_never_probed_or_attributed():
    """The P1 this pins: an entity has no pinnable chunks — probing it by
    bare name would let an unrelated same-named chunk anywhere attribute it
    into the scope. It must be dropped WITHOUT a probe."""

    def probe(_f, _t):  # pragma: no cover - must not run
        raise AssertionError("entity hits must never reach the chunk probe")

    class _D:
        def session(self, **_):
            return _EntitySession({})

    retr = MemgraphRetriever(uri="bolt://x", driver=_D(), chunk_filter_probe=probe)
    assert retr.search("alice", filters={"project": "x"}, include_invalidated=True) == []


def test_entity_hits_with_explicit_none_root_fail_an_index_root_filter():
    """index_root is PRESENT (value None) on entity payloads — a filter on it
    must fail via the own-key tier, key-present-but-None never matches."""

    class _D:
        def session(self, **_):
            return _EntitySession({})

    retr = MemgraphRetriever(
        uri="bolt://x", driver=_D(), chunk_filter_probe=lambda f, t: True
    )
    assert (
        retr.search("alice", filters={"index_root": "/x"}, include_invalidated=True)
        == []
    )


class _RootlessFileSession(_Session):
    """A legacy :File node synced before root tracking: type File, no root."""

    def run(self, cypher, **params):
        if "labels(n)[0]" in cypher:
            w = params.get("w", "")
            rows = (
                [{"name": "note.md", "type": "File", "mc": "", "index_root": None}]
                if w == "note.md"
                else []
            )
            return _Result(rows)
        return _Result([{"from_n": "note.md", "rel": "LINKS_TO", "to_n": "x.md"}])


def test_legacy_rootless_file_hits_are_dropped_not_probed():
    """Without a root there is no pin — a same-named document in another
    root could attribute the hit, so it stays unprovable (fail-closed)."""

    def probe(_f, _t):  # pragma: no cover - must not run
        raise AssertionError("rootless file hits must never reach the probe")

    class _D:
        def session(self, **_):
            return _RootlessFileSession({})

    retr = MemgraphRetriever(uri="bolt://x", driver=_D(), chunk_filter_probe=probe)
    assert (
        retr.search("note.md", filters={"project": "x"}, include_invalidated=True)
        == []
    )


# ------------------------------------------------------------ probe factory


def test_chunk_filter_probe_via_answers_from_the_store():
    s = VectorStore.__new__(VectorStore)
    s.collection = "join"
    s.dimension = 2
    s.client = QdrantClient(":memory:")
    s.client.create_collection(
        collection_name="join", vectors_config={"size": 2, "distance": "Cosine"}
    )
    from qdrant_client.models import PointStruct

    s.client.upsert(
        "join",
        points=[
            PointStruct(
                id="11111111-1111-1111-1111-111111111111",
                vector=[0.1, 0.2],
                payload={"source": "note.md", "index_root": "/corpus/a", "project": "x"},
            )
        ],
    )
    probe = chunk_filter_probe_via(s)
    assert probe is not None
    assert probe({"source": "note.md", "project": "x"}, None) is True
    assert probe({"source": "note.md", "project": "y"}, None) is False


def test_chunk_filter_probe_via_none_for_duck_stores():
    assert chunk_filter_probe_via(None) is None
    assert chunk_filter_probe_via(MagicMock(spec=[])) is None
