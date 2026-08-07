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

    def probe(filters, tenant, include_invalidated, as_of):
        calls.append((dict(filters), tenant, include_invalidated, as_of))
        return True

    retr = _retriever(FILES, probe)
    out = retr.search("note.md", filters={"project": "x"}, include_invalidated=True)

    assert [r.payload["name"] for r in out] == ["note.md"]
    assert calls == [
        (
            {"project": "x", "source": "note.md", "index_root": "/corpus/a"},
            None,
            True,
            None,
        )
    ]


def test_unprovable_hits_are_dropped_not_leaked():
    retr = _retriever(FILES, lambda f, t, inv, ao: False)  # no chunk matches
    assert retr.search("note.md", filters={"project": "x"}, include_invalidated=True) == []


def test_own_payload_keys_short_circuit_without_a_probe_call():
    def probe(*_a):  # pragma: no cover - must not run
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
    document — a caller's source condition checks the node NAME, and a
    satisfied name still needs CHUNK proof (a dangling link target has a
    :File node but no document)."""
    calls: list[dict] = []

    def probe(filters, tenant, include_invalidated, as_of):
        calls.append(dict(filters))
        return True  # the document has chunks

    retr = _retriever(FILES, probe)
    out = retr.search("note.md", filters={"source": "note.md"}, include_invalidated=True)
    assert [r.payload["name"] for r in out] == ["note.md"]
    assert calls and calls[0]["source"] == "note.md"  # chunk-existence proof ran
    assert (
        retr.search("note.md", filters={"source": "other.md"}, include_invalidated=True)
        == []
    )


def test_dangling_link_targets_never_pass_a_source_filter():
    """Round-7 pin: sync creates :File nodes for DANGLING link targets — a
    node whose document has no chunks must not surface under
    filters={"source": ...}, and without a probe it fails closed."""
    retr = _retriever(FILES, lambda f, t, inv, ao: False)  # no chunks exist
    assert (
        retr.search("note.md", filters={"source": "note.md"}, include_invalidated=True)
        == []
    )
    probeless = _retriever(FILES)
    assert (
        probeless.search(
            "note.md", filters={"source": "note.md"}, include_invalidated=True
        )
        == []
    )


def test_probe_failure_fails_closed():
    def probe(*_a):
        raise RuntimeError("store down")

    retr = _retriever(FILES, probe)
    assert retr.search("note.md", filters={"project": "x"}, include_invalidated=True) == []


def test_tenant_is_threaded_into_the_probe():
    calls: list[tuple[dict, str | None]] = []

    def probe(filters, tenant, include_invalidated, as_of):
        calls.append((dict(filters), tenant, include_invalidated, as_of))
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

    def probe(*_a):  # pragma: no cover - must not run
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
        uri="bolt://x", driver=_D(), chunk_filter_probe=lambda f, t, inv, ao: True
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

    def probe(*_a):  # pragma: no cover - must not run
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
    assert probe({"source": "note.md", "project": "x"}, None, True, None) is True
    assert probe({"source": "note.md", "project": "y"}, None, True, None) is False


def test_chunk_filter_probe_via_none_for_duck_stores():
    assert chunk_filter_probe_via(None) is None
    assert chunk_filter_probe_via(MagicMock(spec=[])) is None


def test_source_filter_never_admits_an_entity_named_like_a_document():
    """P1: an entity whose NAME equals the requested source must not pass a
    source-isolation boundary — the name-as-source shortcut is File-only."""

    class _D:
        def session(self, **_):
            return _EntitySession({})

    retr = MemgraphRetriever(
        uri="bolt://x", driver=_D(), chunk_filter_probe=lambda f, t, inv, ao: True
    )
    assert (
        retr.search("alice", filters={"source": "Alice"}, include_invalidated=True)
        == []
    )


def test_tenant_id_filter_is_never_provable_by_chunks():
    """P1: tenant identity is the isolation key itself — an unscoped recall
    with a tenant_id filter must drop graph hits (whose nodes carry no
    stamped tenant), not let another tenant's chunks prove them."""

    def probe(*_a):  # pragma: no cover - must not run
        raise AssertionError("tenant identity must never reach the chunk probe")

    retr = _retriever(FILES, probe)
    assert (
        retr.search("note.md", filters={"tenant_id": "A"}, include_invalidated=True)
        == []
    )


def test_validity_view_is_threaded_into_the_probe():
    calls: list[tuple] = []

    def probe(filters, tenant, include_invalidated, as_of):
        calls.append((include_invalidated, as_of))
        return True

    retr = _retriever(FILES, probe)
    out = retr.search(
        "note.md",
        filters={"project": "x"},
        include_invalidated=True,
        as_of="2026-01-01",
    )
    assert out
    assert calls[0][0] is True
    assert calls[0][1] is not None and calls[0][1].startswith("2026-01-01")


def test_attributed_hits_carry_the_proof_marker():
    retr = _retriever(FILES, lambda f, t, inv, ao: True)
    out = retr.search("note.md", filters={"project": "x"}, include_invalidated=True)
    assert out[0].payload["_attributed_filters"] == {"project": "x"}


def test_result_passes_filters_honors_the_marker_for_graph_hits_only():
    from types import SimpleNamespace

    from mnemostack.recall.filters import result_passes_filters

    filters = {"project": "x"}
    graph_hit = SimpleNamespace(
        payload={"name": "note.md", "_attributed_filters": {"project": "x"}},
        sources=["memgraph"],
    )
    assert result_passes_filters(graph_hit, filters)

    # A DIFFERENT filter set than the one proven — no pass.
    assert not result_passes_filters(graph_hit, {"project": "y"})

    # A vector hit with a planted marker must not bypass the filter.
    spoofed = SimpleNamespace(
        payload={"_attributed_filters": {"project": "x"}}, sources=["vector"]
    )
    assert not result_passes_filters(spoofed, filters)

    # Ordinary payload matching still works.
    plain = SimpleNamespace(payload={"project": "x"}, sources=["vector"])
    assert result_passes_filters(plain, filters)


def test_flow_post_filter_keeps_attributed_graph_hits():
    """The P1 headline: without the marker, recall_flow's post-pipeline
    filter re-dropped every graph hit the retriever had just proven."""
    from mnemostack.recall import RecallResult, recall_flow

    graph_hit = RecallResult(
        id="graph:/corpus/a:note.md",
        text="File: note.md",
        score=1.0,
        payload={"name": "note.md", "_attributed_filters": {"project": "x"}},
        sources=["memgraph"],
    )
    vector_hit = RecallResult(
        id="v1", text="chunk", score=0.9, payload={"project": "x"}, sources=["vector"]
    )
    stray = RecallResult(
        id="v2", text="foreign", score=0.8, payload={"project": "y"}, sources=["vector"]
    )

    class _Recaller:
        def recall(self, query, limit=10, **kwargs):
            return [graph_hit, vector_hit, stray]

    class _Pipeline:
        def apply(self, query, results, **kw):
            return list(results)

        def __iter__(self):
            return iter([])

    out = recall_flow(
        _Recaller(), "q", 10, pipeline=_Pipeline(), filters={"project": "x"}
    )
    ids = [r.id for r in out]
    assert "graph:/corpus/a:note.md" in ids  # proven graph hit survives
    assert "v1" in ids
    assert "v2" not in ids  # unproven stays dropped


def test_factory_converts_timestamp_bounds_and_forwards_validity():
    calls: list[tuple] = []

    class _Store:
        def any_matching_point(self, filters, *, tenant=None, include_invalidated=True, as_of=None):
            calls.append((dict(filters), tenant, include_invalidated, as_of))
            return True

    probe = chunk_filter_probe_via(
        _Store(), timestamp_key="timestamp", timestamp_format="epoch"
    )
    assert probe is not None
    assert probe(
        {"timestamp": {"gte": "2026-01-01T00:00:00+00:00"}, "source": "a.md"},
        None,
        False,
        "2026-02-01",
    )
    forwarded, tenant, inv, as_of = calls[0]
    assert isinstance(forwarded["timestamp"]["gte"], (int, float))  # epoch domain
    assert tenant is None and inv is False and as_of == "2026-02-01"


def test_any_matching_point_respects_validity_views():
    from qdrant_client.models import PointStruct

    s = VectorStore.__new__(VectorStore)
    s.collection = "validity"
    s.dimension = 2
    s.client = QdrantClient(":memory:")
    s.client.create_collection(
        collection_name="validity", vectors_config={"size": 2, "distance": "Cosine"}
    )
    s.client.upsert(
        "validity",
        points=[
            PointStruct(
                id="11111111-1111-1111-1111-111111111111",
                vector=[0.1, 0.2],
                payload={
                    "source": "a.md",
                    "project": "x",
                    "invalidated_at": "2026-01-15T00:00:00+00:00",
                    "valid_from": "2026-01-01T00:00:00+00:00",
                    "valid_until": "2026-01-15T00:00:00+00:00",
                },
            )
        ],
    )
    f = {"source": "a.md", "project": "x"}
    assert s.any_matching_point(f) is True  # neutral view sees it
    assert s.any_matching_point(f, include_invalidated=False) is False  # stale hidden
    # Point-in-time: valid inside the window, not after it.
    assert s.any_matching_point(f, as_of="2026-01-10") is True
    assert s.any_matching_point(f, as_of="2026-02-01") is False


def test_rejected_candidates_do_not_starve_the_node_budget():
    """Round-2 pin: attribution runs during candidate traversal, before the
    max_nodes cut — a higher-ranked unattributable file must not consume the
    budget of an attributable lower-ranked one."""
    files = {
        ("aaa.md", "/corpus/a"): ["x.md"],
        ("bbb.md", "/corpus/a"): ["y.md"],
    }

    def probe(filters, tenant, include_invalidated, as_of):
        return filters.get("source") == "bbb.md"  # only the second attributes

    retr = MemgraphRetriever(
        uri="bolt://x",
        driver=_Driver(files),
        max_nodes=1,  # budget of ONE: the rejected aaa.md must not eat it
        chunk_filter_probe=probe,
    )
    out = retr.search("aaa.md bbb.md", filters={"project": "x"}, include_invalidated=True)
    assert [r.payload["name"] for r in out] == ["bbb.md"]


def test_as_of_probe_scans_past_the_first_pages():
    """Round-2 pin: an as_of probe must scan ALL matching chunks — scroll
    order can put a re-ingested document's new (later-valid) chunks before
    the older chunk that is actually valid at the requested instant."""
    from uuid import uuid4

    from qdrant_client.models import PointStruct

    s = VectorStore.__new__(VectorStore)
    s.collection = "asof-deep"
    s.dimension = 2
    s.client = QdrantClient(":memory:")
    s.client.create_collection(
        collection_name="asof-deep", vectors_config={"size": 2, "distance": "Cosine"}
    )
    points = [
        PointStruct(
            id=str(uuid4()),
            vector=[0.1, 0.2],
            payload={
                "source": "a.md",
                "project": "x",
                "valid_from": "2026-06-01T00:00:00+00:00",  # after the snapshot
            },
        )
        for _ in range(150)
    ]
    points.append(
        PointStruct(
            id=str(uuid4()),
            vector=[0.1, 0.2],
            payload={
                "source": "a.md",
                "project": "x",
                "valid_from": "2026-01-01T00:00:00+00:00",
                "valid_until": "2026-03-01T00:00:00+00:00",
            },
        )
    )
    s.client.upsert("asof-deep", points=points)

    f = {"source": "a.md", "project": "x"}
    # The only chunk valid at 2026-02-01 sits beyond the first pages.
    assert s.any_matching_point(f, as_of="2026-02-01") is True
    assert s.any_matching_point(f, as_of="2025-12-01") is False


def test_deep_candidate_pools_are_traversed_until_attribution():
    """Round-3 pin: the traversal draws from the probes' FULL candidate pool
    — a deep run of unattributable same-named files (other roots) must not
    starve an attributable candidate far beyond the old 3x window."""
    files = {("note.md", f"/corpus/{i:03d}"): [f"n{i}.md"] for i in range(40)}

    def probe(filters, tenant, include_invalidated, as_of):
        return filters.get("index_root") == "/corpus/037"  # deep in the pool

    retr = MemgraphRetriever(
        uri="bolt://x",
        driver=_Driver(files),
        max_nodes=1,
        chunk_filter_probe=probe,
    )
    out = retr.search("note.md", filters={"project": "x"}, include_invalidated=True)
    assert [r.payload["index_root"] for r in out] == ["/corpus/037"]


def test_placeholder_metadata_stays_probe_provable():
    """Round-4 pin: sync writes :File nodes without memory_class; the ""
    placeholder must not be treated as authoritative — the filter key goes
    to the probe, where the file's chunks can prove it."""
    calls: list[dict] = []

    def probe(filters, tenant, include_invalidated, as_of):
        calls.append(dict(filters))
        return True

    retr = _retriever(FILES, probe)
    out = retr.search(
        "note.md", filters={"memory_class": "decision"}, include_invalidated=True
    )
    assert [r.payload["name"] for r in out] == ["note.md"]
    assert calls and calls[0]["memory_class"] == "decision"


def test_filtered_hits_never_serialize_relationship_neighbors():
    """Round-4 pin: an attributed hit's text must not name OTHER files via
    edge serialization — a.md-[LINKS_TO]->b.md inside an attributed b.md hit
    would leak a.md's existence across the filter boundary."""
    retr = _retriever(FILES, lambda f, t, inv, ao: True)
    out = retr.search("note.md", filters={"project": "x"}, include_invalidated=True)
    assert out
    assert "-[" not in out[0].text
    assert "other.md" not in out[0].text  # the linked neighbor stays unnamed


def test_filtered_traversal_issues_no_relationship_queries():
    """Round-4 pin: rejected candidates cost zero graph round trips — the
    relationship expansion is skipped entirely on the filtered path (no
    validity view active)."""
    rel_queries: list[str] = []

    class _CountingSession(_Session):
        def run(self, cypher, **params):
            if "startNode" in cypher:
                rel_queries.append(params.get("name", ""))
            return super().run(cypher, **params)

    class _D:
        def session(self, **_):
            return _CountingSession(FILES)

    retr = MemgraphRetriever(
        uri="bolt://x", driver=_D(), chunk_filter_probe=lambda f, t, inv, ao: False
    )
    assert retr.search("note.md", filters={"project": "x"}, include_invalidated=True) == []
    assert rel_queries == []  # rejection cost: one probe, zero graph queries


def test_probe_budget_bounds_backend_round_trips(monkeypatch):
    """Round-4 pin: probes are capped per call — beyond the budget the
    remaining candidates fail closed instead of hammering the store."""
    import mnemostack.recall.retrievers as retrievers_mod

    monkeypatch.setattr(retrievers_mod, "_MAX_ATTRIBUTION_PROBES", 2)
    files = {(f"doc{i}.md", "/corpus/a"): [] for i in range(6)}
    calls: list[dict] = []

    def probe(filters, tenant, include_invalidated, as_of):
        calls.append(dict(filters))
        return False  # nothing attributes — every candidate wants a probe

    retr = MemgraphRetriever(
        uri="bolt://x", driver=_Driver(files), chunk_filter_probe=probe
    )
    out = retr.search(
        "doc0.md doc1.md doc2.md doc3.md doc4.md doc5.md",
        filters={"project": "x"},
        include_invalidated=True,
    )
    assert out == []
    assert len(calls) == 2  # budget, not candidate count


def test_synthetic_fields_never_self_attribute():
    """Round-5 pin: `text` is synthesized locally ("File: note.md") — a
    filter on it must go through the chunk probe, not self-attribute."""
    calls: list[dict] = []

    def probe(filters, tenant, include_invalidated, as_of):
        calls.append(dict(filters))
        return False  # the chunks do NOT contain this text

    retr = _retriever(FILES, probe)
    out = retr.search(
        "note.md", filters={"text": "File: note.md"}, include_invalidated=True
    )
    assert out == []  # not admitted by its own synthetic text
    assert calls and "text" in calls[0]  # the condition went to the probe


def test_attribution_marker_is_stripped_from_public_payloads():
    from mnemostack.mcp.server import _public_payload
    from mnemostack.server import _memory_of

    payload = {
        "text": "File: note.md",
        "name": "note.md",
        "_attributed_filters": {"project": "x"},
        "_vector_floor_candidates": ["a"],
    }
    assert "_attributed_filters" not in _public_payload(payload)
    assert "_vector_floor_candidates" not in _public_payload(payload)

    from types import SimpleNamespace

    mem = _memory_of(
        SimpleNamespace(id="g1", text="File: note.md", score=1.0, payload=dict(payload), sources=["memgraph"])
    )
    assert "_attributed_filters" not in mem.metadata
    assert "_vector_floor_candidates" not in mem.metadata


def test_self_provable_filters_pass_without_any_probe_configured():
    """Round-6 pin: without a chunk probe, filters the node's trusted
    metadata proves by itself (index_root) still pass — only chunk-proof
    keys fail closed."""
    retr = _retriever(FILES)  # no probe at all
    out = retr.search(
        "note.md", filters={"index_root": "/corpus/a"}, include_invalidated=True
    )
    assert [r.payload["name"] for r in out] == ["note.md"]
    # Residual-needing filters keep the historical fail-closed drop.
    assert retr.search("note.md", filters={"project": "x"}, include_invalidated=True) == []


def test_chunk_proven_hits_survive_the_validity_gate_without_edges():
    """Round-8 pin: a file whose links were all removed keeps its node and
    its current chunks — a validity-aware chunk proof replaces the
    incident-edge gate. Own-metadata-only attribution still requires it."""
    linkless = {("note.md", "/corpus/a"): []}  # node present, zero edges

    # Chunk-proven: survives the default current-facts view without edges.
    retr = _retriever(linkless, lambda f, t, inv, ao: True)
    out = retr.search("note.md", filters={"project": "x"})  # validity active
    assert [r.payload["name"] for r in out] == ["note.md"]

    # Own-metadata-only (no probe ran): the edge gate still applies.
    def must_not_probe(*_a):  # pragma: no cover
        raise AssertionError("index_root attribution must not probe")

    retr2 = _retriever(linkless, must_not_probe)
    assert retr2.search("note.md", filters={"index_root": "/corpus/a"}) == []


# ------------------------------------------ query amplification guards


class _CountingSession(_Session):
    """Fake session that records the token of every node probe issued."""

    def __init__(self, files, probed):
        super().__init__(files)
        self._probed = probed

    def run(self, cypher, **params):
        if "labels(n)[0]" in cypher:  # node probe
            self._probed.append(params.get("w", ""))
        return super().run(cypher, **params)


class _CountingDriver(_Driver):
    def __init__(self, files):
        super().__init__(files)
        self.probed: list[str] = []

    def session(self, **_):
        return _CountingSession(self._files, self.probed)


def test_repeated_query_words_probe_once():
    """Audit pin: the query string is caller-controlled on HTTP/MCP — the
    same word repeated (in any case) must cost ONE probe set, not one per
    occurrence."""
    driver = _CountingDriver(FILES)
    retr = MemgraphRetriever(uri="bolt://x", driver=driver)

    out = retr.search("note.md NOTE.MD note.md", include_invalidated=True)

    assert [r.payload["name"] for r in out] == ["note.md"]
    assert driver.probed == ["note.md"]


def test_distinct_query_tokens_are_capped():
    """Audit pin: a pathological word soup is bounded at _MAX_QUERY_TOKENS
    distinct tokens (order-preserving) — the rest never reach the backend."""
    driver = _CountingDriver(FILES)
    retr = MemgraphRetriever(uri="bolt://x", driver=driver)
    words = [f"word{i:02d}" for i in range(30)]

    retr.search(" ".join(words), include_invalidated=True)

    # An unmatched token legitimately retries several probe shapes — the
    # guard bounds DISTINCT tokens reaching the backend, not probe shapes.
    assert list(dict.fromkeys(driver.probed)) == words[:16]  # _MAX_QUERY_TOKENS


def test_probe_circuit_breaker_trips_on_first_store_failure():
    """Audit pin: a store outage costs ONE failed probe per recall — the
    breaker fails every remaining candidate closed instead of retrying the
    dead store serially up to the full probe allowance."""
    files = {("note.md", "/corpus/a"): [], ("report.md", "/corpus/a"): []}
    calls: list[dict] = []

    def broken_probe(filters, tenant, include_invalidated, as_of):
        calls.append(dict(filters))
        raise RuntimeError("store down")

    retr = _retriever(files, broken_probe)
    out = retr.search(
        "note.md report.md", filters={"project": "x"}, include_invalidated=True
    )

    assert out == []  # fail closed, never leaked
    assert len(calls) == 1  # second candidate never touched the store


def test_probe_breaker_spans_searches_sharing_a_recall_scope():
    """A tripped breaker in a shared recall scope survives into the next
    `search` call — under query expansion each variant re-runs the arm, and
    a store outage must cost one failed probe per PUBLIC recall, not one
    per variant."""
    calls: list[dict] = []

    def broken_probe(filters, tenant, include_invalidated, as_of):
        calls.append(dict(filters))
        raise RuntimeError("store down")

    retr = _retriever(FILES, broken_probe)
    scope: dict = {}
    for variant in ("note.md", "note.md please"):
        out = retr.search(
            variant,
            filters={"project": "x"},
            include_invalidated=True,
            recall_scope=scope,
        )
        assert out == []
    assert len(calls) == 1  # the second variant never touched the store


class _RunCountingSession(_Session):
    """Fake session that records EVERY Cypher round trip issued."""

    def __init__(self, files, runs):
        super().__init__(files)
        self._runs = runs

    def run(self, cypher, **params):
        self._runs.append(cypher[:40])
        return super().run(cypher, **params)


class _RunCountingDriver(_Driver):
    def __init__(self, files):
        super().__init__(files)
        self.runs: list[str] = []

    def session(self, **_):
        return _RunCountingSession(self._files, self.runs)


def test_recall_cypher_calls_bounded_under_expansion():
    """Audit regression (re-audit blocker): a 4-variant expansion with fully
    disjoint 16-token variants used to issue 4 x 16 x 3 = 192 sequential
    Cypher probes (theoretical max 256 with the digit shape) — the per-call
    token cap did nothing across variants. One shared per-recall ceiling now
    bounds the total; variants after the allowance add nothing."""
    from mnemostack.recall import Recaller

    driver = _RunCountingDriver(FILES)
    retr = MemgraphRetriever(uri="bolt://x", driver=driver)
    recaller = Recaller(
        retrievers=[retr], query_expansion=True, expansion_llm=object()
    )
    variants = [
        " ".join(f"tok{v}x{i:02d}" for i in range(16)) for v in range(4)
    ]
    recaller._query_expansion_cache[variants[0]] = variants[1:]

    recaller.recall(variants[0], include_invalidated=True)

    # Deterministic setup: 16 x 3 shapes for variants 1-2 spends the probe
    # ceiling exactly; variants 3-4 add zero. A looser <= bound would let an
    # under-spending accounting bug slip through.
    assert len(driver.runs) == 96  # _MAX_RECALL_NODE_PROBES, not 192


def test_configured_max_nodes_above_pool_cap_is_honored_on_a_single_call():
    """Round-2 pin: the shared discovery pool guards expansion variants —
    it must never clamp an explicitly tuned max_nodes below its historical
    single-call result count."""
    files = {("doc.md", f"/r{i:03d}"): [] for i in range(600)}
    retr = MemgraphRetriever(uri="bolt://x", driver=_Driver(files), max_nodes=600)

    out = retr.search("doc.md", limit=600, include_invalidated=True)

    assert len(out) == 600


def test_digit_tokens_reach_all_four_probe_shapes():
    """The probe ceiling's worst-case arithmetic (16 tokens x 4 shapes = 64)
    is only reachable via numeric contact-id tokens — pin that a >=6-digit
    unmatched token costs exactly 4 probes."""
    driver = _RunCountingDriver(FILES)
    retr = MemgraphRetriever(uri="bolt://x", driver=driver)

    retr.search("1234567", include_invalidated=True)

    assert len(driver.runs) == 4  # contact-id, exact, handle, contains


def test_expansion_variants_share_the_probe_result_cache():
    """A token one variant already probed is served from the per-recall
    cache: identical rows for fusion, zero repeat backend requests."""
    from mnemostack.recall import Recaller

    driver = _CountingDriver(FILES)
    retr = MemgraphRetriever(uri="bolt://x", driver=driver)
    recaller = Recaller(
        retrievers=[retr], query_expansion=True, expansion_llm=object()
    )
    recaller._query_expansion_cache["note.md"] = ["note.md doc"]

    out = recaller.recall("note.md", include_invalidated=True)

    assert [r.payload["name"] for r in out] == ["note.md"]
    assert driver.probed.count("note.md") == 1  # cached for variant 2


def test_candidate_traversal_shares_the_recall_pool(monkeypatch):
    """The discovery-candidate allowance is one pool for the whole recall:
    a variant that spent it leaves nothing for later variants to traverse
    (fail closed) instead of each variant re-drawing a fresh 500."""
    import mnemostack.recall.retrievers as retrievers_mod

    monkeypatch.setattr(retrievers_mod, "_FILTERED_CANDIDATE_BUDGET", 2)
    files = {(f"doc{i}.md", "/corpus/a"): [] for i in range(6)}
    calls: list[dict] = []

    def probe(filters, tenant, include_invalidated, as_of):
        calls.append(dict(filters))
        return False  # nothing attributes — every traversed candidate probes

    # max_nodes=1 keeps this call's node_budget (3) close to the shrunken
    # pool: the shared allowance is max(pool, node_budget) = 3 traversals
    # for the WHOLE recall.
    retr = MemgraphRetriever(
        uri="bolt://x", driver=_Driver(files), max_nodes=1, chunk_filter_probe=probe
    )
    scope: dict = {}
    query = "doc0.md doc1.md doc2.md doc3.md doc4.md doc5.md"
    for _variant in range(2):
        out = retr.search(
            query,
            filters={"project": "x"},
            include_invalidated=True,
            recall_scope=scope,
        )
        assert out == []
    assert len(calls) == 3  # allowance of 3, spent by variant 1; variant 2 adds 0


def test_query_expansion_shares_the_probe_breaker_across_variants():
    """End-to-end pin through the recaller: with query_expansion=True the
    recaller's per-recall scope carries ONE probe budget across variants."""
    from mnemostack.recall import Recaller

    calls: list[dict] = []

    def broken_probe(filters, tenant, include_invalidated, as_of):
        calls.append(dict(filters))
        raise RuntimeError("store down")

    retr = _retriever(FILES, broken_probe)
    recaller = Recaller(
        retrievers=[retr], query_expansion=True, expansion_llm=object()
    )
    # Pre-seed the expansion cache so no LLM call happens: the public query
    # plus two variants — three serial runs of the graph arm.
    recaller._query_expansion_cache["note.md"] = ["note.md doc", "note.md file"]

    out = recaller.recall(
        "note.md", filters={"project": "x"}, include_invalidated=True
    )

    assert out == []
    assert len(calls) == 1  # one failed probe for the whole recall
