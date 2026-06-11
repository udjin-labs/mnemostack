"""Adversarial isolation tests for recall filters.

The contract under test is isolation, not ranking: a recall scoped with
`filters=` must never return a point outside the filtered scope through ANY
retriever. Before this suite, BM25 silently ignored filters — in a
multi-tenant deployment the fused output mixed foreign tenants' chunks in.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance

from mnemostack.embeddings.base import EmbeddingProvider
from mnemostack.recall import BM25Doc, Recaller, payload_matches, recall_flow
from mnemostack.recall.retrievers import (
    BM25Retriever,
    MemgraphRetriever,
    VectorRetriever,
)
from mnemostack.vector import VectorStore


class FakeEmbedder(EmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 8

    @property
    def name(self) -> str:
        return "fake"

    def embed(self, text: str) -> list[float]:
        buckets = [0.0] * 8
        for c in text.lower():
            if c.isalpha():
                buckets[ord(c) % 8] += 1
        s = sum(buckets) or 1.0
        return [b / s for b in buckets]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


# ---------- payload_matches ----------


def test_payload_matches_exact_and_missing_key():
    assert payload_matches({"tenant": "a"}, {"tenant": "a"})
    assert not payload_matches({"tenant": "b"}, {"tenant": "a"})
    # a point that cannot be attributed to the scope must not pass it
    assert not payload_matches({}, {"tenant": "a"})
    assert not payload_matches(None, {"tenant": "a"})


def test_payload_matches_ranges():
    p = {"timestamp": "2026-06-05T10:00:00Z", "count": 7}
    assert payload_matches(p, {"timestamp": {"gte": "2026-06-01", "lte": "2026-06-30"}})
    assert not payload_matches(p, {"timestamp": {"gte": "2026-06-10"}})
    assert payload_matches(p, {"count": {"gte": 5, "lte": 10}})
    assert not payload_matches(p, {"count": {"lte": 5}})
    # incomparable types cannot be proven inside the range — excluded
    assert not payload_matches({"count": "seven"}, {"count": {"gte": 5}})


def test_payload_matches_empty_filters_passes_everything():
    assert payload_matches({}, None)
    assert payload_matches({"x": 1}, {})


# ---------- BM25 ----------


def _bm25_two_tenants() -> BM25Retriever:
    docs = [
        BM25Doc(id="a1", text="quarterly report numbers", payload={"tenant": "a"}),
        BM25Doc(id="a2", text="meeting notes for the report", payload={"tenant": "a"}),
        BM25Doc(id="b1", text="report report report report", payload={"tenant": "b"}),
        BM25Doc(id="b2", text="the secret report of tenant b", payload={"tenant": "b"}),
        BM25Doc(id="b3", text="report draft report final", payload={"tenant": "b"}),
    ]
    return BM25Retriever(docs)


def test_bm25_filters_exclude_foreign_tenant():
    retr = _bm25_two_tenants()

    results = retr.search("report", limit=10, filters={"tenant": "a"})

    assert results
    assert all(r.payload["tenant"] == "a" for r in results)


def test_bm25_filters_before_topk_cut():
    """Foreign docs out-score tenant A's on this query; a post-cut filter
    would return fewer than the available matching docs."""
    retr = _bm25_two_tenants()

    results = retr.search("report", limit=2, filters={"tenant": "a"})

    assert len(results) == 2  # both A docs found despite B's higher scores
    assert {r.id for r in results} == {"a1", "a2"}


def test_bm25_no_filters_unchanged():
    retr = _bm25_two_tenants()
    results = retr.search("report", limit=10)
    assert {r.payload["tenant"] for r in results} == {"a", "b"}


# ---------- Memgraph ----------


def test_memgraph_excluded_under_filters():
    """Graph nodes carry no chunk payload — they cannot be attributed to the
    filtered scope, so under filters the retriever must contribute nothing."""
    driver = MagicMock()
    retr = MemgraphRetriever(driver=driver)

    assert retr.search("some entity name", filters={"tenant": "a"}) == []
    driver.session.assert_not_called()


# ---------- fused isolation ----------


@pytest.fixture
def fused_recaller():
    embedder = FakeEmbedder()
    store = VectorStore.__new__(VectorStore)
    store.collection = "test"
    store.dimension = embedder.dimension
    store.distance = Distance.COSINE
    store.client = QdrantClient(":memory:")
    store.ensure_collection()

    corpus = [
        ("11111111-0000-0000-0000-000000000001", "tenant a quarterly report", "a"),
        ("11111111-0000-0000-0000-000000000002", "tenant a planning notes", "a"),
        ("22222222-0000-0000-0000-000000000001", "tenant b quarterly report", "b"),
        ("22222222-0000-0000-0000-000000000002", "tenant b secret roadmap", "b"),
    ]
    bm25_docs = []
    for pid, text, tenant in corpus:
        payload = {"text": text, "tenant": tenant}
        store.upsert(pid, embedder.embed(text), payload)
        bm25_docs.append(BM25Doc(id=pid, text=text, payload=payload))

    recaller = Recaller(
        retrievers=[
            VectorRetriever(embedding=embedder, vector_store=store),
            BM25Retriever(bm25_docs),
        ]
    )
    return recaller


def test_fused_recall_never_leaks_foreign_tenant(fused_recaller):
    results = fused_recaller.recall("quarterly report", limit=10, filters={"tenant": "a"})

    assert results
    assert all(r.payload.get("tenant") == "a" for r in results)


def test_fused_recall_without_filters_sees_everything(fused_recaller):
    results = fused_recaller.recall("quarterly report", limit=10)
    assert {r.payload.get("tenant") for r in results} == {"a", "b"}


def test_recall_flow_threads_filters(fused_recaller):
    results = recall_flow(
        fused_recaller, "quarterly report", limit=10, filters={"tenant": "b"}
    )

    assert results
    assert all(r.payload.get("tenant") == "b" for r in results)


def test_payload_matches_array_any_element():
    """Qdrant matches array payloads when ANY element satisfies the
    condition — the local helper must mirror that, or filtered hybrid recall
    drops valid lexical candidates the vector retriever returns."""
    p = {"tags": ["urgent", "work"], "scores": [3, 9]}
    assert payload_matches(p, {"tags": "urgent"})
    assert payload_matches(p, {"tags": "work"})
    assert not payload_matches(p, {"tags": "personal"})
    assert payload_matches(p, {"scores": {"gte": 5}})
    assert not payload_matches(p, {"scores": {"gte": 10}})


def test_expansion_retry_stays_inside_filtered_scope():
    """A low-confidence first answer triggers the expansion retry; its
    vector searches must carry the same filters as the primary recall."""
    from unittest.mock import MagicMock

    from mnemostack.llm.base import LLMResponse
    from mnemostack.recall import AnswerGenerator

    llm = MagicMock()
    # draft answer abstains -> retry path engages; retry answer is confident
    llm.generate.side_effect = [
        LLMResponse(text="Not in memory."),
        LLMResponse(text="a confident retry answer"),
    ]
    expansion_llm = MagicMock()
    expansion_llm.generate.return_value = LLMResponse(
        text="variant one\nvariant two\nhypothetical answer"
    )
    def _mem(mid: str, text: str):
        attrs = {"id": mid, "text": text, "score": 0.9, "payload": {"tenant": "a"}, "sources": ["vector"]}
        return type("R", (), attrs)()

    recaller = MagicMock()
    recaller.embedding.embed_batch.return_value = [[0.1], [0.2], [0.3], [0.4]]
    recaller.search_many.return_value = [_mem("a", "in-scope memory")]

    gen = AnswerGenerator(
        llm=llm,
        recaller=recaller,
        retry_with_expansion=True,
        expansion_llm=expansion_llm,
        specificity_resolver=False,
        inference_retry=False,
        category_aware_prompts=False,
    )
    memory = type(
        "R", (), {"id": "m", "text": "some memory", "score": 0.5, "payload": {"tenant": "a"}, "sources": ["vector"]}
    )()

    gen.generate("a question", [memory], recall_filters={"tenant": "a"})

    assert recaller.search_many.call_args.kwargs["filters"] == {"tenant": "a"}


def test_legacy_recaller_path_filters_bm25():
    """Recaller(embedding_provider=..., vector_store=..., bm25_docs=...) uses
    the legacy fusion path — its BM25 arm must honor filters too."""
    embedder = FakeEmbedder()
    store = VectorStore.__new__(VectorStore)
    store.collection = "test"
    store.dimension = embedder.dimension
    store.distance = Distance.COSINE
    store.client = QdrantClient(":memory:")
    store.ensure_collection()

    corpus = [
        ("11111111-0000-0000-0000-00000000000a", "tenant a quarterly report", "a"),
        ("22222222-0000-0000-0000-00000000000b", "tenant b quarterly report", "b"),
    ]
    bm25_docs = []
    for pid, text, tenant in corpus:
        payload = {"text": text, "tenant": tenant}
        store.upsert(pid, embedder.embed(text), payload)
        bm25_docs.append(BM25Doc(id=pid, text=text, payload=payload))

    recaller = Recaller(
        embedding_provider=embedder, vector_store=store, bm25_docs=bm25_docs
    )

    results = recaller.recall("quarterly report", limit=10, filters={"tenant": "a"})

    assert results
    assert all(r.payload.get("tenant") == "a" for r in results)


# ---------- temporal window vs caller timestamp scope ----------


def _temporal_setup(window):
    from mnemostack.recall.retrievers import TemporalRetriever

    embedder = FakeEmbedder()
    store = VectorStore.__new__(VectorStore)
    store.collection = "test"
    store.dimension = embedder.dimension
    store.distance = Distance.COSINE
    store.client = QdrantClient(":memory:")
    store.ensure_collection()

    points = [
        ("33333333-0000-0000-0000-000000000001", "may meeting notes", "2026-05-10T12:00:00+00:00"),
        ("33333333-0000-0000-0000-000000000002", "june meeting notes", "2026-06-10T12:00:00+00:00"),
        ("33333333-0000-0000-0000-000000000003", "july meeting notes", "2026-07-10T12:00:00+00:00"),
    ]
    for pid, text, ts in points:
        store.upsert(pid, embedder.embed(text), {"text": text, "timestamp": ts})

    return TemporalRetriever(embedder, store, extractor=lambda _q: window)


def test_temporal_intersects_caller_timestamp_scope():
    """A caller timestamp filter must narrow the parsed query window, not be
    replaced by it — hits outside the caller's scope must not appear."""
    retr = _temporal_setup(("2026-05-01T00:00:00+00:00", "2026-07-31T23:59:59+00:00"))

    results = retr.search(
        "meeting notes",
        limit=10,
        filters={"timestamp": {"gte": "2026-06-01T00:00:00+00:00"}},
    )

    assert results
    timestamps = {r.payload["timestamp"] for r in results}
    assert "2026-05-10T12:00:00+00:00" not in timestamps  # outside caller scope
    assert "2026-06-10T12:00:00+00:00" in timestamps


def test_temporal_disjoint_scope_returns_nothing():
    retr = _temporal_setup(("2025-05-01T00:00:00+00:00", "2025-05-31T23:59:59+00:00"))

    results = retr.search(
        "meeting notes",
        limit=10,
        filters={"timestamp": {"gte": "2026-01-01T00:00:00+00:00"}},
    )

    assert results == []


def test_mca_prefilter_stays_inside_filtered_scope():
    """The exact-token (MCA) arm queries the BM25 corpus directly — with
    filters it must not inject out-of-scope docs into the RRF pool."""
    docs = [
        BM25Doc(
            id="a1",
            text="tenant a mentions ticket_4711 once",
            payload={"tenant": "a"},
        ),
        BM25Doc(
            id="b1",
            text="ticket_4711 ticket_4711 ticket_4711 in tenant b",
            payload={"tenant": "b"},
        ),
    ]
    embedder = FakeEmbedder()
    store = VectorStore.__new__(VectorStore)
    store.collection = "test"
    store.dimension = embedder.dimension
    store.distance = Distance.COSINE
    store.client = QdrantClient(":memory:")
    store.ensure_collection()

    recaller = Recaller(
        embedding_provider=embedder,
        vector_store=store,
        bm25_docs=docs,
        mca_prefilter=True,
    )

    results = recaller.recall("what about ticket_4711", limit=10, filters={"tenant": "a"})

    assert results
    assert all(r.payload.get("tenant") == "a" for r in results)
    mca_sourced = [r for r in results if "mca" in (r.sources or [])]
    assert all(r.payload.get("tenant") == "a" for r in mca_sourced)


def test_pipeline_appended_results_are_scoped_too(fused_recaller):
    """Pipeline stages (e.g. graph resurrection) can append candidates that
    never passed the filtered retrievers — recall_flow must enforce the
    caller's scope on the pipeline output as well."""

    class _ResurrectingPipeline:
        def apply(self, query, results):
            ghost = type(
                "R",
                (),
                {
                    "id": "ghost",
                    "text": "resurrected graph record",
                    "score": 9.9,
                    "payload": {"source": "memgraph"},  # no tenant attribution
                    "sources": ["memgraph"],
                },
            )()
            return [ghost, *results]

    results = recall_flow(
        fused_recaller,
        "quarterly report",
        limit=10,
        pipeline=_ResurrectingPipeline(),
        filters={"tenant": "a"},
    )

    assert results
    assert all(r.payload.get("tenant") == "a" for r in results)
    assert not any(r.id == "ghost" for r in results)

    # without filters the pipeline's contribution is untouched
    unfiltered = recall_flow(
        fused_recaller, "quarterly report", limit=10, pipeline=_ResurrectingPipeline()
    )
    assert any(r.id == "ghost" for r in unfiltered)
