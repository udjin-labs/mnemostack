"""End-to-end smoke tests — the minimal "does the stack work at all" set.

These exercise the real ingest → recall path, markdown collection, and the
stale-fact validity view against an **in-memory Qdrant** (`:memory:`) with a
deterministic fake embedder, so they need no external services and run in CI
and locally. They are the runnable form of the 1.0 smoke story in
`docs/deployment-quickstart.md`.

Run just the smoke set (target the file so pytest doesn't import the rest of the
tree, which needs the full ``.[dev]`` extra):

    pytest tests/test_smoke.py -q

Graph (Memgraph) retrieval needs a live server, so its smoke is opt-in and
skipped unless MNEMOSTACK_SMOKE_GRAPH_URI is set (see test_smoke_graph_*).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance

from mnemostack import IngestItem, Ingestor
from mnemostack.embeddings import EmbeddingProvider
from mnemostack.markdown import collect_markdown
from mnemostack.recall import BM25Doc, Recaller
from mnemostack.vector import VectorStore

pytestmark = pytest.mark.smoke


class _FakeEmbedder(EmbeddingProvider):
    """Deterministic 8-dim embedder (character-frequency buckets)."""

    @property
    def dimension(self) -> int:
        return 8

    @property
    def name(self) -> str:
        return "fake:smoke"

    def embed(self, text: str) -> list[float]:
        buckets = [0.0] * 8
        for c in text.lower():
            if c.isalpha():
                buckets[ord(c) % 8] += 1
        s = sum(buckets) or 1.0
        return [b / s for b in buckets]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


def _memory_store(collection: str = "smoke") -> VectorStore:
    store = VectorStore.__new__(VectorStore)
    store.collection = collection
    store.dimension = 8
    store.distance = Distance.COSINE
    store.client = QdrantClient(":memory:")
    store.ensure_collection()
    return store


_DOCS = [
    "Python is a programming language used for data science and machine learning",
    "The Qdrant vector database supports similarity search at scale",
    "Machine learning models need large amounts of training data",
    "Memgraph is a graph database compatible with Cypher queries",
    "Hybrid retrieval combines BM25 lexical search with dense vector search",
]


def test_smoke_ingest_then_recall():
    """Ingest through the real Ingestor, then recall — the core happy path."""
    emb = _FakeEmbedder()
    store = _memory_store()
    ingestor = Ingestor(embedding=emb, vector_store=store, batch_size=2)

    items = [IngestItem(text=t, source=f"doc-{i}") for i, t in enumerate(_DOCS)]
    stats = ingestor.ingest(items)

    assert stats.upserted == len(_DOCS)
    assert stats.failed == 0
    assert store.count() == len(_DOCS)

    # Vector path, end-to-end and deterministic regardless of embedder quality:
    # an exact-text query embeds to the same vector as its own chunk (cosine 1.0),
    # so that chunk must rank top-1. This proves ingest stamped the text payload
    # and vector search reads it back — the part a fake embedder can smoke.
    top = store.search(emb.embed(_DOCS[3]), limit=1)
    assert top, "vector search returned nothing on a populated store"
    assert top[0].payload["text"] == _DOCS[3]

    # Hybrid recall path runs and returns relevant results. (The lexical BM25 leg
    # carries the semantic match here — the fake embedder isn't semantic — so this
    # asserts the pipeline fuses and returns, not vector ranking quality.)
    recaller = Recaller(
        embedding_provider=emb,
        vector_store=store,
        bm25_docs=[BM25Doc(id=i, text=t) for i, t in enumerate(_DOCS)],
    )
    results = recaller.recall("graph database with Cypher", limit=5)
    assert results, "recall returned nothing on a populated store"
    joined = " ".join(getattr(r, "text", "") for r in results).lower()
    assert "memgraph" in joined or "graph database" in joined


def test_smoke_stale_facts_hidden_by_default():
    """A fact marked stale is hidden from the default validity view."""
    emb = _FakeEmbedder()
    store = _memory_store()
    store.upsert(1, emb.embed("current fact about vector search"), {"text": "current"})
    store.upsert(2, emb.embed("current fact about vector search"), {"text": "stale"})

    store.invalidate(2)

    hits = store.search(emb.embed("vector search"), limit=10, hide_invalidated=True)
    ids = {h.id for h in hits}
    assert 1 in ids
    assert 2 not in ids
    # Without the validity view, the stale one is still retrievable.
    all_ids = {h.id for h in store.search(emb.embed("vector search"), limit=10)}
    assert {1, 2} <= all_ids


def test_smoke_markdown_collection(tmp_path: Path):
    """Markdown collection folds frontmatter into payload and resolves links."""
    (tmp_path / "alpha.md").write_text(
        "---\ntopic: retrieval\n---\n\n# Alpha\n\nSee [[beta]] for details.\n",
        encoding="utf-8",
    )
    (tmp_path / "beta.md").write_text("# Beta\n\nThe linked note.\n", encoding="utf-8")

    coll = collect_markdown(tmp_path)

    assert coll.files == 2
    assert coll.chunks, "no chunks produced"
    # Frontmatter is a filterable payload field.
    assert any(c.payload.get("topic") == "retrieval" for c in coll.chunks)
    # The [[beta]] wikilink resolves to the sibling note.
    resolved = [(e.source, e.target) for e in coll.edges if e.resolved]
    assert ("alpha.md", "beta.md") in resolved


@pytest.mark.skipif(
    not os.environ.get("MNEMOSTACK_SMOKE_GRAPH_URI"),
    reason="set MNEMOSTACK_SMOKE_GRAPH_URI to run the graph smoke against a live Memgraph",
)
def test_smoke_graph_reachable():
    """Opt-in: a live Memgraph is reachable and answers RETURN 1."""
    from mnemostack.graph.factory import make_graph_store

    store = make_graph_store(
        os.environ["MNEMOSTACK_SMOKE_GRAPH_URI"],
        timeout=5.0,
        user=os.environ.get("MNEMOSTACK_GRAPH_USER", ""),
        password=os.environ.get("MNEMOSTACK_GRAPH_PASSWORD", ""),
        database=os.environ.get("MNEMOSTACK_GRAPH_DATABASE") or None,
    )
    try:
        ok, msg = store.health_check()
        assert ok, f"graph unreachable: {msg}"
    finally:
        store.close()
