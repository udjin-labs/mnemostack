"""Scalable text search over Qdrant — lexical-gated dense + sparse scoring.

Contract under test: both arms replace the in-process BM25 at scale — the
lexical gate filters inside Qdrant (MatchText) with dense ranking, the sparse
arm scores lexically on the server (tf values + IDF modifier) — and both are
payload-schema-aware, tenant-safe, and selectable via recall.text_search.
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance

from mnemostack.config import TEXT_SEARCH_MODES, resolve_text_search_mode
from mnemostack.recall.retrievers import QdrantSparseRetriever, QdrantTextRetriever
from mnemostack.vector import VectorStore
from mnemostack.vector.sparse import SparseTextEncoder, token_index

_V1 = [1.0, 0.0, 0.0, 0.0]
_V2 = [0.0, 1.0, 0.0, 0.0]


class _FakeEmbedder:
    dimension = 4
    name = "fake:text"

    def embed(self, text):
        return _V1

    def embed_batch(self, texts):
        return [_V1 for _ in texts]


def _store(sparse: bool = False, text_key: str = "text") -> VectorStore:
    s = VectorStore.__new__(VectorStore)
    s.collection = "ts"
    s.dimension = 4
    s.distance = Distance.COSINE
    s.client = QdrantClient(":memory:")
    s.sparse_text = sparse
    s.text_key = text_key
    s._sparse_encoder = SparseTextEncoder() if sparse else None
    s.ensure_collection()
    return s


def _seed(s: VectorStore, text_key: str = "text") -> None:
    s.upsert(1, _V1, {text_key: "postgres backup restore procedure"}, tenant="acme")
    s.upsert(2, _V2, {text_key: "kubernetes ingress setup"}, tenant="acme")
    s.upsert(3, _V1, {text_key: "postgres tuning notes"}, tenant="globex")


# ---------- sparse encoder ----------


def test_sparse_encoder_document_and_query_shapes():
    enc = SparseTextEncoder()
    idx, vals = enc.encode_document("alpha beta alpha")
    assert len(idx) == 2 and sorted(vals) == [1.0, 2.0]  # tf per distinct token
    qi, qv = enc.encode_query("alpha beta beta")
    assert len(qi) == 2 and qv == [1.0, 1.0]  # queries are unweighted
    assert enc.encode_document("") == ([], [])
    assert enc.encode_query("   ") == ([], [])
    assert token_index("alpha") == token_index("alpha")  # stable


# ---------- vector store: sparse space ----------


def test_sparse_store_roundtrip_and_ranking():
    s = _store(sparse=True)
    _seed(s)
    hits = s.sparse_search("postgres backup", limit=5)
    assert [h.id for h in hits][:2] == [1, 3]  # both match "postgres", 1 also "backup"
    assert hits[0].score > hits[1].score
    # tenant boundary holds on the sparse arm too
    acme = s.sparse_search("postgres", limit=5, tenant="acme")
    assert [h.id for h in acme] == [1]


def test_sparse_search_requires_sparse_store():
    s = _store(sparse=False)
    with pytest.raises(RuntimeError, match="sparse_text=True"):
        s.sparse_search("anything")


def test_sparse_search_empty_query_tokens_is_empty():
    s = _store(sparse=True)
    _seed(s)
    assert s.sparse_search("!!! ...") == []


# ---------- vector store: lexical gate ----------


def test_text_any_gate_restricts_dense_search():
    s = _store()
    _seed(s)
    hits = s.search(_V1, limit=5, text_any=["postgres"])
    assert {h.id for h in hits} == {1, 3}
    # composes with the tenant boundary
    hits = s.search(_V1, limit=5, text_any=["postgres"], tenant="acme")
    assert [h.id for h in hits] == [1]
    # any-of semantics: either token admits
    hits = s.search(_V1, limit=5, text_any=["kubernetes", "backup"])
    assert {h.id for h in hits} == {1, 2}


def test_ensure_text_index_is_idempotent():
    s = _store()
    s.ensure_text_index()
    s.ensure_text_index("text")  # second run must not raise


# ---------- retrievers ----------


def test_qdrant_text_retriever_gates_and_ranks():
    s = _store(text_key="content")
    _seed(s, text_key="content")
    r = QdrantTextRetriever(
        embedding=_FakeEmbedder(), vector_store=s, text_key="content"
    )
    hits = r.search("how do I restore the postgres backup", limit=5)
    assert hits and all("postgres" in h.text or "backup" in h.text for h in hits)
    assert all(h.sources == ["qdrant_text"] for h in hits)
    assert 2 not in {h.id for h in hits}  # kubernetes chunk gated out


def test_qdrant_text_retriever_gate_tokens():
    r = QdrantTextRetriever(embedding=_FakeEmbedder(), vector_store=_store())
    toks = r._gate_tokens("what is the IP 10.0.0.5 of the staging postgres server")
    assert "10.0.0.5" in toks  # exact technical token always in the gate
    assert "the" not in toks and "is" not in toks  # stopwords never
    assert len(toks) <= r.max_gate_tokens
    assert r.explain_empty("что это") is not None  # nothing salient -> named reason
    assert r.search("что это", limit=5) == []


def test_qdrant_sparse_retriever_end_to_end():
    s = _store(sparse=True, text_key="content")
    _seed(s, text_key="content")
    r = QdrantSparseRetriever(vector_store=s, text_key="content")
    hits = r.search("postgres backup", limit=5)
    assert [h.id for h in hits][:1] == [1]
    assert all(h.sources == ["sparse"] for h in hits)
    assert hits[0].text.startswith("postgres backup")


def test_qdrant_sparse_retriever_requires_sparse_store():
    with pytest.raises(ValueError, match="sparse_text=True"):
        QdrantSparseRetriever(vector_store=_store(sparse=False))


def test_retrievers_declare_the_schema():
    # Both arms participate in the per-field schema derivation from #124.
    from mnemostack.recall.recaller import Recaller

    s = _store(sparse=True, text_key="content")
    r = QdrantSparseRetriever(
        vector_store=s, text_key="content", timestamp_key="updated_at",
        timestamp_format="epoch",
    )
    rec = Recaller(retrievers=[r])
    assert (rec.text_key, rec.timestamp_key, rec.timestamp_format) == (
        "content", "updated_at", "epoch",
    )


# ---------- config / wiring ----------


def test_resolve_text_search_mode():
    assert resolve_text_search_mode("auto", ["/docs"]) == "bm25"
    assert resolve_text_search_mode("auto", []) == "off"
    assert resolve_text_search_mode("sparse", []) == "sparse"
    with pytest.raises(ValueError, match="text_search"):
        resolve_text_search_mode("fulltext", [])
    assert set(TEXT_SEARCH_MODES) >= {"auto", "off", "bm25", "qdrant_bm25", "lexical", "sparse"}


def test_text_search_env_override(monkeypatch):
    from mnemostack.config import Config

    monkeypatch.setenv("MNEMOSTACK_TEXT_SEARCH", "lexical")
    assert Config.load().recall.text_search == "lexical"


def test_doctor_flags_invalid_text_search(monkeypatch, capsys):
    import argparse

    import mnemostack.cli as cli

    monkeypatch.setenv("MNEMOSTACK_TEXT_SEARCH", "fulltext")
    rc = cli.cmd_doctor(
        argparse.Namespace(
            json=True, provider="gemini", embedding_model=None,
            qdrant="http://localhost:1", collection="mt",
            memgraph_uri=None, graph_timeout=1.0, timeout=1,
        )
    )
    out = capsys.readouterr().out
    assert "fulltext" in out and rc == 2


def test_cli_build_recaller_lexical_mode(monkeypatch):
    import argparse

    import mnemostack.cli as cli

    monkeypatch.setenv("MNEMOSTACK_TEXT_SEARCH", "lexical")
    cli._text_search_mode.cache_clear()
    cli._payload_schema.cache_clear()
    store = _store()
    rec = cli._build_recaller(
        argparse.Namespace(collection="ts", qdrant="http://x", bm25_path=[]),
        _FakeEmbedder(),
        store,
    )
    names = {type(r).__name__ for r in rec.retrievers}
    assert "QdrantTextRetriever" in names and "BM25Retriever" not in names
    cli._text_search_mode.cache_clear()
    cli._payload_schema.cache_clear()


def test_cli_text_index_command(monkeypatch, capsys):
    import argparse

    import mnemostack.cli as cli

    store = _store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    cli._payload_schema.cache_clear()
    rc = cli.cmd_text_index(argparse.Namespace(collection="ts", qdrant="http://x"))
    assert rc == 0
    assert "full-text index ensured" in capsys.readouterr().out
