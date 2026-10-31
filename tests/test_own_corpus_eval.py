"""A/B evaluation must retain failures, isolate state, and export scoped data."""

import json
import runpy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient, models

from mnemostack.vector import VectorStore

api = runpy.run_path(str(Path(__file__).resolve().parents[1] / "benchmarks/own_corpus.py"))
compare = api["compare"]
NOW = datetime(2026, 10, 31, tzinfo=timezone.utc)
IDS = ("aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa", "bbbbbbbb-bbbb-4bbb-bbbb-bbbbbbbbbbbb",
       "cccccccc-cccc-4ccc-cccc-cccccccccccc")


@pytest.fixture
def store():
    client = QdrantClient(":memory:")
    store = VectorStore.__new__(VectorStore)
    store.client, store.collection, store.dimension = client, "eval", 2
    store.text_key, store.sparse_text, store.search_params = "text", False, None
    client.create_collection("eval", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
    store.ensure_text_index(scoring=True)
    client.upsert("eval", points=[
        models.PointStruct(id=IDS[0], vector=[1.0, 0.0], payload={"text": "orchard token alpha", "tenant_id": "one", "source": "fixture"}),
        models.PointStruct(id=IDS[1], vector=[0.8, 0.2], payload={"text": "orchard token beta", "tenant_id": "one", "source": "fixture"}),
        models.PointStruct(id=IDS[2], vector=[1.0, 0.0], payload={"text": "FOREIGN PRIVATE TEXT", "tenant_id": "two"}),
    ])
    yield store
    client.close()


def case(expected=None, **kwargs):
    return {"query_id": "q1", "query": "orchard", "expected_ids": [IDS[0], IDS[1]] if expected is None else expected, **kwargs}


def config(**kwargs):
    return {"vector": False, "pipeline": "off", **kwargs}


def test_real_canonical_flow_and_paired_metrics(store):
    report = compare([case()], {"A": config(limit=1), "B": config(limit=2)}, store=store, tenant="one", now=NOW)
    assert report["corpus_unchanged"] and report["valid_comparison"]
    assert report["summary"]["A"]["recall"] == 0.5
    assert report["summary"]["B"]["recall"] == 1.0
    assert report["cases"][0]["changed"]
    assert report["summary"]["A"]["mrr"] == 1.0


def test_retriever_order_matches_the_existing_serving_composition(store):
    recaller, _, _ = api["components"](store, SimpleNamespace(dimension=2), None,
                                       api["validate_config"]({}), "one")
    assert [type(r).__name__ for r in recaller.retrievers] == [
        "VectorRetriever", "QdrantNativeBM25Retriever", "TemporalRetriever"]


def test_foreign_expected_id_is_not_an_authorization_or_export(store):
    report = compare([case([IDS[2]])], {"A": config(), "B": config()}, store=store, tenant="one", now=NOW)
    serialized = json.dumps(report)
    assert IDS[2] not in serialized
    assert "FOREIGN" not in serialized
    assert "orchard" not in serialized
    assert report["cases"][0]["unresolved_expected_count"] == 1
    assert report["summary"]["A"]["hit_rate"] == 0


def test_empty_expected_ids_are_not_automatic_success(store):
    report = compare([case([])], {"A": config(), "B": config()}, store=store, tenant="one", now=NOW)
    assert report["summary"]["A"]["positive_n"] == 0
    assert report["summary"]["A"]["mrr"] is None
    assert report["summary"]["A"]["negative_queries_with_results"] == 1


def test_faults_remain_in_denominator_and_are_visible(store, monkeypatch):
    from mnemostack.recall import QdrantNativeBM25Retriever

    def fail(*args, **kwargs):
        raise RuntimeError("PRIVATE query token must not be exported")
    monkeypatch.setattr(QdrantNativeBM25Retriever, "search", fail)
    report = compare([case()], {"A": config(), "B": config()}, store=store, tenant="one", now=NOW)
    assert report["summary"]["A"]["positive_n"] == 1
    assert report["summary"]["A"]["hit_rate"] == 0
    assert report["summary"]["A"]["degraded_queries"] == 1
    assert "PRIVATE" not in json.dumps(report)


def test_state_order_independence_and_no_persistent_writes(store):
    before = api["snapshot"](store, "one")
    configs = {"A": config(), "B": config(pipeline="full", token_budget=1)}
    first = compare([case()], configs, store=store, tenant="one", now=NOW)
    second = compare([case()], configs, store=store, tenant="one", now=NOW, reverse=True)
    for arm in ("A", "B"):
        assert first["cases"][0]["arms"][arm]["ids"] == second["cases"][0]["arms"][arm]["ids"]
    assert api["snapshot"](store, "one") == before
    assert first["cases"][0]["arms"]["B"]["loss_report"]["losses"]


@pytest.mark.parametrize("bad", [{"limit": 0}, {"vector": "yes"}, {"unknown": 1}, {"limit": 101}, {"vector_floor": -1}])
def test_config_validation(bad):
    with pytest.raises(ValueError):
        api["validate_config"](bad)


def test_text_export_is_explicit_and_scoped(store):
    report = compare([case()], {"A": config(), "B": config()}, store=store, tenant="one", now=NOW, include_text=True)
    assert report["cases"][0]["query"] == "orchard"
    assert report["cases"][0]["arms"]["A"]["texts"]
    assert "FOREIGN" not in json.dumps(report)


def test_no_storage_mutations_are_called(store, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("evaluation attempted a storage mutation")

    for method in ("upsert", "delete", "set_payload", "overwrite_payload", "create_collection", "create_payload_index"):
        monkeypatch.setattr(store.client, method, forbidden)
    report = compare([case()], {"A": config(), "B": config(pipeline="full")}, store=store, tenant="one", now=NOW)
    assert report["corpus_unchanged"]


def test_query_vectors_are_shared_without_touching_provider_afterwards(store):
    from mnemostack.embeddings.base import EmbeddingProvider

    class Embedding(EmbeddingProvider):
        name, dimension = "fixture", 2
        calls = 0

        def embed(self, text):
            self.calls += 1
            return [1.0, 0.0]

        def embed_batch(self, texts):
            return [self.embed(t) for t in texts]

    embedding = Embedding()
    report = compare([case()], {"A": config(vector=True), "B": config(vector=True)}, store=store,
                     tenant="one", embedding=embedding, now=NOW)
    assert embedding.calls == 1
    assert "embed_query" not in vars(embedding)
    assert report["query_vectors_sha256"]
    assert not report["summary"]["A"]["degraded_queries"]


def test_reranker_failure_is_not_reported_as_success(store):
    from mnemostack.llm.base import LLMResponse

    class BadLLM:
        def generate(self, *args, **kwargs):
            return LLMResponse(text="PRIVATE unrelated answer")

    report = compare([case()], {"A": config(), "B": config(rerank=True)}, store=store,
                     tenant="one", llm=BadLLM(), now=NOW)
    assert report["summary"]["B"]["degraded_queries"] == 1
    assert "reranker:fallback" in report["cases"][0]["arms"]["B"]["degraded"]
    assert "PRIVATE" not in json.dumps(report)


def test_snapshot_supports_existing_named_sparse_vectors():
    client = QdrantClient(":memory:")
    try:
        client.create_collection("sparse", vectors_config={"": models.VectorParams(size=2, distance=models.Distance.COSINE)},
                                 sparse_vectors_config={"text-sparse": models.SparseVectorParams()})
        client.upsert("sparse", points=[models.PointStruct(id=1, vector={"": [1.0, 0.0],
                      "text-sparse": models.SparseVector(indices=[1], values=[1.0])},
                      payload={"tenant_id": "one", "text": "orchard"})])
        fingerprint, ids = api["snapshot"](SimpleNamespace(client=client, collection="sparse"), "one")
        assert fingerprint and ids == {"1"}
    finally:
        client.close()


def test_vector_floor_results_are_not_silently_truncated_in_report(store):
    from mnemostack.embeddings.base import EmbeddingProvider

    class Embedding(EmbeddingProvider):
        name, dimension = "fixture", 2

        def embed(self, text):
            return [1.0, 0.0]

        def embed_batch(self, texts):
            return [self.embed(t) for t in texts]

    cfg = config(vector=True, vector_floor=2, limit=1)
    report = compare([case()], {"A": cfg, "B": cfg}, store=store, tenant="one", embedding=Embedding(), now=NOW)
    result = report["cases"][0]["arms"]["A"]
    assert result["returned_count"] == len(result["ids"]) == 2
    assert result["metrics"]["recall"] == 0.5
    assert result["returned_metrics"]["recall"] == 1.0


@pytest.mark.parametrize("use_vector", [False, True])
def test_unused_or_failing_embedding_is_not_modified_or_its_error_masked(store, use_vector):
    from mnemostack.embeddings.base import EmbeddingProvider

    class Embedding(EmbeddingProvider):
        name, dimension = "fixture", 2

        def embed(self, text):
            raise RuntimeError("vector preparation failed")

        def embed_batch(self, texts):
            raise RuntimeError("vector preparation failed")

    embedding = Embedding()
    cfg = config(vector=use_vector)
    if use_vector:
        with pytest.raises(RuntimeError, match="vector preparation failed"):
            compare([case()], {"A": cfg, "B": cfg}, store=store, tenant="one", embedding=embedding, now=NOW)
    else:
        assert compare([case()], {"A": cfg, "B": cfg}, store=store, tenant="one", embedding=embedding, now=NOW)["valid_comparison"]
    assert "embed_query" not in vars(embedding)


def test_failed_open_warmup_is_not_a_clean_warm_sample(store, monkeypatch):
    from mnemostack.recall import QdrantNativeBM25Retriever

    original = QdrantNativeBM25Retriever.search

    def transient(self, *args, **kwargs):
        if not getattr(self, "attempted", False):
            self.attempted = True
            raise RuntimeError("temporary connection failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(QdrantNativeBM25Retriever, "search", transient)
    report = compare([case()], {"A": config(), "B": config()}, store=store,
                     tenant="one", now=NOW, cache_mode="warm")
    for arm in ("A", "B"):
        result = report["cases"][0]["arms"][arm]
        assert not result["warmup_ok"] and result["warmup_degraded"]
        assert result["error_type"] == "WarmupError"
        assert result["latency_ms"] is None
        assert report["summary"][arm]["timed_n"] == 0
        assert report["summary"][arm]["hit_rate"] == 0
        assert report["summary"][arm]["degraded_queries"] == 1


def test_cli_report_created_concurrently_is_not_overwritten(tmp_path, monkeypatch):
    import mnemostack.vector

    query_file, config_file, output = (tmp_path / name for name in ("q.json", "config.json", "report.json"))
    query_file.write_text(json.dumps([case()]))
    config_file.write_text(json.dumps(config()))

    def concurrent_writer(*args, **kwargs):
        output.write_text("other run's completed report")
        return {"valid_comparison": True, "summary": {arm: {"errors": 0, "degraded_queries": 0} for arm in ("A", "B")}}

    monkeypatch.setitem(api["main"].__globals__, "compare", concurrent_writer)
    monkeypatch.setattr(mnemostack.vector, "VectorStore", lambda *a, **kw: SimpleNamespace(client=SimpleNamespace(close=lambda: None)))
    monkeypatch.setattr("sys.argv", ["own_corpus.py", "--queries", str(query_file), "--config-a", str(config_file),
                                   "--config-b", str(config_file), "--collection", "fixture", "--tenant", "one", "--output", str(output)])
    assert api["main"]() == 2
    assert output.read_text() == "other run's completed report"
