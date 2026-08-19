"""Remote write surface: POST /memories, POST /triples, and their shared
ingest helpers (validation, chunk expansion, store-backed dedup, tenant
stamping). The MCP `mnemostack_remember` counterpart is covered in
test_mcp.py; both surfaces share the same validator/ingest path pinned here.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from mnemostack.ingest import (
    REMOTE_CHUNK_SIZE,
    REMOTE_MAX_CHUNKS_PER_REQUEST,
    REMOTE_MAX_TEXT_CHARS,
    IngestItem,
    RemoteRequestTooLarge,
    expand_remote_items,
    ingest_remote_items,
    reserved_metadata_keys,
    stable_chunk_id,
    validate_remote_item,
)
from mnemostack.server import ServerConfig, build_app
from mnemostack.vector import VectorStore


class _CountingEmbedding:
    """Deterministic 3-dim embedder that counts every embedded text.

    Texts containing the marker "|POISON|" embed to an empty vector, which
    the ingest pipeline records as a per-item failure.
    """

    dimension = 3

    def __init__(self):
        self.embedded: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.embedded.append(text)
        if "|POISON|" in text:
            return []
        h = abs(hash(text))
        return [(h % 97) / 97.0, (h % 89) / 89.0, 1.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def health_check(self):
        return True, "ok"


def _mem_store(collection: str = "mem") -> VectorStore:
    s = VectorStore(collection=collection, dimension=3)
    s.client = QdrantClient(":memory:")
    s.ensure_collection()
    return s


# ------------------------------------------------------------ validation


def test_reserved_metadata_keys_rejects_server_namespace():
    bad = reserved_metadata_keys(
        {
            "tenant_id": "x",
            "_id_scheme": "y",
            "_anything": 1,
            "source": "z",
            "indexed_at": "now",
            "project": "ok",
        }
    )
    assert bad == ["_anything", "_id_scheme", "indexed_at", "source", "tenant_id"]
    assert reserved_metadata_keys({"project": "a", "chat_id": "42"}) == []


def test_reserved_metadata_blocks_tags_and_timestamp_smuggling():
    """Review P1 pin: `tags`/`timestamp` have dedicated capped fields — a
    metadata copy would reach the payload through the library-era fallback
    with NO cap or type check, so on the remote surface they are reserved."""
    assert reserved_metadata_keys({"tags": ["x"] * 5000}) == ["tags"]
    assert reserved_metadata_keys({"timestamp": "x" * 15000}) == ["timestamp"]
    assert "tags" in validate_remote_item("t", "s", None, [], {"tags": [1]})


def test_reserved_metadata_matching_is_case_and_unicode_normalized():
    """Review pin: `Tenant_Id` or a full-width-underscore variant must not
    land in the payload visually impersonating a structural key."""
    assert reserved_metadata_keys({"Tenant_Id": "x"}) == ["Tenant_Id"]
    assert reserved_metadata_keys({"＿id_scheme": "x"}) == ["＿id_scheme"]
    assert reserved_metadata_keys({"TAGS": []}) == ["TAGS"]


def test_validate_remote_item_caps_and_reserved():
    ok = validate_remote_item("hello", "s", None, [], {})
    assert ok is None
    assert "non-empty" in validate_remote_item("   ", "s", None, [], {})
    too_long = "x" * (REMOTE_MAX_TEXT_CHARS + 1)
    assert "chunk=true" in validate_remote_item(too_long, "s", None, [], {})
    # chunked: the doc cap applies instead, but a source becomes mandatory
    assert validate_remote_item(too_long, "doc.md", None, [], {}, chunk=True) is None
    assert "source" in validate_remote_item(too_long, "", None, [], {}, chunk=True)
    assert "reserved" in validate_remote_item("t", "s", None, [], {"tenant_id": "b"})
    assert "JSON" in validate_remote_item("t", "s", None, [], {"k": object()})
    # Codex-review pins: negative offsets rejected on BOTH surfaces via the
    # shared validator; a chunked item may not supply a base offset (the
    # server assigns window offsets — a silent base shift would corrupt ids).
    assert "non-negative" in validate_remote_item("t", "s", None, [], {}, offset=-1)
    assert "offset at 0" in validate_remote_item(
        "t", "doc.md", None, [], {}, offset=7, chunk=True
    )


def test_expand_remote_items_matches_cli_prose_split():
    text = "a" * (REMOTE_CHUNK_SIZE * 2 + 10)
    item = IngestItem(text=text, source="doc.md")
    flat, origins = expand_remote_items([(item, True)])
    # Fixed windows at offsets 0, size, 2*size — the CLI `index` split.
    assert [(f.offset, len(f.text)) for f in flat] == [
        (0, REMOTE_CHUNK_SIZE),
        (REMOTE_CHUNK_SIZE, REMOTE_CHUNK_SIZE),
        (REMOTE_CHUNK_SIZE * 2, 10),
    ]
    assert origins == [0, 0, 0]


def test_expand_remote_items_passthrough_and_budget():
    plain = IngestItem(text="hi", source="s", offset=7)
    flat, origins = expand_remote_items([(plain, False)])
    assert [(f.offset, f.text) for f in flat] == [(7, "hi")]
    big = IngestItem(
        text="b" * (REMOTE_CHUNK_SIZE * (REMOTE_MAX_CHUNKS_PER_REQUEST + 1)),
        source="big.md",
    )
    with pytest.raises(RemoteRequestTooLarge):
        expand_remote_items([(big, True)])


def test_expand_skips_whitespace_only_windows():
    text = "x" * REMOTE_CHUNK_SIZE + " " * REMOTE_CHUNK_SIZE + "y"
    flat, _ = expand_remote_items([(IngestItem(text=text, source="d.md"), True)])
    assert [f.offset for f in flat] == [0, REMOTE_CHUNK_SIZE * 2]


# ------------------------------------------------------- ingest_remote_items


def test_remote_ingest_stores_then_dedupes_at_zero_embedding_cost():
    emb, store = _CountingEmbedding(), _mem_store()
    items = [IngestItem(text="alpha fact", source="chat"), IngestItem(text="beta fact", source="chat")]

    first = ingest_remote_items(emb, store, items, tenant="a")
    assert [r.status for r in first] == ["stored", "stored"]
    embedded_after_first = len(emb.embedded)

    second = ingest_remote_items(emb, store, items, tenant="a")
    assert [r.status for r in second] == ["duplicate", "duplicate"]
    assert len(emb.embedded) == embedded_after_first  # retry cost: zero embeds
    assert [r.id for r in second] == [r.id for r in first]


def test_remote_ingest_in_request_repeat_and_failure_statuses():
    emb, store = _CountingEmbedding(), _mem_store()
    items = [
        IngestItem(text="same", source="s"),
        IngestItem(text="same", source="s"),
        IngestItem(text="bad |POISON| text", source="s"),
    ]
    results = ingest_remote_items(emb, store, items, tenant="a")
    assert [r.status for r in results] == ["stored", "duplicate", "failed"]


def test_remote_ingest_stamps_tenant_and_scopes_ids():
    emb, store = _CountingEmbedding(), _mem_store()
    (res_a,) = ingest_remote_items(emb, store, [IngestItem(text="t", source="s")], tenant="a")
    (res_b,) = ingest_remote_items(emb, store, [IngestItem(text="t", source="s")], tenant="b")
    assert res_a.id != res_b.id  # tenant-scoped deterministic ids
    points = store.client.retrieve(store.collection, ids=[res_a.id], with_payload=True)
    assert points[0].payload["tenant_id"] == "a"


def test_remote_ingest_isolation_end_to_end():
    """The audit pin: tenant A writes through the remote path — tenant B's
    recall must not see it, A's must."""
    from mnemostack.recall import Recaller, VectorRetriever

    emb, store = _CountingEmbedding(), _mem_store()
    ingest_remote_items(
        emb, store, [IngestItem(text="the launch code is 42", source="chat")], tenant="a"
    )
    recaller = Recaller(retrievers=[VectorRetriever(embedding=emb, vector_store=store)])
    hits_a = recaller.recall("launch code", limit=5, tenant="a")
    hits_b = recaller.recall("launch code", limit=5, tenant="b")
    assert [h.payload["tenant_id"] for h in hits_a] == ["a"]
    assert hits_b == []


def test_remote_ingest_enforces_the_points_quota():
    from mnemostack.quotas import QuotaExceededError

    emb, store = _CountingEmbedding(), _mem_store()
    ingest_remote_items(emb, store, [IngestItem(text="one", source="s")], tenant="a", max_points=1)
    embedded_before = len(emb.embedded)
    count_before = store.count(tenant="a")
    with pytest.raises(QuotaExceededError):
        ingest_remote_items(
            emb, store, [IngestItem(text="two", source="s")], tenant="a", max_points=1
        )
    # Codex-review P1 pin: the quota is preflighted over the WHOLE request —
    # an over-quota call embeds nothing and commits nothing (no partial
    # write hiding behind the error).
    assert len(emb.embedded) == embedded_before
    assert store.count(tenant="a") == count_before
    # A duplicate of the stored point is NOT growth: never falsely rejected.
    (res,) = ingest_remote_items(
        emb, store, [IngestItem(text="one", source="s")], tenant="a", max_points=1
    )
    assert res.status == "duplicate"


def test_remote_ingest_quota_never_splits_a_multi_flush_request():
    """Codex-review P1: a request expanding past one embed batch must not
    commit its first batch and then 507 on the second — the preflight
    covers the whole request, so nothing lands."""
    from mnemostack.quotas import QuotaExceededError

    emb, store = _CountingEmbedding(), _mem_store()
    text = "d" * (REMOTE_CHUNK_SIZE * 80)  # 80 chunks: above the 64 batch size
    items, _ = expand_remote_items([(IngestItem(text=text, source="doc.md"), True)])
    with pytest.raises(QuotaExceededError):
        ingest_remote_items(emb, store, items, tenant="a", max_points=10)
    assert store.count(tenant="a") == 0  # nothing committed
    assert emb.embedded == []  # nothing embedded either


# ------------------------------------------------------------- HTTP surface


def _ingest_app(monkeypatch, tmp_path, *, auth=True, quotas=None):
    """Build the app with real ingest wiring (in-memory store + counting
    embedder) and the recall layers stubbed out."""
    import mnemostack.server as srv

    emb = _CountingEmbedding()
    store = _mem_store("api")
    monkeypatch.setattr(srv, "VectorStore", lambda **_: store)
    monkeypatch.setattr(srv, "get_provider", lambda _n, **_k: emb)

    class _Probe:
        def get_collections(self):
            return object()

    monkeypatch.setattr(srv, "_make_probe_client", lambda *_a, **_k: _Probe())
    monkeypatch.setattr(srv, "Recaller", lambda **_: object())
    monkeypatch.setattr(srv, "VectorRetriever", lambda **_: object())
    monkeypatch.setattr(srv, "BM25Retriever", lambda **_: object())
    monkeypatch.setattr(srv, "MemgraphRetriever", lambda **_: object())
    monkeypatch.setattr(srv, "TemporalRetriever", lambda **_: object())
    monkeypatch.setattr(srv, "build_full_pipeline", lambda **_: object())
    monkeypatch.setattr(srv, "FileStateStore", lambda path: object())

    def _no_llm(*_a, **_k):
        raise RuntimeError("no llm")

    monkeypatch.setattr(srv, "get_llm", _no_llm)

    keys = {}
    cfg_kw = {}
    if auth:
        from mnemostack.auth import FileKeyStore

        ks = FileKeyStore(tmp_path / "keys.json")
        _, keys["read"] = ks.issue("alpha", ["read"])
        _, keys["write"] = ks.issue("alpha", ["write"])
        _, keys["beta_write"] = ks.issue("beta", ["write"])
        cfg_kw = {"auth_enabled": True, "keys_file": str(tmp_path / "keys.json")}
        if quotas:
            from mnemostack.quotas import FileQuotaStore

            qs = FileQuotaStore(tmp_path / "quotas.json")
            for tenant, mp in quotas.items():
                qs.set(tenant, max_points=mp)
            cfg_kw["quotas_file"] = str(tmp_path / "quotas.json")
    cfg = ServerConfig(provider_name="fake", llm_name="fake", graph_uri=None, **cfg_kw)
    app = build_app(cfg)
    return app, store, emb, keys


def test_memories_requires_write_scope(monkeypatch, tmp_path):
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    body = {"items": [{"text": "hi"}]}
    assert client.post("/memories", json=body).status_code == 401
    r = client.post("/memories", json=body, headers={"X-API-Key": keys["read"]})
    assert r.status_code == 403


def test_memories_stores_under_the_key_tenant(monkeypatch, tmp_path):
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    r = client.post(
        "/memories",
        json={"items": [{"text": "remember me", "source": "chat", "metadata": {"k": "v"}}]},
        headers={"X-API-Key": keys["write"]},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["stored"] == 1 and data["failed"] == 0
    assert data["results"][0]["status"] == "stored" and data["results"][0]["item"] == 0
    pid = data["results"][0]["id"]
    assert pid == stable_chunk_id("chat", 0, "remember me", tenant="alpha")
    point = store.client.retrieve(store.collection, ids=[pid], with_payload=True)[0]
    assert point.payload["tenant_id"] == "alpha"  # tenant from the KEY
    assert point.payload["k"] == "v"


def test_memories_cross_tenant_write_isolation_over_http(monkeypatch, tmp_path):
    """A and B write the same content through the API: two distinct points,
    each stamped with its own key's tenant — B's recall scope never
    contains A's point."""
    app, store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    body = {"items": [{"text": "shared wording", "source": "chat"}]}
    ra = client.post("/memories", json=body, headers={"X-API-Key": keys["write"]})
    rb = client.post("/memories", json=body, headers={"X-API-Key": keys["beta_write"]})
    id_a, id_b = ra.json()["results"][0]["id"], rb.json()["results"][0]["id"]
    assert id_a != id_b
    assert store.count(tenant="alpha") == 1 and store.count(tenant="beta") == 1


def test_memories_rejects_reserved_metadata_with_the_key_name(monkeypatch, tmp_path):
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    r = client.post(
        "/memories",
        json={"items": [{"text": "x", "metadata": {"tenant_id": "evil"}}]},
        headers={"X-API-Key": keys["write"]},
    )
    assert r.status_code == 400
    assert "tenant_id" in r.json()["detail"]


def test_memories_chunked_document_yields_per_chunk_results(monkeypatch, tmp_path):
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    text = "z" * (REMOTE_CHUNK_SIZE + 5)
    r = client.post(
        "/memories",
        json={"items": [{"text": text, "source": "doc.md", "chunk": True}]},
        headers={"X-API-Key": keys["write"]},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["stored"] == 2
    assert [(x["item"], x["offset"]) for x in data["results"]] == [
        (0, 0),
        (0, REMOTE_CHUNK_SIZE),
    ]
    # chunked without a source is a contract violation, not a silent default
    r2 = client.post(
        "/memories",
        json={"items": [{"text": text, "chunk": True}]},
        headers={"X-API-Key": keys["write"]},
    )
    assert r2.status_code == 400 and "source" in r2.json()["detail"]


def test_memories_bounds_request_expansion(monkeypatch, tmp_path):
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    text = "b" * (REMOTE_CHUNK_SIZE * (REMOTE_MAX_CHUNKS_PER_REQUEST + 1))
    r = client.post(
        "/memories",
        json={"items": [{"text": text, "source": "big.md", "chunk": True}]},
        headers={"X-API-Key": keys["write"]},
    )
    assert r.status_code == 400 and "split" in r.json()["detail"]


def test_memories_maps_storage_quota_to_507(monkeypatch, tmp_path):
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path, quotas={"alpha": 1})
    client = TestClient(app)
    hdr = {"X-API-Key": keys["write"]}
    assert client.post("/memories", json={"items": [{"text": "one"}]}, headers=hdr).status_code == 200
    r = client.post("/memories", json={"items": [{"text": "two"}]}, headers=hdr)
    assert r.status_code == 507


def test_memories_works_unscoped_when_auth_disabled(monkeypatch, tmp_path):
    app, store, _emb, _keys = _ingest_app(monkeypatch, tmp_path, auth=False)
    client = TestClient(app)
    r = client.post("/memories", json={"items": [{"text": "legacy"}]})
    assert r.status_code == 200 and r.json()["stored"] == 1
    pid = r.json()["results"][0]["id"]
    point = store.client.retrieve(store.collection, ids=[pid], with_payload=True)[0]
    assert "tenant_id" not in point.payload  # single-tenant: no stamp


# ------------------------------------------------------------------ /triples


def test_triples_503_without_a_graph(monkeypatch, tmp_path):
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    r = client.post(
        "/triples",
        json={"triples": [{"subject": "a", "predicate": "knows", "object": "b"}]},
        headers={"X-API-Key": keys["write"]},
    )
    assert r.status_code == 503


def test_triples_all_failed_is_a_502_not_a_quiet_200(monkeypatch, tmp_path):
    """Review pin: per-triple isolation is for PARTIAL failure — a batch
    where EVERY write failed means the graph path is broken, and a caller
    checking only the status code must see it."""
    import mnemostack.graph.factory as graph_factory

    class _DeadGraph:
        def add_triple(self, **kw):
            raise RuntimeError("schema regression")

        def close(self):
            pass

    monkeypatch.setattr(graph_factory, "make_graph_store", lambda *a, **k: _DeadGraph())
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    # _ingest_app builds with graph_uri=None; rebuild the piece we need by
    # calling the endpoint on an app that has a graph configured.
    import mnemostack.server as srv  # noqa: F401 — patched by _ingest_app
    from mnemostack.auth import FileKeyStore

    ks = FileKeyStore(tmp_path / "keys2.json")
    _, write_key = ks.issue("alpha", ["write"])
    app2 = build_app(
        ServerConfig(
            provider_name="fake",
            llm_name="fake",
            graph_uri="bolt://graph.invalid:7687",
            auth_enabled=True,
            keys_file=str(tmp_path / "keys2.json"),
        )
    )
    client = TestClient(app2)
    r = client.post(
        "/triples",
        json={"triples": [{"subject": "a", "predicate": "p", "object": "b"}]},
        headers={"X-API-Key": write_key},
    )
    assert r.status_code == 502


def test_triples_stamps_the_key_tenant_and_isolates_failures(monkeypatch, tmp_path):
    import mnemostack.graph.factory as graph_factory
    import mnemostack.server as srv

    calls: list[dict] = []

    class _FakeGraph:
        def add_triple(self, **kw):
            if kw["subject"] == "boom":
                raise RuntimeError("backend hiccup")
            calls.append(kw)

        def close(self):
            pass

    monkeypatch.setattr(graph_factory, "make_graph_store", lambda *a, **k: _FakeGraph())

    emb = _CountingEmbedding()
    store = _mem_store("gapi")
    monkeypatch.setattr(srv, "VectorStore", lambda **_: store)
    monkeypatch.setattr(srv, "get_provider", lambda _n, **_k: emb)

    class _Probe:
        def get_collections(self):
            return object()

    monkeypatch.setattr(srv, "_make_probe_client", lambda *_a, **_k: _Probe())
    monkeypatch.setattr(srv, "Recaller", lambda **_: object())
    monkeypatch.setattr(srv, "VectorRetriever", lambda **_: object())
    monkeypatch.setattr(srv, "BM25Retriever", lambda **_: object())
    monkeypatch.setattr(srv, "MemgraphRetriever", lambda **_: object())
    monkeypatch.setattr(srv, "TemporalRetriever", lambda **_: object())
    monkeypatch.setattr(srv, "build_full_pipeline", lambda **_: object())
    monkeypatch.setattr(srv, "FileStateStore", lambda path: object())

    def _no_llm(*_a, **_k):
        raise RuntimeError("no llm")

    monkeypatch.setattr(srv, "get_llm", _no_llm)
    from mnemostack.auth import FileKeyStore

    ks = FileKeyStore(tmp_path / "keys.json")
    _, write_key = ks.issue("alpha", ["write"])
    app = build_app(
        ServerConfig(
            provider_name="fake",
            llm_name="fake",
            graph_uri="bolt://graph.invalid:7687",
            auth_enabled=True,
            keys_file=str(tmp_path / "keys.json"),
        )
    )
    client = TestClient(app)
    r = client.post(
        "/triples",
        json={
            "triples": [
                {"subject": "alice", "predicate": "knows", "object": "bob"},
                {"subject": "boom", "predicate": "x", "object": "y"},
            ]
        },
        headers={"X-API-Key": write_key},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["added"] == 1 and data["failed"] == 1
    assert [x["status"] for x in data["results"]] == ["added", "failed"]
    assert calls[0]["tenant"] == "alpha"  # graph-side tenant stamp from the KEY
    assert data["results"][1]["error"] == "RuntimeError"  # type only, no backend text


# ------------------------------------------------- round-2 review batch pins


def test_remote_chunk_size_matches_the_index_default():
    """The cross-path id claim holds only while the remote split equals the
    CLI's default --chunk-size — pin the coupling so either side moving
    breaks loudly."""
    from mnemostack.config import VectorConfig

    assert REMOTE_CHUNK_SIZE == VectorConfig().chunk_size


def test_remote_ingest_creates_the_collection_on_first_write():
    """A fresh deployment's FIRST write must not 500 on a missing
    collection — no operator pre-ingest required."""
    emb = _CountingEmbedding()
    store = VectorStore(collection="fresh", dimension=3)
    store.client = QdrantClient(":memory:")  # deliberately NO ensure_collection
    (res,) = ingest_remote_items(emb, store, [IngestItem(text="first", source="s")])
    assert res.status == "stored"


def test_remote_ingest_honors_configured_schema_keys():
    """A deployment with non-default recall.text_key/timestamp_key must see
    remote memories: the payload mirrors text and the domain-converted
    event time under the configured keys."""
    emb, store = _CountingEmbedding(), _mem_store()
    (res,) = ingest_remote_items(
        emb,
        store,
        [IngestItem(text="hello", source="s", timestamp="2026-08-20T10:00:00+00:00")],
        tenant="a",
        text_key="content",
        timestamp_key="ts",
        timestamp_format="epoch",
    )
    point = store.client.retrieve(store.collection, ids=[res.id], with_payload=True)[0]
    assert point.payload["content"] == "hello"
    assert point.payload["text"] == "hello"  # historical key stays
    assert isinstance(point.payload["ts"], float)  # epoch domain, not ISO
    assert point.payload["timestamp"] == "2026-08-20T10:00:00+00:00"


def test_validator_rejects_pipe_and_control_chars_in_source():
    assert "source" in validate_remote_item("t", "a|b", None, [], {})
    assert "source" in validate_remote_item("t", "a\x00b", None, [], {})
    assert validate_remote_item("t", "chat/2026-08-20.md", None, [], {}) is None


def test_validator_rejects_non_iso_timestamps():
    assert "ISO-8601" in validate_remote_item("t", "s", "not-a-date", [], {})
    assert validate_remote_item("t", "s", "2026-08-20T10:00:00Z", [], {}) is None


def test_validator_reserves_configured_schema_keys():
    err = validate_remote_item(
        "t", "s", None, [], {"content": "shadow"}, reserved_extra={"content"}
    )
    assert "content" in err
    assert (
        validate_remote_item("t", "s", None, [], {"other": 1}, reserved_extra={"content"})
        is None
    )


def test_validator_distinguishes_non_string_keys_from_reserved():
    assert "must be strings" in validate_remote_item("t", "s", None, [], {1: "x"})


def test_expand_exactly_at_the_chunk_budget_succeeds():
    text = "c" * (REMOTE_CHUNK_SIZE * REMOTE_MAX_CHUNKS_PER_REQUEST)
    flat, _ = expand_remote_items([(IngestItem(text=text, source="d.md"), True)])
    assert len(flat) == REMOTE_MAX_CHUNKS_PER_REQUEST


def test_memories_rejects_configured_schema_key_in_metadata(monkeypatch, tmp_path):
    """HTTP surface reserves the deployment's text/timestamp keys too."""
    import mnemostack.server as srv

    emb = _CountingEmbedding()
    store = _mem_store("schema")
    monkeypatch.setattr(srv, "VectorStore", lambda **_: store)
    monkeypatch.setattr(srv, "get_provider", lambda _n, **_k: emb)

    class _Probe:
        def get_collections(self):
            return object()

    monkeypatch.setattr(srv, "_make_probe_client", lambda *_a, **_k: _Probe())
    monkeypatch.setattr(srv, "Recaller", lambda **_: object())
    monkeypatch.setattr(srv, "VectorRetriever", lambda **_: object())
    monkeypatch.setattr(srv, "BM25Retriever", lambda **_: object())
    monkeypatch.setattr(srv, "MemgraphRetriever", lambda **_: object())
    monkeypatch.setattr(srv, "TemporalRetriever", lambda **_: object())
    monkeypatch.setattr(srv, "build_full_pipeline", lambda **_: object())
    monkeypatch.setattr(srv, "FileStateStore", lambda path: object())

    def _no_llm(*_a, **_k):
        raise RuntimeError("no llm")

    monkeypatch.setattr(srv, "get_llm", _no_llm)
    app = build_app(
        ServerConfig(
            provider_name="fake", llm_name="fake", graph_uri=None, text_key="content"
        )
    )
    client = TestClient(app)
    r = client.post(
        "/memories", json={"items": [{"text": "x", "metadata": {"content": "shadow"}}]}
    )
    assert r.status_code == 400 and "content" in r.json()["detail"]


def test_triples_rejects_blank_components(monkeypatch, tmp_path):
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    # Validation precedes the graph-availability check: a blank component is
    # a caller-fixable 400 even on a graphless deployment.
    r = TestClient(app).post(
        "/triples",
        json={"triples": [{"subject": " ", "predicate": "p", "object": "o"}]},
        headers={"X-API-Key": keys["write"]},
    )
    assert r.status_code == 400 and "non-blank" in r.json()["detail"]


def test_remote_ingest_same_tenant_writes_are_serialized():
    """The per-tenant lock makes preflight+ingest atomic in-process: two
    threads racing one remaining quota slot store exactly one point."""
    import threading as _threading

    from mnemostack.quotas import QuotaExceededError

    emb, store = _CountingEmbedding(), _mem_store()
    outcomes: list[str] = []

    def _write(text: str) -> None:
        try:
            ingest_remote_items(
                emb, store, [IngestItem(text=text, source="s")], tenant="a", max_points=1
            )
            outcomes.append("stored")
        except QuotaExceededError:
            outcomes.append("quota")

    t1 = _threading.Thread(target=_write, args=("one",))
    t2 = _threading.Thread(target=_write, args=("two",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert sorted(outcomes) == ["quota", "stored"]
    assert store.count(tenant="a") == 1  # never over the cap in-process


def test_validator_enforces_tag_and_metadata_caps():
    from mnemostack.ingest import (
        REMOTE_MAX_METADATA_CHARS,
        REMOTE_MAX_METADATA_KEYS,
        REMOTE_MAX_TAG_CHARS,
        REMOTE_MAX_TAGS,
    )

    assert "tags" in validate_remote_item("t", "s", None, ["x"] * (REMOTE_MAX_TAGS + 1), {})
    assert "tag" in validate_remote_item("t", "s", None, ["y" * (REMOTE_MAX_TAG_CHARS + 1)], {})
    many_keys = {f"k{i}": 1 for i in range(REMOTE_MAX_METADATA_KEYS + 1)}
    assert "keys" in validate_remote_item("t", "s", None, [], many_keys)
    fat = {"blob": "v" * (REMOTE_MAX_METADATA_CHARS + 1)}
    assert "serialized" in validate_remote_item("t", "s", None, [], fat)


def test_memories_caps_the_item_count(monkeypatch, tmp_path):
    from mnemostack.ingest import REMOTE_MAX_ITEMS

    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    r = TestClient(app).post(
        "/memories",
        json={"items": [{"text": f"m{i}"} for i in range(REMOTE_MAX_ITEMS + 1)]},
        headers={"X-API-Key": keys["write"]},
    )
    assert r.status_code == 422  # pydantic max_length on the batch
