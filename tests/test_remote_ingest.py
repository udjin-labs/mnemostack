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
    validate_remote_triple,
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
        json={"triples": [{"subject": "a", "predicate": "KNOWS", "object": "b"}]},
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
        json={"triples": [{"subject": "a", "predicate": "P", "object": "b"}]},
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
                {"subject": "alice", "predicate": "KNOWS", "object": "bob"},
                {"subject": "boom", "predicate": "X", "object": "y"},
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


# ------------------------------------------------- round-3 review batch pins


def test_epoch_conversion_applies_to_the_default_timestamp_key():
    """Codex-R3 P1: format drives conversion, not the key name — a numeric
    collection with the DEFAULT `timestamp` key must store epoch, not ISO."""
    emb, store = _CountingEmbedding(), _mem_store()
    (res,) = ingest_remote_items(
        emb,
        store,
        [IngestItem(text="t", source="s", timestamp="2026-08-20T10:00:00+00:00")],
        tenant="a",
        timestamp_format="epoch",
    )
    point = store.client.retrieve(store.collection, ids=[res.id], with_payload=True)[0]
    assert isinstance(point.payload["timestamp"], float)


def test_naive_timestamps_convert_as_utc():
    """Codex/agent R3: a naive ISO input is UTC by stack convention — the
    stored epoch must not depend on the server's local timezone."""
    from datetime import datetime, timezone

    emb, store = _CountingEmbedding(), _mem_store()
    (res,) = ingest_remote_items(
        emb,
        store,
        [IngestItem(text="t", source="s", timestamp="2026-08-20T10:00:00")],
        tenant="a",
        timestamp_format="epoch",
    )
    point = store.client.retrieve(store.collection, ids=[res.id], with_payload=True)[0]
    expected = datetime(2026, 8, 20, 10, 0, 0, tzinfo=timezone.utc).timestamp()
    assert point.payload["timestamp"] == expected


def test_collection_bootstrap_survives_a_concurrent_create_race():
    """R3-R5 evolution: the race has exactly one shape — absent BEFORE,
    present AFTER. The loser then re-runs ensure so the winner's collection
    still passes validation (dimension/sparse), raising honestly if not."""
    emb = _CountingEmbedding()
    real = _mem_store("race")

    class _RacyStore:
        def __init__(self):
            self.ensure_calls = 0
            self.exists_calls = 0

        def collection_exists(self):
            self.exists_calls += 1
            # Absent at the pre-check, present after the winner's create.
            return self.exists_calls > 1

        def ensure_collection(self):
            self.ensure_calls += 1
            if self.ensure_calls == 1:
                raise RuntimeError("already exists")  # lost the create race
            # revalidation pass on the winner's collection succeeds

        def __getattr__(self, name):
            return getattr(real, name)

    racy = _RacyStore()
    (res,) = ingest_remote_items(emb, racy, [IngestItem(text="r", source="s")])
    assert res.status == "stored"
    assert racy.ensure_calls == 2  # loss + honest revalidation of the winner


def test_bootstrap_never_masks_pre_existing_collection_validation():
    """Codex-R5 P1: an ensure failure on a collection that EXISTED BEFORE is
    validation (dimension mismatch, sparse space) — it must propagate, not
    be swallowed as a supposed create race."""
    emb = _CountingEmbedding()

    class _MismatchedStore:
        def __init__(self):
            self.ensure_calls = 0

        def collection_exists(self):
            return True  # pre-existing collection

        def ensure_collection(self):
            self.ensure_calls += 1
            raise ValueError("dimension mismatch: expected 3, found 768")

    bad = _MismatchedStore()
    with pytest.raises(ValueError, match="dimension mismatch"):
        ingest_remote_items(emb, bad, [IngestItem(text="x", source="s")])
    assert bad.ensure_calls == 1  # no second attempt against a broken schema
    assert emb.embedded == []  # nothing embedded before the loud failure


def test_bootstrap_duck_store_surfaces_the_original_error():
    """Agent-R5 P2: a duck store without collection_exists must surface the
    ORIGINAL ensure failure, not an AttributeError about the missing hook."""
    emb = _CountingEmbedding()

    class _DuckStore:
        def ensure_collection(self):
            raise RuntimeError("qdrant unreachable")

        # no collection_exists at all

    with pytest.raises(RuntimeError, match="unreachable"):
        ingest_remote_items(emb, _DuckStore(), [IngestItem(text="x", source="s")])


def test_reserved_extra_matching_is_nfkc_symmetric():
    """Agent-R3 P3: a configured key stored in decomposed Unicode form must
    still catch the composed client variant (both sides normalized)."""
    import unicodedata

    decomposed = unicodedata.normalize("NFD", "café")
    composed = unicodedata.normalize("NFC", "café")
    assert decomposed != composed  # premise: genuinely different strings
    err = validate_remote_item(
        "t", "s", None, [], {composed: "x"}, reserved_extra={decomposed}
    )
    assert err is not None and "reserved" in err


# ------------------------------------------------- round-4 review batch pins


def test_invalidated_at_is_reserved_metadata():
    """Codex-R4: the server-owned stale marker cannot be planted at ingest —
    a client would store memories that default recall silently hides."""
    assert reserved_metadata_keys({"invalidated_at": "2026-01-01"}) == ["invalidated_at"]
    assert "invalidated_at" in validate_remote_item(
        "t", "s", None, [], {"invalidated_at": "2026-01-01"}
    )


def test_unknown_timestamp_format_passes_iso_through():
    """Agent-R4 P1: an unrecognized format must NOT convert (the safe
    default is ISO passthrough, the codebase-wide convention) — a typo'd
    config cannot silently replace event times with bogus numbers."""
    from mnemostack.ingest import _apply_timestamp_domain

    item = IngestItem(text="t", source="s", timestamp="2026-08-20T10:00:00+00:00")
    _apply_timestamp_domain([item], "timestamp", "epoc_typo")
    assert item.timestamp == "2026-08-20T10:00:00+00:00"  # untouched
    assert "timestamp" not in item.metadata  # no bogus numeric landed


def test_bootstrap_does_not_mask_genuine_store_failures():
    """Agent-R4 P2: the create-race recovery is discriminated — when the
    collection does NOT exist after a failure, the original error surfaces
    (store down / dimension mismatch), never a blind second create."""
    emb = _CountingEmbedding()

    class _BrokenStore:
        def __init__(self):
            self.ensure_calls = 0

        def ensure_collection(self):
            self.ensure_calls += 1
            raise RuntimeError("qdrant unreachable")

        def collection_exists(self):
            return False  # genuinely absent: not a lost race

    broken = _BrokenStore()
    with pytest.raises(RuntimeError, match="unreachable"):
        ingest_remote_items(emb, broken, [IngestItem(text="x", source="s")])
    assert broken.ensure_calls == 1  # no blind retry doubling load
    assert emb.embedded == []


def test_bootstrap_runs_once_per_store_instance():
    """Codex-R4 P2: sparse-aware ensure_collection re-verifies coverage with
    collection-wide counts — that cost is paid once per store, not per
    write."""
    emb = _CountingEmbedding()
    real = _mem_store("once")
    calls = {"n": 0}
    orig = real.ensure_collection

    def _counting_ensure():
        calls["n"] += 1
        return orig()

    real.ensure_collection = _counting_ensure  # type: ignore[method-assign]
    ingest_remote_items(emb, real, [IngestItem(text="one", source="s")])
    ingest_remote_items(emb, real, [IngestItem(text="two", source="s")])
    assert calls["n"] == 1  # second write skipped bootstrap


def test_enrich_keys_never_reach_public_payloads():
    """Agent-R4 P3: the enrichment ownership record is structural — both
    public serializers strip it."""
    from types import SimpleNamespace

    from mnemostack.mcp.server import _public_payload
    from mnemostack.server import _memory_of

    payload = {"text": "x", "_enrich_keys": ["content"], "content": "x"}
    assert "_enrich_keys" not in _public_payload(payload)
    mem = _memory_of(
        SimpleNamespace(id="1", text="x", score=1.0, payload=dict(payload), sources=["vector"])
    )
    assert "_enrich_keys" not in mem.metadata


# --------------------------------------------------- bot round-2 batch pins


def test_memories_all_embedding_failures_map_to_502(monkeypatch, tmp_path):
    """Bot-R2: per-item isolation is for PARTIAL failure — when EVERY item
    failed to embed, 200 would hide a dead write path from status checks."""
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    client = TestClient(app)
    r = client.post(
        "/memories",
        json={"items": [{"text": "bad |POISON| one"}, {"text": "bad |POISON| two"}]},
        headers={"X-API-Key": keys["write"]},
    )
    assert r.status_code == 502
    # Partial failure keeps 200 with per-item statuses.
    r2 = client.post(
        "/memories",
        json={"items": [{"text": "bad |POISON| three"}, {"text": "good"}]},
        headers={"X-API-Key": keys["write"]},
    )
    assert r2.status_code == 200
    assert r2.json()["failed"] == 1 and r2.json()["stored"] == 1


def test_offset_is_bounded_to_the_store_integer_domain():
    from mnemostack.ingest import REMOTE_MAX_OFFSET

    assert validate_remote_item("t", "s", None, [], {}, offset=REMOTE_MAX_OFFSET) is None
    assert "exceeds" in validate_remote_item(
        "t", "s", None, [], {}, offset=REMOTE_MAX_OFFSET + 1
    )
    assert "exceeds" in validate_remote_item("t", "s", None, [], {}, offset=2**100)


def test_memories_offset_cap_enforced_at_the_schema(monkeypatch, tmp_path):
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    r = TestClient(app).post(
        "/memories",
        json={"items": [{"text": "x", "offset": 2**100}]},
        headers={"X-API-Key": keys["write"]},
    )
    assert r.status_code == 422  # pydantic le — before any embedding cost


def test_triples_validity_bounds_must_parse(monkeypatch, tmp_path):
    """Bot-R2: an unparseable bound degrades point-in-time queries to
    lexicographic comparison — reject at the API; 'current' stays valid."""
    import mnemostack.graph.factory as graph_factory
    import mnemostack.server as srv  # noqa: F401

    added: list[dict] = []

    class _G:
        def add_triple(self, **kw):
            added.append(kw)

        def close(self):
            pass

    monkeypatch.setattr(graph_factory, "make_graph_store", lambda *a, **k: _G())
    emb = _CountingEmbedding()
    store = _mem_store("tv")
    import mnemostack.server as srv2

    monkeypatch.setattr(srv2, "VectorStore", lambda **_: store)
    monkeypatch.setattr(srv2, "get_provider", lambda _n, **_k: emb)

    class _Probe:
        def get_collections(self):
            return object()

    monkeypatch.setattr(srv2, "_make_probe_client", lambda *_a, **_k: _Probe())
    for name in ("Recaller", "VectorRetriever", "BM25Retriever", "MemgraphRetriever",
                 "TemporalRetriever"):
        monkeypatch.setattr(srv2, name, lambda **_: object())
    monkeypatch.setattr(srv2, "build_full_pipeline", lambda **_: object())
    monkeypatch.setattr(srv2, "FileStateStore", lambda path: object())

    def _no_llm(*_a, **_k):
        raise RuntimeError("no llm")

    monkeypatch.setattr(srv2, "get_llm", _no_llm)
    from mnemostack.auth import FileKeyStore

    ks = FileKeyStore(tmp_path / "k.json")
    _, wk = ks.issue("alpha", ["write"])
    app = build_app(
        ServerConfig(
            provider_name="fake", llm_name="fake",
            graph_uri="bolt://graph.invalid:7687",
            auth_enabled=True, keys_file=str(tmp_path / "k.json"),
        )
    )
    client = TestClient(app)
    hdr = {"X-API-Key": wk}
    bad = client.post(
        "/triples",
        json={"triples": [{"subject": "a", "predicate": "P", "object": "b",
                            "valid_from": "tomorrow"}]},
        headers=hdr,
    )
    assert bad.status_code == 400 and "valid_from" in bad.json()["detail"]
    ok = client.post(
        "/triples",
        json={"triples": [{"subject": "a", "predicate": "P", "object": "b",
                            "valid_from": "2026-01-01", "valid_until": "current"}]},
        headers=hdr,
    )
    assert ok.status_code == 200 and added[0]["valid_until"] == "current"


def test_qdrant_bm25_mode_boots_without_a_collection(monkeypatch, tmp_path):
    """Bot-R2 P1: a fresh deployment must be able to START in qdrant_bm25
    mode — the arm is skipped (startup-snapshot semantics) instead of the
    boot crashing on the missing collection."""
    import mnemostack.server as srv

    emb = _CountingEmbedding()
    store = VectorStore(collection="freshb", dimension=3)
    store.client = QdrantClient(":memory:")  # no ensure_collection: fresh
    monkeypatch.setattr(srv, "VectorStore", lambda **_: store)
    monkeypatch.setattr(srv, "get_provider", lambda _n, **_k: emb)

    class _Probe:
        def get_collections(self):
            return object()

    monkeypatch.setattr(srv, "_make_probe_client", lambda *_a, **_k: _Probe())
    for name in ("Recaller", "VectorRetriever", "MemgraphRetriever", "TemporalRetriever"):
        monkeypatch.setattr(srv, name, lambda **_: object())

    def _must_not_build(*_a, **_k):  # pragma: no cover
        raise AssertionError("BM25 corpus must not load from a missing collection")

    monkeypatch.setattr(srv.BM25Retriever, "from_qdrant", _must_not_build)
    monkeypatch.setattr(srv, "build_full_pipeline", lambda **_: object())
    monkeypatch.setattr(srv, "FileStateStore", lambda path: object())

    def _no_llm(*_a, **_k):
        raise RuntimeError("no llm")

    monkeypatch.setattr(srv, "get_llm", _no_llm)
    app = build_app(
        ServerConfig(
            provider_name="fake", llm_name="fake", graph_uri=None,
            text_search="qdrant_bm25",
        )
    )
    # Boot succeeded; the first write then bootstraps the collection.
    client = TestClient(app)
    r = client.post("/memories", json={"items": [{"text": "first"}]})
    assert r.status_code == 200 and r.json()["stored"] == 1


# --------------------------------------------------- bot round-3 batch pins


def test_metadata_validity_bounds_must_parse():
    """Bot-R3: valid_from/valid_until are allowed content but must parse —
    unparseable bounds degrade as_of recall to lexicographic comparison."""
    assert "valid_from" in validate_remote_item(
        "t", "s", None, [], {"valid_from": "tomorrow"}
    )
    assert "valid_until" in validate_remote_item(
        "t", "s", None, [], {"valid_until": "not-a-date"}
    )
    assert validate_remote_item(
        "t", "s", None, [], {"valid_from": "2026-01-01", "valid_until": "current"}
    ) is None


def test_metadata_numbers_must_fit_the_store_domain():
    """Bot-R3: an int64-overflowing or non-finite number embeds first and
    only fails at upsert — reject before any provider cost, nested too."""
    assert "64-bit" in validate_remote_item("t", "s", None, [], {"priority": 2**100})
    assert "64-bit" in validate_remote_item(
        "t", "s", None, [], {"nested": {"deep": [1, 2, 2**80]}}
    )
    assert "finite" in validate_remote_item("t", "s", None, [], {"score": float("inf")})
    assert "finite" in validate_remote_item("t", "s", None, [], {"score": float("nan")})
    assert validate_remote_item(
        "t", "s", None, [], {"priority": 2**62, "score": 0.5, "flags": [True, 1]}
    ) is None


def test_schema_mirror_is_structural_not_enrichment():
    """Bot-R3: the text_key mirror must NOT travel through the enrichment
    hook — apply_enrichment records its keys in _enrich_keys, and a later
    `index --refresh-payloads` without an enricher would delete the mirror,
    blanking recall for custom-text_key deployments."""
    emb, store = _CountingEmbedding(), _mem_store()
    (res,) = ingest_remote_items(
        emb,
        store,
        [IngestItem(text="hello", source="s")],
        tenant="a",
        text_key="content",
    )
    point = store.client.retrieve(store.collection, ids=[res.id], with_payload=True)[0]
    assert point.payload["content"] == "hello"
    assert "_enrich_keys" not in point.payload  # structural, not enrichment


# --------------------------------------------------- round-8 review batch pins


def test_metadata_nesting_depth_is_bounded(monkeypatch, tmp_path):
    """Agent-R8 P1: pathologically nested metadata must be a clean 400,
    never a RecursionError-turned-500 inside the validator itself.

    Depth 100: parseable by every supported interpreter (3.10/3.11's
    C json coder is recursion-limited around 1000 — deeper structures die
    in the TRANSPORT parser on those versions, before any mnemostack
    code), and far past the 32-level cap the validator enforces."""
    deep: list = []
    cursor = deep
    for _ in range(100):
        nxt: list = []
        cursor.append(nxt)
        cursor = nxt
    assert "nesting" in validate_remote_item("t", "s", None, [], {"a": deep})
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    r = TestClient(app).post(
        "/memories",
        json={"items": [{"text": "x", "metadata": {"a": deep}}]},
        headers={"X-API-Key": keys["write"]},
    )
    assert r.status_code == 400 and "nesting" in r.json()["detail"]


def test_int64_bounds_are_asymmetric():
    """Codex-R8: signed int64 is [-2^63, 2^63-1] — both exact boundaries are
    valid store values; one past either edge is not."""
    assert validate_remote_item("t", "s", None, [], {"n": -(2**63)}) is None
    assert validate_remote_item("t", "s", None, [], {"n": 2**63 - 1}) is None
    assert "64-bit" in validate_remote_item("t", "s", None, [], {"n": -(2**63) - 1})
    assert "64-bit" in validate_remote_item("t", "s", None, [], {"n": 2**63})


def test_re_remembering_an_invalidated_memory_reactivates_it():
    """Codex-R9: invalidate -> remember of the same fact must make it
    recallable again (marker cleared, no re-embedding), reported stored —
    a hidden 'duplicate' would claim success while recall stays empty."""
    emb, store = _CountingEmbedding(), _mem_store()
    (first,) = ingest_remote_items(
        emb, store, [IngestItem(text="the sky is green", source="chat")], tenant="a"
    )
    assert first.status == "stored"
    store.invalidate([first.id], tenant="a")
    point = store.client.retrieve(store.collection, ids=[first.id], with_payload=True)[0]
    assert point.payload.get("invalidated_at")  # retracted
    embedded_before = len(emb.embedded)

    (again,) = ingest_remote_items(
        emb, store, [IngestItem(text="the sky is green", source="chat")], tenant="a"
    )
    assert again.status == "stored"  # reactivated, not a hidden duplicate
    assert len(emb.embedded) == embedded_before  # zero re-embedding
    point = store.client.retrieve(store.collection, ids=[first.id], with_payload=True)[0]
    assert "invalidated_at" not in (point.payload or {})  # recallable again


def test_current_duplicates_still_skip_the_reactivation_patch():
    """A LIVE duplicate must not pay a payload patch round trip."""
    emb, store = _CountingEmbedding(), _mem_store()
    ingest_remote_items(emb, store, [IngestItem(text="live", source="s")], tenant="a")
    patches: list = []
    orig = store.apply_payload_patches

    def _counting(patch_list, **kw):
        patches.append(patch_list)
        return orig(patch_list, **kw)

    store.apply_payload_patches = _counting  # type: ignore[method-assign]
    (res,) = ingest_remote_items(emb, store, [IngestItem(text="live", source="s")], tenant="a")
    assert res.status == "duplicate"
    assert patches == []  # no patch issued for a current point


# -------------------------------------------------- round-10 review batch pins


def test_quota_rejection_leaves_retracted_memories_retracted():
    """Codex-R10 P1: commit-nothing means NOTHING — a 507-rejected request
    must not have reactivated an invalidated duplicate on the way."""
    from mnemostack.quotas import QuotaExceededError

    emb, store = _CountingEmbedding(), _mem_store()
    (first,) = ingest_remote_items(
        emb, store, [IngestItem(text="fact", source="s")], tenant="a", max_points=1
    )
    store.invalidate([first.id], tenant="a")
    with pytest.raises(QuotaExceededError):
        ingest_remote_items(
            emb,
            store,
            [IngestItem(text="fact", source="s"), IngestItem(text="new", source="s")],
            tenant="a",
            max_points=1,
        )
    point = store.client.retrieve(store.collection, ids=[first.id], with_payload=True)[0]
    assert point.payload.get("invalidated_at")  # still retracted


def test_reactivation_preserves_valid_until():
    """Agent-R11 P1: valid_until is dual-use — it may be legitimate
    ingest-declared expiry, indistinguishable from an invalidate-set bound.
    Reactivation clears ONLY invalidated_at; destroying client content to
    fix an as_of edge would be worse than documenting the caveat."""
    emb, store = _CountingEmbedding(), _mem_store()
    (first,) = ingest_remote_items(
        emb,
        store,
        [IngestItem(text="promo", source="s", metadata={"valid_until": "2030-01-01"})],
        tenant="a",
    )
    store.invalidate([first.id], tenant="a")  # no valid_until arg
    ingest_remote_items(emb, store, [IngestItem(text="promo", source="s",
                                                 metadata={"valid_until": "2030-01-01"})], tenant="a")
    point = store.client.retrieve(store.collection, ids=[first.id], with_payload=True)[0]
    assert "invalidated_at" not in (point.payload or {})  # recallable again
    assert point.payload.get("valid_until") == "2030-01-01"  # content survives


def test_vanished_stale_point_is_failed_not_stored():
    """Agent-R10: apply_payload_patches silently skips vanished points — the
    ignored return value must not turn that into a reported 'stored'."""
    emb, store = _CountingEmbedding(), _mem_store()
    (first,) = ingest_remote_items(
        emb, store, [IngestItem(text="gone", source="s")], tenant="a"
    )
    store.invalidate([first.id], tenant="a")

    orig = store.apply_payload_patches

    def _skipping(patches, **kw):
        # Simulate the concurrent-delete race: the store patched nothing.
        store.delete_points([first.id], tenant="a")
        return 0

    store.apply_payload_patches = _skipping  # type: ignore[method-assign]
    (res,) = ingest_remote_items(
        emb, store, [IngestItem(text="gone", source="s")], tenant="a"
    )
    store.apply_payload_patches = orig  # type: ignore[method-assign]
    assert res.status == "failed"  # never a fabricated 'stored'/'duplicate'


def test_colliding_schema_keys_are_rejected_loudly():
    """Codex-R10 P2: text_key='source' would overwrite the provenance field
    (metadata merges last in payload construction) — operator error, loud."""
    emb, store = _CountingEmbedding(), _mem_store()
    with pytest.raises(ValueError, match="pipeline"):
        ingest_remote_items(
            emb, store, [IngestItem(text="x", source="s")], text_key="source"
        )
    with pytest.raises(ValueError, match="pipeline"):
        ingest_remote_items(
            emb, store, [IngestItem(text="x", source="s")], timestamp_key="offset"
        )
    # Codex-R11: DOWNSTREAM pipeline keys too — timestamp_key="tags" would
    # feed the epoch float into the tag materializer and 500 every write.
    with pytest.raises(ValueError, match="pipeline"):
        ingest_remote_items(
            emb, store, [IngestItem(text="x", source="s")], timestamp_key="tags"
        )
    with pytest.raises(ValueError, match="pipeline"):
        ingest_remote_items(
            emb, store, [IngestItem(text="x", source="s")], text_key="indexed_at"
        )
    with pytest.raises(ValueError, match="differ"):
        ingest_remote_items(
            emb, store, [IngestItem(text="x", source="s")],
            text_key="content", timestamp_key="content",
        )


def test_schema_key_misconfiguration_fails_at_boot(monkeypatch, tmp_path):
    """Agent-R11 P2: a colliding schema key must refuse BOOT on both
    surfaces, not 500 every write with an opaque error."""
    import mnemostack.server as srv

    provider_calls: list = []

    def _counting_provider(_n, **_k):
        provider_calls.append(_n)
        return _CountingEmbedding()

    monkeypatch.setattr(srv, "get_provider", _counting_provider)
    with pytest.raises(ValueError, match="pipeline"):
        build_app(
            ServerConfig(provider_name="fake", llm_name="fake", graph_uri=None,
                         text_key="source")
        )
    assert provider_calls == []  # fail-fast: no provider round trip paid


# -------------------------------------------------- round-12 review batch pins


def test_lifecycle_keys_are_forbidden_schema_keys():
    """Agent-R12 P1: text_key='invalidated_at' would stamp EVERY stored
    point as stale — writes report stored while default recall hides them
    all, silently. valid_from/valid_until corrupt as_of the same way."""
    emb, store = _CountingEmbedding(), _mem_store()
    for bad in ("invalidated_at", "valid_from", "valid_until"):
        with pytest.raises(ValueError, match="pipeline"):
            ingest_remote_items(
                emb, store, [IngestItem(text="x", source="s")], text_key=bad
            )
        with pytest.raises(ValueError, match="pipeline"):
            ingest_remote_items(
                emb, store, [IngestItem(text="x", source="s")], timestamp_key=bad
            )


def test_lone_surrogates_are_rejected_not_500(monkeypatch, tmp_path):
    """Codex-R12: '\\ud800' is a valid JSON escape but crashes UTF-8
    encoding in id generation — must be a 400, never a 500."""
    bad = "\ud800"
    assert "UTF-8" in validate_remote_item("x" + bad, "s", None, [], {})
    assert "UTF-8" in validate_remote_item("x", "s" + bad, None, [], {})
    assert "UTF-8" in validate_remote_item("x", "s", None, [bad], {})
    assert "UTF-8" in validate_remote_item("x", "s", None, [], {"k": bad})
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    # A real client delivers the surrogate as a raw JSON escape sequence —
    # the transport bytes are clean ASCII; the decoded STRING is not.
    raw = b'{"items": [{"text": "x\\ud800", "source": "s"}]}'
    r = TestClient(app).post(
        "/memories",
        content=raw,
        headers={"X-API-Key": keys["write"], "Content-Type": "application/json"},
    )
    # pydantic v2's str type itself rejects lone surrogates at the schema
    # layer (422) — the shared validator remains the guard for library and
    # duck-typed callers that bypass pydantic.
    assert r.status_code == 422


def test_underscore_schema_keys_are_reserved():
    """Codex-R13: ownership markers (_enrich_keys/_md_keys) — and the whole
    underscore namespace — are server-structural; a schema key there would
    make refresh iterate garbage or delete unrelated fields."""
    emb, store = _CountingEmbedding(), _mem_store()
    for bad in ("_enrich_keys", "_md_keys", "_anything"):
        with pytest.raises(ValueError, match="underscore"):
            ingest_remote_items(
                emb, store, [IngestItem(text="x", source="s")], text_key=bad
            )
        with pytest.raises(ValueError, match="underscore"):
            ingest_remote_items(
                emb, store, [IngestItem(text="x", source="s")], timestamp_key=bad
            )


def test_blank_schema_keys_are_rejected():
    """Agent-R14: an empty/whitespace schema key slipped every guard branch
    and produced a payload keyed by '' — reject as config error."""
    emb, store = _CountingEmbedding(), _mem_store()
    for bad in ("", "  "):
        with pytest.raises(ValueError, match="non-blank"):
            ingest_remote_items(
                emb, store, [IngestItem(text="x", source="s")], text_key=bad
            )
        with pytest.raises(ValueError, match="non-blank"):
            ingest_remote_items(
                emb, store, [IngestItem(text="x", source="s")], timestamp_key=bad
            )


# --------------------------------------------------- bot round-4 batch pins


def test_metadata_validity_interval_must_be_increasing():
    """Bot-R4: valid_from >= valid_until is an empty [from, until) window —
    the memory would be stored but invisible to every as_of query."""
    assert "precede" in validate_remote_item(
        "t", "s", None, [],
        {"valid_from": "2026-02-01", "valid_until": "2026-01-01"},
    )
    assert "precede" in validate_remote_item(
        "t", "s", None, [],
        {"valid_from": "2026-01-01", "valid_until": "2026-01-01"},
    )
    assert validate_remote_item(
        "t", "s", None, [],
        {"valid_from": "2026-01-01", "valid_until": "2026-02-01"},
    ) is None


def _triples_app(monkeypatch, tmp_path):
    import mnemostack.graph.factory as graph_factory
    import mnemostack.server as srv

    calls: list[dict] = []

    class _G:
        def add_triple(self, **kw):
            calls.append(kw)

        def close(self):
            pass

    monkeypatch.setattr(graph_factory, "make_graph_store", lambda *a, **k: _G())
    emb = _CountingEmbedding()
    store = _mem_store("tr4")
    monkeypatch.setattr(srv, "VectorStore", lambda **_: store)
    monkeypatch.setattr(srv, "get_provider", lambda _n, **_k: emb)

    class _Probe:
        def get_collections(self):
            return object()

    monkeypatch.setattr(srv, "_make_probe_client", lambda *_a, **_k: _Probe())
    for name in ("Recaller", "VectorRetriever", "BM25Retriever", "MemgraphRetriever",
                 "TemporalRetriever"):
        monkeypatch.setattr(srv, name, lambda **_: object())
    monkeypatch.setattr(srv, "build_full_pipeline", lambda **_: object())
    monkeypatch.setattr(srv, "FileStateStore", lambda path: object())

    def _no_llm(*_a, **_k):
        raise RuntimeError("no llm")

    monkeypatch.setattr(srv, "get_llm", _no_llm)
    from mnemostack.auth import FileKeyStore

    ks = FileKeyStore(tmp_path / "k4.json")
    _, wk = ks.issue("alpha", ["write"])
    app = build_app(
        ServerConfig(
            provider_name="fake", llm_name="fake",
            graph_uri="bolt://graph.invalid:7687",
            auth_enabled=True, keys_file=str(tmp_path / "k4.json"),
        )
    )
    return TestClient(app), {"X-API-Key": wk}, calls


def test_triples_reject_surrogates_and_noncanonical_predicates(monkeypatch, tmp_path):
    """Bot-R4: surrogate entities die in the bolt driver (400, not 502);
    predicates the store would NORMALIZE collide silently ('works-at' and
    'works at' -> one WORKS_AT edge) — only canonical identifiers pass."""
    client, hdr, calls = _triples_app(monkeypatch, tmp_path)
    raw = b'{"triples": [{"subject": "a\\ud800", "predicate": "KNOWS", "object": "b"}]}'
    r = client.post("/triples", content=raw,
                    headers={**hdr, "Content-Type": "application/json"})
    assert r.status_code in (400, 422)  # schema or endpoint layer, never 502
    r2 = client.post(
        "/triples",
        json={"triples": [{"subject": "a", "predicate": "works-at", "object": "b"}]},
        headers=hdr,
    )
    assert r2.status_code == 400 and "relation identifier" in r2.json()["detail"]
    # Documented snake_case predicates (works_on/owns/...) stay VALID — the
    # round-16 canonical-form contract wrongly rejected them (and its
    # non-idempotent suggestion looped forever on digit-leading input).
    r3 = client.post(
        "/triples",
        json={"triples": [{"subject": "a", "predicate": "works_on", "object": "b"}]},
        headers=hdr,
    )
    assert r3.status_code == 200 and calls[-1]["predicate"] == "works_on"
    r4 = client.post(
        "/triples",
        json={"triples": [{"subject": "a", "predicate": "1X", "object": "b"}]},
        headers=hdr,
    )
    assert r4.status_code == 400  # rejected outright — no unreachable suggestion


def test_triples_reject_inverted_intervals(monkeypatch, tmp_path):
    client, hdr, _calls = _triples_app(monkeypatch, tmp_path)
    r = client.post(
        "/triples",
        json={"triples": [{"subject": "a", "predicate": "KNOWS", "object": "b",
                            "valid_from": "2026-02-01", "valid_until": "2026-01-01"}]},
        headers=hdr,
    )
    assert r.status_code == 400 and "precede" in r.json()["detail"]


def test_predicate_rejects_unicode_number_characters():
    """Round-18 (codex+agent): categories No/Nl (superscripts ², Roman
    numerals Ⅳ, circled digits ①) pass a ``\\w``-based regex — ``\\d`` only
    covers Nd — but leading they get silently underscore-prefixed by
    GraphStore._safe_rel (not isalpha), and anywhere they survive _safe_rel
    unchanged (isalnum) yet crash Memgraph's unescaped rel-type grammar.
    Reject them up front on both positions."""
    for bad in ("²abc", "Ⅳabc", "①abc", "a²bc", "aⅣ", "a①bc", "١abc"):
        assert "relation identifier" in (
            validate_remote_triple("s", bad, "o", None, None) or ""
        ), bad
    # Letters of any script + decimal digits + underscore stay valid.
    for good in ("works_on", "работает_в", "中文", "a9", "Éto_1"):
        assert validate_remote_triple("s", good, "o", None, None) is None, good


def test_422_echo_keeps_distinct_surrogate_keys(monkeypatch, tmp_path):
    """Round-18 (agent P3): U+FFFD replacement collapsed metadata keys that
    differ only by their lone surrogate — the dict comprehension then kept a
    single entry, so the 422 echo misrepresented what the caller sent.
    backslashreplace keeps each key distinct (and shows WHICH surrogate)."""
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    # No "items" → the missing-field 422 echoes the WHOLE body as `input`,
    # surrogate keys included.
    raw = b'{"probe": {"a\\ud800": "1", "a\\ud801": "2", "a\\ud802": "3"}}'
    r = TestClient(app).post(
        "/memories",
        content=raw,
        headers={"X-API-Key": keys["write"], "Content-Type": "application/json"},
    )
    assert r.status_code == 422
    body = r.text
    for echoed in ("a\\\\ud800", "a\\\\ud801", "a\\\\ud802"):
        assert echoed in body, echoed


def test_422_echo_never_drops_colliding_sanitized_keys(monkeypatch, tmp_path):
    """Round-19 (agent P2): a key holding the LITERAL text 'a\\ud800'
    (backslash + letters) and a key holding the real lone surrogate
    sanitize to the same string — the echo dict must disambiguate, not
    silently overwrite one entry with the other."""
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    # JSON: first key = escaped backslash + text; second = real surrogate.
    raw = b'{"probe": {"a\\\\ud800": "1", "a\\ud800": "2"}}'
    r = TestClient(app).post(
        "/memories",
        content=raw,
        headers={"X-API-Key": keys["write"], "Content-Type": "application/json"},
    )
    assert r.status_code == 422
    echoed = r.json()["detail"][0]["input"]["probe"]
    assert sorted(echoed.values()) == ["1", "2"]
    assert len(echoed) == 2


def test_422_echo_is_bounded(monkeypatch, tmp_path):
    """Round-20 (agent P1): the 422 echo runs pre-business-caps on the event
    loop — unbounded, a batch of crafted all-colliding keys made the
    disambiguation loop quadratic (seconds of stalled loop per request).
    The echo is a diagnostic: cap entries per container and string length."""
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    many = ", ".join(f'"k{i}": "v"' for i in range(200))
    long_s = "x" * 5000
    arr = ", ".join('"e"' for _ in range(200))
    raw = f'{{"probe": {{{many}}}, "big": "{long_s}", "arr": [{arr}]}}'.encode()
    r = TestClient(app).post(
        "/memories",
        content=raw,
        headers={"X-API-Key": keys["write"], "Content-Type": "application/json"},
    )
    assert r.status_code == 422
    echoed = r.json()["detail"][0]["input"]
    assert len(echoed["probe"]) == 65  # 64 entries + omission marker
    assert any("omitted" in str(v) for v in echoed["probe"].values())
    assert len(echoed["arr"]) == 65 and "omitted" in echoed["arr"][-1]
    assert len(echoed["big"]) < 3000 and echoed["big"].endswith("…[truncated]")


def test_ingest_failure_leaves_retracted_memories_retracted(monkeypatch):
    """Codex-R20 P2: reactivation runs AFTER the failure-prone new-item
    ingest (embedding-space guard, provider, upsert) — a mixed batch whose
    ingest step raises must not leave the invalidated duplicate visibly
    reactivated behind a 5xx response."""
    import mnemostack.ingest as ingest_mod

    emb, store = _CountingEmbedding(), _mem_store()
    (first,) = ingest_remote_items(
        emb, store, [IngestItem(text="fact", source="s")], tenant="a"
    )
    store.invalidate([first.id], tenant="a")

    class _Boom(Exception):
        pass

    def _fail(self, items):
        raise _Boom("space conflict")

    monkeypatch.setattr(ingest_mod.Ingestor, "ingest", _fail)
    with pytest.raises(_Boom):
        ingest_remote_items(
            emb,
            store,
            [IngestItem(text="fact", source="s"), IngestItem(text="new", source="s")],
            tenant="a",
        )
    point = store.client.retrieve(store.collection, ids=[first.id], with_payload=True)[0]
    assert point.payload.get("invalidated_at")  # still retracted — no side effect
