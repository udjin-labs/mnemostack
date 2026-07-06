"""Read-only web inspector — endpoint + tenant-isolation tests.

Uses a real in-memory Qdrant (`:memory:`) seeded across two tenants so the
isolation assertions are genuine (a tenant's view must never include another's
records), and a fake embedder for the vector-search path.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import Distance  # noqa: E402

import mnemostack.inspector as insp  # noqa: E402
from mnemostack.server import ServerConfig  # noqa: E402
from mnemostack.vector import VectorStore  # noqa: E402

_VEC = [1.0, 0.0, 0.0, 0.0]


class _FakeProvider:
    dimension = 4
    name = "fake:insp"

    def embed(self, text):
        return _VEC

    def embed_batch(self, texts):
        return [_VEC for _ in texts]


def _seeded_store() -> VectorStore:
    s = VectorStore.__new__(VectorStore)
    s.collection = "mt"
    s.dimension = 4
    s.distance = Distance.COSINE
    s.client = QdrantClient(":memory:")
    s.ensure_collection()
    s.upsert(1, _VEC, {"text": "alpha one", "source": "a1.md"}, tenant="alpha")
    s.upsert(2, _VEC, {"text": "alpha two", "source": "a2.md"}, tenant="alpha")
    s.upsert(3, _VEC, {"text": "beta one", "source": "b1.md"}, tenant="beta")
    s.invalidate([2], tenant="alpha")  # mark one alpha record stale
    return s


@pytest.fixture
def client(monkeypatch):
    store = _seeded_store()
    monkeypatch.setattr(insp, "get_provider", lambda *a, **k: _FakeProvider())
    monkeypatch.setattr(insp, "VectorStore", lambda **_: store)
    # Probe reachability reflects the in-memory store (no real Qdrant to reach).
    monkeypatch.setattr(insp, "_make_probe_client", lambda *a, **k: store.client)
    app = insp.build_inspector_app(
        ServerConfig(provider_name="fake", collection="mt", graph_uri=None)
    )
    return TestClient(app)


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "mnemostack inspector" in r.text
    assert "read-only" in r.text


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_tenants_lists_distinct_tenants_with_counts(client):
    d = client.get("/api/tenants").json()
    by_id = {t["id"]: t["count"] for t in d["tenants"]}
    assert by_id == {"alpha": 2, "beta": 1}


def test_overview_is_tenant_scoped(client):
    d = client.get("/api/overview?tenant=alpha").json()
    assert d["tenant"] == "alpha"
    assert d["points"] == 2
    assert d["invalidated"] == 1  # id 2 was invalidated
    assert d["qdrant"] is True
    assert d["memgraph"] is None  # graph not configured

    d2 = client.get("/api/overview?tenant=beta").json()
    assert d2["points"] == 1
    assert d2["invalidated"] == 0


def test_records_browse_isolated_by_tenant(client):
    alpha = client.get("/api/records?tenant=alpha").json()
    sources = {r["source"] for r in alpha["records"]}
    assert alpha["mode"] == "browse"
    assert sources == {"a1.md", "a2.md"}
    assert "b1.md" not in sources
    # The stale record is flagged.
    assert any(r["invalidated"] for r in alpha["records"])

    beta = client.get("/api/records?tenant=beta").json()
    assert {r["source"] for r in beta["records"]} == {"b1.md"}


def test_records_vector_search_scoped(client):
    d = client.get("/api/records?tenant=beta&q=anything").json()
    assert d["mode"] == "vector search"
    # A vector search from beta must never surface alpha's points.
    assert all(r["source"] == "b1.md" for r in d["records"])


def test_records_bad_filters_json_reports_error(client):
    d = client.get("/api/records?tenant=alpha&filters=not-json").json()
    assert d["records"] == []
    assert "invalid filters JSON" in d["error"]


def test_records_filters_and_tenant_combine(client):
    d = client.get('/api/records?tenant=alpha&filters={"source":"a1.md"}').json()
    assert {r["source"] for r in d["records"]} == {"a1.md"}


def test_filters_cannot_override_tenant_scope(client):
    # A crafted filters={"tenant_id":"beta"} while tenant=alpha ANDs, it can't
    # widen scope — it just contradicts the mandatory tenant filter -> empty.
    d = client.get('/api/records?tenant=alpha&filters={"tenant_id":"beta"}').json()
    assert d["records"] == []


def test_records_bad_filter_shape_returns_clean_error_not_500(client):
    # A filter value Qdrant rejects must degrade to a clean error, not a 500.
    r = client.get('/api/records?tenant=alpha&filters={"source":[1,2]}')
    assert r.status_code == 200
    assert r.json()["records"] == []
    assert "error" in r.json()


def test_console_escapes_attribute_context():
    # The inline console must escape quotes (data-id lives in an HTML attribute)
    # so a non-hash record id can't break out into an XSS vector.
    assert "&quot;" in insp.INSPECTOR_HTML
    assert '[<&>"\']' in insp.INSPECTOR_HTML


def test_browse_works_without_embedding_provider(monkeypatch):
    # Browse-only use must not require an embedding provider (no GEMINI_API_KEY):
    # build_inspector_app is lazy, so get_provider is never called for browsing.
    store = _seeded_store()

    def _boom(*a, **k):
        raise RuntimeError("GEMINI_API_KEY not set")

    monkeypatch.setattr(insp, "get_provider", _boom)
    monkeypatch.setattr(insp, "VectorStore", lambda **_: store)
    app = insp.build_inspector_app(
        ServerConfig(provider_name="gemini", collection="mt", graph_uri=None)
    )
    c = TestClient(app)
    assert {t["id"] for t in c.get("/api/tenants").json()["tenants"]} == {"alpha", "beta"}
    assert c.get("/api/overview?tenant=alpha").status_code == 200
    assert c.get("/api/records?tenant=alpha").status_code == 200  # scroll — no embed


def _app_no_facet(monkeypatch, store, probe_up: bool):
    def _boom(*a, **k):
        raise RuntimeError("facet not supported")

    monkeypatch.setattr(store.client, "facet", _boom)  # simulate old Qdrant / failure
    monkeypatch.setattr(insp, "get_provider", lambda *a, **k: _FakeProvider())
    monkeypatch.setattr(insp, "VectorStore", lambda **_: store)

    class _Probe:
        def get_collections(self):
            if not probe_up:
                raise RuntimeError("connection refused")
            return store.client.get_collections()

    monkeypatch.setattr(insp, "_make_probe_client", lambda *a, **k: _Probe())
    app = insp.build_inspector_app(
        ServerConfig(provider_name="fake", collection="mt", graph_uri=None)
    )
    return TestClient(app)


def test_tenants_falls_back_to_scroll_when_facet_unavailable(monkeypatch):
    # Old Qdrant (< 1.12 Facet API) but reachable → scroll-based discovery.
    store = _seeded_store()
    d = _app_no_facet(monkeypatch, store, probe_up=True).get("/api/tenants").json()
    assert d["ok"] is True and d.get("scanned") is True
    assert {t["id"]: t["count"] for t in d["tenants"]} == {"alpha": 2, "beta": 1}


def test_tenants_reports_error_when_qdrant_unreachable(monkeypatch):
    # Facet fails AND the probe can't reach Qdrant → surface the outage.
    store = _seeded_store()
    d = _app_no_facet(monkeypatch, store, probe_up=False).get("/api/tenants").json()
    assert d["ok"] is False and d["tenants"] == []
    assert "Qdrant unreachable" in d["error"]


def test_records_returns_clean_error_on_bad_filter(client):
    # /api/records returns a 200 error payload (records:[] + error), which the UI
    # must surface instead of rendering "0 record(s)".
    d = client.get("/api/records?tenant=alpha&filters=not-json").json()
    assert d["records"] == []
    assert d.get("error")


def test_inspector_html_has_manual_tenant_and_larger_limit():
    from mnemostack.inspector import INSPECTOR_HTML

    assert 'id="tenant-manual"' in INSPECTOR_HTML  # inspect any tenant, not just page 1
    assert "/api/tenants?limit=1000" in INSPECTOR_HTML


def test_cmd_inspect_wires_graph_timeout_to_probe(monkeypatch):
    uvicorn = pytest.importorskip("uvicorn")  # server extra; cmd_inspect imports it

    import mnemostack.cli as cli
    import mnemostack.inspector as insp_mod

    captured = {}
    monkeypatch.setattr(
        insp_mod, "build_inspector_app", lambda cfg: (captured.__setitem__("cfg", cfg), object())[1]
    )
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    args = cli.build_parser().parse_args(["inspect", "--graph-timeout", "7"])
    cli.cmd_inspect(args)
    assert captured["cfg"].graph_health_timeout == 7.0


def test_graph_reachable_closes_driver_on_failure(monkeypatch):
    pytest.importorskip("neo4j")
    import neo4j

    from mnemostack.inspector import _graph_reachable

    closed = {"n": 0}

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def run(self, *a):
            raise RuntimeError("graph down")

    class _Driver:
        def session(self, **k):
            return _Session()

        def close(self):
            closed["n"] += 1

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", lambda *a, **k: _Driver())
    cfg = ServerConfig(graph_uri="bolt://x", graph_user="", graph_password="", graph_database=None)
    assert _graph_reachable(cfg) is False
    assert closed["n"] == 1  # driver closed despite the failure


def test_unscoped_browse_sees_all_including_legacy(monkeypatch):
    # A legacy/default single-tenant collection (points without tenant_id) must be
    # inspectable via the unscoped (tenant="") mode, not blank.
    store = _seeded_store()
    store.upsert(9, _VEC, {"text": "legacy", "source": "legacy.md"})  # no tenant
    monkeypatch.setattr(insp, "get_provider", lambda *a, **k: _FakeProvider())
    monkeypatch.setattr(insp, "VectorStore", lambda **_: store)
    monkeypatch.setattr(insp, "_make_probe_client", lambda *a, **k: store.client)
    c = TestClient(
        insp.build_inspector_app(ServerConfig(provider_name="fake", collection="mt", graph_uri=None))
    )
    assert c.get("/api/overview").json()["points"] == 4  # alpha 2 + beta 1 + legacy 1
    sources = {r["source"] for r in c.get("/api/records").json()["records"]}
    assert {"a1.md", "b1.md", "legacy.md"} <= sources


def test_data_endpoints_gate_on_unreachable_qdrant(monkeypatch):
    store = _seeded_store()
    monkeypatch.setattr(insp, "get_provider", lambda *a, **k: _FakeProvider())
    monkeypatch.setattr(insp, "VectorStore", lambda **_: store)

    class _DownProbe:
        def get_collections(self):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(insp, "_make_probe_client", lambda *a, **k: _DownProbe())
    c = TestClient(
        insp.build_inspector_app(ServerConfig(provider_name="fake", collection="mt", graph_uri=None))
    )
    assert c.get("/api/tenants").json()["error"] == "Qdrant unreachable"
    assert c.get("/api/overview").json()["qdrant"] is False
    assert c.get("/api/records").json()["error"] == "Qdrant unreachable"
