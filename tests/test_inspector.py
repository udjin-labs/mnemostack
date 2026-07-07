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
    monkeypatch.setattr(insp, "_make_probe_client", lambda *a, **k: store.client)
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


def test_unscoped_is_legacy_only_never_cross_tenant(monkeypatch):
    # On a MIXED collection, unscoped (tenant="") shows ONLY untenanted points —
    # never another tenant's data (upholds the never-cross-tenant contract).
    store = _seeded_store()
    store.upsert(9, _VEC, {"text": "legacy", "source": "legacy.md"})  # no tenant_id
    monkeypatch.setattr(insp, "get_provider", lambda *a, **k: _FakeProvider())
    monkeypatch.setattr(insp, "VectorStore", lambda **_: store)
    monkeypatch.setattr(insp, "_make_probe_client", lambda *a, **k: store.client)
    c = TestClient(
        insp.build_inspector_app(ServerConfig(provider_name="fake", collection="mt", graph_uri=None))
    )
    assert c.get("/api/overview").json()["points"] == 1  # only the legacy point
    assert {r["source"] for r in c.get("/api/records").json()["records"]} == {"legacy.md"}
    # A vector search under unscoped is legacy-only too (never alpha/beta).
    assert {r["source"] for r in c.get("/api/records?q=x").json()["records"]} == {"legacy.md"}


def test_unscoped_pure_single_tenant_sees_all(monkeypatch):
    # A default single-tenant collection (no point has tenant_id): unscoped = all.
    s = VectorStore.__new__(VectorStore)
    s.collection = "mt2"
    s.dimension = 4
    s.distance = Distance.COSINE
    s.client = QdrantClient(":memory:")
    s.ensure_collection()
    s.upsert(1, _VEC, {"text": "a", "source": "a.md"})  # no tenant
    s.upsert(2, _VEC, {"text": "b", "source": "b.md"})  # no tenant
    monkeypatch.setattr(insp, "get_provider", lambda *a, **k: _FakeProvider())
    monkeypatch.setattr(insp, "VectorStore", lambda **_: s)
    monkeypatch.setattr(insp, "_make_probe_client", lambda *a, **k: s.client)
    c = TestClient(
        insp.build_inspector_app(ServerConfig(provider_name="fake", collection="mt2", graph_uri=None))
    )
    assert c.get("/api/overview").json()["points"] == 2
    assert {r["source"] for r in c.get("/api/records").json()["records"]} == {"a.md", "b.md"}


def test_cmd_inspect_wires_qdrant_health_timeout(monkeypatch):
    uvicorn = pytest.importorskip("uvicorn")

    import mnemostack.cli as cli
    import mnemostack.inspector as insp_mod

    captured = {}
    monkeypatch.setattr(
        insp_mod, "build_inspector_app", lambda cfg: (captured.__setitem__("cfg", cfg), object())[1]
    )
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    cli.cmd_inspect(cli.build_parser().parse_args(["inspect", "--qdrant-health-timeout", "9"]))
    assert captured["cfg"].qdrant_health_timeout == 9


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


# ---------- admin console (--auth): key + quota management ----------


def _admin_client(monkeypatch, tmp_path):
    """Inspector built with --auth: real key/quota stores + the seeded data store.
    Returns (client, admin_key, read_key, keys_file, quotas_file)."""
    from mnemostack.auth import FileKeyStore

    store = _seeded_store()
    monkeypatch.setattr(insp, "get_provider", lambda *a, **k: _FakeProvider())
    monkeypatch.setattr(insp, "VectorStore", lambda **_: store)
    monkeypatch.setattr(insp, "_make_probe_client", lambda *a, **k: store.client)
    kf, qf = str(tmp_path / "keys.json"), str(tmp_path / "quotas.json")
    ks = FileKeyStore(kf)
    _, admin_key = ks.issue("ops", ["admin"])
    _, read_key = ks.issue("acme", ["read"])
    app = insp.build_inspector_app(
        ServerConfig(
            provider_name="fake", collection="mt", graph_uri=None,
            auth_enabled=True, keys_file=kf, quotas_file=qf,
        )
    )
    return TestClient(app), admin_key, read_key, kf, qf


def _hdr(key):
    return {"X-API-Key": key}


def test_admin_console_requires_a_key(monkeypatch, tmp_path):
    c, *_ = _admin_client(monkeypatch, tmp_path)
    # under --auth EVERY /api call needs an admin key (the whole console is admin-only)
    assert c.get("/api/tenants").status_code == 401
    assert c.get("/api/keys").status_code == 401
    assert c.get("/api/quotas").status_code == 401


def test_non_admin_key_is_forbidden(monkeypatch, tmp_path):
    c, _admin, read_key, *_ = _admin_client(monkeypatch, tmp_path)
    assert c.get("/api/tenants", headers=_hdr(read_key)).status_code == 403
    assert c.get("/api/keys", headers=_hdr(read_key)).status_code == 403


def test_admin_key_unlocks_browse_and_management(monkeypatch, tmp_path):
    c, admin_key, *_ = _admin_client(monkeypatch, tmp_path)
    assert c.get("/api/tenants", headers=_hdr(admin_key)).status_code == 200
    assert c.get("/api/keys", headers=_hdr(admin_key)).status_code == 200
    assert c.get("/api/quotas", headers=_hdr(admin_key)).status_code == 200


def test_management_forbidden_without_auth(monkeypatch):
    # the default read-only inspector: browse open, management endpoints 403
    store = _seeded_store()
    monkeypatch.setattr(insp, "get_provider", lambda *a, **k: _FakeProvider())
    monkeypatch.setattr(insp, "VectorStore", lambda **_: store)
    monkeypatch.setattr(insp, "_make_probe_client", lambda *a, **k: store.client)
    c = TestClient(insp.build_inspector_app(
        ServerConfig(provider_name="fake", collection="mt", graph_uri=None)
    ))
    assert c.get("/api/tenants").status_code == 200  # browse open, no key
    assert c.get("/api/keys").status_code == 403      # management needs --auth
    assert c.post("/api/keys", json={"tenant": "x", "scopes": ["read"]}).status_code == 403
    assert c.put("/api/quotas/x", json={"max_points": 1}).status_code == 403


def test_issue_key_returns_plaintext_once_and_lists_redacted(monkeypatch, tmp_path):
    from mnemostack.auth import FileKeyStore

    c, admin_key, _read, kf, _qf = _admin_client(monkeypatch, tmp_path)
    r = c.post(
        "/api/keys", headers=_hdr(admin_key),
        json={"tenant": "acme", "scopes": ["read", "write"], "label": "svc"},
    )
    assert r.status_code == 201
    plaintext = r.json()["key"]
    # the issued key actually authenticates as that tenant/scopes
    p = FileKeyStore(kf).verify(plaintext)
    assert p is not None and p.tenant == "acme" and p.can("write")
    # the listing never leaks a plaintext or hash
    listed = c.get("/api/keys", headers=_hdr(admin_key)).json()["keys"]
    row = next(k for k in listed if k["id"] == r.json()["id"])
    assert row["tenant"] == "acme" and set(row["scopes"]) == {"read", "write"}
    assert "key" not in row and "hash" not in row


def test_issue_key_rejects_unknown_scope(monkeypatch, tmp_path):
    c, admin_key, *_ = _admin_client(monkeypatch, tmp_path)
    r = c.post("/api/keys", headers=_hdr(admin_key),
               json={"tenant": "acme", "scopes": ["root"]})
    assert r.status_code == 400 and "scope" in r.json()["detail"].lower()


def test_revoke_key(monkeypatch, tmp_path):
    c, admin_key, *_ = _admin_client(monkeypatch, tmp_path)
    kid = c.post("/api/keys", headers=_hdr(admin_key),
                 json={"tenant": "acme", "scopes": ["read"]}).json()["id"]
    assert c.delete(f"/api/keys/{kid}", headers=_hdr(admin_key)).json()["revoked"] is True
    ids = [k["id"] for k in c.get("/api/keys", headers=_hdr(admin_key)).json()["keys"]]
    assert kid not in ids
    assert c.delete("/api/keys/nope", headers=_hdr(admin_key)).status_code == 404


def test_revoke_last_admin_key_refused(monkeypatch, tmp_path):
    # the seeded store has exactly one admin key ("ops"); revoking it would lock out
    c, admin_key, *_ = _admin_client(monkeypatch, tmp_path)
    keys = c.get("/api/keys", headers=_hdr(admin_key)).json()["keys"]
    admin_id = next(k["id"] for k in keys if "admin" in k["scopes"])
    r = c.delete(f"/api/keys/{admin_id}", headers=_hdr(admin_key))
    assert r.status_code == 409 and "last admin" in r.json()["detail"]
    # a SECOND admin key makes the first revocable again
    c.post("/api/keys", headers=_hdr(admin_key),
           json={"tenant": "ops2", "scopes": ["admin"]})
    assert c.delete(f"/api/keys/{admin_id}", headers=_hdr(admin_key)).json()["revoked"] is True


def test_set_quota_is_partial_and_validated(monkeypatch, tmp_path):
    c, admin_key, *_ = _admin_client(monkeypatch, tmp_path)
    assert c.put("/api/quotas/acme", headers=_hdr(admin_key),
                 json={"max_points": 1000}).status_code == 200
    # setting the rate must NOT wipe the size cap (partial update)
    q = c.put("/api/quotas/acme", headers=_hdr(admin_key),
              json={"max_rps": 5}).json()
    assert q["max_points"] == 1000 and q["max_rps"] == 5.0 and q["burst"] == 5
    # a bad value is a clean 400, not a 500
    assert c.put("/api/quotas/acme", headers=_hdr(admin_key),
                 json={"max_rps": -1}).status_code == 400
    listed = c.get("/api/quotas", headers=_hdr(admin_key)).json()["quotas"]
    assert any(x["tenant"] == "acme" and x["max_points"] == 1000 for x in listed)


def test_set_quota_empty_body_does_not_provision(monkeypatch, tmp_path):
    # a no-op PUT (no fields) must 400, not create an empty quota row for a new tenant
    c, admin_key, *_ = _admin_client(monkeypatch, tmp_path)
    assert c.put("/api/quotas/ghost", headers=_hdr(admin_key), json={}).status_code == 400
    listed = c.get("/api/quotas", headers=_hdr(admin_key)).json()["quotas"]
    assert not any(x["tenant"] == "ghost" for x in listed)


def test_quota_tenant_with_slash(monkeypatch, tmp_path):
    # tenant ids may contain '/', which the stores accept — the quota route must
    # handle them (path converter), or such a tenant is unmanageable via the API.
    c, admin_key, *_ = _admin_client(monkeypatch, tmp_path)
    r = c.put("/api/quotas/team%2Facme", headers=_hdr(admin_key), json={"max_points": 7})
    assert r.status_code == 200 and r.json()["tenant"] == "team/acme"
    listed = c.get("/api/quotas", headers=_hdr(admin_key)).json()["quotas"]
    assert any(x["tenant"] == "team/acme" and x["max_points"] == 7 for x in listed)
    assert c.delete("/api/quotas/team%2Facme", headers=_hdr(admin_key)).json()["removed"] is True


def test_remove_quota(monkeypatch, tmp_path):
    c, admin_key, *_ = _admin_client(monkeypatch, tmp_path)
    c.put("/api/quotas/acme", headers=_hdr(admin_key), json={"max_points": 5})
    assert c.delete("/api/quotas/acme", headers=_hdr(admin_key)).json()["removed"] is True
    assert c.delete("/api/quotas/acme", headers=_hdr(admin_key)).json()["removed"] is False


def test_tenant_list_unions_config_tenants(monkeypatch, tmp_path):
    # a tenant that has a key but no data still shows up (count 0)
    c, admin_key, *_ = _admin_client(monkeypatch, tmp_path)
    c.post("/api/keys", headers=_hdr(admin_key),
           json={"tenant": "gamma", "scopes": ["read"]})
    tenants = {t["id"]: t["count"] for t in
               c.get("/api/tenants", headers=_hdr(admin_key)).json()["tenants"]}
    assert tenants.get("alpha") == 2       # from data
    assert tenants.get("gamma") == 0       # config-only (keystore), no data yet
