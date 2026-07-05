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
