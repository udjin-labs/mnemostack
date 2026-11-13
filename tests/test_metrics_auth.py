"""Metrics view selection without tenant request accounting."""

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException
from test_remote_ingest import _ingest_app

from mnemostack.auth import FileKeyStore, KeyStoreError
from mnemostack.observability.recorder import get_recorder
from mnemostack.ratelimit import RateLimiter


@pytest.fixture
def metrics_app(monkeypatch, tmp_path):
    app, _, _, keys = _ingest_app(monkeypatch, tmp_path)
    _, keys["admin"] = FileKeyStore(tmp_path / "keys.json").issue("operator", ["admin"])
    rec = get_recorder()
    for tenant, value in (("private-alpha", 2), ("private-beta", 5)):
        labels = {"tenant": tenant}
        rec.record_counter("mnemostack.tenant.requests", value, labels)
        rec.record_histogram("latency", value, labels)
    endpoint = next(route.endpoint for route in app.routes if route.path == "/metrics")
    return endpoint, keys, rec


def test_anonymous_metrics_totals(metrics_app):
    endpoint, _, _ = metrics_app
    body = endpoint(authorization=None, x_api_key=None).body.decode()
    assert "tenant=" not in body
    assert "private-" not in body
    assert "mnemostack_tenant_requests_total 7.0" in body


@pytest.mark.parametrize("header", ["authorization", "x_api_key"])
def test_admin_metrics_preserve_tenants_without_metering(metrics_app, monkeypatch, header):
    endpoint, keys, rec = metrics_app

    def reject_check(*args):
        pytest.fail("scrapes must bypass the rate limiter")

    monkeypatch.setattr(RateLimiter, "check", reject_check)
    before = rec.snapshot_counters()
    headers = {"authorization": None, "x_api_key": None}
    headers[header] = f"Bearer {keys['admin']}" if header == "authorization" else keys["admin"]
    for _ in range(2):
        body = endpoint(**headers).body.decode()
        for tenant, value in (("private-alpha", 2), ("private-beta", 5)):
            assert f'mnemostack_tenant_requests_total{{tenant="{tenant}"}} {value}.0' in body
            assert f'latency_count{{tenant="{tenant}"}} 1' in body
            assert f'latency_sum{{tenant="{tenant}"}} {value}.0' in body
    assert rec.snapshot_counters() == before


@pytest.mark.parametrize(
    ("key", "status", "detail"),
    [
        ("read", 403, "key lacks 'admin' scope"),
        ("write", 403, "key lacks 'admin' scope"),
        ("invalid", 401, "invalid service key"),
    ],
)
def test_metrics_rejects_non_operator_keys(metrics_app, key, status, detail):
    endpoint, keys, _ = metrics_app
    with pytest.raises(HTTPException) as exc:
        endpoint(authorization=None, x_api_key=keys.get(key, "invalid"))
    assert exc.value.status_code == status
    assert exc.value.detail == detail


@pytest.mark.parametrize("header", ["authorization", "x_api_key"])
def test_metrics_auth_disabled_ignores_key(monkeypatch, tmp_path, header):
    app, _, _, _ = _ingest_app(monkeypatch, tmp_path, auth=False)
    rec = get_recorder()
    for tenant, value in (("private-alpha", 2), ("private-beta", 5)):
        rec.record_counter("requests", value, {"tenant": tenant})
    endpoint = next(route.endpoint for route in app.routes if route.path == "/metrics")
    headers = {"authorization": None, "x_api_key": None}
    headers[header] = "Bearer invalid" if header == "authorization" else "invalid"
    body = endpoint(**headers).body.decode()
    assert "tenant" not in body
    assert "private-" not in body
    assert "requests_total 7.0" in body


def test_metrics_verification_errors_propagate(metrics_app, monkeypatch):
    endpoint, keys, _ = metrics_app

    def broken_verify(self, key):
        raise KeyStoreError("unavailable")

    monkeypatch.setattr(FileKeyStore, "verify", broken_verify)
    with pytest.raises(KeyStoreError, match="unavailable"):
        endpoint(authorization=None, x_api_key=keys["admin"])


def test_metrics_views_over_http(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    app, _, _, keys = _ingest_app(monkeypatch, tmp_path)
    _, admin = FileKeyStore(tmp_path / "keys.json").issue("operator", ["admin"])
    get_recorder().record_counter("mnemostack.tenant.requests", 2, {"tenant": "private-alpha"})
    client = TestClient(app)

    anonymous = client.get("/metrics")
    assert anonymous.status_code == 200
    assert "private-alpha" not in anonymous.text
    assert anonymous.headers["cache-control"] == "no-store"

    labelled = client.get("/metrics", headers={"Authorization": f"Bearer {admin}"})
    assert labelled.status_code == 200
    assert 'tenant="private-alpha"' in labelled.text
    assert labelled.headers["cache-control"] == "no-store"
    by_header = client.get("/metrics", headers={"X-API-Key": admin})
    assert by_header.status_code == 200
    assert 'tenant="private-alpha"' in by_header.text
    assert by_header.headers["cache-control"] == "no-store"

    tenant_key = next(iter(keys.values()))
    denied = client.get("/metrics", headers={"X-API-Key": tenant_key})
    assert denied.status_code == 403
    assert client.get("/metrics", headers={"X-API-Key": "msk_bogus"}).status_code == 401
