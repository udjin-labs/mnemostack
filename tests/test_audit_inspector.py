"""Audit log — inspector (admin console) wiring.

Split from test_audit.py so the module/CLI audit tests still run on a
core-only install: FastAPI is an optional `server` extra, and a module-level
importorskip would skip the whole file at collection.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from mnemostack.auth import FileKeyStore  # noqa: E402


def _events(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def audit_file(monkeypatch, tmp_path):
    p = tmp_path / "audit.jsonl"
    monkeypatch.setenv("MNEMOSTACK_AUDIT_FILE", str(p))
    return p


from fastapi.testclient import TestClient  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import Distance  # noqa: E402

import mnemostack.inspector as insp  # noqa: E402
from mnemostack.server import ServerConfig  # noqa: E402
from mnemostack.vector import VectorStore  # noqa: E402


def _mini_store() -> VectorStore:
    s = VectorStore.__new__(VectorStore)
    s.collection = "mt"
    s.dimension = 4
    s.distance = Distance.COSINE
    s.client = QdrantClient(":memory:")
    s.ensure_collection()
    s.upsert(1, [1.0, 0, 0, 0], {"text": "one"}, tenant="acme")
    return s


def _admin_client(monkeypatch, tmp_path):
    """Inspector under --auth with real key/quota stores.
    Returns (client, admin_key, admin_id, read_key)."""
    store = _mini_store()
    monkeypatch.setattr(insp, "VectorStore", lambda **_: store)
    monkeypatch.setattr(insp, "_make_probe_client", lambda *a, **k: store.client)
    ks = FileKeyStore(tmp_path / "keys.json")
    admin_id, admin_key = ks.issue("ops", ["admin"])
    _, read_key = ks.issue("acme", ["read"])
    app = insp.build_inspector_app(
        ServerConfig(
            provider_name="fake", collection="mt", graph_uri=None,
            auth_enabled=True,
            keys_file=str(tmp_path / "keys.json"),
            quotas_file=str(tmp_path / "quotas.json"),
        )
    )
    return TestClient(app), admin_key, admin_id, read_key


def _hdr(key):
    return {"X-API-Key": key}


def test_inspector_key_issue_audits_with_admin_actor(monkeypatch, tmp_path, audit_file):
    c, admin_key, admin_id, _ = _admin_client(monkeypatch, tmp_path)
    r = c.post(
        "/api/keys",
        json={"tenant": "acme", "scopes": ["read"], "label": "ci"},
        headers=_hdr(admin_key),
    )
    assert r.status_code == 201
    ev = [e for e in _events(audit_file) if e["action"] == "keys.issue"]
    assert len(ev) == 1
    assert ev[0]["surface"] == "inspector" and ev[0]["actor"] == f"key:{admin_id}"
    assert ev[0]["tenant"] == "acme" and ev[0]["details"]["key_id"] == r.json()["id"]
    # never the plaintext returned by the endpoint
    assert r.json()["key"] not in audit_file.read_text()


def test_inspector_denials_are_audited_but_probes_are_not(monkeypatch, tmp_path, audit_file):
    c, _admin, _aid, read_key = _admin_client(monkeypatch, tmp_path)
    assert c.get("/api/keys").status_code == 401  # missing key: probe, NOT audited
    assert c.get("/api/keys", headers=_hdr("msk_bogus")).status_code == 401
    assert c.get("/api/keys", headers=_hdr(read_key)).status_code == 403
    ev = _events(audit_file)
    assert [e["details"]["reason"] for e in ev] == ["invalid_key", "not_admin"]
    assert all(e["action"] == "auth.denied" and e["outcome"] == "denied" for e in ev)
    assert ev[0]["actor"].startswith("ip:")
    assert ev[1]["actor"].startswith("key:")  # rejected non-admin is attributable
    assert ev[1]["tenant"] == "acme"  # ...and tenant-filterable
    assert "msk_bogus" not in audit_file.read_text()  # never presented key material


def test_inspector_last_admin_revoke_audits_denied(monkeypatch, tmp_path, audit_file):
    c, admin_key, admin_id, _ = _admin_client(monkeypatch, tmp_path)
    r = c.delete(f"/api/keys/{admin_id}", headers=_hdr(admin_key))
    assert r.status_code == 409
    ev = [e for e in _events(audit_file) if e["action"] == "keys.revoke"]
    assert len(ev) == 1 and ev[0]["outcome"] == "denied"
    assert ev[0]["details"]["reason"] == "last_admin"
    assert ev[0]["tenant"] == "ops"  # the key's tenant, from the pre-lookup


def test_inspector_revoke_success_attributes_the_tenant(monkeypatch, tmp_path, audit_file):
    c, admin_key, _aid, _read = _admin_client(monkeypatch, tmp_path)
    read_id = next(
        k["id"] for k in FileKeyStore(tmp_path / "keys.json").list_keys()
        if k["tenant"] == "acme"
    )
    assert c.delete(f"/api/keys/{read_id}", headers=_hdr(admin_key)).status_code == 200
    ev = [e for e in _events(audit_file) if e["action"] == "keys.revoke"]
    assert len(ev) == 1 and ev[0]["outcome"] == "success"
    assert ev[0]["tenant"] == "acme" and ev[0]["details"]["key_id"] == read_id


def test_inspector_audit_endpoint_503_on_symlinked_trail(monkeypatch, tmp_path):
    # Configured-but-diverted trail: /api/audit fails loud (503), and the UI
    # keeps the tab visible rendering the error rather than hiding the outage.
    target = tmp_path / "spoofed.jsonl"
    target.write_text("{}\n")
    link = tmp_path / "audit.jsonl"
    link.symlink_to(target)
    monkeypatch.setenv("MNEMOSTACK_AUDIT_FILE", str(link))
    c, admin_key, *_ = _admin_client(monkeypatch, tmp_path)
    assert c.get("/api/audit", headers=_hdr(admin_key)).status_code == 503


def test_inspector_quota_set_and_remove_audit(monkeypatch, tmp_path, audit_file):
    c, admin_key, *_ = _admin_client(monkeypatch, tmp_path)
    assert (
        c.put("/api/quotas/acme", json={"max_points": 9}, headers=_hdr(admin_key)).status_code
        == 200
    )
    assert c.delete("/api/quotas/acme", headers=_hdr(admin_key)).status_code == 200
    ev = [e for e in _events(audit_file) if e["action"].startswith("quota.")]
    assert [e["action"] for e in ev] == ["quota.set", "quota.remove"]
    assert ev[0]["details"]["max_points"] == 9 and ev[0]["tenant"] == "acme"


def test_inspector_audit_endpoint(monkeypatch, tmp_path, audit_file):
    c, admin_key, *_ = _admin_client(monkeypatch, tmp_path)
    c.put("/api/quotas/acme", json={"max_points": 9}, headers=_hdr(admin_key))
    d = c.get("/api/audit", headers=_hdr(admin_key)).json()
    assert d["enabled"] is True and d["skipped"] == 0
    assert any(e["action"] == "quota.set" for e in d["events"])
    # admin-gated like every management endpoint
    assert c.get("/api/audit").status_code == 401


def test_inspector_audit_endpoint_reports_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("MNEMOSTACK_AUDIT_FILE", raising=False)
    c, admin_key, *_ = _admin_client(monkeypatch, tmp_path)
    d = c.get("/api/audit", headers=_hdr(admin_key)).json()
    assert d == {"enabled": False, "events": [], "skipped": 0}


def test_inspector_audit_endpoint_forbidden_without_auth(monkeypatch):
    store = _mini_store()
    monkeypatch.setattr(insp, "VectorStore", lambda **_: store)
    monkeypatch.setattr(insp, "_make_probe_client", lambda *a, **k: store.client)
    app = insp.build_inspector_app(
        ServerConfig(provider_name="fake", collection="mt", graph_uri=None)
    )
    assert TestClient(app).get("/api/audit").status_code == 403

