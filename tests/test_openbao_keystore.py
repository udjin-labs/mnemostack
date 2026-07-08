"""OpenBao verify-only key store — transport-faked tests + factory selection."""

from __future__ import annotations

import json

import pytest

from mnemostack.auth import KeyStoreError, hash_key, make_key_store
from mnemostack.openbao import OpenBaoKeyStore


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class _FakeBao:
    """Records requests; serves KV-v2 reads from a dict and AppRole logins."""

    def __init__(self, records=None, *, valid_tokens=("tok",), login_token="tok"):
        self.records = dict(records or {})  # hash -> record dict
        self.valid_tokens = set(valid_tokens)
        self.login_token = login_token
        self.calls: list[tuple[str, str]] = []
        self.boom = False  # simulate an unreachable store

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url))
        if self.boom:
            raise OSError("connection refused")
        if url.endswith("/v1/auth/approle/login"):
            creds = json.loads(body)
            if creds.get("role_id") == "rid" and creds.get("secret_id") == "sid":
                return 200, json.dumps({"auth": {"client_token": self.login_token}}).encode()
            return 400, b"{}"
        if headers.get("X-Vault-Token") not in self.valid_tokens:
            return 403, b"{}"
        h = url.rsplit("/", 1)[-1]
        rec = self.records.get(h)
        if rec is None:
            return 404, b"{}"
        return 200, json.dumps({"data": {"data": rec}}).encode()


def _store(bao, clock=None, **kw):
    kw.setdefault("token", "tok")
    kw.setdefault("cache_ttl", 5.0)
    return OpenBaoKeyStore(
        "https://bao.example:8200", transport=bao,
        clock=clock or _Clock(), **kw,
    )


def test_verify_known_key_resolves_principal():
    bao = _FakeBao({hash_key("msk_a"): {"tenant": "acme", "scopes": ["read", "write"]}})
    p = _store(bao).verify("msk_a")
    assert p is not None and p.tenant == "acme" and p.can("write") and not p.can("admin")
    # the lookup used the KV-v2 data path with the key's hash, never the plaintext
    method, url = bao.calls[-1]
    assert method == "GET" and url.endswith(f"/v1/secret/data/mnemostack/keys/{hash_key('msk_a')}")
    assert "msk_a" not in url


def test_verify_accepts_comma_string_scopes():
    # `bao kv put ... scopes=read,write` stores a string — must normalize.
    bao = _FakeBao({hash_key("k"): {"tenant": "t", "scopes": "read,write"}})
    p = _store(bao).verify("k")
    assert p is not None and p.can("read") and p.can("write")


def test_verify_unknown_key_is_none_and_not_cached():
    bao = _FakeBao()
    store = _store(bao)
    assert store.verify("msk_new") is None
    reads_before = len(bao.calls)
    # a just-added key must work immediately: the miss was NOT negatively cached
    bao.records[hash_key("msk_new")] = {"tenant": "t", "scopes": ["read"]}
    assert store.verify("msk_new") is not None
    assert len(bao.calls) > reads_before  # it re-queried the store


def test_positive_cache_bounds_reads_and_revocation():
    clock = _Clock()
    h = hash_key("k")
    bao = _FakeBao({h: {"tenant": "t", "scopes": ["read"]}})
    store = _store(bao, clock=clock, cache_ttl=5.0)
    assert store.verify("k") is not None
    n = len(bao.calls)
    clock.t = 3.0
    assert store.verify("k") is not None  # within TTL: served from cache
    assert len(bao.calls) == n
    # revoke in the store; visible after the TTL expires (bounded latency)
    del bao.records[h]
    clock.t = 4.0
    assert store.verify("k") is not None  # still cached (revocation not yet visible)
    clock.t = 6.0
    assert store.verify("k") is None  # TTL elapsed -> re-read -> revoked


@pytest.mark.parametrize(
    "rec",
    [
        {"tenant": "", "scopes": ["read"]},  # empty tenant
        {"tenant": 5, "scopes": ["read"]},  # non-string tenant
        {"tenant": "t", "scopes": {"admin": True}},  # dict scopes must never grant
        {"tenant": "t", "scopes": ["superuser"]},  # unknown scope
        {"tenant": "t"},  # no scopes
        "not-a-dict",
    ],
)
def test_malformed_record_denies(rec):
    bao = _FakeBao({hash_key("k"): rec})
    assert _store(bao).verify("k") is None


def test_unreachable_store_fails_closed():
    bao = _FakeBao({hash_key("k"): {"tenant": "t", "scopes": ["read"]}})
    store = _store(bao)
    bao.boom = True
    assert store.verify("k") is None  # deny, not crash


def test_non_200_response_fails_closed():
    bao = _FakeBao({hash_key("k"): {"tenant": "t", "scopes": ["read"]}},
                   valid_tokens=("other",))  # our token -> 403, no approle to retry
    assert _store(bao).verify("k") is None


def test_approle_login_and_relogin_on_expiry():
    h = hash_key("k")
    bao = _FakeBao({h: {"tenant": "t", "scopes": ["read"]}}, valid_tokens=("tok",))
    store = OpenBaoKeyStore(
        "https://bao.example:8200", transport=bao, clock=_Clock(),
        role_id="rid", secret_id="sid", cache_ttl=0.0,  # no cache: every call reads
    )
    assert store.verify("k") is not None  # lazy login happened first
    assert any(u.endswith("/auth/approle/login") for _m, u in bao.calls)
    # token rotates server-side: old token now invalid, login yields the new one
    bao.valid_tokens = {"tok2"}
    bao.login_token = "tok2"
    assert store.verify("k") is not None  # 403 -> one re-login -> retry succeeds


def test_requires_token_or_approle():
    with pytest.raises(KeyStoreError):
        OpenBaoKeyStore("https://bao.example:8200")
    with pytest.raises(KeyStoreError):
        OpenBaoKeyStore("")


# ---------- make_key_store factory ----------


def test_factory_defaults_to_file(tmp_path, monkeypatch):
    from mnemostack.auth import FileKeyStore

    monkeypatch.delenv("MNEMOSTACK_KEYSTORE", raising=False)
    store = make_key_store(tmp_path / "keys.json")
    assert isinstance(store, FileKeyStore)


def test_factory_builds_openbao_from_env(monkeypatch):
    monkeypatch.setenv("MNEMOSTACK_KEYSTORE", "openbao")
    monkeypatch.setenv("MNEMOSTACK_OPENBAO_URL", "https://bao.example:8200")
    monkeypatch.setenv("MNEMOSTACK_OPENBAO_TOKEN", "tok")
    monkeypatch.setenv("MNEMOSTACK_OPENBAO_MOUNT", "kv")
    monkeypatch.setenv("MNEMOSTACK_OPENBAO_PATH_PREFIX", "svc/keys")
    store = make_key_store()
    assert isinstance(store, OpenBaoKeyStore)
    assert store._mount == "kv" and store._prefix == "svc/keys"


def test_factory_openbao_misconfig_fails_loud(monkeypatch):
    monkeypatch.setenv("MNEMOSTACK_KEYSTORE", "openbao")
    monkeypatch.delenv("MNEMOSTACK_OPENBAO_URL", raising=False)
    with pytest.raises(KeyStoreError):  # boot must fail, not silently fall back
        make_key_store()
    monkeypatch.setenv("MNEMOSTACK_OPENBAO_URL", "https://bao.example:8200")
    for var in ("MNEMOSTACK_OPENBAO_TOKEN", "BAO_TOKEN", "VAULT_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(KeyStoreError):  # url but no credentials
        make_key_store()


def test_factory_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("MNEMOSTACK_KEYSTORE", "etcd")
    with pytest.raises(KeyStoreError):
        make_key_store()


def test_kv2_soft_delete_denies():
    # `bao kv delete` (the COMMON revocation path) soft-deletes: the read returns
    # HTTP 200 with data.data = null, not a 404 — must deny, not crash.
    def bao(method, url, headers, body):
        return 200, json.dumps({"data": {"data": None}}).encode()

    assert _store(bao).verify("k") is None


def test_redirects_are_refused_by_the_real_transport():
    # urllib's default opener follows 3xx and re-sends ALL headers — including
    # X-Vault-Token — to the redirect target, which would hand the store
    # credential to whatever host a hostile/compromised endpoint points at.
    # The real transport must therefore refuse redirects: the 3xx comes back as
    # a plain non-200 response (-> deny), the Location is never fetched.
    import http.server
    import threading

    leaked: list[str] = []

    class _Redirector(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/leak"):
                leaked.append(self.headers.get("X-Vault-Token", ""))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")
                return
            self.send_response(302)  # try to bounce the client (and its token)
            self.send_header("Location", f"http://{self.server.server_address[0]}:{self.server.server_address[1]}/leak")
            self.end_headers()

        def log_message(self, *a):  # keep test output quiet
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), _Redirector)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        host, port = srv.server_address
        store = OpenBaoKeyStore(f"http://{host}:{port}", token="tok", cache_ttl=0.0)
        assert store.verify("k") is None  # 302 -> refused -> non-200 -> deny
        assert leaked == []  # the token was never re-sent to the redirect target
    finally:
        srv.shutdown()
        t.join(timeout=5)


def test_factory_whitespace_backend_is_file(monkeypatch, tmp_path):
    from mnemostack.auth import FileKeyStore

    monkeypatch.setenv("MNEMOSTACK_KEYSTORE", "   ")
    assert isinstance(make_key_store(tmp_path / "k.json"), FileKeyStore)
