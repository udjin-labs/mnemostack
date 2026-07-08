"""OpenBao (Vault-compatible) key-store backend — verify-only, stdlib HTTP.

An alternative ``KeyStore`` for deployments that already run a secret store:
service-key records live in OpenBao's KV-v2 engine instead of a local JSON
file, so key lifecycle (rotation, audit, replication across nodes) belongs to
the store's own tooling rather than being reinvented on a flat file.

Layout: one KV-v2 entry per key, at ``<mount>/<path_prefix>/<sha256-of-key>``,
with the same record shape ``FileKeyStore`` uses::

    bao kv put secret/mnemostack/keys/<hash> tenant=acme scopes=read,write

The adapter is **verify-only** — it implements ``verify(key)`` and nothing
else, and only ever needs a read-capable token/AppRole. Issue/revoke/list stay
in the store's own tooling (``bao kv put`` / ``delete`` / ``list``); the
``mnemostack keys`` CLI keeps managing the local file store only.

Semantics mirror ``FileKeyStore.verify`` exactly: plaintext never leaves the
process (the lookup key is the SHA-256 digest), a malformed record (empty
tenant, non-list/unknown scopes) denies, and any store error — unreachable,
bad token, malformed response — **fails closed** (deny) with a loud log.

The MCP server re-verifies the key on *every* tool call, so positive results
are cached for a short ``cache_ttl``. The TTL bounds how long **any** grant
change takes effect — revocation, a scope downgrade, or moving a key to
another tenant keep serving the old ``Principal`` for at most ``cache_ttl``
seconds. Negative results are never cached, so a just-added key works
immediately. Only the KV **v2** engine is supported (the ``/data/`` read path);
pointing at a v1 mount denies every key.

Uses only the standard library (``urllib``) — no new dependency; TLS
certificates are verified by default.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

from .auth import KeyStoreError, Principal, _normalize_scopes, hash_key

log = logging.getLogger(__name__)

#: (status_code, body_bytes) returned by a transport call.
_Response = tuple[int, bytes]
#: transport(method, url, headers, body) -> (status, body). Injectable for tests.
_Transport = Callable[[str, str, dict[str, str], bytes | None], _Response]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse to follow ANY redirect. urllib's default redirect handler copies
    every request header — including ``X-Vault-Token`` — onto the redirected
    request, so a hostile/compromised endpoint (or an open redirect in a proxy
    in front of it) answering 3xx could exfiltrate the store credential to an
    attacker host. A KV read never legitimately redirects; treating 3xx as a
    plain (deny) response keeps the token pinned to the configured origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None  # -> urlopen raises HTTPError(code=3xx), handled below


def _urllib_transport(timeout: float) -> _Transport:
    opener = urllib.request.build_opener(_NoRedirect())

    def _call(method: str, url: str, headers: dict[str, str], body: bytes | None) -> _Response:
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with opener.open(req, timeout=timeout) as resp:  # noqa: S310 — operator-configured https URL
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            # An HTTP status (404 unknown key, 403 bad token, refused 3xx) is a
            # *response*, not a transport failure — return it for the caller to
            # interpret (any non-200 denies).
            return e.code, e.read()

    return _call


class OpenBaoKeyStore:
    """A verify-only ``KeyStore`` reading KV-v2 records from OpenBao/Vault.

    Auth: pass a ``token`` directly, or ``role_id``/``secret_id`` for AppRole —
    the adapter logs in lazily and re-logs-in once when the token stops being
    accepted (expiry/rotation). Fails closed on any error.
    """

    def __init__(
        self,
        url: str,
        *,
        mount: str = "secret",
        path_prefix: str = "mnemostack/keys",
        token: str | None = None,
        role_id: str | None = None,
        secret_id: str | None = None,
        cache_ttl: float = 5.0,
        timeout: float = 3.0,
        clock: Callable[[], float] = time.monotonic,
        transport: _Transport | None = None,
    ):
        if not url:
            raise KeyStoreError("OpenBao key store requires a url")
        if token is None and not (role_id and secret_id):
            raise KeyStoreError(
                "OpenBao key store requires a token or an AppRole role_id + secret_id"
            )
        if url.startswith("http://"):
            # Allowed (localhost/dev stores are common) but a plaintext transport
            # carries the store token — and AppRole secrets — in the clear.
            log.warning(
                "OpenBao key store URL uses plaintext http:// — the store token "
                "travels unencrypted; use https:// for anything non-local"
            )
        self._base = url.rstrip("/")
        self._mount = mount.strip("/")
        self._prefix = path_prefix.strip("/")
        self._static_token = token
        self._role_id = role_id
        self._secret_id = secret_id
        self._cache_ttl = cache_ttl
        self._clock = clock
        self._transport = transport if transport is not None else _urllib_transport(timeout)
        self._token: str | None = token  # current working token (static or AppRole-issued)
        #: hash -> (expiry, Principal). Positive results only — a miss/deny is
        #: never cached, so a just-added key authenticates immediately.
        self._cache: dict[str, tuple[float, Principal]] = {}
        self._lock = threading.Lock()

    # ---- HTTP ----

    def _login(self) -> str | None:
        """AppRole login → client token, or None (fail closed) on any failure."""
        body = json.dumps({"role_id": self._role_id, "secret_id": self._secret_id}).encode()
        try:
            status, raw = self._transport(
                "POST",
                f"{self._base}/v1/auth/approle/login",
                {"Content-Type": "application/json"},
                body,
            )
            if status != 200:
                log.error("openbao approle login failed (HTTP %s) — denying", status)
                return None
            token = json.loads(raw).get("auth", {}).get("client_token")
            if not isinstance(token, str) or not token:
                log.error("openbao approle login returned no client_token — denying")
                return None
            return token
        except Exception as e:  # noqa: BLE001 — any transport/parse failure denies
            log.error("openbao approle login error (%s) — denying", e)
            return None

    def _read(self, path: str) -> _Response | None:
        """GET a KV path with the current token; one AppRole re-login on 403.

        Returns None (fail closed) when no token can be obtained or the
        transport itself fails. Works on a LOCAL snapshot of the token so a
        concurrent thread clearing/rotating ``self._token`` can't inject
        ``None`` into this request's header mid-flight (worst case both threads
        log in — benign).
        """
        token = self._token
        if token is None:
            token = self._token = self._login()
            if token is None:
                return None
        # mount/prefix are operator config, not attacker input — quoted anyway so
        # an odd character can't silently reshape the request path.
        url = f"{self._base}/v1/{quote(self._mount)}/data/{quote(path, safe='/')}"
        try:
            status, raw = self._transport("GET", url, {"X-Vault-Token": token}, None)
        except Exception as e:  # noqa: BLE001 — unreachable store denies
            log.error("openbao unreachable (%s) — denying (fail closed)", e)
            return None
        if status == 403 and self._role_id and self._secret_id:
            # Token expired/rotated: one fresh AppRole login, one retry.
            token = self._token = self._login()
            if token is None:
                return None
            try:
                status, raw = self._transport("GET", url, {"X-Vault-Token": token}, None)
            except Exception as e:  # noqa: BLE001
                log.error("openbao unreachable (%s) — denying (fail closed)", e)
                return None
        return status, raw

    # ---- KeyStore ----

    def verify(self, key: str) -> Principal | None:
        h = hash_key(key)
        now = self._clock()
        with self._lock:
            cached = self._cache.get(h)
            if cached is not None and now < cached[0]:
                return cached[1]
        resp = self._read(f"{self._prefix}/{quote(h, safe='')}")
        if resp is None:
            return None  # transport/auth failure — already logged, fail closed
        status, raw = resp
        if status == 404:
            return None  # unknown/revoked key — never cached (a new key works at once)
        if status != 200:
            log.error("openbao read failed (HTTP %s) — denying (fail closed)", status)
            return None
        try:
            rec = json.loads(raw)["data"]["data"]
        except Exception:  # noqa: BLE001 — malformed body denies
            log.error("openbao returned a malformed KV response — denying")
            return None
        principal = _record_principal(rec)
        if principal is None:
            log.error("openbao record for a presented key is malformed — denying")
            return None
        with self._lock:
            self._cache[h] = (self._clock() + self._cache_ttl, principal)
        return principal


def _record_principal(rec: Any) -> Principal | None:
    """Build a Principal from a KV record — the same validity rules as
    ``FileKeyStore.verify``: a non-empty string tenant and a scopes value that
    normalizes to known scopes. ``scopes`` may be a list or the comma-string the
    ``bao kv put ... scopes=read,write`` CLI naturally produces. Anything else
    (a dict, an empty tenant) denies — never normalize a foreign shape into a
    grant."""
    if not isinstance(rec, dict):
        return None
    tenant = rec.get("tenant")
    if not isinstance(tenant, str) or not tenant:
        return None
    raw_scopes = rec.get("scopes")
    if not isinstance(raw_scopes, (list, str)):
        return None
    try:
        scopes = frozenset(_normalize_scopes(raw_scopes))
    except (ValueError, TypeError):
        return None
    return Principal(tenant=tenant, scopes=scopes)
