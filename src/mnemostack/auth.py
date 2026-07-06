"""Service keys + scopes — authenticated tenant resolution for multi-tenant.

A service key maps to a ``Principal(tenant, scopes)``. This is the piece that
lets the multi-tenant boundary resolve *which* tenant a request belongs to from
the caller's credential — the client presents a key, the server derives the
tenant and scopes; the client can never assert a tenant it wasn't issued.

Keys are stored **hashed** (SHA-256): the plaintext is shown once at creation
and never persisted, so a leaked key file can't be replayed. ``KeyStore`` is a
Protocol; ``FileKeyStore`` (a JSON file of hashed records) is the default —
implement ``verify`` to plug a real secret store.

Scopes gate operations: ``read`` (recall/inspect), ``write`` (ingest/invalidate/
feedback), ``admin`` (backup/export/delete, key management). ``admin`` implies
the others. This module is credential storage only — wiring it into the HTTP /
MCP surfaces (auth middleware, default-deny) is a separate step.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)


class KeyStoreError(RuntimeError):
    """The key store is unreadable/corrupt — management ops surface this loudly."""

#: The full set of scopes. ``admin`` is a superset (see ``Principal.can``).
SCOPES = frozenset({"read", "write", "admin"})

_KEY_PREFIX = "msk_"  # mnemostack service key


@dataclass(frozen=True)
class Principal:
    """The authenticated identity behind a request."""

    tenant: str
    scopes: frozenset[str]

    def can(self, scope: str) -> bool:
        """Whether this principal is allowed a scope. ``admin`` implies all."""
        return "admin" in self.scopes or scope in self.scopes


class KeyStore(Protocol):
    """Resolve a presented key to a Principal, or None if unknown/revoked."""

    def verify(self, key: str) -> Principal | None: ...


def hash_key(key: str) -> str:
    """SHA-256 of a key — what's stored, so plaintext never touches disk."""
    return hashlib.sha256(key.encode()).hexdigest()


def _normalize_scopes(scopes: list[str] | frozenset[str] | str) -> list[str]:
    if isinstance(scopes, str):
        scopes = [s.strip() for s in scopes.split(",")]
    out = sorted({s for s in scopes if s})
    bad = [s for s in out if s not in SCOPES]
    if bad:
        raise ValueError(f"unknown scope(s): {', '.join(bad)}; valid: {', '.join(sorted(SCOPES))}")
    if not out:
        raise ValueError("at least one scope is required")
    return out


def default_keys_path() -> Path:
    """Where FileKeyStore lives by default (override with MNEMOSTACK_KEYS_FILE)."""
    env = os.environ.get("MNEMOSTACK_KEYS_FILE")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "mnemostack" / "keys.json"


class FileKeyStore:
    """A KeyStore backed by a JSON file of hashed key records.

    Record shape: ``{id, hash, tenant, scopes, label, created_at}``. The ``id``
    is a public, non-secret handle for listing/revoking; ``hash`` is the SHA-256
    of the plaintext key. Reads are lazy-cached; writes are atomic (temp +
    replace). Not a high-throughput store — fine for the key volumes a
    self-hosted deployment has.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else default_keys_path()

    # ---- persistence ----

    def _load(self) -> list[dict[str, Any]]:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return []
        except json.JSONDecodeError as e:
            # Fail loudly for management ops; verify() catches this and denies.
            raise KeyStoreError(f"key store {self.path} is corrupt: {e}") from e
        except OSError as e:
            # A directory, a permission error, etc. — treat as unreadable (not
            # empty) so verify() fails closed instead of silently denying every key.
            raise KeyStoreError(f"key store {self.path} is unreadable: {e}") from e
        # Valid JSON but the wrong shape must surface, not reset to empty: an empty
        # view would deny all keys AND let a later write clobber the real store.
        if not isinstance(data, dict) or not isinstance(data.get("keys", []), list):
            raise KeyStoreError(
                f"key store {self.path} has an unexpected shape "
                "(expected an object with a 'keys' list)"
            )
        records = data.get("keys", [])
        return [r for r in records if isinstance(r, dict)]

    def _save(self, records: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # mkstemp creates a UNIQUE file (O_CREAT|O_EXCL, mode 0600) in the target
        # dir, so a pre-created symlink at a guessable ".tmp" path can't be followed
        # to clobber another file (the store may live in a shared dir like /tmp).
        # 0600 from the start means the secret is never briefly world-readable, and
        # os.replace preserves the mode so the landed file is 0600.
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=self.path.name + ".", suffix=".tmp"
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"keys": records}, f, indent=2)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)  # re-assert (defensive; already 0600)
        except OSError as e:
            log.warning("could not set 0600 permissions on %s: %s", self.path, e)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Advisory exclusive lock around a load-modify-save (POSIX flock).

        Serializes concurrent issue()/revoke() so a lost write can't silently
        drop a just-issued key. Best-effort: a no-op where flock is unavailable.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX
            yield
            return
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)

    # ---- KeyStore ----

    def verify(self, key: str) -> Principal | None:
        h = hash_key(key)
        try:
            records = self._load()
        except KeyStoreError:
            # Corrupt/unreadable store -> deny everything (fail closed), never crash.
            log.error("key store %s unreadable — denying (fail closed)", self.path)
            return None
        for rec in records:
            # constant-time compare so a timing side-channel can't probe hashes
            if hmac.compare_digest(str(rec.get("hash", "")), h):
                tenant = rec.get("tenant")
                if not tenant:
                    return None
                try:
                    # `or []` handles a null/missing scopes; the except handles a
                    # non-iterable or invalid value — a malformed record denies,
                    # it never crashes verify for every other key.
                    scopes = frozenset(_normalize_scopes(rec.get("scopes") or []))
                except (ValueError, TypeError):
                    return None
                return Principal(tenant=tenant, scopes=scopes)
        return None

    # ---- management ----

    def issue(
        self, tenant: str, scopes: list[str] | str, label: str = ""
    ) -> tuple[str, str]:
        """Create a key for a tenant. Returns (key_id, plaintext_key).

        The plaintext is returned ONCE and never stored — only its hash is
        persisted. Show it to the operator immediately; it can't be recovered.
        """
        if not tenant:
            raise ValueError("tenant is required")
        norm = _normalize_scopes(scopes)
        key = _KEY_PREFIX + secrets.token_urlsafe(24)
        with self._locked():
            records = self._load()
            # Guarantee a unique public id so revoke() can never remove a second,
            # unrelated key (possibly another tenant's) that shares an id.
            existing = {r.get("id") for r in records}
            key_id = secrets.token_hex(6)
            while key_id in existing:
                key_id = secrets.token_hex(6)
            records.append(
                {
                    "id": key_id,
                    "hash": hash_key(key),
                    "tenant": tenant,
                    "scopes": norm,
                    "label": label,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            self._save(records)
        return key_id, key

    def revoke(self, key_id: str) -> bool:
        """Remove a key by its public id. Returns True if one was removed."""
        with self._locked():
            records = self._load()
            kept = [r for r in records if r.get("id") != key_id]
            if len(kept) == len(records):
                return False
            self._save(kept)
        return True

    def list_keys(self) -> list[dict[str, Any]]:
        """List keys WITHOUT their hashes (safe to print)."""
        return [
            {k: v for k, v in r.items() if k != "hash"}
            for r in sorted(self._load(), key=lambda r: str(r.get("created_at", "")))
        ]
