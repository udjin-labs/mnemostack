"""Per-tenant resource quotas — a storage-size limit today (rate limits later).

A quota is a per-*tenant* policy (not per-key), so it lives in its own small
store keyed by tenant, separate from the service-key store. Unlike auth — which
is a security boundary and fails **closed** — a quota is a resource guardrail, so
a corrupt/unreadable store fails **open** (no limit, logged loudly) rather than
blocking all ingest: a broken quota file must not take a deployment down.

    store = FileQuotaStore()
    store.set("acme", max_points=100_000)
    q = store.get("acme")          # TenantQuota(max_points=100000) or None
    # enforced at ingest via enforce_points_quota(...) below.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)


class QuotaStoreError(RuntimeError):
    """The quota store is unreadable/corrupt — management ops surface this loudly."""


class QuotaExceededError(RuntimeError):
    """A write would push a tenant over its storage quota."""

    def __init__(self, tenant: str, limit: int, attempted: int):
        self.tenant = tenant
        self.limit = limit
        self.attempted = attempted
        super().__init__(
            f"tenant '{tenant}' storage quota exceeded: "
            f"{attempted} points would exceed the limit of {limit}"
        )


@dataclass(frozen=True)
class TenantQuota:
    """Resource limits for one tenant. ``None`` means unlimited for that field."""

    #: Maximum number of vector points the tenant may store (None = unlimited).
    max_points: int | None = None

    def to_record(self) -> dict[str, Any]:
        rec: dict[str, Any] = {}
        if self.max_points is not None:
            rec["max_points"] = self.max_points
        return rec


def default_quotas_path() -> Path:
    """Where FileQuotaStore lives by default (override with MNEMOSTACK_QUOTAS_FILE)."""
    env = os.environ.get("MNEMOSTACK_QUOTAS_FILE")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "mnemostack" / "quotas.json"


class QuotaStore(Protocol):
    """Resolve a tenant to its quota, or None when the tenant has no quota set."""

    def get(self, tenant: str) -> TenantQuota | None: ...


def _coerce_quota(rec: Any) -> TenantQuota:
    """Build a TenantQuota from a stored record, tolerating a malformed field.

    A bad ``max_points`` (non-int, negative) is dropped to ``None`` (no limit) with
    a warning rather than raising — one broken record must not block a tenant's
    ingest (quotas fail open).
    """
    mp = rec.get("max_points") if isinstance(rec, dict) else None
    if mp is not None and not (isinstance(mp, int) and not isinstance(mp, bool) and mp >= 0):
        log.warning("ignoring malformed max_points %r in quota store", mp)
        mp = None
    return TenantQuota(max_points=mp)


class FileQuotaStore:
    """A QuotaStore backed by a JSON file: ``{"quotas": {tenant: {max_points}}}``.

    Writes are atomic (temp + replace) and flock-serialized, like the key store.
    The file is not secret (no 0600 requirement), but the same robustness applies
    so a concurrent ``quota set`` can't lose a write.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else default_quotas_path()

    # ---- persistence ----

    def _load(self) -> dict[str, Any]:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise QuotaStoreError(f"quota store {self.path} is corrupt: {e}") from e
        except OSError as e:
            raise QuotaStoreError(f"quota store {self.path} is unreadable: {e}") from e
        if not isinstance(data, dict) or not isinstance(data.get("quotas"), dict):
            raise QuotaStoreError(
                f"quota store {self.path} has an unexpected shape "
                "(expected an object with a 'quotas' map)"
            )
        return {t: r for t, r in data["quotas"].items() if isinstance(t, str)}

    def _save(self, quotas: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=str(self.path.parent), prefix=self.path.name + ".", suffix=".tmp"
            )
        except OSError as e:
            raise QuotaStoreError(f"cannot write quota store {self.path}: {e}") from e
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"quotas": quotas}, f, indent=2)
            os.replace(tmp, self.path)
        except OSError as e:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise QuotaStoreError(f"cannot write quota store {self.path}: {e}") from e
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Advisory exclusive lock around a load-modify-save (POSIX flock)."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise QuotaStoreError(
                f"cannot create quota store directory {self.path.parent}: {e}"
            ) from e
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX
            yield
            return
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        try:
            fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o600)
        except OSError as e:
            raise QuotaStoreError(f"cannot open quota store lock {lock_path}: {e}") from e
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)

    # ---- QuotaStore ----

    def get(self, tenant: str) -> TenantQuota | None:
        try:
            quotas = self._load()
        except QuotaStoreError:
            # Fail OPEN: a broken quota file must not block ingest. Loud log so the
            # operator notices the guardrail is off.
            log.error("quota store %s unreadable — treating as no limits", self.path)
            return None
        rec = quotas.get(tenant)
        if rec is None:
            return None
        return _coerce_quota(rec)

    # ---- management ----

    def set(self, tenant: str, *, max_points: int | None = None) -> TenantQuota:
        """Set (replace) a tenant's quota. Returns the stored quota."""
        if not tenant:
            raise ValueError("tenant is required")
        # bool is an int subclass — reject it explicitly so True isn't stored as 1.
        if max_points is not None and (
            not isinstance(max_points, int) or isinstance(max_points, bool) or max_points < 0
        ):
            raise ValueError("max_points must be a non-negative integer")
        quota = TenantQuota(max_points=max_points)
        with self._locked():
            quotas = self._load()
            quotas[tenant] = quota.to_record()
            self._save(quotas)
        return quota

    def remove(self, tenant: str) -> bool:
        """Drop a tenant's quota. Returns True if one existed."""
        with self._locked():
            quotas = self._load()
            if tenant not in quotas:
                return False
            del quotas[tenant]
            self._save(quotas)
        return True

    def list_quotas(self) -> list[dict[str, Any]]:
        """List quotas as ``[{tenant, max_points}]``, sorted by tenant."""
        quotas = self._load()  # load once — a second load could race a concurrent rm
        out: list[dict[str, Any]] = []
        for tenant in sorted(quotas):
            q = _coerce_quota(quotas[tenant])
            out.append({"tenant": tenant, "max_points": q.max_points})
        return out


def enforce_points_quota(
    tenant: str | None, current: int, adding: int, max_points: int | None
) -> None:
    """Raise ``QuotaExceededError`` if a net-positive change would exceed ``max_points``.

    A no-op when the tenant is unscoped (``None``) or the limit is unset — so a
    single-tenant deployment and any tenant without a quota are unaffected.

    Only genuine growth is rejected: a non-increasing change (``adding <= 0`` — a
    markdown ``--prune`` that removes at least as many points as it adds) always
    passes, even for a tenant already over its cap. Otherwise lowering a quota
    below current usage would block the very ``--prune`` cleanup that brings the
    tenant back under the limit.

    Best-effort: ``current`` is read before the write, so two ingesters flushing
    the same tenant concurrently can both pass this check and land the tenant a
    little over the cap (there's no cross-writer reservation/lock). The cap holds
    exactly for the common single-writer ingest; concurrent ingest can overshoot
    by roughly one batch per extra writer, then self-corrects on the next flush.
    """
    if tenant is None or max_points is None:
        return
    if adding > 0 and current + adding > max_points:
        raise QuotaExceededError(tenant, max_points, current + adding)
