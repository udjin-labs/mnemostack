"""Tenant-level audit log for control-plane operations.

Records WHO did WHAT to WHICH tenant's resources: service-key issue/revoke,
quota changes, tenant offboarding/backup/migration, and admin-console denials.
One JSON object per line (JSONL), appended under an exclusive lock so concurrent
writers (CLI + inspector) never interleave partial lines.

Contract — read this before extending:

- **Opt-in.** Auditing is enabled by setting ``MNEMOSTACK_AUDIT_FILE`` to a
  path; unset means no events are written anywhere (``NullAuditLog``). No
  surprise files on upgrade.
- **Best-effort operational trail, not a tamper-evident compliance ledger.**
  ``record()`` never raises: an unwritable audit file logs a loud error and the
  audited operation still proceeds — a broken log must not take key management
  or an emergency offboarding down. An appendable local file offers no
  cryptographic integrity; if you need tamper evidence, ship the file to an
  external collector (it's plain JSONL — any log shipper works).
- **Data plane is out of scope.** Recall/answer/feedback traffic is served
  requests, not administration — auditing it belongs to HTTP access logs. Only
  operations that change a tenant's resources (and failed attempts at them) are
  recorded here.
- **Never any key material.** Events carry public key *ids* only — never a
  plaintext key, never a hash (a hash is an offline-crackable fingerprint of
  the key). This rule binds every call site.
- **Rotation is external.** The file grows without bound; point logrotate (or
  equivalent) at it. Reads tolerate a mid-line truncation from a copytruncate
  rotation by skipping unparseable lines (counted, not silent).
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)

#: Actions recorded today (call sites keep this in sync; the log itself does
#: not validate — an old reader must tolerate actions added by a newer writer).
KNOWN_ACTIONS = frozenset(
    {
        "keys.issue",
        "keys.revoke",
        "quota.set",
        "quota.remove",
        "tenant.rm",
        "tenant.export",
        "tenant.migrate",
        "auth.denied",
    }
)


class AuditLogError(RuntimeError):
    """The audit log is unreadable — management READS surface this loudly.

    (Writes never raise — see the module contract.)"""


class AuditLog(Protocol):
    """Anything that can record an audit event."""

    def record(
        self,
        action: str,
        *,
        tenant: str | None = None,
        actor: str = "operator:cli",
        surface: str = "cli",
        outcome: str = "success",
        details: dict[str, Any] | None = None,
    ) -> bool: ...


class NullAuditLog:
    """Auditing disabled — every record is a no-op (returns False)."""

    enabled = False

    def record(
        self,
        action: str,
        *,
        tenant: str | None = None,
        actor: str = "operator:cli",
        surface: str = "cli",
        outcome: str = "success",
        details: dict[str, Any] | None = None,
    ) -> bool:
        return False


class FileAuditLog:
    """Append-only JSONL audit log at ``path``.

    Each event line: ``{"ts", "action", "actor", "surface", "outcome",
    "tenant", "details"}`` — ``ts`` is stamped at write time (UTC ISO-8601).
    """

    enabled = True

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def record(
        self,
        action: str,
        *,
        tenant: str | None = None,
        actor: str = "operator:cli",
        surface: str = "cli",
        outcome: str = "success",
        details: dict[str, Any] | None = None,
    ) -> bool:
        """Append one event. Returns True if it was durably written.

        NEVER raises (module contract): any failure — unwritable directory,
        full disk, unserializable detail value — logs a loud error and returns
        False, and the audited operation proceeds.
        """
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "actor": actor,
            "surface": surface,
            "outcome": outcome,
            "tenant": tenant,
            "details": details or {},
        }
        try:
            # default=str: an exotic detail value (Path, datetime) degrades to its
            # string form rather than losing the whole event to a TypeError.
            line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # 0o640 ceiling (umask may restrict further): the trail names tenants
            # and key ids — operational metadata other local users needn't read.
            # O_NOFOLLOW: refuse a symlink pre-planted at the configured path (the
            # same shared-dir threat FileKeyStore guards against) — otherwise an
            # attacker with write access to the parent dir could silently divert
            # the trail into an arbitrary file while record() reports success.
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(str(self.path), flags, 0o640)
            try:
                # Exclusive lock so a CLI op and an inspector op appending at the
                # same moment can't interleave bytes of their lines.
                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    os.write(fd, line.encode("utf-8"))
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        except Exception as e:  # noqa: BLE001 — contract: a broken log never blocks the op
            log.error("audit log %s: failed to record %s: %s", self.path, action, e)
            return False
        return True

    def tail(self, limit: int = 200) -> tuple[list[dict[str, Any]], int]:
        """The last ``limit`` events (oldest→newest) and how many lines were
        skipped as unparseable (a truncated line from a copytruncate rotation,
        or hand-editing). A missing file is an empty log, not an error; an
        unreadable one raises :class:`AuditLogError` — this is a management
        read, so it fails loud like the key/quota stores' list operations."""
        if limit < 1:
            raise ValueError("limit must be >= 1")
        events: deque[dict[str, Any]] = deque(maxlen=limit)
        skipped = 0
        try:
            with open(self.path, encoding="utf-8", errors="replace") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    if isinstance(parsed, dict):
                        events.append(parsed)
                    else:
                        skipped += 1
        except FileNotFoundError:
            return [], 0
        except OSError as e:
            raise AuditLogError(f"audit log {self.path} is unreadable: {e}") from e
        return list(events), skipped


def audit_log_from_env() -> FileAuditLog | NullAuditLog:
    """The process-wide audit sink: a :class:`FileAuditLog` at
    ``MNEMOSTACK_AUDIT_FILE`` when that is set (non-empty), else the no-op
    :class:`NullAuditLog`. Cheap to call per event — the file is opened per
    ``record``, so a long-lived server picks up rotation without a restart."""
    path = (os.environ.get("MNEMOSTACK_AUDIT_FILE") or "").strip()
    if path:
        return FileAuditLog(path)
    return NullAuditLog()
