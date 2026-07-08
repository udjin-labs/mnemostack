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
- **Hardened file handling.** Both read and write open the trail with every
  path component refusing symlinks (an ``openat`` walk — a pre-planted link
  anywhere in the path can neither divert appends nor feed the Audit tab
  spoofed events); the target must be a regular file (a FIFO can't hang an
  audited operation) **owned by the running user or root and no looser than
  ``0o640``** (a pre-created world-accessible file is refused, not silently
  appended to); the append lock is acquired with a bounded ~2s budget (a hung
  lock-holder can't stall key revocation); and short writes are completed or
  reported as failures. Deliberate consequences: a *legitimately* symlinked
  directory in the path is refused — point ``MNEMOSTACK_AUDIT_FILE`` at a real
  path — and a pre-provisioned trail file must be owner-correct and
  ``chmod 0640`` (or stricter).
"""

from __future__ import annotations

import json
import logging
import os
import stat
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)

#: Whether os.open supports dir_fd (POSIX) — enables the openat() walk below.
_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd

#: Bounded wait for the append lock: ~2s total, then the write is reported
#: failed. A blocking LOCK_EX would let a hung/hostile lock-holder stall key
#: revocation or an offboarding forever — the exact hang the contract forbids.
_LOCK_ATTEMPTS = 20
_LOCK_SLEEP = 0.1


def _require_regular(fd: int, path: Path) -> int:
    """Refuse an unsafe trail target. Closes ``fd`` on refusal.

    Must be a regular file (a FIFO would hang audited operations; a device node
    is not an append-only trail), owned by the running user or root, and not
    group-writable or world-accessible — a pre-created ``0666`` file in a shared
    log directory would otherwise leak tenant names/key ids to every local user
    (and let any of them spoof or truncate the trail) while ``record()`` reports
    success. Fresh files are created ``0o640``; an existing file must be at
    least that strict."""
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OSError(f"audit path {path} is not a regular file")
        # Owner/mode checks are POSIX-semantics: on Windows st_uid is 0 and the
        # permission bits are synthesized (a normal writable file reports
        # group/world bits chmod can't clear), so enforcing them there would
        # reject every freshly-created trail. Windows access control is ACLs —
        # out of scope for this best-effort trail; documented.
        if os.name == "posix":
            geteuid = getattr(os, "geteuid", None)
            if geteuid is not None and st.st_uid not in (geteuid(), 0):
                raise OSError(f"audit path {path} is owned by uid {st.st_uid} — not trusted")
            if st.st_mode & 0o027:  # group-write or any world access
                raise OSError(
                    f"audit path {path} has insecure mode {stat.S_IMODE(st.st_mode):#o} "
                    "(group-writable or world-accessible) — fix it, e.g. chmod 0640"
                )
    except BaseException:
        os.close(fd)
        raise
    return fd


def _open_trail_fd(path: Path, *, write: bool) -> int:
    """Open the trail with EVERY path component refusing symlinks.

    ``O_NOFOLLOW`` on a single open only protects the final component — a
    symlinked *ancestor* (``audit-dir -> attacker-dir``) would still divert the
    trail. So walk the path with ``openat`` semantics (``dir_fd``), each
    component ``O_NOFOLLOW``; the final open adds ``O_NONBLOCK`` (a FIFO with no
    reader fails with ENXIO instead of hanging the audited operation) and the
    result must be a regular file. The trade-off is deliberate: a legitimately
    symlinked directory in the configured path is ALSO refused (loudly) —
    point ``MNEMOSTACK_AUDIT_FILE`` at a real path. Raises OSError on refusal.
    """
    final_flags = (
        ((os.O_WRONLY | os.O_APPEND | os.O_CREAT) if write else os.O_RDONLY)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    if not _SUPPORTS_DIR_FD:  # pragma: no cover — non-POSIX platform
        # No openat walk available (Windows): best remaining defense is the
        # final-component flags + the regular-file check.
        return _require_regular(os.open(str(path), final_flags, 0o640), path)
    parts = path.parts
    if path.is_absolute():
        anchor, walk = parts[0], parts[1:]
    else:
        anchor, walk = ".", parts
    if not walk:
        raise OSError(f"audit path {path} is not a file path")
    dfd = os.open(anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for comp in walk[:-1]:
            ndfd = os.open(comp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dfd)
            os.close(dfd)
            dfd = ndfd
        return _require_regular(os.open(walk[-1], final_flags, 0o640, dir_fd=dfd), path)
    finally:
        os.close(dfd)

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
            # Hardened open (see _open_trail_fd): every component refuses
            # symlinks — the shared-dir threat FileKeyStore guards against,
            # extended to ancestors — and the target must be a regular file
            # created 0o640 (umask may restrict further; the trail names
            # tenants and key ids, which other local users needn't read).
            fd = _open_trail_fd(self.path, write=True)
            try:
                # Exclusive lock so a CLI op and an inspector op appending at the
                # same moment can't interleave bytes of their lines. fcntl is
                # POSIX-only (lazy import, like the key/quota stores' locks) —
                # without it (Windows) fall back to the bare O_APPEND write
                # rather than letting an ImportError break the audited op.
                try:
                    import fcntl
                except ImportError:  # pragma: no cover — non-POSIX platform
                    fcntl = None  # type: ignore[assignment]
                if fcntl is not None:
                    # Bounded, NON-blocking acquisition: a hung (or hostile)
                    # holder of this advisory lock must stall an audited
                    # operation for at most ~2s, not forever — after the budget
                    # the write is reported failed (loud, best-effort), and key
                    # revocation / offboarding proceeds.
                    for attempt in range(_LOCK_ATTEMPTS):
                        try:
                            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except (BlockingIOError, InterruptedError):
                            if attempt == _LOCK_ATTEMPTS - 1:
                                raise OSError(
                                    "audit log lock is held by another process "
                                    f"(waited ~{_LOCK_ATTEMPTS * _LOCK_SLEEP:.0f}s)"
                                ) from None
                            time.sleep(_LOCK_SLEEP)
                try:
                    # os.write may accept only part of the buffer (disk filling
                    # mid-line, interrupted write) WITHOUT raising — reporting
                    # success on a truncated event would violate the loud
                    # best-effort contract. Loop until every byte landed or the
                    # write raises; the flock keeps the pieces contiguous.
                    data = line.encode("utf-8")
                    while data:
                        n = os.write(fd, data)
                        if n <= 0:  # defensive: a 0-byte "success" must not spin
                            raise OSError("short write to the audit log")
                        data = data[n:]
                finally:
                    if fcntl is not None:
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
            # Same hardened open as record() (symlink-refusing walk + regular
            # file only): a pre-planted link must not let the Audit tab display
            # spoofed events from an attacker-chosen file while writes are
            # (correctly) refused — the read fails loud instead.
            fd = _open_trail_fd(self.path, write=False)
            with os.fdopen(fd, encoding="utf-8", errors="replace") as f:
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
