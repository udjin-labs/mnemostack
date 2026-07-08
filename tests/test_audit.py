"""Audit log — module contract + CLI and inspector wiring.

The module contract under test: opt-in via MNEMOSTACK_AUDIT_FILE, best-effort
writes that never raise into the audited operation, JSONL lines with no key
material ever, and reads that tolerate corrupt lines (counted, not silent).
"""

from __future__ import annotations

import argparse
import json
import os
import stat

import pytest

from mnemostack.audit import (
    AuditLogError,
    FileAuditLog,
    NullAuditLog,
    audit_log_from_env,
)


def _events(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------- FileAuditLog ----------


def test_record_appends_jsonl_with_expected_fields(tmp_path):
    log = FileAuditLog(tmp_path / "audit.jsonl")
    assert log.record("keys.issue", tenant="acme", details={"key_id": "abc123"}) is True
    assert log.record("quota.set", tenant="globex", outcome="error") is True
    ev = _events(tmp_path / "audit.jsonl")
    assert len(ev) == 2
    assert ev[0]["action"] == "keys.issue" and ev[0]["tenant"] == "acme"
    assert ev[0]["actor"] == "operator:cli" and ev[0]["surface"] == "cli"
    assert ev[0]["outcome"] == "success" and ev[0]["details"] == {"key_id": "abc123"}
    assert ev[0]["ts"].endswith("+00:00")  # UTC, ISO-8601
    assert ev[1]["outcome"] == "error" and ev[1]["details"] == {}


def test_record_never_raises_on_unwritable_path(tmp_path):
    # Parent "dir" is a file — mkdir/open must fail; record reports False, no raise.
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir")
    log = FileAuditLog(blocker / "audit.jsonl")
    assert log.record("keys.issue", tenant="acme") is False


def test_record_refuses_a_symlinked_audit_path(tmp_path):
    # A symlink pre-planted at the configured path (shared-dir attack) must not
    # divert the trail into an arbitrary file — O_NOFOLLOW refuses it, and the
    # best-effort contract turns that into False, not a crash.
    target = tmp_path / "diverted.log"
    target.write_text("")
    link = tmp_path / "audit.jsonl"
    link.symlink_to(target)
    assert FileAuditLog(link).record("keys.issue", tenant="acme") is False
    assert target.read_text() == ""  # nothing was appended through the link


def test_record_refuses_a_symlinked_ancestor_dir(tmp_path):
    # O_NOFOLLOW on the final open doesn't cover a symlinked PARENT
    # (audit-dir -> attacker-dir) — the openat walk must refuse it too.
    real = tmp_path / "real"
    real.mkdir()
    linkdir = tmp_path / "linkdir"
    linkdir.symlink_to(real, target_is_directory=True)
    assert FileAuditLog(linkdir / "audit.jsonl").record("keys.issue") is False
    assert not (real / "audit.jsonl").exists()  # nothing landed behind the link


def test_tail_refuses_a_symlinked_ancestor_dir(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "audit.jsonl").write_text('{"action": "keys.issue", "actor": "spoof"}\n')
    linkdir = tmp_path / "linkdir"
    linkdir.symlink_to(real, target_is_directory=True)
    with pytest.raises(AuditLogError):
        FileAuditLog(linkdir / "audit.jsonl").tail()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="mkfifo is POSIX-only")
def test_fifo_at_the_audit_path_fails_fast_not_hangs(tmp_path):
    # A pre-created FIFO with no reader would hang a plain open() forever,
    # taking key management down with it — O_NONBLOCK + the regular-file check
    # must turn it into a fast, loud failure on both paths.
    p = tmp_path / "audit.jsonl"
    os.mkfifo(p)
    assert FileAuditLog(p).record("keys.issue") is False  # ENXIO, immediate
    with pytest.raises(AuditLogError):
        FileAuditLog(p).tail()  # opens nonblocking, refused as non-regular


def test_record_and_tail_refuse_an_insecure_precreated_file(tmp_path):
    # A file pre-planted 0666 in a shared log dir would leak tenant names/key
    # ids to every local user (and let any of them spoof the trail) — both
    # paths must refuse it loudly, not append/read and report success.
    p = tmp_path / "audit.jsonl"
    p.write_text("")
    os.chmod(p, 0o666)
    assert FileAuditLog(p).record("keys.issue", tenant="acme") is False
    assert p.read_text() == ""  # nothing was appended to the insecure file
    with pytest.raises(AuditLogError):
        FileAuditLog(p).tail()
    os.chmod(p, 0o640)  # fixed by the operator -> works again
    assert FileAuditLog(p).record("keys.issue", tenant="acme") is True
    assert FileAuditLog(p).tail()[0][0]["tenant"] == "acme"


def test_record_gives_up_on_a_held_lock_within_budget(tmp_path, monkeypatch):
    # A hung (or hostile) process holding the advisory lock must not stall an
    # audited operation forever — record() waits a bounded budget, then reports
    # the write failed and the operation proceeds.
    import fcntl

    import mnemostack.audit as audit_mod

    monkeypatch.setattr(audit_mod, "_LOCK_ATTEMPTS", 3)
    monkeypatch.setattr(audit_mod, "_LOCK_SLEEP", 0.01)
    log = FileAuditLog(tmp_path / "a.jsonl")
    assert log.record("keys.issue") is True  # create the file first
    holder = os.open(str(tmp_path / "a.jsonl"), os.O_WRONLY)
    try:
        fcntl.flock(holder, fcntl.LOCK_EX)
        assert log.record("quota.set", tenant="acme") is False  # bounded give-up
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        os.close(holder)
    assert log.record("quota.set", tenant="acme") is True  # lock released -> works
    events, _ = log.tail()
    assert [e["action"] for e in events] == ["keys.issue", "quota.set"]


def test_record_completes_short_writes(tmp_path, monkeypatch):
    # os.write may accept only part of the buffer without raising — record()
    # must loop until the whole line landed, not report success on a truncated
    # (and later skipped-as-corrupt) event.
    real_write = os.write
    monkeypatch.setattr(os, "write", lambda fd, data: real_write(fd, bytes(data)[:3]))
    log = FileAuditLog(tmp_path / "a.jsonl")
    assert log.record("keys.issue", tenant="acme", details={"key_id": "abc123"}) is True
    monkeypatch.undo()
    (ev,) = _events(tmp_path / "a.jsonl")  # parses fully — no torn line
    assert ev["tenant"] == "acme" and ev["details"] == {"key_id": "abc123"}


def test_record_creates_missing_parents_inside_the_walk(tmp_path):
    # Parent creation happens dir_fd-relative inside the validated walk (the
    # old path-string mkdir(parents=True) is gone) — a clean nested path still
    # self-provisions.
    log = FileAuditLog(tmp_path / "a" / "b" / "audit.jsonl")
    assert log.record("keys.issue", tenant="acme") is True
    (ev,) = _events(tmp_path / "a" / "b" / "audit.jsonl")
    assert ev["tenant"] == "acme"


def test_record_never_mkdirs_through_a_symlinked_ancestor(tmp_path):
    # Round-8 repro: linkdir -> real, configured path linkdir/sub/audit.jsonl.
    # The old pre-walk mkdir(parents=True) would create real/sub before the
    # walk refused — a filesystem write outside the configured path. Now the
    # mkdir itself is dir_fd-relative and refused at linkdir.
    real = tmp_path / "real"
    real.mkdir()
    linkdir = tmp_path / "linkdir"
    linkdir.symlink_to(real, target_is_directory=True)
    assert FileAuditLog(linkdir / "sub" / "audit.jsonl").record("keys.issue") is False
    assert not (real / "sub").exists()  # no directory planted behind the link


def test_record_degrades_exotic_detail_values(tmp_path):
    # default=str: a non-JSON-serializable detail must not lose the event.
    log = FileAuditLog(tmp_path / "a.jsonl")
    assert log.record("tenant.export", details={"path": tmp_path}) is True
    (ev,) = _events(tmp_path / "a.jsonl")
    assert str(tmp_path) in ev["details"]["path"]


def test_audit_file_not_world_accessible(tmp_path):
    log = FileAuditLog(tmp_path / "a.jsonl")
    log.record("keys.issue")
    mode = stat.S_IMODE(os.stat(tmp_path / "a.jsonl").st_mode)
    assert mode & 0o007 == 0  # 0o640 ceiling: no world access regardless of umask


def test_tail_returns_last_n_oldest_first_and_counts_corrupt(tmp_path):
    p = tmp_path / "a.jsonl"
    log = FileAuditLog(p)
    for i in range(5):
        log.record("quota.set", tenant=f"t{i}")
    # a truncated line (copytruncate rotation) and a non-object line
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"ts": "2026-07-08T', )
        f.write("\n[1, 2]\n")
    events, skipped = log.tail(3)
    assert [e["tenant"] for e in events] == ["t2", "t3", "t4"]
    assert skipped == 2


def test_tail_missing_file_is_empty_not_error(tmp_path):
    events, skipped = FileAuditLog(tmp_path / "missing.jsonl").tail()
    assert events == [] and skipped == 0


def test_tail_refuses_a_symlinked_audit_path(tmp_path):
    # Read-side twin of the write-side O_NOFOLLOW: a pre-planted link must not
    # let the Audit tab render spoofed events from an attacker-chosen file —
    # the management read fails loud instead.
    target = tmp_path / "spoofed.jsonl"
    target.write_text('{"action": "keys.issue", "actor": "attacker"}\n')
    link = tmp_path / "audit.jsonl"
    link.symlink_to(target)
    with pytest.raises(AuditLogError):
        FileAuditLog(link).tail()


def test_tail_unreadable_file_raises(tmp_path):
    p = tmp_path / "a.jsonl"
    log = FileAuditLog(p)
    log.record("keys.issue")
    os.chmod(p, 0)
    if os.access(p, os.R_OK):  # running as root: chmod 0 doesn't bite
        pytest.skip("cannot make the file unreadable under this user")
    try:
        with pytest.raises(AuditLogError):
            log.tail()
    finally:
        os.chmod(p, 0o600)


def test_tail_rejects_nonpositive_limit(tmp_path):
    with pytest.raises(ValueError):
        FileAuditLog(tmp_path / "a.jsonl").tail(0)


def test_sink_redacts_key_shaped_strings_everywhere(tmp_path):
    # An operator pasting a plaintext key where a public id belongs (e.g.
    # `keys revoke msk_...` -> not-found event) must not land a usable
    # credential in the trail — the SINK redacts any msk_-shaped string,
    # wherever it sits in the event.
    log = FileAuditLog(tmp_path / "a.jsonl")
    assert log.record(
        "keys.revoke",
        tenant="msk_pasted_as_tenant",
        actor="msk_pasted_as_actor",
        details={"key_id": "msk_secret123", "nested": {"ids": ["ok", "msk_deep"]}},
    )
    raw = (tmp_path / "a.jsonl").read_text()
    assert "msk_" not in raw
    (ev,) = _events(tmp_path / "a.jsonl")
    assert ev["details"]["key_id"] == "[redacted:key-shaped]"
    assert ev["details"]["nested"]["ids"] == ["ok", "[redacted:key-shaped]"]
    assert ev["tenant"] == "[redacted:key-shaped]" and ev["actor"] == "[redacted:key-shaped]"
    # normal ids/tenants are untouched
    log2 = FileAuditLog(tmp_path / "b.jsonl")
    log2.record("keys.revoke", tenant="acme", details={"key_id": "abc123"})
    (ev2,) = _events(tmp_path / "b.jsonl")
    assert ev2["tenant"] == "acme" and ev2["details"]["key_id"] == "abc123"


def test_sink_redacts_key_shaped_substrings(tmp_path):
    # Prefix-only redaction misses a key EMBEDDED in a longer string — e.g. a
    # denied request audited with details.path="/api/keys/msk_..." — which
    # would still land usable material in the trail.
    log = FileAuditLog(tmp_path / "a.jsonl")
    log.record(
        "auth.denied",
        actor="key:msk_sometoken",
        details={"path": "/api/keys/msk_abc-123", "reason": "not_admin"},
    )
    raw = (tmp_path / "a.jsonl").read_text()
    assert "msk_" not in raw
    (ev,) = _events(tmp_path / "a.jsonl")
    assert ev["actor"] == "key:[redacted:key-shaped]"
    assert ev["details"]["path"] == "/api/keys/[redacted:key-shaped]"
    assert ev["details"]["reason"] == "not_admin"  # untouched


def test_sink_redacts_dict_keys_and_stringified_values(tmp_path):
    # Two shapes a structural pre-serialization walk would miss: key material
    # as a dict KEY, and inside a non-string value that default=str converts
    # during dumping. The line-level redaction catches both.
    from pathlib import Path as _P

    log = FileAuditLog(tmp_path / "a.jsonl")
    log.record("keys.revoke", details={"msk_pasted_as_key": "x", "path": _P("msk_in_path")})
    raw = (tmp_path / "a.jsonl").read_text()
    assert "msk_" not in raw
    (ev,) = _events(tmp_path / "a.jsonl")  # and the line is still valid JSON
    assert ev["details"]["[redacted:key-shaped]"] == "x"


def test_tail_counts_non_decode_parse_failures_as_skipped(tmp_path):
    # json.loads can fail with a bare ValueError (huge int on 3.11+), not just
    # JSONDecodeError — one such hand-edited line must count as skipped, not
    # 500 the Audit tab.
    p = tmp_path / "a.jsonl"
    log = FileAuditLog(p)
    log.record("keys.issue", tenant="acme")
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"n": ' + "9" * 5000 + "}\n")
    events, skipped = log.tail()
    assert [e["tenant"] for e in events] == ["acme"]
    assert skipped == 1


def test_record_repairs_a_torn_tail(tmp_path):
    # A file ending mid-line (copytruncate rotation cut a write short) must not
    # swallow the NEXT event by gluing it onto the corrupt bytes — record()
    # starts on a fresh line, and only the torn line is lost (counted).
    p = tmp_path / "a.jsonl"
    log = FileAuditLog(p)
    log.record("keys.issue", tenant="acme")
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"ts": "trunc')  # torn tail, no newline
    assert log.record("quota.set", tenant="globex") is True
    events, skipped = log.tail()
    assert [e["action"] for e in events] == ["keys.issue", "quota.set"]
    assert events[1]["tenant"] == "globex"  # the post-truncation event survives
    assert skipped == 1  # only the torn line itself is the casualty


def test_no_dir_fd_fallback_still_refuses_symlinks(tmp_path, monkeypatch):
    # The fallback branch (no openat walk — Windows) must lstat-refuse a
    # symlink anywhere in the path instead of silently following it.
    import mnemostack.audit as audit_mod

    monkeypatch.setattr(audit_mod, "_SUPPORTS_DIR_FD", False)
    target = tmp_path / "diverted.jsonl"
    target.write_text("")
    link = tmp_path / "audit.jsonl"
    link.symlink_to(target)
    assert FileAuditLog(link).record("keys.issue") is False
    assert target.read_text() == ""
    real = tmp_path / "realdir"
    real.mkdir()
    linkdir = tmp_path / "linkdir"
    linkdir.symlink_to(real, target_is_directory=True)
    assert FileAuditLog(linkdir / "audit.jsonl").record("keys.issue") is False
    with pytest.raises(AuditLogError):
        FileAuditLog(link).tail()
    # and a clean path still works through the fallback
    assert FileAuditLog(tmp_path / "ok.jsonl").record("keys.issue") is True


# ---------- env plumbing ----------


def test_audit_log_from_env(monkeypatch, tmp_path):
    monkeypatch.delenv("MNEMOSTACK_AUDIT_FILE", raising=False)
    assert isinstance(audit_log_from_env(), NullAuditLog)
    monkeypatch.setenv("MNEMOSTACK_AUDIT_FILE", "  ")
    assert isinstance(audit_log_from_env(), NullAuditLog)  # blank = unset
    monkeypatch.setenv("MNEMOSTACK_AUDIT_FILE", str(tmp_path / "a.jsonl"))
    sink = audit_log_from_env()
    assert isinstance(sink, FileAuditLog)
    assert sink.path == tmp_path / "a.jsonl"


def test_null_audit_log_is_a_noop():
    assert NullAuditLog().record("keys.issue", tenant="acme") is False


# ---------- CLI wiring ----------

import mnemostack.cli as cli  # noqa: E402
from mnemostack.auth import FileKeyStore  # noqa: E402
from mnemostack.quotas import FileQuotaStore  # noqa: E402


@pytest.fixture
def audit_file(monkeypatch, tmp_path):
    p = tmp_path / "audit.jsonl"
    monkeypatch.setenv("MNEMOSTACK_AUDIT_FILE", str(p))
    return p


def _keys_ns(tmp_path, **kw):
    base = {"keys_file": str(tmp_path / "keys.json"), "label": None}
    base.update(kw)
    return argparse.Namespace(**base)


def test_cli_keys_add_audits_key_id_never_the_key(tmp_path, audit_file, capsys):
    rc = cli.cmd_keys_add(_keys_ns(tmp_path, tenant="acme", scopes="read,write"))
    assert rc == 0
    (ev,) = _events(audit_file)
    assert ev["action"] == "keys.issue" and ev["tenant"] == "acme"
    assert ev["details"]["scopes"] == "read,write"
    key_id = ev["details"]["key_id"]
    assert FileKeyStore(tmp_path / "keys.json").list_keys()[0]["id"] == key_id
    # The plaintext key (printed once) and its hash must NEVER reach the log.
    plaintext = next(
        line.strip() for line in capsys.readouterr().out.splitlines()
        if line.strip().startswith("msk_")
    )
    raw = audit_file.read_text()
    assert plaintext not in raw
    from mnemostack.auth import hash_key

    assert hash_key(plaintext) not in raw


def test_cli_keys_revoke_audits_success_and_not_found(tmp_path, audit_file):
    ks = FileKeyStore(tmp_path / "keys.json")
    key_id, _ = ks.issue("acme", ["read"])
    assert cli.cmd_keys_revoke(_keys_ns(tmp_path, id=key_id)) == 0
    assert cli.cmd_keys_revoke(_keys_ns(tmp_path, id="nope")) == 1
    ev = _events(audit_file)
    assert [e["outcome"] for e in ev] == ["success", "error"]
    assert ev[0]["details"]["key_id"] == key_id
    assert ev[0]["tenant"] == "acme"  # attributed via pre-deletion lookup
    assert ev[1]["details"]["reason"] == "not_found"
    assert ev[1]["tenant"] is None  # unknown id: nothing to attribute


def test_cli_quota_set_and_rm_audit(tmp_path, audit_file):
    ns = argparse.Namespace(
        quotas_file=str(tmp_path / "q.json"), tenant="acme", max_points=100
    )
    assert cli.cmd_quota_set(ns) == 0
    assert cli.cmd_quota_rm(argparse.Namespace(quotas_file=str(tmp_path / "q.json"), tenant="acme")) == 0
    ev = _events(audit_file)
    assert ev[0]["action"] == "quota.set" and ev[0]["details"]["max_points"] == 100
    assert ev[1]["action"] == "quota.remove" and ev[1]["outcome"] == "success"


def test_cli_disabled_audit_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv("MNEMOSTACK_AUDIT_FILE", raising=False)
    assert cli.cmd_keys_add(_keys_ns(tmp_path, tenant="acme", scopes="read")) == 0
    assert not list(tmp_path.glob("*.jsonl"))


class _FakeHit:
    def __init__(self, i):
        self.id = str(i)
        self.payload = {"text": f"row {i}", "tenant_id": "acme"}
        self.vector = None


class _FakeExportStore:
    def __init__(self, **_):
        pass

    def collection_exists(self):
        return True

    def scroll(self, **_):
        return iter([_FakeHit(1), _FakeHit(2)])


def test_cli_tenant_export_audits_points_and_destination(monkeypatch, tmp_path, audit_file):
    monkeypatch.setattr(cli, "VectorStore", lambda **_: _FakeExportStore())
    out = tmp_path / "dump.jsonl"
    ns = argparse.Namespace(
        collection="mt", qdrant="http://localhost:6333",
        tenant="acme", output=str(out), no_vectors=True,
    )
    assert cli.cmd_tenant_export(ns) == 0
    (ev,) = _events(audit_file)
    assert ev["action"] == "tenant.export" and ev["tenant"] == "acme"
    assert ev["details"] == {"points": 2, "output": str(out)}


def _migrate_ns(tmp_path, **kw):
    base = {
        "collection": "mt", "qdrant": "http://localhost:6333",
        "tenant": "acme", "all": False, "yes": False, "dry_run": False,
        "memgraph_uri": None, "graph_timeout": 5.0,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def test_cli_tenant_migrate_preflight_failure_is_audited(monkeypatch, tmp_path, audit_file):
    # A non-dry-run migration that dies before stamp_tenant (unreachable
    # Qdrant / missing collection) is a failed ATTEMPT — the trail must show
    # it, same rule as tenant-export's early errors.
    class _Down:
        def __init__(self, **_):
            raise ConnectionError("qdrant down")

    monkeypatch.setattr(cli, "VectorStore", _Down)
    assert cli.cmd_tenant_migrate(_migrate_ns(tmp_path)) == 1
    (ev,) = _events(audit_file)
    assert ev["action"] == "tenant.migrate" and ev["outcome"] == "error"
    assert "qdrant down" in ev["details"]["error"]


def test_cli_tenant_migrate_absent_collection_is_audited(monkeypatch, tmp_path, audit_file):
    class _NoColl:
        def __init__(self, **_):
            pass

        def collection_exists(self):
            return False

    monkeypatch.setattr(cli, "VectorStore", lambda **_: _NoColl())
    assert cli.cmd_tenant_migrate(_migrate_ns(tmp_path)) == 1
    (ev,) = _events(audit_file)
    assert ev["outcome"] == "error" and ev["details"]["reason"] == "collection_absent"


def test_cli_tenant_migrate_dry_run_preflight_failure_not_audited(
    monkeypatch, tmp_path, audit_file
):
    class _Down:
        def __init__(self, **_):
            raise ConnectionError("qdrant down")

    monkeypatch.setattr(cli, "VectorStore", _Down)
    assert cli.cmd_tenant_migrate(_migrate_ns(tmp_path, dry_run=True)) == 1
    assert not audit_file.exists()  # dry-run: nothing attempted, nothing logged


def test_mode_check_skipped_on_non_posix(monkeypatch, tmp_path):
    # Windows reports synthesized group/world permission bits that chmod can't
    # clear — the POSIX mode check must be gated on os.name, or every fresh
    # trail file would be rejected as "world-accessible" and record() would
    # always fail. Tested at the _require_regular level (patching os.name
    # globally breaks pathlib, which the higher-level record() path uses).
    import mnemostack.audit as audit_mod

    p = tmp_path / "a.jsonl"
    p.write_text("")
    os.chmod(p, 0o666)
    fd = os.open(str(p), os.O_RDONLY)
    monkeypatch.setattr(os, "name", "nt")
    assert audit_mod._require_regular(fd, p) == fd  # accepted: bits not enforceable
    monkeypatch.undo()
    os.close(fd)
    fd2 = os.open(str(p), os.O_RDONLY)
    with pytest.raises(OSError, match="insecure mode"):  # POSIX: still refused
        audit_mod._require_regular(fd2, p)
    os.chmod(p, 0o640)


def test_cli_tenant_rm_audits_the_sweep_outcome(monkeypatch, tmp_path, audit_file):
    # Full sweep with fakes: success -> one tenant.rm success event.
    class _FakeRmStore:
        def __init__(self, **_):
            pass

        def collection_exists(self):
            return True

        def count(self, *, tenant=None):
            return 1

        def delete_tenant(self, tenant):
            return 1

    monkeypatch.setattr(cli, "VectorStore", lambda **_: _FakeRmStore())
    ks = FileKeyStore(tmp_path / "keys.json")
    ks.issue("acme", ["read"])
    ks.issue("ops", ["admin"])  # another tenant's admin survives the sweep
    FileQuotaStore(tmp_path / "quotas.json").set("acme", max_points=5)
    ns = argparse.Namespace(
        collection="mt", qdrant="http://localhost:6333",
        tenant="acme", memgraph_uri=None, no_graph=False, graph_timeout=5.0,
        dry_run=False, yes=True,
        keys_file=str(tmp_path / "keys.json"),
        quotas_file=str(tmp_path / "quotas.json"),
        state_path=str(tmp_path / "state.json"),
        external_keys_revoked=False,
    )
    assert cli.cmd_tenant_rm(ns) == 0
    (ev,) = _events(audit_file)
    assert ev["action"] == "tenant.rm" and ev["tenant"] == "acme"
    assert ev["outcome"] == "success"


def test_cli_tenant_rm_abort_audits_aborted(tmp_path, audit_file, capsys):
    rc = cli._abort_keys_alive("acme", ["service keys (last admin key)"], None)
    assert rc == 1
    (ev,) = _events(audit_file)
    assert ev["action"] == "tenant.rm" and ev["outcome"] == "aborted"
    assert ev["details"]["failed"] == ["service keys (last admin key)"]



# ---------- Principal.key_id (audit attribution) ----------


def test_verify_carries_the_key_id(tmp_path):
    ks = FileKeyStore(tmp_path / "keys.json")
    key_id, key = ks.issue("acme", ["read"])
    p = ks.verify(key)
    assert p is not None and p.key_id == key_id
