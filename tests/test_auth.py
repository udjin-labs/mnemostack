"""Service-key store (multi-tenant auth): issue / verify / scope / revoke."""

from __future__ import annotations

import json
import os

import pytest

from mnemostack.auth import (
    SCOPES,
    FileKeyStore,
    Principal,
    default_keys_path,
    hash_key,
)


def _store(tmp_path):
    return FileKeyStore(tmp_path / "keys.json")


def test_issue_returns_plaintext_once_and_stores_only_hash(tmp_path):
    ks = _store(tmp_path)
    key_id, key = ks.issue("acme", ["read"], label="app")
    assert key.startswith("msk_")
    # The file must contain the hash, never the plaintext.
    raw = (tmp_path / "keys.json").read_text()
    assert hash_key(key) in raw
    assert key not in raw
    # list_keys never exposes the hash.
    listed = ks.list_keys()
    assert listed[0]["id"] == key_id
    assert "hash" not in listed[0]


def test_verify_resolves_principal(tmp_path):
    ks = _store(tmp_path)
    _, key = ks.issue("acme", ["read", "write"])
    p = ks.verify(key)
    assert isinstance(p, Principal)
    assert p.tenant == "acme"
    assert p.scopes == frozenset({"read", "write"})


def test_verify_unknown_key_is_none(tmp_path):
    ks = _store(tmp_path)
    ks.issue("acme", ["read"])
    assert ks.verify("msk_not-a-real-key") is None


def test_verify_across_instances_reads_from_disk(tmp_path):
    _, key = _store(tmp_path).issue("acme", ["read"])
    # A fresh store instance (e.g. a server process) must resolve it from disk.
    assert _store(tmp_path).verify(key).tenant == "acme"


def test_scopes_admin_implies_all():
    admin = Principal("acme", frozenset({"admin"}))
    assert admin.can("read") and admin.can("write") and admin.can("admin")
    reader = Principal("acme", frozenset({"read"}))
    assert reader.can("read")
    assert not reader.can("write")
    assert not reader.can("admin")


def test_two_tenants_get_distinct_principals(tmp_path):
    ks = _store(tmp_path)
    _, ka = ks.issue("alpha", ["read"])
    _, kb = ks.issue("beta", ["read", "write"])
    assert ks.verify(ka).tenant == "alpha"
    assert ks.verify(kb).tenant == "beta"
    assert ks.verify(ka).tenant != ks.verify(kb).tenant


def test_revoke_removes_key(tmp_path):
    ks = _store(tmp_path)
    key_id, key = ks.issue("acme", ["read"])
    assert ks.verify(key) is not None
    assert ks.revoke(key_id) is True
    assert ks.verify(key) is None
    assert ks.revoke(key_id) is False  # already gone


def test_revoke_guarded_statuses_and_last_admin(tmp_path):
    ks = _store(tmp_path)
    read_id, _ = ks.issue("acme", ["read"])
    admin1, _ = ks.issue("ops", ["admin"])

    # a non-admin key is always revocable
    assert ks.revoke_guarded(read_id, protect_last_admin=True) == "revoked"
    assert ks.revoke_guarded("missing", protect_last_admin=True) == "not_found"
    # the only admin key is protected...
    assert ks.revoke_guarded(admin1, protect_last_admin=True) == "last_admin"
    # ...until a second admin exists, then either is revocable
    admin2, _ = ks.issue("ops2", ["admin"])
    assert ks.revoke_guarded(admin1, protect_last_admin=True) == "revoked"
    # now admin2 is the last one again -> protected
    assert ks.revoke_guarded(admin2, protect_last_admin=True) == "last_admin"
    # without the guard, even the last admin goes
    assert ks.revoke_guarded(admin2) == "revoked"


def test_revoke_guarded_ignores_malformed_admin_records(tmp_path):
    # A shaped-but-invalid "admin" record (empty tenant -> verify() would deny it)
    # must NOT count as a surviving admin, or the last *usable* admin could be
    # revoked, locking everyone out.
    import json

    p = tmp_path / "keys.json"
    ks = FileKeyStore(p)
    real_admin, real_key = ks.issue("ops", ["admin"])
    # hand-write malformed admin shells alongside the real one: one with an empty
    # tenant, one with a well-formed tenant but a hash that isn't a real SHA-256
    # digest (so no key can ever hash to it). Neither can authenticate as admin.
    data = json.loads(p.read_text())
    data["keys"].append({"id": "g1", "hash": "deadbeef", "tenant": "", "scopes": ["admin"]})
    data["keys"].append({"id": "g2", "hash": "deadbeef", "tenant": "x", "scopes": ["admin"]})
    p.write_text(json.dumps(data))
    # neither ghost authenticates, so the real admin is still the LAST usable admin
    assert ks.revoke_guarded(real_admin, protect_last_admin=True) == "last_admin"
    assert ks.verify(real_key) is not None  # the real admin is still there


def test_invalid_scope_rejected(tmp_path):
    ks = _store(tmp_path)
    with pytest.raises(ValueError, match="unknown scope"):
        ks.issue("acme", ["read", "superuser"])
    with pytest.raises(ValueError, match="at least one scope"):
        ks.issue("acme", [])
    with pytest.raises(ValueError, match="tenant is required"):
        ks.issue("", ["read"])


def test_missing_file_verifies_to_none(tmp_path):
    # A store pointed at a nonexistent file resolves nothing (no crash).
    ks = FileKeyStore(tmp_path / "nope.json")
    assert ks.verify("msk_whatever") is None
    assert ks.list_keys() == []


def test_scopes_string_parsing(tmp_path):
    ks = _store(tmp_path)
    _, key = ks.issue("acme", "read, write")  # comma string with spaces
    assert ks.verify(key).scopes == frozenset({"read", "write"})


def test_default_path_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MNEMOSTACK_KEYS_FILE", str(tmp_path / "custom.json"))
    assert default_keys_path() == tmp_path / "custom.json"


def test_scopes_constant():
    assert SCOPES == frozenset({"read", "write", "admin"})


def test_file_is_json_with_keys_array(tmp_path):
    ks = _store(tmp_path)
    ks.issue("acme", ["read"])
    data = json.loads((tmp_path / "keys.json").read_text())
    assert isinstance(data["keys"], list) and len(data["keys"]) == 1
    assert set(data["keys"][0]) == {"id", "hash", "tenant", "scopes", "label", "created_at"}


@pytest.mark.skipif(os.name != "posix", reason="POSIX file mode")
def test_keys_file_is_owner_only(tmp_path):
    import stat

    ks = _store(tmp_path)
    ks.issue("acme", ["read"])
    mode = stat.S_IMODE(os.stat(tmp_path / "keys.json").st_mode)
    assert mode == 0o600


def test_corrupt_store_fails_closed_for_verify_and_raises_for_mgmt(tmp_path):
    from mnemostack.auth import KeyStoreError

    p = tmp_path / "keys.json"
    p.write_text("{ not valid json")
    ks = FileKeyStore(p)
    # verify never crashes — a corrupt store denies (fail closed).
    assert ks.verify("msk_anything") is None
    # management ops surface the corruption loudly instead of silently clobbering.
    with pytest.raises(KeyStoreError):
        ks.list_keys()


@pytest.mark.parametrize("bad_scopes", [None, 5, True])
def test_malformed_scopes_record_denies_not_crashes(tmp_path, bad_scopes):
    from mnemostack.auth import hash_key

    key = "msk_probe"
    p = tmp_path / "keys.json"
    p.write_text(
        json.dumps(
            {"keys": [{"id": "1", "hash": hash_key(key), "tenant": "acme", "scopes": bad_scopes}]}
        )
    )
    # A malformed scopes value denies that key without raising for every key.
    assert FileKeyStore(p).verify(key) is None


def test_revoke_only_removes_its_target(tmp_path):
    ks = _store(tmp_path)
    id_a, _ = ks.issue("alpha", ["read"])
    id_b, key_b = ks.issue("beta", ["read"])
    assert id_a != id_b  # ids are unique
    assert ks.revoke(id_a) is True
    # beta's unrelated key must survive (no id-collision blast radius).
    assert ks.verify(key_b) is not None


# ----- CLI -----


def _ns(**kw):
    import argparse

    return argparse.Namespace(**kw)


def test_cli_keys_add_list_revoke_roundtrip(tmp_path, capsys):
    from mnemostack.cli import cmd_keys_add, cmd_keys_list, cmd_keys_revoke

    kf = str(tmp_path / "k.json")
    assert cmd_keys_add(_ns(keys_file=kf, tenant="acme", scopes="read,write", label="app")) == 0
    out = capsys.readouterr().out
    assert "msk_" in out and "shown once" in out

    assert cmd_keys_list(_ns(keys_file=kf, json=True)) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["tenant"] == "acme"
    assert "hash" not in listed[0]
    kid = listed[0]["id"]

    assert cmd_keys_revoke(_ns(keys_file=kf, id=kid)) == 0
    assert cmd_keys_revoke(_ns(keys_file=kf, id=kid)) == 1  # already gone


def test_cli_keys_add_rejects_bad_scope(tmp_path, capsys):
    from mnemostack.cli import cmd_keys_add

    rc = cmd_keys_add(_ns(keys_file=str(tmp_path / "k.json"), tenant="acme", scopes="root", label=""))
    assert rc == 2
    assert "unknown scope" in capsys.readouterr().err


def test_verify_fails_closed_on_unreadable_store(tmp_path):
    from mnemostack.auth import KeyStoreError

    # A directory at the store path: open() raises OSError, not FileNotFoundError.
    store_path = tmp_path / "keys.json"
    store_path.mkdir()
    ks = FileKeyStore(store_path)
    assert ks.verify("msk_whatever") is None  # fail closed, never raises
    with pytest.raises(KeyStoreError):
        ks.list_keys()  # management surfaces it loudly


def test_malformed_shape_surfaces_not_reset(tmp_path):
    from mnemostack.auth import KeyStoreError

    p = tmp_path / "keys.json"
    p.write_text('{"keys": "not-a-list"}')  # valid JSON, wrong shape
    ks = FileKeyStore(p)
    assert ks.verify("msk_x") is None
    with pytest.raises(KeyStoreError):
        ks.list_keys()
    p.write_text("[]")  # top-level array is also invalid
    with pytest.raises(KeyStoreError):
        FileKeyStore(p).list_keys()


def test_save_does_not_follow_tmp_symlink(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    p = tmp_path / "keys.json"
    # An attacker pre-creates a symlink at the old fixed temp path.
    (tmp_path / "keys.json.tmp").symlink_to(victim)
    FileKeyStore(p).issue("alpha", "read")  # writes via mkstemp (random name)
    assert victim.read_text() == "precious"  # untouched
    assert p.is_file() and not p.is_symlink()  # store is a real file


def test_verify_denies_non_list_scopes_no_privilege_escalation(tmp_path):
    p = tmp_path / "keys.json"
    key = "msk_test"
    # A malformed scopes object would iterate to its keys — {"admin": false} must
    # NOT grant admin; the record is denied.
    p.write_text(
        json.dumps(
            {"keys": [{"id": "x", "hash": hash_key(key), "tenant": "acme", "scopes": {"admin": False}}]}
        )
    )
    assert FileKeyStore(p).verify(key) is None


def test_verify_skips_malformed_hash_and_finds_valid(tmp_path):
    p = tmp_path / "keys.json"
    key = "msk_good"
    # A non-ASCII hash on an earlier record must not crash verify (TypeError from
    # compare_digest) — it's skipped and the valid record still authenticates.
    p.write_text(
        json.dumps(
            {
                "keys": [
                    {"id": "bad", "hash": "деadbeef", "tenant": "x", "scopes": ["read"]},
                    {"id": "ok", "hash": hash_key(key), "tenant": "acme", "scopes": ["read"]},
                ]
            },
            ensure_ascii=False,
        )
    )
    principal = FileKeyStore(p).verify(key)
    assert principal is not None
    assert principal.tenant == "acme"
    assert principal.can("read")


def test_object_without_keys_member_is_rejected(tmp_path):
    from mnemostack.auth import KeyStoreError

    p = tmp_path / "keys.json"
    p.write_text('{"other": 1}')  # a valid object but not a key store (e.g. a config file)
    assert FileKeyStore(p).verify("msk_x") is None
    with pytest.raises(KeyStoreError):
        FileKeyStore(p).list_keys()


def test_invalid_utf8_store_is_corrupt(tmp_path):
    from mnemostack.auth import KeyStoreError

    p = tmp_path / "keys.json"
    p.write_bytes(b"\xff\xfe not utf-8")
    assert FileKeyStore(p).verify("msk_x") is None  # fail closed, no crash
    with pytest.raises(KeyStoreError):
        FileKeyStore(p).list_keys()


def test_verify_denies_non_string_tenant(tmp_path):
    key = "msk_t"
    p = tmp_path / "keys.json"
    p.write_text(
        json.dumps(
            {"keys": [{"id": "x", "hash": hash_key(key), "tenant": ["a", "b"], "scopes": ["read"]}]}
        )
    )
    assert FileKeyStore(p).verify(key) is None


def test_issue_errors_cleanly_when_parent_is_a_file(tmp_path):
    from mnemostack.auth import KeyStoreError

    blocker = tmp_path / "afile"
    blocker.write_text("blocking")  # the store's parent path is a regular file
    ks = FileKeyStore(blocker / "keys.json")
    with pytest.raises(KeyStoreError):
        ks.issue("acme", "read")


def test_save_write_failure_is_store_error(tmp_path, monkeypatch):
    from mnemostack.auth import KeyStoreError

    def boom(*a, **k):
        raise OSError("disk full")

    ks = FileKeyStore(tmp_path / "keys.json")
    monkeypatch.setattr("mnemostack.auth.os.replace", boom)
    with pytest.raises(KeyStoreError):
        ks.issue("acme", "read")


def test_list_keys_sanitizes_malformed_record(tmp_path):
    p = tmp_path / "keys.json"
    p.write_text(
        json.dumps(
            {"keys": [{"id": 5, "hash": "abc", "tenant": ["x"], "scopes": [1, 2], "created_at": None}]}
        )
    )
    rows = FileKeyStore(p).list_keys()
    assert rows[0]["id"] == "5"
    assert rows[0]["scopes"] == ["1", "2"]  # coerced to strings; formatter won't crash
    ",".join(rows[0]["scopes"])  # the exact op cmd_keys_list does


def test_keys_list_works_with_malformed_stack_config(tmp_path, monkeypatch):
    from mnemostack.cli import main

    # An unrelated malformed stack config/env must not block key management.
    monkeypatch.setenv("MNEMOSTACK_TOKEN_BUDGET", "not-a-number")
    rc = main(["keys", "list", "--keys-file", str(tmp_path / "keys.json")])
    assert rc == 0


def test_revoke_tenant_atomic_all_and_last_admin(tmp_path):
    ks = _store(tmp_path)
    ks.issue("acme", ["read"])
    ks.issue("acme", ["write"])
    ks.issue("beta", ["read"])
    # drops ALL of the tenant's keys in one shot, leaves others
    res = ks.revoke_tenant("acme")
    assert res == {"revoked": 2, "last_admin_kept": False}
    assert [k["tenant"] for k in ks.list_keys()] == ["beta"]

    # the tenant's key that is the only usable admin is KEPT and flagged
    ks2 = _store(tmp_path / "sub")
    ks2.issue("ops", ["admin"])   # the only admin, owned by 'ops'
    ks2.issue("ops", ["write"])
    res2 = ks2.revoke_tenant("ops", protect_last_admin=True)
    assert res2["last_admin_kept"] is True and res2["revoked"] == 1  # write dropped
    assert any("admin" in k["scopes"] for k in ks2.list_keys())

    with pytest.raises(ValueError):
        ks.revoke_tenant("")


def test_revoke_tenant_keeps_one_admin_when_tenant_owns_them_all(tmp_path):
    # If the offboarded tenant owns EVERY usable admin key (2+), one must survive
    # or key management is locked out — a global snapshot count missed this.
    ks = _store(tmp_path)
    ks.issue("acme", ["admin"])
    ks.issue("acme", ["admin"])  # two admins, both acme; no other admin anywhere
    ks.issue("acme", ["write"])
    res = ks.revoke_tenant("acme", protect_last_admin=True)
    assert res["last_admin_kept"] is True
    assert res["revoked"] == 2  # one admin + the write key dropped; one admin kept
    remaining = ks.list_keys()
    assert len(remaining) == 1 and "admin" in remaining[0]["scopes"]

    # but with an admin owned by ANOTHER tenant, all of acme's keys can go
    ks2 = _store(tmp_path / "b")
    ks2.issue("acme", ["admin"])
    ks2.issue("acme", ["admin"])
    ks2.issue("ops", ["admin"])  # a surviving admin elsewhere
    res2 = ks2.revoke_tenant("acme", protect_last_admin=True)
    assert res2 == {"revoked": 2, "last_admin_kept": False}
    assert [k["tenant"] for k in ks2.list_keys()] == ["ops"]
