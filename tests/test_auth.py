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
