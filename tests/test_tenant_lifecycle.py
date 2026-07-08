"""Tenant offboarding (tenant-rm) and backup (tenant-export) — cross-store tests.

Uses a real in-memory Qdrant seeded across two tenants plus a legacy unscoped
point, so the isolation assertions are genuine: removing tenant A must never
touch tenant B's records or the unscoped legacy data.
"""

from __future__ import annotations

import argparse
import json

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance

import mnemostack.cli as cli
from mnemostack.auth import FileKeyStore
from mnemostack.quotas import FileQuotaStore
from mnemostack.recall.pipeline import FileStateStore
from mnemostack.recall.pipeline.state import tenant_state_key
from mnemostack.vector import VectorStore

_VEC = [1.0, 0.0, 0.0, 0.0]


def _seeded_store() -> VectorStore:
    s = VectorStore.__new__(VectorStore)
    s.collection = "mt"
    s.dimension = 4
    s.distance = Distance.COSINE
    s.client = QdrantClient(":memory:")
    s.ensure_collection()
    s.upsert(1, _VEC, {"text": "alpha one", "source": "a1.md"}, tenant="alpha")
    s.upsert(2, _VEC, {"text": "alpha two", "source": "a2.md"}, tenant="alpha")
    s.upsert(3, _VEC, {"text": "beta one", "source": "b1.md"}, tenant="beta")
    s.upsert(4, _VEC, {"text": "legacy unscoped"})  # no tenant_id
    return s


# ---------- VectorStore.delete_tenant ----------


def test_delete_tenant_removes_only_that_tenant():
    s = _seeded_store()
    assert s.delete_tenant("alpha") == 2
    assert s.count(tenant="alpha") == 0
    assert s.count(tenant="beta") == 1  # untouched
    # the legacy unscoped point survives too (total = beta + legacy)
    assert s.client.count(collection_name="mt").count == 2
    assert s.delete_tenant("alpha") == 0  # idempotent


def test_delete_tenant_rejects_empty():
    s = _seeded_store()
    with pytest.raises(ValueError):
        s.delete_tenant("")


# ---------- Hit.vector via scroll ----------


def test_scroll_with_vectors_carries_the_embedding():
    s = _seeded_store()
    hits = list(s.scroll(with_vectors=True, tenant="beta"))
    assert len(hits) == 1 and hits[0].vector is not None
    assert len(hits[0].vector) == 4
    # default scroll stays vector-free (no payload bloat for browse paths)
    assert all(h.vector is None for h in s.scroll(tenant="beta"))


# ---------- GraphStore.delete_tenant (recorded Cypher) ----------


def test_graph_delete_tenant_cypher_is_tenant_confined():
    from unittest.mock import MagicMock

    from mnemostack.graph.store import GraphStore

    calls: list[tuple[str, dict]] = []

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def run(self, cypher, **params):
            calls.append((cypher, params))

            class _R:
                def single(self):
                    return {"n": 3}

            return _R()

    store = GraphStore.__new__(GraphStore)
    driver = MagicMock()
    driver.session.return_value = _Session()
    store.driver = driver
    store.database = None

    counts = store.delete_tenant("acme")
    assert counts == {"nodes": 3, "relationships": 3, "detached": 3}
    # every statement is confined to the tenant property — no unscoped MATCH
    assert len(calls) == 5  # 3 counts + delete rels + detach delete nodes
    for cypher, params in calls:
        assert "{tenant: $tenant}" in cypher
        assert params == {"tenant": "acme"}
    assert any("DETACH DELETE" in c for c, _ in calls)

    # dry_run only counts
    calls.clear()
    store.delete_tenant("acme", dry_run=True)
    assert len(calls) == 3 and all("DETACH" not in c and not c.strip().endswith("DELETE r") for c, _ in calls)

    with pytest.raises(ValueError):
        store.delete_tenant("")


# ---------- state store delete ----------


def test_state_store_delete(tmp_path):
    fs = FileStateStore(tmp_path / "state.json")
    key = tenant_state_key("q_table", "acme")
    fs.set(key, {"x": 1})
    fs.set("q_table", {"y": 2})  # unscoped partition
    assert fs.delete(key) is True
    assert fs.get(key) is None
    assert fs.get("q_table") == {"y": 2}  # unscoped survives
    assert fs.delete(key) is False  # already gone


# ---------- CLI: tenant-export ----------


def _ns(**kw) -> argparse.Namespace:
    base = {"collection": "mt", "qdrant": "http://localhost:6333"}
    base.update(kw)
    return argparse.Namespace(**base)


def test_tenant_export_writes_only_the_tenant(monkeypatch, tmp_path):
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    out = tmp_path / "dump.jsonl"
    rc = cli.cmd_tenant_export(_ns(tenant="alpha", output=str(out), no_vectors=False))
    assert rc == 0
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert lines[0]["kind"] == "meta" and lines[0]["tenant"] == "alpha"
    points = [row for row in lines if row["kind"] == "point"]
    assert len(points) == 2
    assert all(r["payload"]["tenant_id"] == "alpha" for r in points)
    assert all(len(r["vector"]) == 4 for r in points)  # vectors included
    texts = {r["payload"]["text"] for r in points}
    assert "beta one" not in texts and "legacy unscoped" not in texts


def test_tenant_export_no_vectors(monkeypatch, tmp_path):
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    out = tmp_path / "dump.jsonl"
    assert cli.cmd_tenant_export(_ns(tenant="beta", output=str(out), no_vectors=True)) == 0
    points = [json.loads(line) for line in out.read_text().splitlines()][1:]
    assert points and all("vector" not in r for r in points)


def test_tenant_export_rejects_empty_tenant(monkeypatch, tmp_path, capsys):
    assert cli.cmd_tenant_export(_ns(tenant="  ", output="-", no_vectors=False)) == 2
    assert "non-empty tenant" in capsys.readouterr().err


# ---------- CLI: tenant-rm ----------


def _rm_ns(tmp_path, **kw) -> argparse.Namespace:
    base = {
        "collection": "mt",
        "qdrant": "http://localhost:6333",
        "memgraph_uri": None,
        "graph_timeout": 5.0,
        "dry_run": False,
        "yes": False,
        "keys_file": str(tmp_path / "keys.json"),
        "quotas_file": str(tmp_path / "quotas.json"),
        "state_path": str(tmp_path / "state.json"),
        "external_keys_revoked": False,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def _seed_config(tmp_path):
    ks = FileKeyStore(tmp_path / "keys.json")
    ks.issue("alpha", ["read"])
    ks.issue("alpha", ["write"])
    ks.issue("beta", ["read"])
    FileQuotaStore(tmp_path / "quotas.json").set("alpha", max_points=100)
    fs = FileStateStore(tmp_path / "state.json")
    fs.set(tenant_state_key("q_table", "alpha"), {"w": 1})
    fs.set(tenant_state_key("ior_log", "alpha"), [1])
    fs.set(tenant_state_key("q_table", "beta"), {"w": 2})
    return ks, fs


def test_tenant_rm_dry_run_deletes_nothing(monkeypatch, tmp_path, capsys):
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    ks, fs = _seed_config(tmp_path)
    rc = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha", dry_run=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert "vector points:    2" in out and "service keys:     2" in out
    assert "quota:            set" in out and "2 partition(s)" in out
    # nothing was deleted
    assert store.count(tenant="alpha") == 2
    assert len(ks.list_keys()) == 3
    assert fs.get(tenant_state_key("q_table", "alpha")) is not None


def test_tenant_rm_requires_yes(monkeypatch, tmp_path, capsys):
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    _seed_config(tmp_path)
    rc = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha"))
    assert rc == 2
    assert "--yes" in capsys.readouterr().err
    assert store.count(tenant="alpha") == 2  # nothing deleted


def test_tenant_rm_removes_everything_and_spares_others(monkeypatch, tmp_path):
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    ks, fs = _seed_config(tmp_path)
    rc = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha", yes=True))
    assert rc == 0
    # alpha is gone across every store
    assert store.count(tenant="alpha") == 0
    assert all(k["tenant"] != "alpha" for k in ks.list_keys())
    assert FileQuotaStore(tmp_path / "quotas.json").get("alpha") is None
    assert fs.get(tenant_state_key("q_table", "alpha")) is None
    assert fs.get(tenant_state_key("ior_log", "alpha")) is None
    # beta and the legacy unscoped data are untouched
    assert store.count(tenant="beta") == 1
    assert store.client.count(collection_name="mt").count == 2  # beta + legacy
    assert any(k["tenant"] == "beta" for k in ks.list_keys())
    assert fs.get(tenant_state_key("q_table", "beta")) == {"w": 2}


def test_tenant_rm_deletes_graph_when_uri_given(monkeypatch, tmp_path):
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    _seed_config(tmp_path)

    class _FakeGraph:
        def __init__(self):
            self.calls: list[tuple[str, bool]] = []

        def delete_tenant(self, tenant, *, dry_run=False):
            self.calls.append((tenant, dry_run))
            return {"nodes": 5, "relationships": 7}

        def close(self):
            pass

    fake = _FakeGraph()
    monkeypatch.setattr("mnemostack.graph.factory.make_graph_store", lambda *a, **k: fake)
    rc = cli.cmd_tenant_rm(
        _rm_ns(tmp_path, tenant="alpha", yes=True, memgraph_uri="bolt://x:7687")
    )
    assert rc == 0
    # counted (dry_run=True) then deleted (dry_run=False), tenant-confined
    assert fake.calls == [("alpha", True), ("alpha", False)]


def test_tenant_rm_external_keystore_requires_flag_then_sweeps(monkeypatch, tmp_path, capsys):
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    ks, _fs = _seed_config(tmp_path)
    monkeypatch.setenv("MNEMOSTACK_KEYSTORE", "openbao")
    # WITHOUT the flag: tenant-rm can't revoke the external key, so it refuses to
    # sweep (a live key would re-write the data back) — nothing deleted.
    rc = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha", yes=True))
    assert rc == 1
    err = capsys.readouterr().err
    assert "external store" in err and "--external-keys-revoked" in err
    assert store.count(tenant="alpha") == 2  # data untouched
    # WITH the flag (operator revoked the key in OpenBao first): sweep proceeds.
    rc2 = cli.cmd_tenant_rm(
        _rm_ns(tmp_path, tenant="alpha", yes=True, external_keys_revoked=True)
    )
    assert rc2 == 0
    assert "treated as revoked" in capsys.readouterr().out
    assert store.count(tenant="alpha") == 0
    # the LOCAL file store was never touched (keys live in the external store)
    assert sum(1 for k in ks.list_keys() if k["tenant"] == "alpha") == 2


def test_tenant_rm_partial_failure_reports_and_continues(monkeypatch, tmp_path, capsys):
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    ks, fs = _seed_config(tmp_path)

    def _boom(tenant):
        raise RuntimeError("qdrant exploded")

    monkeypatch.setattr(store, "delete_tenant", _boom)
    rc = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha", yes=True))
    assert rc == 1  # nonzero: something failed
    err = capsys.readouterr().err
    assert "vector delete failed" in err and "FAILED: vector points" in err
    # the other stores were still cleaned (best-effort, not all-or-nothing)
    assert all(k["tenant"] != "alpha" for k in ks.list_keys())
    assert fs.get(tenant_state_key("q_table", "alpha")) is None


def test_tenant_rm_refuses_to_revoke_last_admin_key(monkeypatch, tmp_path, capsys):
    # If the offboarded tenant holds the LAST usable admin key, revoking it would
    # lock the operator out of key management — same guard the admin console uses.
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    ks = FileKeyStore(tmp_path / "keys.json")
    admin_id, _ = ks.issue("alpha", ["admin"])  # alpha holds the ONLY admin key
    ks.issue("alpha", ["read"])
    rc = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha", yes=True))
    assert rc == 1  # partial: the admin key survived
    err = capsys.readouterr().err
    assert "LAST usable admin key" in err and "last admin key" in err
    remaining = ks.list_keys()
    assert [k["id"] for k in remaining] == [admin_id]  # read key revoked, admin kept
    # with a SECOND admin key elsewhere, the same offboarding completes cleanly
    ks.issue("ops", ["admin"])
    rc2 = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha", yes=True))
    assert rc2 == 0
    assert all(k["tenant"] != "alpha" for k in ks.list_keys())


def test_tenant_export_unwritable_output_is_clean_error(monkeypatch, tmp_path, capsys):
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    bad = tmp_path / "no-such-dir" / "dump.jsonl"
    rc = cli.cmd_tenant_export(_ns(tenant="alpha", output=str(bad), no_vectors=False))
    assert rc == 1
    assert "cannot write" in capsys.readouterr().err  # clean error, no traceback


def test_tenant_export_zero_points_warns(monkeypatch, tmp_path, capsys):
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    out = tmp_path / "empty.jsonl"
    rc = cli.cmd_tenant_export(_ns(tenant="ghost", output=str(out), no_vectors=False))
    assert rc == 0  # an empty tenant is a valid (if suspicious) dump
    assert "0 points matched" in capsys.readouterr().err


# ---------- best-effort under store outages (review round 1) ----------


def test_tenant_rm_continues_when_qdrant_down(monkeypatch, tmp_path, capsys):
    # Qdrant unreachable must NOT strand keys/quota/state cleanup — that outage is
    # exactly what best-effort is for. Vector store lands in FAILED, exit nonzero.
    class _DownStore:
        def __init__(self, **_):
            raise ConnectionError("qdrant down")

    monkeypatch.setattr(cli, "VectorStore", _DownStore)
    ks, fs = _seed_config(tmp_path)
    rc = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha", yes=True))
    assert rc == 1
    err = capsys.readouterr().err
    assert "FAILED: vector points" in err
    # ...but everything reachable was still cleaned
    assert all(k["tenant"] != "alpha" for k in ks.list_keys())
    assert FileQuotaStore(tmp_path / "quotas.json").get("alpha") is None
    assert fs.get(tenant_state_key("q_table", "alpha")) is None


def test_tenant_rm_continues_when_graph_down(monkeypatch, tmp_path, capsys):
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    ks, _fs = _seed_config(tmp_path)

    def _boom(*a, **k):
        raise ConnectionError("memgraph down")

    monkeypatch.setattr("mnemostack.graph.factory.make_graph_store", _boom)
    rc = cli.cmd_tenant_rm(
        _rm_ns(tmp_path, tenant="alpha", yes=True, memgraph_uri="bolt://x:7687")
    )
    assert rc == 1
    assert "FAILED: graph" in capsys.readouterr().err
    # vector + keys still cleaned despite the graph outage
    assert store.count(tenant="alpha") == 0
    assert all(k["tenant"] != "alpha" for k in ks.list_keys())


def test_tenant_rm_unreadable_key_store_aborts_sweep(monkeypatch, tmp_path, capsys):
    # Keys that can't be inspected might still be verified by serve --auth, so a
    # live key could re-write swept data — abort, don't sweep with keys of
    # unknown state.
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    _seed_config(tmp_path)
    (tmp_path / "keys.json").write_text("{ not json")  # corrupt the key store
    rc = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha", yes=True))
    assert rc == 1
    err = capsys.readouterr().err
    assert "could not be inspected" in err and "NOT removed" in err
    assert store.count(tenant="alpha") == 2  # data sweep never ran


def test_tenant_rm_unreadable_quota_store_is_partial(monkeypatch, tmp_path, capsys):
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    _seed_config(tmp_path)
    (tmp_path / "quotas.json").write_text("{ not json")  # get() would fail OPEN
    rc = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha", yes=True))
    assert rc == 1
    assert "FAILED: quota" in capsys.readouterr().err


def test_tenant_rm_corrupt_state_file_is_partial(monkeypatch, tmp_path, capsys):
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    _seed_config(tmp_path)
    (tmp_path / "state.json").write_text("{ not json")  # _read_all would fail OPEN
    rc = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha", yes=True))
    assert rc == 1
    assert "FAILED: learning state" in capsys.readouterr().err


def test_tenant_rm_absent_collection_is_not_a_failure(monkeypatch, tmp_path, capsys):
    # A never-created/already-dropped collection means nothing to delete — the
    # offboarding of keys/quota/state must complete cleanly (exit 0).
    class _NoCollection:
        def __init__(self, **_):
            pass

        def collection_exists(self):
            return False

    monkeypatch.setattr(cli, "VectorStore", lambda **_: _NoCollection())
    ks, _fs = _seed_config(tmp_path)
    rc = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha", yes=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert "collection absent" in out and "fully removed" in out
    assert all(k["tenant"] != "alpha" for k in ks.list_keys())


def test_state_store_delete_raises_on_corrupt_file(tmp_path):
    # delete() is a management op: unlike get() (fail-open for recall), a corrupt
    # file must raise, not read as "nothing to delete".
    p = tmp_path / "state.json"
    p.write_text("{ not json")
    fs = FileStateStore(p)
    assert fs.get("anything") is None  # read path still fails open
    with pytest.raises(ValueError):  # json.JSONDecodeError is a ValueError
        fs.delete("anything")


def test_tenant_rm_dry_run_incomplete_counts_exit_nonzero(monkeypatch, tmp_path, capsys):
    class _DownStore:
        def __init__(self, **_):
            raise ConnectionError("qdrant down")

    monkeypatch.setattr(cli, "VectorStore", _DownStore)
    _seed_config(tmp_path)
    rc = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha", dry_run=True))
    assert rc == 1  # a script must notice the counts are partial
    out = capsys.readouterr()
    assert "unknown — store unavailable" in out.out
    assert "counts are incomplete" in out.err


# ---------- review round 2 ----------


def test_tenant_rm_revokes_keys_before_data_deletion(monkeypatch, tmp_path):
    # On a live authed deployment the tenant's key must die FIRST, or it can
    # keep writing into stores this command already cleaned.
    order: list[str] = []

    class _OrderStore:
        def __init__(self, **_):
            pass

        def collection_exists(self):
            return True

        def count(self, tenant=None):
            return 1

        def delete_tenant(self, tenant):
            order.append("vector")
            return 1

    monkeypatch.setattr(cli, "VectorStore", lambda **_: _OrderStore())
    ks, _fs = _seed_config(tmp_path)
    class _OrderKeys:
        def list_keys(self):
            return ks.list_keys()

        def revoke_tenant(self, tenant, **kw):
            order.append("keys")
            return ks.revoke_tenant(tenant, **kw)

    monkeypatch.setattr(cli, "_keys_store", lambda _a: _OrderKeys())
    rc = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha", yes=True))
    assert rc == 0
    assert "keys" in order and "vector" in order
    assert order.index("keys") < order.index("vector")  # revocation cuts writes first


def test_tenant_rm_bad_state_path_is_partial_not_abort(monkeypatch, tmp_path, capsys):
    # --state-path whose parent can't be created (a path component is a FILE)
    # must degrade to a failed store, not abort the whole offboarding.
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    ks, _fs = _seed_config(tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file")  # mkdir(parents=True) under it will fail
    rc = cli.cmd_tenant_rm(
        _rm_ns(tmp_path, tenant="alpha", yes=True, state_path=str(blocker / "x" / "state.json"))
    )
    assert rc == 1
    assert "FAILED: learning state" in capsys.readouterr().err
    # the rest still proceeded
    assert store.count(tenant="alpha") == 0
    assert all(k["tenant"] != "alpha" for k in ks.list_keys())


def test_lifecycle_commands_tolerate_malformed_stack_config(monkeypatch, tmp_path, capsys):
    # A malformed unrelated env value must not block emergency backup/offboarding.
    monkeypatch.setenv("MNEMOSTACK_TOKEN_BUDGET", "notint")
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    out = tmp_path / "dump.jsonl"
    rc = cli.main([
        "tenant-export", "--tenant", "alpha", "-o", str(out), "--no-vectors",
        "--qdrant", "http://localhost:6333", "--collection", "mt",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "config failed to load" in captured.err  # warned, not aborted
    assert out.exists()
    # sanity: a non-lifecycle command still fails loud on the same env
    with pytest.raises(ValueError):  # Config.load: invalid literal for int()
        cli.main(["search", "q"])


def test_config_fallback_refuses_destructive_defaults(monkeypatch, tmp_path, capsys):
    # With the config unreadable, the built-in localhost/"mnemostack" defaults
    # could point tenant-rm --yes (or a trusted backup) at the WRONG store —
    # refuse unless the target is named explicitly.
    monkeypatch.setenv("MNEMOSTACK_TOKEN_BUDGET", "notint")
    deleted = []

    class _Guard:
        def __init__(self, **_):
            deleted.append(True)

    monkeypatch.setattr(cli, "VectorStore", _Guard)
    rc = cli.main(["tenant-rm", "--tenant", "alpha", "--yes"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "refusing to fall back" in err and "--qdrant" in err
    assert deleted == []  # no store was even constructed


def test_tenant_export_is_atomic_on_midway_failure(monkeypatch, tmp_path, capsys):
    # Qdrant dropping mid-scroll must not truncate an existing dump at the
    # trusted backup path: the old file survives, no partial temp remains.
    class _FlakyStore:
        def __init__(self, **_):
            pass

        def collection_exists(self):
            return True

        def scroll(self, **_):
            yield type("H", (), {"id": "1", "payload": {"text": "x"}, "vector": None})()
            raise ConnectionError("qdrant dropped mid-scroll")

    monkeypatch.setattr(cli, "VectorStore", lambda **_: _FlakyStore())
    out = tmp_path / "backup.jsonl"
    out.write_text("PRECIOUS OLD BACKUP\n")
    rc = cli.cmd_tenant_export(_ns(tenant="alpha", output=str(out), no_vectors=True))
    assert rc == 1
    assert "export failed after 1 point(s)" in capsys.readouterr().err
    assert out.read_text() == "PRECIOUS OLD BACKUP\n"  # old dump untouched
    assert not list(tmp_path.glob("*.tmp"))  # no partial temp left behind


# ---------- review round 4 ----------


def test_tenant_rm_unknown_keystore_backend_is_failure_not_skip(monkeypatch, tmp_path, capsys):
    # A typo like MNEMOSTACK_KEYSTORE=flie must not read as "externally managed"
    # and silently skip local keys — that would report full removal with the
    # tenant's keys alive.
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    ks, _fs = _seed_config(tmp_path)
    monkeypatch.setenv("MNEMOSTACK_KEYSTORE", "flie")
    rc = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha", yes=True))
    assert rc == 1
    err = capsys.readouterr().err
    assert "unknown MNEMOSTACK_KEYSTORE backend" in err and "FAILED: service keys" in err
    # keys were not touched (unknown backend), but the run is loudly partial
    assert sum(1 for k in ks.list_keys() if k["tenant"] == "alpha") == 2


def test_tenant_rm_last_admin_aborts_data_sweep(monkeypatch, tmp_path, capsys):
    # If the tenant retains a usable (write-capable) key, sweeping data stores
    # would leave a window for fresh writes right after cleaning — abort instead.
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    ks = FileKeyStore(tmp_path / "keys.json")
    ks.issue("alpha", ["admin"])  # the ONLY admin key — revocation will refuse
    FileQuotaStore(tmp_path / "quotas.json").set("alpha", max_points=5)
    rc = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha", yes=True))
    assert rc == 1
    err = capsys.readouterr().err
    assert "retains an active key" in err and "NOT removed" in err
    # nothing was swept: points and quota still present for the re-run
    assert store.count(tenant="alpha") == 2
    assert FileQuotaStore(tmp_path / "quotas.json").get("alpha") is not None


def test_tenant_export_temp_is_exclusive(monkeypatch, tmp_path):
    # A pre-existing "<output>.tmp" (another user's file / stale export) must
    # survive: the temp is mkstemp-unique, never a predictable truncate target.
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    out = tmp_path / "dump.jsonl"
    bystander = tmp_path / "dump.jsonl.tmp"
    bystander.write_text("SOMEONE ELSE'S FILE\n")
    rc = cli.cmd_tenant_export(_ns(tenant="alpha", output=str(out), no_vectors=True))
    assert rc == 0
    assert out.exists()
    assert bystander.read_text() == "SOMEONE ELSE'S FILE\n"  # untouched


# ---------- review round 5 ----------


def test_tenant_rm_revocation_write_failure_aborts_sweep(monkeypatch, tmp_path, capsys):
    # A failed revocation WRITE (disk full / unwritable dir) leaves the tenant's
    # keys alive — same active-key window as last-admin, same abort.
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    ks, _fs = _seed_config(tmp_path)

    class _BrokenWrite:
        def list_keys(self):
            return ks.list_keys()  # readable...

        def revoke_tenant(self, tenant, **kw):
            from mnemostack.auth import KeyStoreError
            raise KeyStoreError("read-only file system")  # ...but not writable

    monkeypatch.setattr(cli, "_keys_store", lambda _a: _BrokenWrite())
    rc = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha", yes=True))
    assert rc == 1
    err = capsys.readouterr().err
    assert "key revocation failed" in err and "NOT removed" in err
    assert store.count(tenant="alpha") == 2  # data sweep never ran


def test_config_fallback_requires_graph_flag_when_env_graph_set(monkeypatch, tmp_path, capsys):
    # With the config unreadable and a graph configured via env, tenant-rm must
    # not silently drop the graph from the sweep — require --memgraph-uri too.
    monkeypatch.setenv("MNEMOSTACK_TOKEN_BUDGET", "notint")
    monkeypatch.setenv("MNEMOSTACK_MEMGRAPH_URI", "bolt://prod-graph:7687")
    rc = cli.main([
        "tenant-rm", "--tenant", "alpha", "--yes",
        "--qdrant", "http://localhost:6333", "--collection", "mt",
    ])
    assert rc == 2
    assert "--memgraph-uri" in capsys.readouterr().err


# ---------- review round 6 ----------


def test_config_fallback_seeds_graph_auth_from_env(monkeypatch, tmp_path, capsys):
    # With the config unreadable but graph auth in env, the fallback must still
    # pass those creds through — else graph deletion fails on an authed Memgraph
    # and the sweep proceeds reporting removal while graph records survive.
    monkeypatch.setenv("MNEMOSTACK_TOKEN_BUDGET", "notint")
    monkeypatch.setenv("MNEMOSTACK_MEMGRAPH_URI", "bolt://g:7687")
    monkeypatch.setenv("MNEMOSTACK_GRAPH_USER", "neo4j")
    monkeypatch.setenv("MNEMOSTACK_GRAPH_PASSWORD", "s3cret")
    seen: dict = {}

    class _FakeGraph:
        def delete_tenant(self, tenant, *, dry_run=False):
            return {"nodes": 0, "relationships": 0, "detached": 0}

        def close(self):
            pass

    def _capture(uri, *, timeout=5.0, user=None, password=None, database=None):
        seen.update(user=user, password=password)
        return _FakeGraph()

    monkeypatch.setattr("mnemostack.graph.factory.make_graph_store", _capture)
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    rc = cli.main([
        "tenant-rm", "--tenant", "alpha", "--yes",
        "--qdrant", "http://localhost:6333", "--collection", "mt",
        "--memgraph-uri", "bolt://g:7687",
    ])
    assert rc == 0
    assert seen == {"user": "neo4j", "password": "s3cret"}  # env creds threaded through


# ---------- review round 7 ----------


def test_config_fallback_requires_graph_flag_for_GRAPH_URI_alias(monkeypatch, tmp_path, capsys):
    # config.py reads MNEMOSTACK_GRAPH_URI (canonical) as well as _MEMGRAPH_URI —
    # both must force --memgraph-uri on the fallback, or an env-configured graph
    # silently drops out of the sweep.
    monkeypatch.setenv("MNEMOSTACK_TOKEN_BUDGET", "notint")
    monkeypatch.setenv("MNEMOSTACK_GRAPH_URI", "bolt://prod-graph:7687")
    rc = cli.main([
        "tenant-rm", "--tenant", "alpha", "--yes",
        "--qdrant", "http://localhost:6333", "--collection", "mt",
    ])
    assert rc == 2
    assert "--memgraph-uri" in capsys.readouterr().err


def test_tenant_rm_revokes_malformed_id_record_by_tenant(monkeypatch, tmp_path):
    # revoke_tenant matches by RECORD (tenant), not id, so a malformed-id record
    # that revoke-by-id would miss is still removed — no "unrevocable" survivor.
    import json as _json

    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    kf = tmp_path / "keys.json"
    ks = FileKeyStore(kf)
    ks.issue("alpha", ["write"])
    data = _json.loads(kf.read_text())
    data["keys"][0]["id"] = 12345  # numeric id: revoke-by-id (str) would miss it
    kf.write_text(_json.dumps(data))
    rc = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha", yes=True))
    assert rc == 0
    assert not any(k["tenant"] == "alpha" for k in FileKeyStore(kf).list_keys())
    assert store.count(tenant="alpha") == 0


# ---------- review round 8 ----------


def test_tenant_rm_empty_memgraph_uri_is_rejected(monkeypatch, tmp_path, capsys):
    # An explicit but empty --memgraph-uri signals intent to clean the graph; it
    # must not coerce to None and silently skip it under a "fully removed" report.
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    _seed_config(tmp_path)
    rc = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha", yes=True, memgraph_uri="  "))
    assert rc == 2
    assert "empty value" in capsys.readouterr().err
    assert store.count(tenant="alpha") == 2  # nothing deleted


def test_tenant_rm_unknown_backend_aborts_before_sweep(monkeypatch, tmp_path, capsys):
    # An unknown MNEMOSTACK_KEYSTORE backend leaves key state unknown -> abort the
    # data sweep too (not just report keys failed).
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    _seed_config(tmp_path)
    monkeypatch.setenv("MNEMOSTACK_KEYSTORE", "flie")
    rc = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha", yes=True))
    assert rc == 1
    err = capsys.readouterr().err
    assert "could not be inspected" in err and "NOT removed" in err
    assert store.count(tenant="alpha") == 2  # sweep never ran


# ---------- review round 9 ----------


def test_tenant_rm_catches_key_issued_after_count(monkeypatch, tmp_path, capsys):
    # The concurrent-offboarding race: a key issued AFTER the count phase must
    # still be revoked (revoke_tenant re-reads under lock, not a snapshot).
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    kf = tmp_path / "keys.json"
    ks = FileKeyStore(kf)  # start with NO keys for alpha (count sees zero)

    class _RacyKeys:
        def __init__(self):
            self._store = ks

        def list_keys(self):
            return self._store.list_keys()

        def revoke_tenant(self, tenant, **kw):
            return self._store.revoke_tenant(tenant, **kw)

    monkeypatch.setattr(cli, "_keys_store", lambda _a: _RacyKeys())
    # revoke_tenant re-reads under lock, so a key present at sweep time is caught
    # regardless of the count-phase snapshot.
    ks.issue("alpha", ["write"])
    rc = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha", yes=True))
    assert rc == 0
    assert not any(k["tenant"] == "alpha" for k in ks.list_keys())  # caught & revoked


def test_tenant_rm_external_warns_about_verify_cache(monkeypatch, tmp_path, capsys):
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    _seed_config(tmp_path)
    monkeypatch.setenv("MNEMOSTACK_KEYSTORE", "openbao")
    rc = cli.cmd_tenant_rm(
        _rm_ns(tmp_path, tenant="alpha", yes=True, external_keys_revoked=True)
    )
    assert rc == 0
    assert "verify cache expires" in capsys.readouterr().err


# ---------- self-review batch (primary graph path, sweep-window, export mode) ----------


def test_tenant_rm_primary_path_uses_configured_graph(monkeypatch, tmp_path, capsys):
    # THE hole Codex missed: on a normal (config loads fine) deployment with a
    # graph configured, omitting --memgraph-uri must NOT print "fully removed"
    # while graph records survive — the flag now defaults to the configured URI.
    from mnemostack.config import Config, GraphConfig

    def _load_with_graph(*a, **k):
        cfg = Config()
        cfg.graph = GraphConfig(uri="bolt://configured:7687")
        return cfg

    monkeypatch.setattr(Config, "load", classmethod(lambda cls: _load_with_graph()))
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    _seed_config(tmp_path)
    called = {}

    class _G:
        def delete_tenant(self, tenant, *, dry_run=False):
            called["uri_used"] = True
            return {"nodes": 1, "relationships": 0, "detached": 0}

        def close(self):
            pass

    monkeypatch.setattr("mnemostack.graph.factory.make_graph_store", lambda *a, **k: _G())
    # no --memgraph-uri on the CLI, but the config has one -> it's swept
    rc = cli.main([
        "tenant-rm", "--tenant", "alpha", "--yes",
        "--qdrant", "http://localhost:6333", "--collection", "mt",
    ])
    assert rc == 0
    assert called.get("uri_used") is True  # the configured graph was actually swept


def test_tenant_rm_key_issued_during_sweep_is_caught(monkeypatch, tmp_path, capsys):
    # A key that appears mid-sweep (concurrent onboarding) is caught by the final
    # re-scan and reported as a partial removal, not a silent "fully removed".
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    ks, _fs = _seed_config(tmp_path)  # alpha has 2 keys at count time

    class _RacerKeys:
        def __init__(self):
            self.calls = 0

        def list_keys(self):
            return ks.list_keys()

        def revoke_tenant(self, tenant, **kw):
            self.calls += 1
            res = ks.revoke_tenant(tenant, **kw)
            if self.calls == 1:
                ks.issue("alpha", ["write"])  # a racer onboards mid-sweep
            return res

    racer = _RacerKeys()  # ONE instance so `calls` persists across the two revoke passes
    monkeypatch.setattr(cli, "_keys_store", lambda _a: racer)
    rc = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha", yes=True))
    assert rc == 1  # not "fully removed"
    err = capsys.readouterr().err
    assert "during the sweep" in err.lower() or "DURING" in err
    assert not any(k["tenant"] == "alpha" for k in ks.list_keys())  # racer key revoked


def test_tenant_export_preserves_existing_file_mode(monkeypatch, tmp_path):
    import os
    import stat

    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    out = tmp_path / "dump.jsonl"
    out.write_text("old\n")
    os.chmod(out, 0o644)
    assert cli.cmd_tenant_export(_ns(tenant="alpha", output=str(out), no_vectors=True)) == 0
    assert stat.S_IMODE(os.stat(out).st_mode) == 0o644  # not silently narrowed to 0600


# ---------- review round 12 ----------


def test_tenant_rm_honors_state_path_env(monkeypatch, tmp_path):
    # tenant-rm without --state-path must clean the SAME state file the servers
    # use (MNEMOSTACK_STATE_PATH), not the XDG default.
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    ks = FileKeyStore(tmp_path / "keys.json")
    ks.issue("alpha", ["read"])
    FileQuotaStore(tmp_path / "quotas.json").set("alpha", max_points=1)
    env_state = tmp_path / "env-state.json"
    fs = FileStateStore(env_state)
    fs.set(tenant_state_key("q_table", "alpha"), {"w": 1})
    monkeypatch.setenv("MNEMOSTACK_STATE_PATH", str(env_state))
    ns = _rm_ns(tmp_path, tenant="alpha", yes=True)
    del ns.state_path  # force the env/default resolution path
    rc = cli.cmd_tenant_rm(ns)
    assert rc == 0
    assert fs.get(tenant_state_key("q_table", "alpha")) is None  # env file was cleaned


def test_tenant_rm_removes_quota_created_after_count(monkeypatch, tmp_path):
    # A quota created for the tenant AFTER the count snapshot must still be
    # removed (remove() is idempotent and now called unconditionally).
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    ks = FileKeyStore(tmp_path / "keys.json")
    ks.issue("alpha", ["read"])
    # NO quota at count time
    qf = tmp_path / "quotas.json"

    real_remove = FileQuotaStore.remove

    class _RacyQuota(FileQuotaStore):
        def list_quotas(self):
            return []  # count sees nothing

        def remove(self, tenant):
            return real_remove(self, tenant)

    holder = _RacyQuota(qf)
    monkeypatch.setattr(cli, "_quota_store", lambda _a: holder)
    # a racer sets a quota during the sweep
    FileQuotaStore(qf).set("alpha", max_points=99)
    rc = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha", yes=True))
    assert rc == 0
    assert FileQuotaStore(qf).get("alpha") is None  # removed despite empty snapshot


def test_tenant_rm_late_admin_during_sweep_is_partial(monkeypatch, tmp_path, capsys):
    # A late-issued key that becomes the last usable admin (revoked==0,
    # last_admin_kept) must be a partial removal, not a silent success.
    store = _seeded_store()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    ks = FileKeyStore(tmp_path / "keys.json")
    ks.issue("alpha", ["write"])  # one non-admin key at count time
    FileQuotaStore(tmp_path / "quotas.json").set("alpha", max_points=1)

    class _LateAdmin:
        def __init__(self):
            self.calls = 0

        def list_keys(self):
            return ks.list_keys()

        def revoke_tenant(self, tenant, **kw):
            self.calls += 1
            res = ks.revoke_tenant(tenant, **kw)  # pass 1 drops the write key
            if self.calls == 1:
                ks.issue("alpha", ["admin"])  # a racer onboards an admin mid-sweep
            return res

    monkeypatch.setattr(cli, "_keys_store", lambda _a: _LateAdmin())
    rc = cli.cmd_tenant_rm(_rm_ns(tmp_path, tenant="alpha", yes=True))
    assert rc == 1  # not "fully removed"
    err = capsys.readouterr().err
    assert "last usable admin" in err and "DURING" in err
