"""Sliding-window chunks for memories written through the remote surfaces
(`ingest.remote_window_size`): parity with `Ingestor(window_size=N)`,
isolation, retries, out-of-order writes, lifecycle propagation, quota and
the `rebuild-windows` command.
"""

from __future__ import annotations

import argparse
import json
import random

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from test_remote_ingest import _CountingEmbedding, _ingest_app, _mem_store

from mnemostack.ingest import IngestItem, Ingestor, ingest_remote_items, stable_chunk_id
from mnemostack.remote_windows import (
    WINDOW_MEMBERS_KEY,
    WINDOWED_KEY,
    sync_source_windows,
    window_chunk_id,
)

_VOLATILE = ("indexed_at", WINDOW_MEMBERS_KEY, WINDOWED_KEY)


def _items(source="chat/a", n=5, **extra):
    return [
        IngestItem(
            text=f"{source} message {i}",
            source=source,
            offset=i,
            timestamp=f"2026-01-0{i + 1}T00:00:00+00:00",
            tags=[f"t{i}"],
            metadata={"speaker": "user" if i % 2 else "assistant", **extra},
        )
        for i in range(n)
    ]


def _records(store, tenant=None):
    """Every retrievable record, minus the write stamp and the bookkeeping
    keys that only the remote path adds."""
    kw = {"tenant": tenant} if tenant is not None else {}
    records = {}
    for hit in store.scroll(**kw):
        payload = {k: v for k, v in hit.payload.items() if k not in _VOLATILE}
        # Memories keep their ids on both paths. Window chunks are compared
        # by position: API-built windows have their own id space.
        key = str(hit.id)
        if payload.get("chunk_kind") == "sliding_window":
            key = f"window:{payload['source']}:{payload['chunk_start_offset']}:{payload['chunk_window']}"
        assert key not in records
        records[key] = payload
    return records


def _windows(store, source=None):
    return [
        hit
        for hit in store.scroll()
        if hit.payload.get("chunk_kind") == "sliding_window"
        and (source is None or hit.payload["source"] == source)
    ]


def _write(store, emb, items, size=3, **kw):
    return ingest_remote_items(emb, store, items, window_size=size, **kw)


# ------------------------------------------------------------------ parity


@pytest.mark.parametrize("size", [2, 3])
@pytest.mark.parametrize("tenant", [None, "acme"])
@pytest.mark.parametrize("batching", ["one", "each", "pairs"])
def test_remote_writes_store_the_records_the_ingestor_stores(size, tenant, batching):
    """Complete records: text, member order, timestamps, metadata, tags and
    visibility keys must match the measured path. Memory ids match too;
    window ids are compared separately."""
    reference = _mem_store("ref")
    Ingestor(_CountingEmbedding(), reference, window_size=size, tenant=tenant).ingest(_items())
    store, emb = _mem_store("api"), _CountingEmbedding()
    step = {"one": 5, "each": 1, "pairs": 2}[batching]
    for start in range(0, 5, step):
        _write(store, emb, _items()[start : start + step], size, tenant=tenant)
    assert _records(store, tenant) == _records(reference, tenant)
    for hit in _windows(store):
        assert len(hit.payload[WINDOW_MEMBERS_KEY]) == size
        args = (hit.payload["source"], hit.payload["offset"], hit.payload["text"])
        assert str(hit.id) == window_chunk_id(*args, tenant=tenant)
        assert str(hit.id) != stable_chunk_id(*args, tenant=tenant)
        assert hit.payload["text"].split("\n") == [
            f"chat/a message {i}"
            for i in range(hit.payload["chunk_start_offset"], hit.payload["chunk_end_offset"] + 1)
        ]


def test_custom_schema_keys_mirror_the_window_text():
    store, emb = _mem_store(), _CountingEmbedding()
    _write(store, emb, _items(n=3), text_key="body", timestamp_key="ts", timestamp_format="epoch")
    (window,) = _windows(store)
    assert window.payload["body"] == window.payload["text"]
    assert window.payload["text"].count("\n") == 2
    assert isinstance(window.payload["ts"], float)


@pytest.mark.parametrize("size", [0, 2.5, True])
def test_invalid_size_is_refused_before_any_write(size):
    store, emb = _mem_store(), _CountingEmbedding()
    with pytest.raises(ValueError, match="window_size"):
        ingest_remote_items(emb, store, _items(n=3), window_size=size)
    assert store.count() == 0 and emb.embedded == []


def test_off_by_default_writes_nothing_extra():
    store, emb = _mem_store(), _CountingEmbedding()
    results = ingest_remote_items(emb, store, _items())
    assert results.windows is None
    assert store.count() == 5 and len(emb.embedded) == 5
    assert all(WINDOWED_KEY not in hit.payload for hit in store.scroll())


def test_out_of_order_writes_converge_to_the_in_order_records():
    reference = _mem_store("ref")
    Ingestor(_CountingEmbedding(), reference, window_size=3).ingest(_items(n=7))
    for seed in range(5):
        store, emb = _mem_store(f"api{seed}"), _CountingEmbedding()
        shuffled = _items(n=7)
        random.Random(seed).shuffle(shuffled)
        for item in shuffled:
            _write(store, emb, [item])
        assert _records(store) == _records(reference)


# --------------------------------------------------------------- isolation


def test_windows_never_cross_a_source_or_a_tenant():
    store, emb = _mem_store(), _CountingEmbedding()
    a, b = _items("chat/a", 3), _items("chat/b", 3)
    for pair in zip(a, b, strict=True):
        _write(store, emb, list(pair), tenant="acme")
    _write(store, emb, _items("chat/a", 3), tenant="beta")
    for hit in _windows(store):
        tenant, source = hit.payload["tenant_id"], hit.payload["source"]
        members = store.retrieve_payload_fields(
            hit.payload[WINDOW_MEMBERS_KEY], ["source", "tenant_id", "text"]
        )
        assert len(members) == 3
        assert {m["source"] for m in members.values()} == {source}
        assert {m["tenant_id"] for m in members.values()} == {tenant}
        assert all(text.startswith(source) for text in hit.payload["text"].split("\n"))
    assert len(_windows(store)) == 3  # one per (tenant, source)


def test_unscoped_writes_do_not_join_a_tenants_memories():
    store, emb = _mem_store(), _CountingEmbedding()
    _write(store, emb, _items(n=2), tenant="acme")
    _write(store, emb, [_items(n=3)[2]])
    assert _windows(store) == []


def test_memories_without_a_source_get_no_window():
    store, emb = _mem_store(), _CountingEmbedding()
    results = _write(store, emb, [IngestItem(text=f"note {i}", offset=i) for i in range(4)])
    assert results.windows.created == 0 and _windows(store) == []


# ----------------------------------------------------------------- retries


def test_repeated_request_adds_nothing_and_embeds_nothing():
    store, emb = _mem_store(), _CountingEmbedding()
    _write(store, emb, _items())
    count, embedded = store.count(), len(emb.embedded)
    again = _write(store, emb, _items())
    assert store.count() == count and len(emb.embedded) == embedded
    assert again.windows.created == 0 and again.windows.complete


class _WindowFailingEmbedding(_CountingEmbedding):
    """Fails every joined (multi-line) text while ``broken`` is set."""

    broken = True

    def embed(self, text: str) -> list[float]:
        if self.broken and "\n" in text:
            self.embedded.append(text)
            return []
        return super().embed(text)


def test_failed_window_step_is_reported_and_a_repeat_repairs_it():
    store, emb = _mem_store(), _WindowFailingEmbedding()
    first = _write(store, emb, _items(n=3))
    assert [r.status for r in first] == ["stored"] * 3
    assert first.windows.complete is False and first.windows.reason == "embedding"
    assert _windows(store) == []
    emb.broken = False
    second = _write(store, emb, _items(n=3))
    assert [r.status for r in second] == ["duplicate"] * 3
    assert second.windows.created == 1 and second.windows.complete


def test_store_failure_in_the_window_step_does_not_fail_the_memories(monkeypatch):
    import mnemostack.remote_windows as rw

    store, emb = _mem_store(), _CountingEmbedding()

    def boom(*_a, **_k):
        raise RuntimeError("store down")

    monkeypatch.setattr(rw, "sync_source_windows", boom)
    results = _write(store, emb, _items(n=3))
    assert [r.status for r in results] == ["stored"] * 3
    assert results.windows.complete is False and results.windows.reason == "error"


# ------------------------------------------------------ ambiguous positions


def test_reused_offset_removes_the_old_windows_and_is_reported():
    store, emb = _mem_store(), _CountingEmbedding()
    _write(store, emb, _items())
    assert len(_windows(store)) == 3
    edited = IngestItem(text="rewritten message 2", source="chat/a", offset=2)
    result = _write(store, emb, [edited])
    assert result.windows.ambiguous_offsets == 1
    # Every window of size 3 over offsets 0..4 holds offset 2.
    assert _windows(store) == [] and result.windows.removed == 3
    # Only the affected windows go: a longer source keeps the others.
    _write(store, emb, [IngestItem(text=f"late {i}", source="chat/a", offset=i) for i in (5, 6, 7)])
    assert sorted(h.payload["chunk_start_offset"] for h in _windows(store)) == [3, 4, 5]
    for hit in _windows(store):
        assert "rewritten" not in hit.payload["text"] and "message 2" not in hit.payload["text"]


def test_retracting_the_old_text_makes_the_offset_usable_again():
    store, emb = _mem_store(), _CountingEmbedding()
    first = _write(store, emb, _items(n=3))
    _write(store, emb, [IngestItem(text="rewritten 1", source="chat/a", offset=1)])
    store.invalidate([first[1].id])
    result = sync_source_windows(
        store, "chat/a", tenant=None, window_size=3, mode="rebuild", embedding=emb
    )
    assert result.ambiguous_offsets == 0 and result.created == 1
    (window,) = _windows(store)
    assert window.payload["text"].split("\n")[1] == "rewritten 1"


def test_memory_with_a_validity_bound_stays_out_of_windows_and_is_not_bridged():
    store, emb = _mem_store(), _CountingEmbedding()
    items = _items(n=5)
    items[2].metadata["valid_until"] = "2026-06-01T00:00:00+00:00"
    _write(store, emb, items, size=2)
    spans = sorted(
        (h.payload["chunk_start_offset"], h.payload["chunk_end_offset"]) for h in _windows(store)
    )
    assert spans == [(0, 1), (3, 4)]


# --------------------------------------------------------------- lifecycle


def _app(monkeypatch, tmp_path, size=3):
    app, store, emb, keys = _ingest_app(
        monkeypatch, tmp_path, cfg_extra={"remote_window_size": size}
    )
    return TestClient(app), store, emb, {"X-API-Key": keys["write"]}


def _post(client, headers, items):
    body = [
        {"text": i.text, "source": i.source, "offset": i.offset, "timestamp": i.timestamp}
        for i in items
    ]
    response = client.post("/memories", headers=headers, json={"items": body})
    assert response.status_code == 200, response.text
    return response.json()


def _texts(store):
    return "\n".join(
        h.payload["text"] for h in store.scroll() if not h.payload.get("invalidated_at")
    )


def test_http_write_reports_the_window_step(monkeypatch, tmp_path):
    client, store, _emb, headers = _app(monkeypatch, tmp_path)
    out = _post(client, headers, _items())
    assert out["windows"] == {
        "created": 3, "removed": 0, "ambiguous_offsets": 0, "skipped_sources": 0,
        "blocked_by_secret_policy": 0, "complete": True, "reason": None,
    }
    assert len(_windows(store)) == 3


def test_http_write_reports_a_window_blocked_by_the_secret_policy(monkeypatch, tmp_path):
    app, store, _emb, keys = _ingest_app(
        monkeypatch, tmp_path, cfg_extra={"remote_window_size": 2, "secret_policy": "block"}
    )
    body = [{"text": i.text, "source": i.source, "offset": i.offset} for i in _split_key()]
    response = TestClient(app).post(
        "/memories", headers={"X-API-Key": keys["write"]}, json={"items": body}
    )
    assert response.status_code == 200 and response.json()["stored"] == 2
    windows = response.json()["windows"]
    assert (windows["blocked_by_secret_policy"], windows["created"], windows["complete"]) == (1, 0, True)
    assert _windows(store) == [] and "PRIVATE KEY" not in response.text


def test_http_response_has_null_windows_when_off(monkeypatch, tmp_path):
    client, store, _emb, headers = _app(monkeypatch, tmp_path, size=1)
    assert _post(client, headers, _items())["windows"] is None
    assert store.count() == 5


@pytest.mark.parametrize("method", ["invalidate", "delete"])
def test_by_id_retraction_removes_the_windows_that_hold_the_memory(monkeypatch, tmp_path, method):
    client, store, _emb, headers = _app(monkeypatch, tmp_path)
    out = _post(client, headers, _items())
    victim = out["results"][2]["id"]
    if method == "invalidate":
        response = client.post("/invalidate", headers=headers, json={"ids": [victim]})
    else:
        response = client.request("DELETE", "/memories", headers=headers, json={"ids": [victim]})
    assert response.status_code == 200 and response.json()["requested"] == 1
    assert "message 2" not in _texts(store)
    assert _windows(store) == []  # all three windows of 0..4 held offset 2


def test_retraction_keeps_the_windows_that_do_not_hold_the_memory(monkeypatch, tmp_path):
    client, store, _emb, headers = _app(monkeypatch, tmp_path, size=2)
    out = _post(client, headers, _items())
    client.post("/invalidate", headers=headers, json={"ids": [out["results"][0]["id"]]})
    spans = sorted(h.payload["chunk_start_offset"] for h in _windows(store))
    assert spans == [1, 2, 3]


def test_retraction_propagates_after_the_setting_is_turned_off(monkeypatch, tmp_path):
    client, store, emb, headers = _app(monkeypatch, tmp_path, size=1)
    written = ingest_remote_items(emb, store, _items(n=3), tenant="alpha", window_size=3)
    assert len(_windows(store)) == 1
    client.post("/invalidate", headers=headers, json={"ids": [written[1].id]})
    assert _windows(store) == []


def test_source_scoped_retraction_and_deletion_cover_the_windows(monkeypatch, tmp_path):
    client, store, _emb, headers = _app(monkeypatch, tmp_path)
    _post(client, headers, _items())
    r = client.post("/invalidate", headers=headers, json={"source": "chat/a"})
    # The windows are deleted, not retracted: recall of retracted memories
    # (include_invalidated / as_of) must not return them either.
    assert r.json() == {"requested": 5, "invalidated": 5, "complete": True}
    assert _windows(store) == [] and _texts(store) == "" and store.count() == 5
    _post(client, headers, _items("chat/b"))
    r = client.request("DELETE", "/memories", headers=headers, json={"source": "chat/b"})
    assert r.json()["deleted"] == 8 and store.count() == 5


def test_source_scoped_calls_in_batches_leave_no_window_over_retracted_text(monkeypatch, tmp_path):
    import mnemostack.server as srv

    monkeypatch.setattr(srv, "REMOTE_SOURCE_BATCH", 2)
    for call in ("invalidate", "delete"):
        client, store, _emb, headers = _app(monkeypatch, tmp_path)
        _post(client, headers, _items())
        for _ in range(10):
            if call == "invalidate":
                r = client.post("/invalidate", headers=headers, json={"source": "chat/a"})
            else:
                r = client.request("DELETE", "/memories", headers=headers, json={"source": "chat/a"})
            live = [h.payload["text"] for h in store.scroll() if not h.payload.get("invalidated_at")]
            singles = {t for t in live if "\n" not in t}
            # After every batch: each live window holds live memories only.
            assert all(set(t.split("\n")) <= singles for t in live)
            if r.json()["complete"]:
                break
        assert r.json()["complete"] and _windows(store) == []


def test_write_during_a_source_scoped_call_leaves_no_window_over_its_text(monkeypatch, tmp_path):
    """A write that lands between the selection and the effect of a
    source-scoped call (another process: no shared lock)."""
    import mnemostack.server as srv

    client, store, emb, headers = _app(monkeypatch, tmp_path)
    _post(client, headers, _items(n=2))
    real = srv.find_source_points

    def find_then_write(*a, **k):
        found = real(*a, **k)
        third = Ingestor(emb, store, tenant="alpha").ingest([_items(n=3)[2]])
        sync_source_windows(
            store, "chat/a", tenant="alpha", window_size=3, mode="write",
            embedding=emb, focus_ids=third.ids,
        )
        return found

    monkeypatch.setattr(srv, "find_source_points", find_then_write)
    r = client.post("/invalidate", headers=headers, json={"source": "chat/a"})
    assert r.json()["invalidated"] == 2 and _windows(store) == []


def test_by_id_calls_survive_an_unhashable_legacy_tenant_value():
    from mnemostack.remote_windows import member_retraction

    store = _mem_store()
    store.upsert("11111111-1111-1111-1111-111111111111", [0.1, 0.2, 0.3], {"text": "old", "source": "s"})
    store.client.set_payload("mem", payload={"tenant_id": ["a", "b"]}, points=["11111111-1111-1111-1111-111111111111"])
    with member_retraction(store, ["11111111-1111-1111-1111-111111111111"], tenant=None):
        assert store.invalidate(["11111111-1111-1111-1111-111111111111"]) == 1


def test_a_window_never_shares_an_id_with_a_memory():
    """A memory with the source, the middle offset and the joined text of a
    window is an independent record: neither write touches the other."""
    store, emb = _mem_store(), _CountingEmbedding()
    _write(store, emb, _items("chat/a", 3))
    (window,) = _windows(store)
    joined = "\n".join(i.text for i in _items("chat/a", 3))
    assert str(window.id) != stable_chunk_id("chat/a", 1, joined)
    # The same text as a memory of another source position set: stored, and
    # the window keeps its record.
    before = dict(window.payload)
    twin = IngestItem(text=joined, source="chat/a", offset=1, metadata={"kind": "summary"})
    result = ingest_remote_items(emb, store, [twin])  # setting off: no window pass
    assert result[0].status == "stored" and result[0].id == stable_chunk_id("chat/a", 1, joined)
    assert store.retrieve_payload(result[0].id)["kind"] == "summary"
    assert store.retrieve_payload(str(window.id)) == before
    # And the reverse order: a memory first, the window after it.
    other, emb2 = _mem_store("other"), _CountingEmbedding()
    first = ingest_remote_items(emb2, other, [IngestItem(text=joined, source="chat/a", offset=1)])
    other.invalidate([first[0].id])  # retracted, so offset 1 has one live memory
    _write(other, emb2, _items("chat/a", 3))
    assert other.retrieve_payload(first[0].id)["invalidated_at"] and len(_windows(other)) == 1
    # A tenant name cannot steer a memory onto the id of an unscoped window.
    for tenant in ("mnemostack-window", "window", ""):
        assert stable_chunk_id("chat/a", 1, joined, tenant=tenant) != str(window.id)


def test_refused_or_failed_request_leaves_memories_and_windows_unchanged():
    from mnemostack.quotas import QuotaExceededError

    store, emb = _mem_store(), _CountingEmbedding()
    _write(store, emb, _items(n=3), tenant="acme", max_points=4)
    before = _records(store, "acme")
    assert len(before) == 4
    late = [IngestItem(text=f"late {i}", source="chat/a", offset=i) for i in (3, 4)]
    with pytest.raises(QuotaExceededError):
        _write(store, emb, late, tenant="acme", max_points=4)
    assert _records(store, "acme") == before
    poisoned = [IngestItem(text="bad |POISON|", source="chat/a", offset=3)]
    result = _write(store, emb, poisoned, tenant="acme", max_points=10)
    assert [r.status for r in result] == ["failed"]
    assert _records(store, "acme") == before and result.windows.created == 0


def test_complete_is_false_whenever_a_wanted_window_is_missing():
    """Every way a wanted window can stay unwritten sets `complete: false`."""
    for fault, reason in (("embedding", "embedding"), ("quota", "quota")):
        store = _mem_store(fault)
        emb = _WindowFailingEmbedding() if fault == "embedding" else _CountingEmbedding()
        kw = {"tenant": "acme", "max_points": 3} if fault == "quota" else {}
        result = _write(store, emb, _items(n=3), **kw)
        assert [r.status for r in result] == ["stored"] * 3
        assert (result.windows.complete, result.windows.reason) == (False, reason)
        assert result.windows.created == 0 and _windows(store) == []


def test_secret_policy_checks_the_joined_text_of_a_window():
    import base64

    label = "RSA " + "PRIVATE KEY"
    body = base64.b64encode(bytes(range(128))).decode()
    halves = [
        IngestItem(text=f"-----BEGIN {label}-----\n{body}", source="chat/a", offset=0),
        IngestItem(text=f"-----END {label}-----", source="chat/a", offset=1),
    ]
    store, emb = _mem_store(), _CountingEmbedding()
    result = _write(store, emb, halves, size=2, secret_policy="block")
    assert [r.status for r in result] == ["stored", "stored"]
    assert result.windows.policy_blocked == 1 and result.windows.created == 0
    # A blocked window is not a window that should exist.
    assert result.windows.complete and result.windows.reason is None
    assert _windows(store) == [] and all("-----END" not in t or "BEGIN" not in t for t in emb.embedded)
    # Without the policy the same messages get their window.
    plain_store, plain_emb = _mem_store("plain"), _CountingEmbedding()
    assert _write(plain_store, plain_emb, halves, size=2).windows.created == 1


def _split_key():
    import base64

    label = "RSA " + "PRIVATE KEY"
    body = base64.b64encode(bytes(range(128))).decode()
    return [
        IngestItem(text=f"-----BEGIN {label}-----\n{body}", source="chat/key", offset=0),
        IngestItem(text=f"-----END {label}-----", source="chat/key", offset=1),
    ]


def test_rebuild_applies_the_current_policy_to_stored_windows(cli_store, capsys):
    """A window built while the policy was off: the dry run reports it, the
    rebuild removes it, neither prints its text, and it is not built again."""
    cli, store, emb = cli_store
    ingest_remote_items(emb, store, _split_key() + _items(n=2), window_size=2)
    assert len(_windows(store)) == 2
    before = _records(store)
    args = {"window_size": 2, "secret_policy": "block"}
    assert cli.cmd_rebuild_windows(_cli_args(dry_run=True, **args)) == 0
    out = capsys.readouterr()
    dry = json.loads(out.out)
    assert (dry["blocked_by_secret_policy"], dry["removed"], dry["created"], dry["kept"]) == (1, 1, 0, 1)
    assert _records(store) == before and "PRIVATE KEY" not in out.out + out.err
    assert cli.cmd_rebuild_windows(_cli_args(**args)) == 0
    out = capsys.readouterr()
    real = json.loads(out.out)
    assert (real["blocked_by_secret_policy"], real["removed"], real["created"]) == (1, 1, 0)
    assert [h.payload["source"] for h in _windows(store)] == ["chat/a"]
    assert "PRIVATE KEY" not in out.out + out.err
    # Without the policy nothing is removed: the cleanup belongs to the policy.
    again = _mem_store("again")
    ingest_remote_items(emb, again, _split_key(), window_size=2)
    assert sync_source_windows(
        again, "chat/key", tenant=None, window_size=2, mode="rebuild", embedding=emb
    ).removed == 0


def test_dry_run_counts_what_the_policy_would_block(cli_store, capsys):
    cli, store, emb = cli_store
    ingest_remote_items(emb, store, _split_key() + _items(n=2))
    assert cli.cmd_rebuild_windows(
        _cli_args(dry_run=True, window_size=2, secret_policy="block")
    ) == 0
    dry = json.loads(capsys.readouterr().out)
    assert (dry["created"], dry["blocked_by_secret_policy"]) == (1, 1)
    assert _windows(store) == []
    assert not any("BEGIN" in t and "-----END" in t for t in emb.embedded)


def test_failing_policy_check_is_reported_not_passed(monkeypatch):
    from mnemostack.ingest_policy import SecretPolicy

    store, emb = _mem_store(), _CountingEmbedding()

    real = SecretPolicy.check_item

    def broken(self, item):
        if WINDOW_MEMBERS_KEY in item.metadata:  # the check of a joined text
            raise RuntimeError("policy engine down")
        return real(self, item)

    monkeypatch.setattr(SecretPolicy, "check_item", broken)
    result = ingest_remote_items(emb, store, _items(n=3), window_size=3)
    assert [r.status for r in result] == ["stored"] * 3
    assert (result.windows.complete, result.windows.reason) == (False, "error")
    assert _windows(store) == []


def test_retraction_covers_an_owner_found_only_under_the_locks(monkeypatch):
    """An unscoped retraction whose first read ran before a tenant's write."""
    import mnemostack.remote_windows as rw

    store, emb = _mem_store(), _CountingEmbedding()
    written = _write(store, emb, _items(), tenant="acme")
    real, calls = rw._member_scopes, []

    def late(*a, **k):
        calls.append(1)
        return {} if len(calls) == 1 else real(*a, **k)

    monkeypatch.setattr(rw, "_member_scopes", late)
    with rw.member_retraction(store, [written[2].id], tenant=None):
        store.invalidate([written[2].id])
    assert _windows(store) == [] and len(calls) >= 3


def test_re_remembering_a_retracted_memory_brings_its_windows_back(monkeypatch, tmp_path):
    client, store, _emb, headers = _app(monkeypatch, tmp_path)
    out = _post(client, headers, _items(n=3))
    client.post("/invalidate", headers=headers, json={"ids": [out["results"][1]["id"]]})
    assert _windows(store) == []
    again = _post(client, headers, [_items(n=3)[1]])
    assert again["results"][0]["status"] == "stored" and again["windows"]["created"] == 1
    (window,) = _windows(store)
    assert "message 1" in window.payload["text"]


def test_rebuild_never_resurrects_retracted_or_deleted_text(monkeypatch, tmp_path):
    client, store, emb, headers = _app(monkeypatch, tmp_path)
    out = _post(client, headers, _items(n=7))
    client.post("/invalidate", headers=headers, json={"ids": [out["results"][1]["id"]]})
    client.request("DELETE", "/memories", headers=headers, json={"ids": [out["results"][5]["id"]]})
    tenant = next(iter(store.scroll())).payload["tenant_id"]
    result = sync_source_windows(
        store, "chat/a", tenant=tenant, window_size=3, mode="rebuild", embedding=emb
    )
    assert result.complete
    for hit in _windows(store):
        assert "message 1" not in hit.payload["text"] and "message 5" not in hit.payload["text"]
        assert not hit.payload.get("invalidated_at")
    # The remaining live memories 0,2,3,4,6 are neighbours by offset.
    assert len(_windows(store)) == 3


def test_stale_window_from_an_interleaved_writer_is_removed_by_the_next_call():
    """Another process stored a window over a member that was retracted in
    between: the next write or retraction in the source removes it."""
    store, emb = _mem_store(), _CountingEmbedding()
    results = _write(store, emb, _items(n=3))
    store.invalidate([results[0].id])  # a retraction that skipped the window pass
    assert len(_windows(store)) == 1
    _write(store, emb, [IngestItem(text="next", source="chat/a", offset=9)])
    assert all("message 0" not in h.payload["text"] for h in _windows(store))


def test_mcp_remember_and_invalidate_follow_the_same_rules(monkeypatch):
    pytest.importorskip("fastmcp")
    import asyncio

    import mnemostack.mcp.server as mcp_srv

    store, emb = _mem_store(), _CountingEmbedding()
    monkeypatch.setattr(mcp_srv, "VectorStore", lambda **_: store)
    monkeypatch.setattr(mcp_srv, "get_provider", lambda _n, **_k: emb)
    server = mcp_srv.build_server(collection="mem", remote_window_size=3)

    def call(name, **arguments):
        result = asyncio.run(server.call_tool(name, arguments))
        return getattr(result, "structured_content", None) or json.loads(result.content[0].text)

    outs = [
        call("mnemostack_remember", text=i.text, source=i.source, offset=i.offset)
        for i in _items(n=3)
    ]
    assert outs[-1]["windows"]["created"] == 1 and len(_windows(store)) == 1
    reference = _mem_store("ref")
    Ingestor(_CountingEmbedding(), reference, window_size=3).ingest(
        [IngestItem(text=i.text, source=i.source, offset=i.offset) for i in _items(n=3)]
    )
    assert _records(store) == _records(reference)
    call("mnemostack_invalidate", ids=[outs[1]["results"][0]["id"]])
    assert _windows(store) == []


# -------------------------------------------------------------------- quota


def test_quota_counts_windows_and_keeps_the_memories_when_they_do_not_fit():
    store, emb = _mem_store(), _CountingEmbedding()
    results = _write(store, emb, _items(n=4), tenant="acme", max_points=5)
    assert [r.status for r in results] == ["stored"] * 4
    assert results.windows.complete is False and results.windows.reason == "quota"
    assert results.windows.created == 0 and store.count(tenant="acme") == 4
    # With room, a repeat of the request writes the windows.
    again = _write(store, emb, _items(n=4), tenant="acme", max_points=6)
    assert again.windows.created == 2 and store.count(tenant="acme") == 6


# ------------------------------------------------------------------ rebuild


def _cli_args(**kw):
    base = {
        "collection": "mem", "qdrant": None, "provider": "fake", "tenant": None,
        "source": None, "window_size": 3, "dry_run": False, "json": True, "quotas_file": None,
        "secret_policy": "off",
    }
    args = {**base, **kw}
    args.setdefault("all_sources", not args["source"])
    return argparse.Namespace(**args)


def test_rebuild_requires_an_explicit_scope(cli_store, capsys):
    cli, store, emb = cli_store
    ingest_remote_items(emb, store, _items(n=3))
    assert cli.cmd_rebuild_windows(_cli_args(all_sources=False)) == 2
    assert cli.cmd_rebuild_windows(_cli_args(source=["chat/a"], all_sources=True)) == 2
    assert "--all-sources" in capsys.readouterr().err and _windows(store) == []
    parsed = cli.build_parser().parse_args(["rebuild-windows", "--all-sources", "--dry-run"])
    assert parsed.all_sources and parsed.source is None


@pytest.fixture
def cli_store(monkeypatch):
    import mnemostack.cli as cli

    store, emb = _mem_store(), _CountingEmbedding()
    monkeypatch.setattr(cli, "VectorStore", lambda **_: store)
    monkeypatch.setattr(cli, "get_provider", lambda _n, **_k: emb)
    monkeypatch.setattr(cli, "_provider_kwargs_from_args", lambda _a: {})
    return cli, store, emb


def test_rebuild_backfills_existing_memories_and_is_idempotent(cli_store, capsys):
    cli, store, emb = cli_store
    ingest_remote_items(emb, store, _items("chat/a") + _items("chat/b", 2))
    ingest_remote_items(emb, store, [IngestItem(text="loose note")])
    assert cli.cmd_rebuild_windows(_cli_args(dry_run=True)) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["created"] == 3 and dry["skipped_without_source"] == 1 and _windows(store) == []
    assert cli.cmd_rebuild_windows(_cli_args()) == 0
    assert json.loads(capsys.readouterr().out)["created"] == 3
    reference = _mem_store("ref")
    Ingestor(_CountingEmbedding(), reference, window_size=3).ingest(
        _items("chat/a") + _items("chat/b", 2) + [IngestItem(text="loose note")]
    )
    assert _records(store) == _records(reference)
    embedded = len(emb.embedded)
    assert cli.cmd_rebuild_windows(_cli_args()) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["created"] == 0 and second["removed"] == 0 and second["kept"] == 3
    assert len(emb.embedded) == embedded


def test_rebuild_is_scoped_to_the_tenant_and_the_source(cli_store, capsys):
    cli, store, emb = cli_store
    for tenant in ("acme", "beta"):
        ingest_remote_items(emb, store, _items("chat/a") + _items("chat/b"), tenant=tenant)
    assert cli.cmd_rebuild_windows(_cli_args(tenant="acme", source=["chat/a"])) == 0
    assert json.loads(capsys.readouterr().out)["created"] == 3
    assert {(h.payload["tenant_id"], h.payload["source"]) for h in _windows(store)} == {
        ("acme", "chat/a")
    }


def test_rebuild_changes_the_size_and_size_one_removes_the_windows(cli_store, capsys):
    cli, store, emb = cli_store
    ingest_remote_items(emb, store, _items(), window_size=3)
    assert cli.cmd_rebuild_windows(_cli_args(window_size=2)) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["removed"], out["created"]) == (3, 4)
    assert {h.payload["chunk_window"] for h in _windows(store)} == {2}
    assert cli.cmd_rebuild_windows(_cli_args(window_size=1)) == 0
    assert _windows(store) == [] and store.count() == 5


def test_rebuild_leaves_operator_built_windows_alone(cli_store, capsys):
    cli, store, emb = cli_store
    Ingestor(emb, store, window_size=3).ingest(_items("docs/file.md"))
    before = _records(store)
    assert cli.cmd_rebuild_windows(_cli_args(window_size=1)) == 0
    assert _records(store) == before
    # Same size: building would write over them (same members, same ids).
    embedded = len(emb.embedded)
    assert cli.cmd_rebuild_windows(_cli_args(window_size=3)) == 0
    out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert out["created"] == 0 and out["skipped_sources_with_other_windows"] == 1
    assert _records(store) == before and len(emb.embedded) == embedded
    # An API write into that source builds nothing there either.
    late = IngestItem(text="late", source="docs/file.md", offset=9)
    step = ingest_remote_items(emb, store, [late], window_size=3).windows
    assert step.created == 0 and step.foreign_sources == 1 and step.complete
    assert all(WINDOW_MEMBERS_KEY not in h.payload for h in _windows(store))


def test_windows_of_another_writer_replace_the_ones_built_here(cli_store, capsys):
    """The API built windows first, then `Ingestor` wrote its own over the
    same conversation: the two sets must not stay side by side."""
    cli, store, emb = cli_store
    ingest_remote_items(emb, store, _items(), window_size=3)
    Ingestor(emb, store, window_size=3).ingest(_items())
    assert len(_windows(store)) == 6
    assert cli.cmd_rebuild_windows(_cli_args(dry_run=True)) == 0
    assert json.loads(capsys.readouterr().out)["removed"] == 3 and len(_windows(store)) == 6
    assert cli.cmd_rebuild_windows(_cli_args()) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["removed"], out["created"], out["kept"]) == (3, 0, 0)
    assert len(_windows(store)) == 3
    assert all(WINDOW_MEMBERS_KEY not in h.payload for h in _windows(store))
    # A write does the same, without a rebuild.
    ingest_remote_items(emb, store, _items("chat/z"), window_size=3)
    Ingestor(emb, store, window_size=3).ingest(_items("chat/z"))
    late = IngestItem(text="late", source="chat/z", offset=9)
    assert ingest_remote_items(emb, store, [late], window_size=3).windows.removed == 3
    assert len(_windows(store, "chat/z")) == 3


def test_rebuild_visits_a_source_whose_memories_are_all_gone(cli_store, capsys):
    cli, store, emb = cli_store
    written = ingest_remote_items(emb, store, _items(n=3), window_size=3)
    store.delete_points([r.id for r in written])  # a deletion that skipped the window pass
    assert len(_windows(store)) == 1
    assert cli.cmd_rebuild_windows(_cli_args()) == 0
    assert json.loads(capsys.readouterr().out)["removed"] == 1 and store.count() == 0


def test_cli_invalidate_removes_the_windows_of_a_tenants_memory(cli_store, capsys):
    """The CLI is unscoped: the windows to remove belong to the tenant that
    owns the memory, not to the caller."""
    cli, store, emb = cli_store
    written = ingest_remote_items(emb, store, _items(), tenant="acme", window_size=3)
    args = argparse.Namespace(
        ids=[written[2].id], collection="mem", qdrant=None, invalidated_at=None,
        valid_until=None, index_root=None, json=True,
    )
    assert cli.cmd_invalidate(args) == 0
    assert json.loads(capsys.readouterr().out)["invalidated"] == 1
    assert _windows(store) == []


def test_unscoped_deletion_removes_the_windows_of_a_tenants_memory():
    from mnemostack.remote_windows import member_retraction

    store, emb = _mem_store(), _CountingEmbedding()
    written = _write(store, emb, _items(), tenant="acme")
    with member_retraction(store, [written[2].id], tenant=None):
        store.delete_points([written[2].id])
    assert _windows(store) == []


def test_window_stored_by_another_process_during_a_retraction_is_removed():
    """The member flag is read before the other writer sets it. With the
    setting on, the pass after the retraction does not depend on the flag."""
    from mnemostack.remote_windows import member_retraction

    store, emb = _mem_store(), _CountingEmbedding()
    written = _write(store, emb, _items(n=2))  # no window yet, no flag
    third = Ingestor(emb, store).ingest([_items(n=3)[2]])
    with member_retraction(store, [written[1].id], tenant=None, active=True):
        # The other process: it holds no lock of this process.
        sync_source_windows(
            store, "chat/a", tenant=None, window_size=3, mode="write",
            embedding=emb, focus_ids=third.ids,
        )
        assert len(_windows(store)) == 1
        store.invalidate([written[1].id])
    assert _windows(store) == []


def test_failed_check_after_a_retraction_does_not_fail_the_call(monkeypatch):
    import mnemostack.remote_windows as rw

    store, emb = _mem_store(), _CountingEmbedding()
    written = _write(store, emb, _items(n=3))

    def boom(*_a, **_k):
        raise RuntimeError("store down")

    monkeypatch.setattr(rw, "sync_source_windows", boom)
    with rw.member_retraction(store, [written[1].id], tenant=None, active=True):
        store.invalidate([written[1].id])
    assert _windows(store) == []  # removed before the retraction


def test_rebuild_reports_ambiguous_offsets(cli_store, capsys):
    cli, store, emb = cli_store
    ingest_remote_items(emb, store, _items(n=3))
    ingest_remote_items(emb, store, [IngestItem(text="other 1", source="chat/a", offset=1)])
    assert cli.cmd_rebuild_windows(_cli_args()) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["skipped_ambiguous_offsets"] == 1 and out["created"] == 0


# ------------------------------------------------------------------ resolve


def test_resolve_does_not_verify_an_api_built_window(tmp_path):
    from mnemostack.provenance import resolve_citation

    store, emb = _mem_store(), _CountingEmbedding()
    _write(store, emb, _items(n=3))
    (window,) = _windows(store)
    resolution = resolve_citation(store, str(window.id), root=tmp_path)
    assert resolution.verdict in ("unresolvable", "missing")


# ------------------------------------------------------------------- config


def test_setting_is_read_from_config_and_env(monkeypatch, tmp_path):
    from mnemostack import config
    from mnemostack.server import ServerConfig

    monkeypatch.setenv("MNEMOSTACK_CONFIG", str(tmp_path / "missing.yaml"))
    assert config.Config.load().ingest.remote_window_size == 1
    monkeypatch.setenv("MNEMOSTACK_REMOTE_WINDOW_SIZE", "3")
    assert config.Config.load().ingest.remote_window_size == 3
    assert ServerConfig.from_env().remote_window_size == 3
    monkeypatch.setenv("MNEMOSTACK_REMOTE_WINDOW_SIZE", "0")
    with pytest.raises(ValueError, match="remote_window_size"):
        config.Config.load()
