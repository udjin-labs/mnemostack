"""CLI audit coverage for collection administration."""

import argparse
import json

import pytest

from mnemostack import cli
from mnemostack.audit import KNOWN_ACTIONS
from mnemostack.vector.qdrant import PayloadIndexConflictError


@pytest.fixture
def audit_file(monkeypatch, tmp_path):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("MNEMOSTACK_AUDIT_FILE", str(path))
    return path


class _Store:
    def collection_exists(self):
        return True

    def ensure_text_index(self, field, **kwargs):
        pass

    def ensure_payload_index(self, field, schema):
        return schema

    def payload_indexes(self):
        return {}

    def ensure_sparse_space(self, **kwargs):
        pass

    def backfill_sparse_text(self, on_batch=None):
        if on_batch is not None:
            on_batch(7)
        return 7


@pytest.fixture
def store(monkeypatch):
    store = _Store()
    monkeypatch.setattr(cli, "VectorStore", lambda **kwargs: store)
    monkeypatch.setattr(cli, "_lexical_gate_fields", lambda: ["text"])
    monkeypatch.setattr(cli, "_payload_schema", lambda: ("text",))
    return store


def _args(**kwargs):
    return argparse.Namespace(collection="mt", qdrant="https://secret@example.org", **kwargs)


CASES = [
    ("text_index", {}, {"fields": ["text"], "schema": "text"}, "ensure_text_index"),
    ("payload_index", {"field": "kind", "schema": "keyword"},
     {"field": "kind", "schema": "keyword"}, "ensure_payload_index"),
    ("sparse_backfill", {}, {"points": 7}, "backfill_sparse_text"),
]


@pytest.mark.parametrize("command,kwargs,details,method", CASES)
def test_success(store, audit_file, command, kwargs, details, method):
    assert getattr(cli, "cmd_" + command)(_args(**kwargs)) == 0
    (event,) = [json.loads(line) for line in audit_file.read_text().splitlines()]
    assert event["action"] == "collection." + command
    assert event["action"] in KNOWN_ACTIONS
    assert event["outcome"] == "success"
    assert event.get("tenant") is None
    assert event["details"] == {"collection": "mt", **details}
    assert "secret@example.org" not in audit_file.read_text()


@pytest.mark.parametrize("command,kwargs,details,method", CASES)
def test_failure(monkeypatch, store, audit_file, command, kwargs, details, method):
    def fail(*args, **kwargs):
        raise ConnectionError("https://private:password@example.org")

    monkeypatch.setattr(store, method, fail)
    assert getattr(cli, "cmd_" + command)(_args(**kwargs)) == 1
    (event,) = [json.loads(line) for line in audit_file.read_text().splitlines()]
    assert event["action"] == "collection." + command
    assert event["outcome"] == "error"
    assert event["details"] == {
        "collection": "mt", **{k: v for k, v in details.items() if k != "points"},
        "error": "ConnectionError",
        **({"points": 0, "stage": "backfill"} if command == "sparse_backfill" else {}),
    }
    assert "private:password" not in audit_file.read_text()


def test_partial_backfill_records_written_points(monkeypatch, store, audit_file):
    def fail_after_first_batch(on_batch=None):
        on_batch(256)
        raise ConnectionError("lost")

    monkeypatch.setattr(store, "backfill_sparse_text", fail_after_first_batch)
    assert cli.cmd_sparse_backfill(_args()) == 1
    (event,) = [json.loads(line) for line in audit_file.read_text().splitlines()]
    assert event["outcome"] == "partial"
    assert event["details"] == {
        "collection": "mt", "error": "ConnectionError", "points": 256, "stage": "backfill",
    }


@pytest.mark.parametrize("command,kwargs,details,method", CASES)
def test_disabled(monkeypatch, store, tmp_path, command, kwargs, details, method):
    monkeypatch.delenv("MNEMOSTACK_AUDIT_FILE", raising=False)
    assert getattr(cli, "cmd_" + command)(_args(**kwargs)) == 0
    assert not list(tmp_path.glob("*.jsonl"))


@pytest.mark.parametrize("kwargs", [
    {"field": None, "schema": "keyword"},
    {"field": "kind", "schema": None},
    {"field": "kind", "schema": "text"},
])
def test_validation_writes_nothing(audit_file, kwargs):
    assert cli.cmd_payload_index(_args(**kwargs)) == 2
    assert not audit_file.exists()


def test_list_writes_nothing(store, audit_file):
    assert cli.cmd_payload_index(_args(field=None, schema=None)) == 0
    assert not audit_file.exists()


@pytest.mark.parametrize("command,kwargs,details,method", CASES)
def test_absent_collection(monkeypatch, store, audit_file, command, kwargs, details, method):
    monkeypatch.setattr(store, "collection_exists", lambda: False)
    assert getattr(cli, "cmd_" + command)(_args(**kwargs)) == 1
    (event,) = [json.loads(line) for line in audit_file.read_text().splitlines()]
    assert event["outcome"] == "error"
    assert event["details"]["reason"] == "collection_absent"


def test_conflict(monkeypatch, store, audit_file):
    def fail(*args):
        raise PayloadIndexConflictError("private message")

    monkeypatch.setattr(store, "ensure_payload_index", fail)
    assert cli.cmd_payload_index(_args(field="kind", schema="keyword")) == 2
    (event,) = [json.loads(line) for line in audit_file.read_text().splitlines()]
    assert event["details"]["error"] == "PayloadIndexConflictError"
    assert event["outcome"] == "error"
    assert "private message" not in audit_file.read_text()


def test_index_writes_no_admin_actions(monkeypatch, tmp_path, audit_file):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance
    from test_cli_enrich import _args, _CountingProvider, _patch_stack

    from mnemostack.vector import VectorStore

    store = VectorStore.__new__(VectorStore)
    store.collection = "ordinary"
    store.dimension = 4
    store.distance = Distance.COSINE
    store.client = QdrantClient(":memory:")
    store.ensure_collection()
    _patch_stack(monkeypatch, store, _CountingProvider())
    (tmp_path / "note.md").write_text("Ordinary ingestion content.")
    assert cli.cmd_index(_args(tmp_path)) == 0
    events = [json.loads(line) for line in audit_file.read_text().splitlines()] if audit_file.exists() else []
    assert not {event["action"] for event in events} & {
        "collection.text_index", "collection.payload_index", "collection.sparse_backfill",
    }


@pytest.mark.parametrize("command,kwargs,details,method", CASES)
@pytest.mark.parametrize("stage", ["constructor", "collection_exists", "ensure_sparse_space"])
def test_preflight_failure(monkeypatch, store, audit_file, command, kwargs, details, method, stage):
    if stage == "ensure_sparse_space" and command != "sparse_backfill":
        return

    def fail(*args, **kwargs):
        raise ConnectionError("private connection URL")

    if stage == "constructor":
        monkeypatch.setattr(cli, "VectorStore", fail)
    else:
        monkeypatch.setattr(store, stage, fail)
    assert getattr(cli, "cmd_" + command)(_args(**kwargs)) == 1
    (event,) = [json.loads(line) for line in audit_file.read_text().splitlines()]
    assert event["action"] == "collection." + command
    assert event["outcome"] == "error"
    assert event["details"]["error"] == "ConnectionError"
    if command == "sparse_backfill":
        assert event["details"]["stage"] == "preflight"
    assert "private connection URL" not in audit_file.read_text()
