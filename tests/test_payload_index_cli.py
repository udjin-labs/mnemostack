"""Payload-index operator surface: store methods, CLI command, doctor row."""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdrant_client.models import PayloadSchemaType

import mnemostack.cli as cli
from mnemostack.vector import VectorStore


def _store_with_client(client) -> VectorStore:
    s = VectorStore.__new__(VectorStore)
    s.collection = "c"
    s.dimension = 4
    s.client = client
    return s


def _info_with_schema(schema: dict) -> SimpleNamespace:
    return SimpleNamespace(payload_schema=schema)


# ------------------------------------------------------------ store methods


def test_payload_indexes_parses_the_collection_schema():
    client = MagicMock()
    client.get_collection.return_value = _info_with_schema(
        {
            "tenant_id": SimpleNamespace(data_type=PayloadSchemaType.KEYWORD),
            "timestamp": SimpleNamespace(data_type=PayloadSchemaType.DATETIME),
            "flat": "integer",  # some client versions expose bare strings
        }
    )
    s = _store_with_client(client)
    assert s.payload_indexes() == {
        "tenant_id": "keyword",
        "timestamp": "datetime",
        "flat": "integer",
    }


def test_payload_indexes_empty_when_backend_records_none():
    client = MagicMock()
    client.get_collection.return_value = _info_with_schema({})
    assert _store_with_client(client).payload_indexes() == {}


def test_ensure_payload_index_creates_and_reports():
    client = MagicMock()
    client.get_collection.side_effect = [
        _info_with_schema({}),  # pre-check: nothing indexed yet
        _info_with_schema(
            {"project": SimpleNamespace(data_type=PayloadSchemaType.KEYWORD)}
        ),  # post-create verification
    ]
    s = _store_with_client(client)

    assert s.ensure_payload_index("project", "keyword") == "keyword"
    client.create_payload_index.assert_called_once_with(
        collection_name="c", field_name="project", field_schema=PayloadSchemaType.KEYWORD
    )


def test_ensure_payload_index_same_type_is_a_no_op():
    client = MagicMock()
    client.get_collection.return_value = _info_with_schema(
        {"project": SimpleNamespace(data_type=PayloadSchemaType.KEYWORD)}
    )
    s = _store_with_client(client)

    assert s.ensure_payload_index("project", "keyword") == "keyword"
    client.create_payload_index.assert_not_called()


def test_ensure_payload_index_refuses_a_conflicting_type():
    """A real server silently REPLACES an index re-created with another type
    (verified live) — the operator surface must refuse before mutating."""
    client = MagicMock()
    client.get_collection.return_value = _info_with_schema(
        {"project": SimpleNamespace(data_type=PayloadSchemaType.KEYWORD)}
    )
    s = _store_with_client(client)

    with pytest.raises(ValueError, match="already indexed as 'keyword'"):
        s.ensure_payload_index("project", "integer")
    client.create_payload_index.assert_not_called()


def test_ensure_payload_index_rejects_unknown_schema():
    s = _store_with_client(MagicMock())
    with pytest.raises(ValueError, match="unknown payload index schema"):
        s.ensure_payload_index("project", "geo")
    s.client.create_payload_index.assert_not_called()


def test_ensure_payload_index_is_loud_on_backend_rejection():
    client = MagicMock()
    client.get_collection.return_value = _info_with_schema({})
    client.create_payload_index.side_effect = RuntimeError("backend rejected")
    s = _store_with_client(client)
    with pytest.raises(RuntimeError, match="backend rejected"):
        s.ensure_payload_index("project", "keyword")


def test_ensure_payload_index_falls_back_to_requested_name_locally():
    # The in-memory client records nothing — the reported type is then the
    # requested one, not a KeyError.
    client = MagicMock()
    client.get_collection.return_value = _info_with_schema({})
    s = _store_with_client(client)
    assert s.ensure_payload_index("project", "datetime") == "datetime"


# ------------------------------------------------------------ CLI command


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        collection="c", qdrant="http://localhost:6333", field=None, schema=None
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _patch_store(monkeypatch, store):
    monkeypatch.setattr(cli, "VectorStore", lambda **_kw: store)


def test_cmd_payload_index_lists_existing_indexes(monkeypatch, capsys):
    client = MagicMock()
    client.get_collection.return_value = _info_with_schema(
        {"tenant_id": SimpleNamespace(data_type=PayloadSchemaType.KEYWORD)}
    )
    store = _store_with_client(client)
    monkeypatch.setattr(store, "collection_exists", lambda: True)
    _patch_store(monkeypatch, store)

    assert cli.cmd_payload_index(_args()) == 0
    assert "tenant_id: keyword" in capsys.readouterr().out


def test_cmd_payload_index_creates_an_index(monkeypatch, capsys):
    client = MagicMock()
    client.get_collection.side_effect = [
        _info_with_schema({}),
        _info_with_schema(
            {"project": SimpleNamespace(data_type=PayloadSchemaType.KEYWORD)}
        ),
    ]
    store = _store_with_client(client)
    monkeypatch.setattr(store, "collection_exists", lambda: True)
    _patch_store(monkeypatch, store)

    rc = cli.cmd_payload_index(_args(field="project", schema="keyword"))

    assert rc == 0
    client.create_payload_index.assert_called_once()
    assert "payload index ensured on 'project' (keyword)" in capsys.readouterr().out


def test_cmd_payload_index_requires_schema_for_creation(monkeypatch, capsys):
    store = _store_with_client(MagicMock())
    monkeypatch.setattr(store, "collection_exists", lambda: True)
    _patch_store(monkeypatch, store)

    assert cli.cmd_payload_index(_args(field="project")) == 2
    assert "--schema is required" in capsys.readouterr().err


def test_cmd_payload_index_redirects_text_schema(monkeypatch, capsys):
    store = _store_with_client(MagicMock())
    monkeypatch.setattr(store, "collection_exists", lambda: True)
    _patch_store(monkeypatch, store)

    assert cli.cmd_payload_index(_args(field="body", schema="text")) == 2
    err = capsys.readouterr().err
    assert "text-index" in err
    store.client.create_payload_index.assert_not_called()


def test_cmd_payload_index_missing_collection(monkeypatch, capsys):
    store = _store_with_client(MagicMock())
    monkeypatch.setattr(store, "collection_exists", lambda: False)
    _patch_store(monkeypatch, store)

    assert cli.cmd_payload_index(_args(field="project", schema="keyword")) == 1
    assert "does not exist" in capsys.readouterr().err


def test_cmd_payload_index_backend_failure_is_loud(monkeypatch, capsys):
    client = MagicMock()
    client.get_collection.return_value = _info_with_schema({})
    client.create_payload_index.side_effect = RuntimeError("boom")
    store = _store_with_client(client)
    monkeypatch.setattr(store, "collection_exists", lambda: True)
    _patch_store(monkeypatch, store)

    assert cli.cmd_payload_index(_args(field="project", schema="keyword")) == 1
    assert "cannot create payload index" in capsys.readouterr().err


def test_conflict_with_a_full_text_index_is_refused():
    """The realistic conflict: `text-index` indexed the field as TEXT, then
    the operator asks for keyword — refused, not silently replaced."""
    from mnemostack.vector.qdrant import PayloadIndexConflictError

    client = MagicMock()
    client.get_collection.return_value = _info_with_schema(
        {"title": SimpleNamespace(data_type=PayloadSchemaType.TEXT)}
    )
    s = _store_with_client(client)

    with pytest.raises(PayloadIndexConflictError, match="already indexed as 'text'"):
        s.ensure_payload_index("title", "keyword")
    client.create_payload_index.assert_not_called()


def test_backend_value_error_is_exit_1_not_config_error(monkeypatch, capsys):
    """A backend failure that surfaces as ValueError (auth, reachability)
    must exit 1 like text-index's — only the deliberate conflict refusal is
    a usage error (exit 2)."""
    client = MagicMock()
    client.get_collection.side_effect = ValueError("Forbidden: 403")
    store = _store_with_client(client)
    monkeypatch.setattr(store, "collection_exists", lambda: True)
    _patch_store(monkeypatch, store)

    assert cli.cmd_payload_index(_args(field="project", schema="keyword")) == 1
    assert "cannot create payload index" in capsys.readouterr().err


def test_cmd_payload_index_conflict_is_exit_2(monkeypatch, capsys):
    client = MagicMock()
    client.get_collection.return_value = _info_with_schema(
        {"project": SimpleNamespace(data_type=PayloadSchemaType.KEYWORD)}
    )
    store = _store_with_client(client)
    monkeypatch.setattr(store, "collection_exists", lambda: True)
    _patch_store(monkeypatch, store)

    assert cli.cmd_payload_index(_args(field="project", schema="integer")) == 2
    assert "already indexed as 'keyword'" in capsys.readouterr().err


def test_usage_errors_are_exit_2_even_with_the_backend_down(monkeypatch, capsys):
    """Argument validation happens BEFORE any backend contact: a malformed
    invocation exits 2 whether or not Qdrant is reachable."""

    def _boom(**_kw):
        raise AssertionError("usage validation must not touch the backend")

    monkeypatch.setattr(cli, "VectorStore", _boom)

    assert cli.cmd_payload_index(_args(field="project")) == 2  # no --schema
    assert "--schema is required" in capsys.readouterr().err
    assert cli.cmd_payload_index(_args(field="body", schema="text")) == 2
    assert "text-index" in capsys.readouterr().err


def test_schema_without_a_field_is_a_usage_error(monkeypatch, capsys):
    def _boom(**_kw):
        raise AssertionError("usage validation must not touch the backend")

    monkeypatch.setattr(cli, "VectorStore", _boom)

    assert cli.cmd_payload_index(_args(schema="keyword")) == 2
    assert "requires a field" in capsys.readouterr().err


def test_lost_creation_race_is_reported_as_conflict():
    """If a concurrent creation wins with another type, the post-create
    verification must refuse to report the requested index as ensured."""
    from mnemostack.vector.qdrant import PayloadIndexConflictError

    client = MagicMock()
    client.get_collection.side_effect = [
        _info_with_schema({}),  # pre-check: nothing indexed
        _info_with_schema(  # post-create: a rival integer index landed
            {"project": SimpleNamespace(data_type=PayloadSchemaType.INTEGER)}
        ),
    ]
    s = _store_with_client(client)

    with pytest.raises(PayloadIndexConflictError, match="concurrent index creation"):
        s.ensure_payload_index("project", "keyword")


def test_store_construction_failure_is_a_cli_error_not_a_traceback(
    monkeypatch, capsys
):
    def _bad_url(**_kw):
        raise ValueError("Unsupported scheme: ftp")

    monkeypatch.setattr(cli, "VectorStore", _bad_url)

    assert cli.cmd_payload_index(_args(field="project", schema="keyword")) == 1
    assert "cannot create payload index" in capsys.readouterr().err


def test_parser_accepts_payload_index_command():
    parser = cli.build_parser()
    args = parser.parse_args(["payload-index", "project", "--schema", "keyword"])
    assert args.field == "project" and args.schema == "keyword"
    args = parser.parse_args(["payload-index"])
    assert args.field is None
