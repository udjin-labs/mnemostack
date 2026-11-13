"""Request-scoped graph sessions and application-scoped graph stores."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from test_server import _patched_app

from mnemostack.graph.store import GraphStore
from mnemostack.server import ServerConfig, TriplesRequest


def _store():
    store = object.__new__(GraphStore)
    store.database = "facts"
    store.driver = MagicMock()
    return store


def _facts():
    return [
        dict(
            subject=f"Имя{i}",
            predicate="USES",
            obj="Object",
            valid_from="2026-01-01",
            tenant="alpha",
        )
        for i in range(5)
    ]


@pytest.mark.parametrize("tenant", [None, "alpha"])
def test_batch_preserves_individual_queries_and_partial_success(tenant):
    facts = [dict(fact, tenant=tenant) for fact in _facts()]
    old = _store()
    for fact in facts:
        old.add_triple(**fact)
    store = _store()
    session = store.driver.session.return_value.__enter__.return_value
    session.run.side_effect = [
        MagicMock(),
        MagicMock(),
        ValueError("bad"),
        MagicMock(),
        MagicMock(),
    ]
    errors = store.add_triples(facts)
    assert [type(e) if e else None for e in errors] == [None, None, ValueError, None, None]
    assert store.driver.session.call_count == 1
    assert (
        session.run.call_args_list
        == old.driver.session.return_value.__enter__.return_value.run.call_args_list
    )


def test_batch_reports_commit_error_and_continues():
    store = _store()
    session = store.driver.session.return_value.__enter__.return_value
    results = [MagicMock() for _ in range(5)]
    results[2].consume.side_effect = ValueError("commit failed")
    session.run.side_effect = results
    errors = store.add_triples(_facts())
    assert [type(e) if e else None for e in errors] == [None, None, ValueError, None, None]
    for result in results:
        result.consume.assert_called_once()
    assert store.driver.session.call_count == 1


def test_batch_reports_session_failure_for_every_triple():
    store = _store()
    store.driver.session.side_effect = ConnectionError("offline")
    errors = store.add_triples(_facts())
    assert [type(e) for e in errors] == [ConnectionError] * 5


def _endpoint(monkeypatch):
    app, _ = _patched_app(
        monkeypatch,
        config=ServerConfig(provider_name="fake", llm_name="fake", graph_uri="bolt://fake"),
    )
    endpoint = next(r.endpoint for r in app.routes if r.path == "/triples")
    req = TriplesRequest(
        triples=[dict(subject=f"s{i}", predicate="USES", object="o") for i in range(5)]
    )
    return app, lambda: endpoint(req, principal=None)


def test_store_reused_and_closed_by_lifespan(monkeypatch):
    store = _store()
    factory = MagicMock(return_value=store)
    monkeypatch.setattr("mnemostack.graph.factory.make_graph_store", factory)
    app, call = _endpoint(monkeypatch)

    async def run():
        async with app.router.lifespan_context(app):
            for _ in range(3):
                result = call()
                assert (result.added, result.failed) == (5, 0)
            assert factory.call_count == 1
            assert store.driver.session.call_count == 3
            store.driver.close.assert_not_called()
        store.driver.close.assert_called_once()

    asyncio.run(run())


def test_unavailable_store_and_failed_writes_keep_status_codes(monkeypatch):
    store, replacement = _store(), _store()
    store.driver.session.side_effect = ConnectionError("offline")
    factory = MagicMock(side_effect=[ConnectionError("offline"), store, replacement])
    monkeypatch.setattr("mnemostack.graph.factory.make_graph_store", factory)
    app, call = _endpoint(monkeypatch)
    with pytest.raises(HTTPException) as unavailable:
        call()
    assert (unavailable.value.status_code, unavailable.value.detail) == (503, "graph unavailable")
    with pytest.raises(HTTPException) as failed:
        call()
    assert (failed.value.status_code, failed.value.detail) == (502, "all 5 triple write(s) failed")
    store.driver.close.assert_called_once()
    assert call().added == 5

    async def stop():
        async with app.router.lifespan_context(app):
            pass

    asyncio.run(stop())
    replacement.driver.close.assert_called_once()


def test_failed_store_replaced_without_closing_concurrent_user(monkeypatch):
    entered, release = Event(), Event()
    old, new = _store(), _store()

    def write(facts):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
            return [None] * len(facts)
        return [None, None, ValueError("bad"), None, None]

    old.add_triples = write
    factory = MagicMock(side_effect=[old, new])
    monkeypatch.setattr("mnemostack.graph.factory.make_graph_store", factory)
    app, call = _endpoint(monkeypatch)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(call)
        try:
            assert entered.wait(5)
            result = call()
            assert (result.added, result.failed) == (4, 1)
            assert result.results[2].error == "ValueError"
            old.driver.close.assert_not_called()
            assert call().added == 5
            assert factory.call_count == 2
        finally:
            release.set()
        assert pending.result().added == 5
    old.driver.close.assert_called_once()

    async def stop():
        async with app.router.lifespan_context(app):
            pass

    asyncio.run(stop())
    new.driver.close.assert_called_once()


def test_name_indexes_only_explicit_migration(monkeypatch):
    from argparse import Namespace

    from mnemostack.cli import cmd_graph_migrate_current

    store = _store()
    store.backfill_current_markers = MagicMock(return_value=dict(nodes=0, relationships=0))
    monkeypatch.setattr("mnemostack.graph.factory.make_graph_store", lambda *a, **k: store)
    args = Namespace(memgraph_uri="bolt://fake", timeout=5, dry_run=True)
    cmd_graph_migrate_current(args)
    store.driver.session.assert_not_called()
    args.dry_run = False
    cmd_graph_migrate_current(args)
    session = store.driver.session.return_value.__enter__.return_value
    assert [c.args[0] for c in session.run.call_args_list] == [
        "CREATE INDEX ON :Entity(name)",
        "CREATE INDEX ON :File(name)",
        "CREATE INDEX ON :File(name_lower)",
    ]


def test_rejected_index_statement_is_a_warning(monkeypatch, capsys):
    from argparse import Namespace

    from mnemostack.cli import cmd_graph_migrate_current

    store = _store()
    store.backfill_current_markers = MagicMock(return_value=dict(nodes=2, relationships=1))
    session = store.driver.session.return_value.__enter__.return_value
    session.run.side_effect = [MagicMock(), RuntimeError("bolt://user:secret@host"), MagicMock()]
    monkeypatch.setattr("mnemostack.graph.factory.make_graph_store", lambda *a, **k: store)
    assert cmd_graph_migrate_current(Namespace(memgraph_uri="bolt://fake", timeout=5, dry_run=False)) == 0
    captured = capsys.readouterr()
    assert "Updated 2 node(s) and 1 relationship(s)." in captured.out
    assert captured.err == "warning: index not created (RuntimeError): CREATE INDEX ON :File(name)\n"
    assert session.run.call_count == 3


def test_single_triple_write_reports_a_late_failure():
    store = _store()
    session = store.driver.session.return_value.__enter__.return_value
    session.run.return_value.consume.side_effect = ValueError("commit failed")
    with pytest.raises(ValueError):
        store.add_triple("a", "knows", "b")
