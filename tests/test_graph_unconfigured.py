"""An unconfigured or unreachable graph must not cost every request (issue #181).

Two mechanisms, tested separately:

- An EXPLICITLY empty ``MNEMOSTACK_GRAPH_URI`` / ``MNEMOSTACK_MEMGRAPH_URI``
  disables the graph on the server path. An absent variable still expands to
  the documented ``bolt://localhost:7687`` default — that default is a stable
  contract and is not changed here.
- When the store proves UNREACHABLE, the graph arm trips a cooldown: one
  warning, then silence and zero connection attempts until the window expires,
  after which it retries on its own (a Memgraph that boots after the server —
  the ordinary compose case — rejoins without a restart). A bad query or any
  other operator mistake keeps the loud per-call path and never trips it.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from mnemostack.recall import retrievers as retrievers_mod
from mnemostack.recall.retrievers import GRAPH_UNAVAILABLE_COOLDOWN_S, MemgraphRetriever


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    for key in list(os.environ.keys()):
        if key.startswith("MNEMOSTACK_"):
            monkeypatch.delenv(key)
    # No config file discovery: HOME points at an empty dir.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    yield monkeypatch


# --- the env off switch -----------------------------------------------------


def _from_env():
    from mnemostack.server import ServerConfig

    return ServerConfig.from_env()


def test_absent_env_keeps_documented_default(isolated_env):
    assert _from_env().graph_uri == "bolt://localhost:7687"


@pytest.mark.parametrize("name", ["MNEMOSTACK_GRAPH_URI", "MNEMOSTACK_MEMGRAPH_URI"])
def test_explicitly_empty_env_disables_graph(isolated_env, name):
    isolated_env.setenv(name, "")
    cfg = _from_env()
    # Falsy is the contract every guard reads (`if not cfg.graph_uri`).
    assert not cfg.graph_uri


def test_empty_primary_beats_alias(isolated_env):
    # Same precedence as the config layer: the first PRESENT name decides.
    isolated_env.setenv("MNEMOSTACK_GRAPH_URI", "")
    isolated_env.setenv("MNEMOSTACK_MEMGRAPH_URI", "bolt://alias:7687")
    assert not _from_env().graph_uri


def test_set_env_still_wins(isolated_env):
    isolated_env.setenv("MNEMOSTACK_MEMGRAPH_URI", "bolt://remote:7687")
    assert _from_env().graph_uri == "bolt://remote:7687"


# --- the unreachable-store cooldown ----------------------------------------


class _Result:
    def __iter__(self):
        return iter(())

    def single(self):
        return None

    def data(self):
        return []


class _CountingDriver:
    """Counts session attempts; each one refuses like a dead store."""

    def __init__(self, exc: BaseException):
        self.exc = exc
        self.sessions = 0

    def session(self, **kwargs: Any):
        self.sessions += 1
        raise self.exc


def test_unreachable_store_trips_cooldown_once(monkeypatch):
    driver = _CountingDriver(ConnectionRefusedError("refused"))
    r = MemgraphRetriever(uri="bolt://dead:7687", driver=driver)

    assert r.search("deploy window", limit=3) == []
    assert driver.sessions == 1
    assert r._unavailable_until > 0.0

    # While the window is open: no session attempts at all, still empty.
    assert r.search("deploy window", limit=3) == []
    assert driver.sessions == 1


def test_cooldown_expires_and_retries(monkeypatch):
    driver = _CountingDriver(ConnectionRefusedError("refused"))
    r = MemgraphRetriever(uri="bolt://dead:7687", driver=driver)
    r.search("deploy window", limit=3)
    assert driver.sessions == 1

    # Expire the window: the arm must try again, then re-trip.
    now = [0.0]
    monkeypatch.setattr(retrievers_mod.time, "monotonic", lambda: now[0])
    now[0] = r._unavailable_until + 1.0
    r.search("deploy window", limit=3)
    assert driver.sessions == 2
    assert r._unavailable_until == pytest.approx(now[0] + GRAPH_UNAVAILABLE_COOLDOWN_S)


def test_injected_driver_is_not_closed_on_trip():
    class _ClosableDriver(_CountingDriver):
        def __init__(self, exc):
            super().__init__(exc)
            self.closed = False

        def close(self):
            self.closed = True

    driver = _ClosableDriver(ConnectionRefusedError("refused"))
    r = MemgraphRetriever(uri="bolt://dead:7687", driver=driver)
    r.search("deploy window", limit=3)
    # The caller owns an injected driver: cooldown applies, the driver survives.
    assert r._unavailable_until > 0.0
    assert driver.closed is False
    assert r._driver is driver


def test_own_driver_is_dropped_on_trip(monkeypatch):
    class _OwnDriver(_CountingDriver):
        def __init__(self, exc):
            super().__init__(exc)
            self.closed = False

        def close(self):
            self.closed = True

    made = _OwnDriver(ConnectionRefusedError("refused"))

    class _GraphDatabase:
        @staticmethod
        def driver(*a, **k):
            return made

    monkeypatch.setattr(retrievers_mod, "GraphDatabase", _GraphDatabase)
    monkeypatch.setattr(retrievers_mod, "_NEO4J_AVAILABLE", True)
    r = MemgraphRetriever(uri="bolt://dead:7687")
    r.search("deploy window", limit=3)
    assert made.closed is True
    assert r._driver is None


def test_bad_query_stays_loud_and_does_not_trip():
    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def run(self, *a, **k):
            raise ValueError("bad cypher")

    class _Driver:
        def session(self, **kwargs: Any):
            return _Session()

    r = MemgraphRetriever(uri="bolt://live:7687", driver=_Driver())
    assert r.search("deploy window", limit=3) == []
    assert r._unavailable_until == 0.0


def test_service_unavailable_counts_as_unreachable():
    neo4j_exc = pytest.importorskip("neo4j.exceptions")
    driver = _CountingDriver(neo4j_exc.ServiceUnavailable("no route"))
    r = MemgraphRetriever(uri="bolt://dead:7687", driver=driver)
    r.search("deploy window", limit=3)
    assert r._unavailable_until > 0.0
