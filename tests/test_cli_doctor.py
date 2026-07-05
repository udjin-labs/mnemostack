"""Tests for `mnemostack doctor` — the read-only deployment diagnostic.

The stack is faked (no real Qdrant / provider / LLM): we assert the check
statuses, the tri-state exit code (0 healthy / 1 down / 2 misconfigured), the
read-only guarantee (a missing collection is reported, never created), and the
--json shape.
"""

from __future__ import annotations

import argparse
import json

import pytest

import mnemostack.cli as cli
from mnemostack.config import Config


class _FakeProvider:
    def __init__(self, dim: int = 8, healthy: bool = True):
        self.dimension = dim
        self.name = "fake:embed"
        self._healthy = healthy

    def embed(self, text):
        return [0.0] * self.dimension

    def health_check(self):
        if self._healthy:
            return True, f"ok, dim={self.dimension}"
        return False, f"unexpected dim: got 0, expected {self.dimension}"


class _FakeLLM:
    def __init__(self, healthy: bool = True):
        self._healthy = healthy

    def health_check(self):
        return (True, "ok") if self._healthy else (False, "no key")


def _fake_qdrant_cls(*, reachable=True, exists=True, count=5, size=8, created=None, coll_error=None):
    """A QdrantClient stand-in. `created` (a list) records ensure_collection-style
    side effects so a test can assert the diagnostic never creates a collection.
    `coll_error` raises an arbitrary exception from get_collection (e.g. an auth
    failure) to exercise the non-"not found" branch."""

    class _Client:
        def __init__(self, *a, **k):
            pass

        def get_collections(self):
            if not reachable:
                raise ConnectionError("connection refused")
            return object()

        def get_collection(self, name):
            if coll_error is not None:
                raise coll_error
            if not exists:
                raise ValueError(f"Collection {name} not found")
            vectors = type("V", (), {"size": size})()
            params = type("P", (), {"vectors": vectors})()
            config = type("C", (), {"params": params})()
            return type("Info", (), {"points_count": count, "config": config})()

        # A diagnostic must never call these; present so a call would be visible.
        def create_collection(self, *a, **k):  # pragma: no cover - must not run
            (created if created is not None else []).append(("create", a, k))
            raise AssertionError("doctor must not create a collection")

    return _Client


def _args(**over):
    d = dict(
        provider="gemini",
        embedding_model=None,
        collection="mnemostack",
        qdrant="http://localhost:6333",
        json=False,
        check_llm=False,
    )
    d.update(over)
    return argparse.Namespace(**d)


@pytest.fixture
def patched(monkeypatch):
    """Patch Config.load to a deterministic default config (graph disabled)."""

    def _install(cfg: Config | None = None, provider=None, qdrant_cls=None, llm=None, graph=None):
        monkeypatch.setattr(cli.Config, "load", classmethod(lambda cls: cfg or Config()))
        if provider is not None:
            monkeypatch.setattr(cli, "get_provider", lambda name, **kw: provider)
        if qdrant_cls is not None:
            import qdrant_client

            monkeypatch.setattr(qdrant_client, "QdrantClient", qdrant_cls)
        if llm is not None:
            monkeypatch.setattr(cli, "get_llm", lambda name, **kw: llm)
        if graph is not None:
            import mnemostack.graph.factory as factory

            monkeypatch.setattr(factory, "make_graph_store", lambda *a, **k: graph)

    return _install


def test_doctor_all_healthy(patched, capsys):
    patched(
        provider=_FakeProvider(dim=8, healthy=True),
        qdrant_cls=_fake_qdrant_cls(reachable=True, exists=True, count=5, size=8),
        llm=_FakeLLM(healthy=True),
    )
    rc = cli.cmd_doctor(_args())
    out = capsys.readouterr().out
    assert rc == 0
    assert "doctor: healthy" in out
    assert "[OK" in out


def test_doctor_qdrant_down_exit_1(patched, capsys):
    patched(
        provider=_FakeProvider(healthy=True),
        qdrant_cls=_fake_qdrant_cls(reachable=False),
        llm=_FakeLLM(),
    )
    rc = cli.cmd_doctor(_args())
    out = capsys.readouterr().out
    assert rc == 1
    assert "DOWN" in out and "qdrant" in out


def test_doctor_dimension_mismatch_exit_1(patched, capsys):
    patched(
        provider=_FakeProvider(dim=8, healthy=True),
        qdrant_cls=_fake_qdrant_cls(reachable=True, exists=True, count=5, size=16),
        llm=_FakeLLM(),
    )
    rc = cli.cmd_doctor(_args())
    out = capsys.readouterr().out
    assert rc == 1
    assert "MISMATCH" in out
    assert "stored=16" in out and "provider=8" in out


def test_doctor_missing_collection_is_warn_not_created(patched, capsys):
    created: list = []
    patched(
        provider=_FakeProvider(healthy=True),
        qdrant_cls=_fake_qdrant_cls(reachable=True, exists=False, created=created),
        llm=_FakeLLM(),
    )
    rc = cli.cmd_doctor(_args())
    out = capsys.readouterr().out
    # embedding ok + qdrant reachable-but-empty -> only warnings -> exit 0
    assert rc == 0
    assert "does not exist yet" in out
    assert created == []  # never created the collection


def test_doctor_qdrant_collection_error_is_down_not_warn(patched, capsys):
    # A non-"not found" error from get_collection (e.g. auth failure) must read
    # as DOWN (exit 1), not be masked as "collection missing" (which is a warn).
    patched(
        provider=_FakeProvider(healthy=True),
        qdrant_cls=_fake_qdrant_cls(reachable=True, coll_error=ValueError("Forbidden: 403")),
        llm=_FakeLLM(),
    )
    rc = cli.cmd_doctor(_args())
    out = capsys.readouterr().out
    assert rc == 1
    assert "DOWN" in out and "collection query failed" in out
    assert "does not exist yet" not in out


def test_doctor_graph_down_is_warn_not_exit_1(patched, capsys):
    # A configured-but-unreachable graph is optional/fail-soft: reported as WARN,
    # never fails the exit code (recall works without the graph).
    cfg = Config()
    cfg.graph.uri = "bolt://localhost:7687"

    class _DownGraph:
        def health_check(self):
            return False, "unreachable: connection refused"

        def close(self):
            pass

    patched(
        cfg=cfg,
        provider=_FakeProvider(healthy=True),
        qdrant_cls=_fake_qdrant_cls(reachable=True, exists=True, size=8),
        llm=_FakeLLM(),
        graph=_DownGraph(),
    )
    rc = cli.cmd_doctor(_args())
    out = capsys.readouterr().out
    assert rc == 0
    assert "WARN" in out and "configured but unreachable" in out


def test_doctor_unknown_provider_is_misconfig_exit_2(patched, monkeypatch, capsys):
    # An invalid config provider name -> misconfig -> exit 2. get_provider is
    # patched to fail loudly to prove the unknown name is never constructed.
    def _boom(*a, **k):
        raise AssertionError("should not construct an unknown provider")

    patched(qdrant_cls=_fake_qdrant_cls(reachable=True), llm=_FakeLLM())
    monkeypatch.setattr(cli, "get_provider", _boom)
    rc = cli.cmd_doctor(_args(provider="bogus-provider"))
    out = capsys.readouterr().out
    assert rc == 2
    assert "MISCONFIG" in out
    assert "unknown provider" in out


def test_doctor_json_output(patched, capsys):
    patched(
        provider=_FakeProvider(dim=8, healthy=True),
        qdrant_cls=_fake_qdrant_cls(reachable=True, exists=True, count=3, size=8),
        llm=_FakeLLM(healthy=True),
    )
    rc = cli.cmd_doctor(_args(json=True))
    out = capsys.readouterr().out
    doc = json.loads(out)
    assert rc == 0
    assert doc["exit_code"] == 0
    assert doc["version"]
    assert isinstance(doc["checks"], list) and doc["checks"]
    assert set(doc["summary"]) >= {"ok", "warn", "down", "misconfig", "disabled"}
    sections = {c["section"] for c in doc["checks"]}
    assert {"embedding", "qdrant", "llm", "graph"} <= sections


def test_doctor_graph_reported_when_configured(patched, capsys):
    cfg = Config()
    cfg.graph.uri = "bolt://localhost:7687"

    class _FakeGraphStore:
        def health_check(self):
            return True, "ok"

        def count_nodes(self):
            return 12

        def count_edges(self):
            return 7

        def close(self):
            pass

    patched(
        cfg=cfg,
        provider=_FakeProvider(healthy=True),
        qdrant_cls=_fake_qdrant_cls(reachable=True, exists=True, size=8),
        llm=_FakeLLM(),
        graph=_FakeGraphStore(),
    )
    rc = cli.cmd_doctor(_args())
    out = capsys.readouterr().out
    assert rc == 0
    assert "nodes=12" in out and "edges=7" in out
