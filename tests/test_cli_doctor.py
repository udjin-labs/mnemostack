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


def _fake_qdrant_cls(
    *, reachable=True, exists=True, count=5, size=8, created=None, coll_error=None,
    payload_schema=None, memory=None, memory_error=None, memory_calls=None,
):
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
            return type(
                "Info",
                (),
                {
                    "points_count": count,
                    "config": config,
                    "payload_schema": dict(payload_schema or {}),
                },
            )()

        @property
        def http(self):
            from types import SimpleNamespace

            def request(**kwargs):
                if memory_calls is not None:
                    memory_calls.append(kwargs)
                if memory_error is not None:
                    raise memory_error
                return {"result": memory}

            return SimpleNamespace(client=SimpleNamespace(request=request))

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


def _memory_counters(disk=1000, ram=100, cached=200, expected=300):
    return dict(disk_bytes=disk, ram_bytes=ram, cached_bytes=cached, expected_cache_bytes=expected)


def test_doctor_memory_json_is_collection_wide_and_read_only(patched, capsys):
    calls = []
    created = []
    data = {
        "total": _memory_counters(),
        "vectors": [{"name": "semantic", "storage": _memory_counters(),
                     "quantized": _memory_counters(250, 250, 0, 0)}],
        "sparse_vectors": [{"name": "keywords", "index": _memory_counters()}],
        "payload_index": [{"name": "tenant", "usage": _memory_counters(50, 10, 20, 30)}],
        "payload": _memory_counters(),
        "other": {"id_tracker": _memory_counters()},
    }
    patched(provider=_FakeProvider(), llm=_FakeLLM(), qdrant_cls=_fake_qdrant_cls(
        memory=data, memory_calls=calls, created=created,
    ))
    assert cli.cmd_doctor(_args(json=True)) == 0
    report = json.loads(capsys.readouterr().out)
    check = next(x for x in report["checks"] if x["section"] == "qdrant.memory")
    assert check["status"] == "ok"
    assert check["data"]["total"] == data["total"]
    assert "vectors.semantic.quantized" in check["data"]["components"]
    assert "sparse_vectors.keywords.index" in check["data"]["components"]
    assert check["data"]["components"]["payload_index.tenant.usage"] == _memory_counters(50, 10, 20, 30)
    assert "not process RSS" in check["detail"]
    assert len(calls) == 1 and calls[0]["method"] == "GET"
    assert calls[0]["path_params"] == {"collection_name": "mnemostack"}
    assert created == []


@pytest.mark.parametrize("total", [None, {}, {"disk_bytes": True}, _memory_counters(ram=-1)])
def test_doctor_invalid_memory_warns_without_failing(patched, capsys, total):
    patched(provider=_FakeProvider(), llm=_FakeLLM(), qdrant_cls=_fake_qdrant_cls(memory={"total": total}))
    assert cli.cmd_doctor(_args(json=True)) == 0
    report = json.loads(capsys.readouterr().out)
    check = next(x for x in report["checks"] if x["section"] == "qdrant.memory")
    assert check["status"] == "warn" and "data" not in check


@pytest.mark.parametrize("json_output", [False, True])
def test_doctor_incomplete_memory_warns_without_exposing_paths(patched, capsys, json_output):
    patched(provider=_FakeProvider(), llm=_FakeLLM(), qdrant_cls=_fake_qdrant_cls(
        memory={"total": _memory_counters(), "warnings": ["cannot measure /private/storage/segment"]},
    ))
    assert cli.cmd_doctor(_args(json=json_output)) == 0
    out = capsys.readouterr().out
    assert "incomplete collection estimates" in out
    assert "/private/storage" not in out
    if json_output:
        check = next(x for x in json.loads(out)["checks"] if x["section"] == "qdrant.memory")
        assert check["status"] == "warn"
        assert check["data"]["total"] == _memory_counters()
    else:
        assert "[WARN" in out


def test_doctor_memory_failure_never_echoes_response(patched, capsys):
    patched(provider=_FakeProvider(), llm=_FakeLLM(), qdrant_cls=_fake_qdrant_cls(
        memory_error=RuntimeError("private response text"),
    ))
    assert cli.cmd_doctor(_args()) == 0
    out = capsys.readouterr().out
    assert "memory estimates unavailable (RuntimeError)" in out
    assert "private response text" not in out


def test_missing_collection_does_not_request_memory(patched, capsys):
    calls = []
    patched(provider=_FakeProvider(), llm=_FakeLLM(), qdrant_cls=_fake_qdrant_cls(
        exists=False, memory_calls=calls,
    ))
    assert cli.cmd_doctor(_args()) == 0
    capsys.readouterr()
    assert calls == []


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


def test_doctor_mixed_case_provider_is_not_misconfig(patched, capsys):
    # get_provider() lowercases names, so a mixed-case config value (e.g.
    # MNEMOSTACK_PROVIDER=OLLAMA) is valid and must not be flagged MISCONFIG.
    patched(
        provider=_FakeProvider(healthy=True),
        qdrant_cls=_fake_qdrant_cls(reachable=True, exists=True, size=8),
        llm=_FakeLLM(),
    )
    rc = cli.cmd_doctor(_args(provider="OLLAMA"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "MISCONFIG" not in out


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


def test_doctor_lists_payload_indexes(patched, capsys):
    from types import SimpleNamespace

    from qdrant_client.models import PayloadSchemaType

    patched(
        provider=_FakeProvider(dim=8, healthy=True),
        qdrant_cls=_fake_qdrant_cls(
            reachable=True, exists=True, count=5, size=8,
            payload_schema={
                "tenant_id": SimpleNamespace(data_type=PayloadSchemaType.KEYWORD),
                "captured": SimpleNamespace(data_type=PayloadSchemaType.DATETIME),
            },
        ),
        llm=_FakeLLM(healthy=True),
    )
    rc = cli.cmd_doctor(_args())
    out = capsys.readouterr().out
    assert rc == 0
    assert "qdrant.payload_indexes" in out
    assert "captured (datetime)" in out
    assert "tenant_id (keyword)" in out


def test_doctor_warns_when_no_payload_indexes_without_flipping_exit(patched, capsys):
    patched(
        provider=_FakeProvider(dim=8, healthy=True),
        qdrant_cls=_fake_qdrant_cls(reachable=True, exists=True, count=5, size=8),
        llm=_FakeLLM(healthy=True),
    )
    rc = cli.cmd_doctor(_args())
    out = capsys.readouterr().out
    assert rc == 0  # informational warn never flips the exit code
    assert "qdrant.payload_indexes" in out
    assert "mnemostack payload-index" in out


@pytest.mark.parametrize("scoring, expected_rc", [(True, 0), (False, 2)])
def test_doctor_checks_native_bm25_scoring(patched, capsys, scoring, expected_rc):
    from types import SimpleNamespace

    from qdrant_client.models import PayloadSchemaType

    cfg = Config()
    cfg.recall.text_search = "native_bm25"
    patched(
        cfg=cfg,
        provider=_FakeProvider(dim=8, healthy=True),
        qdrant_cls=_fake_qdrant_cls(
            reachable=True,
            exists=True,
            count=5,
            size=8,
            payload_schema={
                "text": SimpleNamespace(
                    data_type=PayloadSchemaType.TEXT,
                    params=SimpleNamespace(scoring=scoring),
                )
            },
        ),
        llm=_FakeLLM(healthy=True),
    )
    cli._payload_schema.cache_clear()
    rc = cli.cmd_doctor(_args())
    out = capsys.readouterr().out
    assert rc == expected_rc
    assert "BM25" in out
    cli._payload_schema.cache_clear()
