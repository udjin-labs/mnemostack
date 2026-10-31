"""Quickstart automation must check evidence and faults, with explicit side effects."""

import asyncio
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

api = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/verify_quickstart.py"))
SmokeError = api["SmokeError"]


def results():
    return [{"id": "demo-chunk", "text": "CEDAR-482 from demo-001", "payload": {"source": "workshop-chat.txt"}}]


def test_commands_come_from_current_published_quickstart():
    guide = api["walkthrough"](Path(__file__).resolve().parents[1] / "examples/quickstart.md")
    assert guide["commands"]["search"][-1] == "--json"
    assert api["flag_value"](guide["commands"]["mcp"], "--rerank-mode") == "full_reorder"
    assert "CEDAR-482" in guide["fixture"]
    assert all("--recreate" not in tokens for tokens in guide["commands"].values())


def test_duplicate_flags_cannot_override_the_forced_test_collection(tmp_path):
    text = (Path(__file__).resolve().parents[1] / "examples/quickstart.md").read_text()
    changed = text.replace("--collection archive-demo", "--collection archive-demo --collection existing-archive", 1)
    path = tmp_path / "quickstart.md"
    path.write_text(changed)
    with pytest.raises(SmokeError, match="duplicate_document_flag"):
        api["walkthrough"](path)


@pytest.mark.parametrize("fault", ["reranker:fallback", "reranker:unavailable", "retriever:vector:failed"])
def test_expected_fact_does_not_hide_degraded_path(fault):
    with pytest.raises(SmokeError, match="degraded"):
        api["mcp_evidence"]({"ok": True, "results": results(), "degraded": [fault], "notes": []})


def test_routine_temporal_note_is_not_a_failure():
    assert api["mcp_evidence"]({"ok": True, "results": results(), "degraded": ["temporal:no_parse"],
                                "notes": ["temporal:no_parse"]}) == ["demo-chunk"]


@pytest.mark.parametrize("missing", ["code", "message", "source"])
def test_code_alone_does_not_prove_source_evidence(missing):
    records = results()
    if missing == "source":
        records[0]["payload"]["source"] = "different.txt"
    else:
        records[0]["text"] = "demo-001" if missing == "code" else "CEDAR-482"
    with pytest.raises(SmokeError):
        api["evidence"](records)


def test_user_config_and_state_are_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSTACK_COLLECTION", "user-real-collection")
    monkeypatch.setenv("MNEMOSTACK_TENANT", "other-user")
    env = api["isolated_env"](tmp_path)
    assert "MNEMOSTACK_COLLECTION" not in env and "MNEMOSTACK_TENANT" not in env
    assert env["MNEMOSTACK_CONFIG"] == str(tmp_path / "empty-config.yaml")
    assert env["XDG_STATE_HOME"] == str(tmp_path / "state")


def test_isolated_state_does_not_import_legacy_feedback(tmp_path, monkeypatch):
    from mnemostack.recall.pipeline import state

    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"q_table": {"fixture": "must not inherit"}}))
    monkeypatch.setattr(state, "_LEGACY_STATE_PATH", legacy)
    env = api["isolated_env"](tmp_path / "owned")
    monkeypatch.setenv("XDG_STATE_HOME", env["XDG_STATE_HOME"])
    assert json.loads(Path(state.default_state_path()).read_text()) == {}


@pytest.mark.parametrize("url", ["http://user:placeholder@localhost", "http://localhost/?key=example", "file:///tmp/test"])
def test_endpoint_credentials_and_non_http_urls_are_rejected(url):
    with pytest.raises(SmokeError):
        api["safe_endpoint"](url)


def test_two_independent_mcp_clients_share_collection_not_state(tmp_path, monkeypatch):
    import fastmcp
    import fastmcp.client.transports

    transports = []
    def transport(**kwargs):
        target = Path(kwargs["env"]["XDG_STATE_HOME"]) / "mnemostack/server-state.json"
        assert json.loads(target.read_text()) == {}
        transports.append(kwargs)
        return kwargs

    class Client:
        def __init__(self, transport):
            self.transport = transport

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def call_tool(self, name, params):
            assert name == "mnemostack_search" and params["include_trace"]
            return SimpleNamespace(is_error=False, structured_content={"ok": True, "results": results(), "degraded": [], "notes": []})

    monkeypatch.setattr(fastmcp, "Client", Client)
    monkeypatch.setattr(fastmcp.client.transports, "StdioTransport", transport)
    out = asyncio.run(api["mcp_clients"]("mnemostack", ["mcp-serve", "--collection", "test"], {}, tmp_path, 5))
    assert len(out) == len(transports) == 2
    assert transports[0]["args"] == transports[1]["args"]
    assert transports[0]["env"]["XDG_STATE_HOME"] != transports[1]["env"]["XDG_STATE_HOME"]


@pytest.mark.parametrize("scenario", ["success", "missing_model", "existing_collection", "index_failure",
                                    "missing_evidence", "mcp_failure", "cleanup_failure", "repeat_creates_duplicates",
                                    "index_failure_before_creation", "cleanup_probe_failed"])
def test_complete_orchestration_and_scoped_cleanup(tmp_path, monkeypatch, scenario):
    ns = api["verify"].__globals__
    deletes, commands = [], []

    class HTTP:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url):
            status, body = 200, {"version": "fixture"}
            if "/collections/" in url:
                exists = scenario == "existing_collection" or any(c[0] == "index" for c in commands)
                count = 2 if scenario == "repeat_creates_duplicates" and sum(c[0] == "index" for c in commands) >= 2 else 1
                status, body = (200 if exists else 404), {"result": {"points_count": count}}
                if any(c[0] == "index" for c in commands):
                    if scenario == "index_failure_before_creation":
                        status = 404
                    elif scenario == "cleanup_probe_failed":
                        status = 500
            elif url.endswith("/api/tags"):
                body = {"models": [{"name": "nomic-embed-text:latest", "digest": "embed"},
                                   {"name": "llama3.2:3b", "digest": "llm"}]}
                if scenario == "missing_model":
                    body = {"models": []}
            return httpx.Response(status, json=body, request=httpx.Request("GET", url))

        def delete(self, url):
            deletes.append(url)
            return httpx.Response(500 if scenario == "cleanup_failure" else 200, json={}, request=httpx.Request("DELETE", url))

    def cli(command, tokens, root, env, timeout):
        assert command == str(Path("fixture/mnemostack").resolve())
        commands.append(tokens)
        if tokens[0] == "--version":
            return "mnemostack 3.1.0\n", 0.01
        if tokens[0] == "search":
            assert "--require-rerank" in tokens
            return json.dumps([] if scenario == "missing_evidence" else results()), 0.01
        if tokens[0] == "index" and scenario in ("index_failure", "index_failure_before_creation", "cleanup_probe_failed"):
            raise SmokeError("cli_index_failed")
        if tokens[0] == "answer":
            assert "--require-rerank" in tokens
            return "CEDAR-482 SOURCES: workshop-chat.txt", 0.01
        return "index success", 0.01

    async def clients(*args, **kwargs):
        if scenario == "mcp_failure":
            raise SmokeError("mcp_search_degraded")
        return [{"step": label, "ids": ["demo-chunk"], "ok": True} for label in ("client_a", "client_b")]

    monkeypatch.setattr(httpx, "Client", HTTP)
    monkeypatch.setitem(ns, "run_cli", cli)
    monkeypatch.setitem(ns, "mcp_clients", clients)
    monkeypatch.setattr(ns["shutil"], "which", lambda command: "fixture/mnemostack")
    args = SimpleNamespace(qdrant="http://127.0.0.1:6333", command="mnemostack", ollama_host=None,
                           embedding_model=None, llm_model=None, timeout=5, check_answer=True, cleanup=True)
    report = api["verify"](args)
    assert report["status"] == ("PASS" if scenario == "success" else "FAIL")
    if scenario in ("missing_model", "existing_collection", "index_failure_before_creation", "cleanup_probe_failed"):
        assert not deletes
        assert report["cleanup"] == ("failed" if scenario == "cleanup_probe_failed" else "creation_not_confirmed_nothing_deleted")
    else:
        assert deletes == [args.qdrant + "/collections/" + report["collection"]]
        assert report["cleanup"] == ("failed" if scenario == "cleanup_failure" else "deleted_own_test_collection")
    assert report["collection"].startswith("mnemostack_quickstart_")
    assert all(report["collection"] in tokens for tokens in commands if tokens[0] != "--version")


def test_cli_requires_explicit_opt_in(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["verify_quickstart.py", "--output", str(tmp_path / "report.json")])
    with pytest.raises(SystemExit) as caught:
        api["main"]()
    assert caught.value.code == 2
    assert not (tmp_path / "report.json").exists()


def test_existing_report_is_not_overwritten(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    path.write_text("existing artifact")
    monkeypatch.setattr("sys.argv", ["verify_quickstart.py", "--run", "--output", str(path)])
    with pytest.raises(SystemExit) as caught:
        api["main"]()
    assert caught.value.code == 2
    assert path.read_text() == "existing artifact"


def test_report_created_during_run_is_not_overwritten(tmp_path, monkeypatch):
    path = tmp_path / "report.json"

    def concurrent_writer(args):
        path.write_text("other run's completed report")
        return {"status": "PASS"}

    monkeypatch.setitem(api["main"].__globals__, "verify", concurrent_writer)
    monkeypatch.setattr("sys.argv", ["verify_quickstart.py", "--run", "--output", str(path)])
    assert api["main"]() == 1
    assert path.read_text() == "other run's completed report"
