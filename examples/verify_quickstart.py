"""Opt-in verification of the published local quickstart, not a quality benchmark."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import httpx

CODE, MESSAGE_ID, SOURCE = "CEDAR-482", "demo-001", "workshop-chat.txt"
ALLOWED_FLAGS = {"--provider", "--embedding-model", "--ollama-host", "--qdrant", "--collection",
                 "--llm", "--llm-model", "--rerank-mode", "--json", "--transport"}


class SmokeError(RuntimeError):
    """A safe fixed failure code; never print model or process response bodies."""


def walkthrough(path):
    text = path.read_text()
    commands = {}
    for block in re.findall(r"```bash\n(.*?)```", text, re.S):
        for line in block.replace("\\\n", " ").splitlines():
            if line.startswith("mnemostack "):
                tokens = shlex.split(line)
                if len(tokens) < 3 or tokens[1] not in ("index", "search", "answer") or tokens[1] in commands:
                    raise SmokeError("unexpected_document_command")
                commands[tokens[1]] = tokens[1:]
    if set(commands) != {"index", "search", "answer"}:
        raise SmokeError("missing_document_commands")
    block = re.search(r"```json\n(.*?)```", text, re.S)
    if block is None:
        raise SmokeError("missing_mcp_configuration")
    mcp = json.loads(block.group(1))["mcpServers"]["mnemostack"]
    commands["mcp"] = mcp["args"]
    for tokens in commands.values():
        if not isinstance(tokens, list) or any(not isinstance(t, str) for t in tokens):
            raise SmokeError("invalid_document_arguments")
        if any(t.startswith("-") and t not in ALLOWED_FLAGS for t in tokens):
            raise SmokeError("unreviewed_document_flag")
        flags = [t for t in tokens if t.startswith("-")]
        if len(flags) != len(set(flags)):
            raise SmokeError("duplicate_document_flag")
    fixture = re.search(r"cat > demo-notes/workshop-chat\.txt <<'MESSAGE'\n(.*?)\nMESSAGE", text, re.S)
    if fixture is None or any(value not in fixture.group(1) for value in (CODE, MESSAGE_ID)):
        raise SmokeError("documentation_fixture_changed")
    if commands["index"][1] != "./demo-notes/" or commands["mcp"][0] != "mcp-serve":
        raise SmokeError("unexpected_document_target")
    return {"commands": commands, "fixture": fixture.group(1) + "\n",
            "sha256": hashlib.sha256(text.encode()).hexdigest()}


def replace_flag(tokens, flag, value):
    out = list(tokens)
    if value is None:
        return out
    if flag not in out or out.index(flag) + 1 >= len(out):
        raise SmokeError("required_document_flag_missing")
    out[out.index(flag) + 1] = value
    return out


def flag_value(tokens, flag):
    if flag not in tokens or tokens.index(flag) + 1 >= len(tokens):
        raise SmokeError("required_document_flag_missing")
    return tokens[tokens.index(flag) + 1]


def safe_endpoint(value):
    url = urlsplit(value)
    if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password or url.query or url.fragment or url.path not in ("", "/"):
        raise SmokeError("endpoint_must_be_a_plain_http_origin_without_credentials")
    return value.rstrip("/")


def evidence(results):
    if not isinstance(results, list):
        raise SmokeError("invalid_search_response")
    matching = [str(r["id"]) for r in results if CODE in r.get("text", "")
                and MESSAGE_ID in r.get("text", "") and r.get("payload", {}).get("source") == SOURCE]
    if not matching:
        raise SmokeError("expected_fact_source_or_message_id_missing")
    return matching


def mcp_evidence(raw):
    if not isinstance(raw, dict) or not raw.get("ok"):
        raise SmokeError("mcp_search_failed")
    faults = sorted(set(raw.get("degraded", [])) - set(raw.get("notes", [])))
    if faults:
        raise SmokeError("mcp_search_degraded")
    return evidence(raw.get("results"))


def initialize_state(base):
    # Prevent default_state_path() from migrating a shared legacy /tmp file.
    target = Path(base) / "mnemostack/server-state.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        stream.write("{}\n")


def isolated_env(root):
    env = {key: value for key, value in os.environ.items() if not key.startswith("MNEMOSTACK_")}
    env.update(MNEMOSTACK_CONFIG=str(root / "empty-config.yaml"), MNEMOSTACK_GRAPH_URI="",
               XDG_CONFIG_HOME=str(root / "config"), XDG_STATE_HOME=str(root / "state"),
               PYTHONDONTWRITEBYTECODE="1")
    initialize_state(env["XDG_STATE_HOME"])
    return env


async def mcp_clients(command, args, env, cwd, timeout):
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    results = []
    for label in ("client_a", "client_b"):
        client_env = {**env, "XDG_STATE_HOME": str(cwd / label)}
        initialize_state(client_env["XDG_STATE_HOME"])
        transport = StdioTransport(command=command, args=args, env=client_env, cwd=str(cwd),
                                   keep_alive=False, log_file=cwd / (label + ".stderr"))
        started = time.perf_counter()

        async def call(transport=transport):
            async with Client(transport) as client:
                response = await client.call_tool("mnemostack_search", {
                    "query": "What is the equipment pickup code for the Cedar workshop?", "include_trace": True})
                if response.is_error:
                    raise SmokeError("mcp_tool_error")
                return mcp_evidence(response.structured_content)

        ids = await asyncio.wait_for(call(), timeout=timeout)
        results.append({"step": label, "ids": ids, "seconds": time.perf_counter() - started, "ok": True})
    return results


def run_cli(command, tokens, root, env, timeout):
    started = time.perf_counter()
    result = subprocess.run([command, *tokens], cwd=root, env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise SmokeError("cli_" + tokens[0] + "_failed")
    if re.search(r"rerank[^\n]*(?:fail|no usable|fallback)", result.stderr, re.I):
        raise SmokeError("cli_reranker_fallback")
    return result.stdout, time.perf_counter() - started


def verify(args):
    guide = walkthrough(Path(__file__).with_name("quickstart.md"))
    qdrant = safe_endpoint(args.qdrant)
    collection = "mnemostack_quickstart_" + uuid.uuid4().hex
    report = {"report_version": 1, "status": "FAIL", "collection": collection, "steps": [],
              "walkthrough_sha256": guide["sha256"], "agent_ui": "NOT_RUN", "remote_clients": "NOT_RUN",
              "model_overrides": bool(args.embedding_model or args.llm_model), "cleanup": "not_requested"}
    command = shutil.which(args.command)
    if not command:
        raise SmokeError("mnemostack_executable_not_found")
    command = str(Path(command).resolve())
    commands = {}
    for name, tokens in guide["commands"].items():
        tokens = replace_flag(tokens, "--qdrant", qdrant)
        tokens = replace_flag(tokens, "--collection", collection)
        tokens = replace_flag(tokens, "--ollama-host", args.ollama_host)
        tokens = replace_flag(tokens, "--embedding-model", args.embedding_model)
        if name != "index":
            tokens = replace_flag(tokens, "--llm-model", args.llm_model)
        # Require runtime evidence from the CLI's canonical recall trace;
        # stderr alone cannot detect silently unavailable reranking.
        commands[name] = tokens + ["--require-rerank"] if name in ("search", "answer") else tokens
    model_host = safe_endpoint(flag_value(commands["search"], "--ollama-host"))
    embedding_model = flag_value(commands["search"], "--embedding-model")
    llm_model = flag_value(commands["search"], "--llm-model")
    created, index_attempted = False, False
    with httpx.Client(trust_env=False, timeout=10) as http, tempfile.TemporaryDirectory(prefix="mnemostack-quickstart-") as temporary:
        root = Path(temporary)
        try:
            check = http.get(qdrant + "/collections/" + collection)
            if check.status_code != 404:
                raise SmokeError("test_collection_absence_not_confirmed")
            models = http.get(model_host + "/api/tags").raise_for_status().json()["models"]
            selected = {}
            for model in (embedding_model, llm_model):
                entry = next((m for m in models if m["name"] in (model, model + ":latest")), None)
                if entry is None:
                    raise SmokeError("required_model_not_installed")
                selected[model] = entry["digest"]
            report["model_digests"] = selected
            (root / "empty-config.yaml").write_text("{}\n")
            (root / "demo-notes").mkdir()
            (root / "demo-notes" / SOURCE).write_text(guide["fixture"])
            env = isolated_env(root)
            version, _ = run_cli(command, ["--version"], root, env, args.timeout)
            if not re.fullmatch(r"mnemostack [0-9][0-9A-Za-z.+-]*\s*", version):
                raise SmokeError("unexpected_cli_version_response")
            report["versions"] = {
                "mnemostack_cli": version.strip(),
                "qdrant": http.get(qdrant + "/").raise_for_status().json().get("version"),
                "ollama": http.get(model_host + "/api/version").raise_for_status().json().get("version"),
            }
            index_attempted = True
            _, seconds = run_cli(command, commands["index"], root, env, args.timeout)
            created = True
            report["steps"].append({"step": "index", "ok": True, "seconds": seconds})
            raw, seconds = run_cli(command, commands["search"], root, env, args.timeout)
            ids = evidence(json.loads(raw))
            report["steps"].append({"step": "cli_search", "ok": True, "seconds": seconds, "ids": ids, "reranker_required": True})
            count_before = http.get(qdrant + "/collections/" + collection).raise_for_status().json()["result"]["points_count"]
            _, seconds = run_cli(command, commands["index"], root, env, args.timeout)
            count_after = http.get(qdrant + "/collections/" + collection).raise_for_status().json()["result"]["points_count"]
            if count_before < 1 or count_after != count_before:
                raise SmokeError("repeat_index_changed_record_count")
            report["steps"].append({"step": "index_repeat", "ok": True, "seconds": seconds,
                                    "points_before": count_before, "points_after": count_after})
            if args.check_answer:
                answer, seconds = run_cli(command, commands["answer"], root, env, args.timeout)
                if CODE not in answer or SOURCE not in answer:
                    raise SmokeError("answer_evidence_missing")
                report["steps"].append({"step": "cli_answer", "ok": True, "seconds": seconds, "reranker_required": True})
            clients = asyncio.run(mcp_clients(command, commands["mcp"], env, root, args.timeout))
            if any(set(client["ids"]) != set(ids) for client in clients):
                raise SmokeError("clients_did_not_read_the_same_indexed_record")
            report["steps"].extend(clients)
            report["status"] = "PASS"
        except Exception as exc:
            report["failure"] = str(exc) if isinstance(exc, SmokeError) else type(exc).__name__
        finally:
            if args.cleanup:
                try:
                    # The name was absent before our index attempt and is a
                    # fresh UUID. Index can create it before failing to embed.
                    if index_attempted and not created:
                        probe = http.get(qdrant + "/collections/" + collection)
                        if probe.status_code != 404:
                            probe.raise_for_status()
                            created = True
                    if created:
                        http.delete(qdrant + "/collections/" + collection).raise_for_status()
                        report["cleanup"] = "deleted_own_test_collection"
                    else:
                        report["cleanup"] = "creation_not_confirmed_nothing_deleted"
                except httpx.HTTPError:
                    report["cleanup"] = "failed"
                    report["status"] = "FAIL"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="permit local model calls and create a unique test collection")
    parser.add_argument("--command", default="mnemostack")
    parser.add_argument("--qdrant", default="http://127.0.0.1:6333")
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--check-answer", action="store_true")
    parser.add_argument("--cleanup", action="store_true", help="delete only this run's confirmed new test collection")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required; this check uses models and writes a synthetic test collection")
    if not 0 < args.timeout <= 600:
        parser.error("timeout must be in (0, 600]")
    output = Path(args.output)
    if output.exists() or not output.parent.is_dir():
        parser.error("choose a new output file in an existing directory")
    try:
        report = verify(args)
    except Exception as exc:
        report = {"status": "FAIL", "failure": str(exc) if isinstance(exc, SmokeError) else type(exc).__name__}
    try:
        with output.open("x") as stream:
            stream.write(json.dumps(report, indent=2))
    except OSError as exc:
        print(json.dumps({"status": "FAIL", "failure": "report_write_failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps(report))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
