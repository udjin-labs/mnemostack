"""CLI failure visibility and exact runtime-option forwarding without services."""
import argparse
import ast
import copy
import inspect
import os
from types import SimpleNamespace

import pytest

import mnemostack.cli as cli
from mnemostack.recall.pipeline import default_state_path
from mnemostack.server import _env_bool


def _config_argument(name):
    tree = ast.parse(inspect.getsource(cli.cmd_serve))
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "ServerConfig")
    return copy.deepcopy(next(k.value for k in call.keywords if k.arg == name))


@pytest.mark.parametrize("flag,value,expected", [(False, "1", True), (False, "true", True), (False, "0", False), (True, "0", True)])
def test_serve_auto_ior_forwards_flag_or_environment(monkeypatch, flag, value, expected):
    monkeypatch.setenv("MNEMOSTACK_AUTO_RECORD_IOR", value)
    result = eval(compile(ast.fix_missing_locations(ast.Expression(_config_argument("auto_record_ior"))), "actual-server-config-option", "eval"), {"args": SimpleNamespace(auto_record_ior=flag), "_env_bool": _env_bool})
    assert result is expected


@pytest.mark.parametrize("explicit", [False, True])
def test_all_state_flags_use_environment_unless_explicit(monkeypatch, tmp_path, explicit):
    selected = str(tmp_path / "environment.json")
    override = str(tmp_path / "explicit.json")
    monkeypatch.setenv("MNEMOSTACK_STATE_PATH", selected)
    tree = ast.parse(inspect.getsource(cli.build_parser))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument" and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--state-path"]
    # Tenant removal has an intentional None sentinel,not a shared default.
    calls = [call for call in calls if not any(k.arg == "default" and isinstance(k.value, ast.Constant) and k.value.value is None for k in call.keywords)]
    assert len(calls) == 3
    for call in calls:
        parser = argparse.ArgumentParser(add_help=False)
        env = {call.func.value.id: parser, "os": os, "default_state_path": default_state_path}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(copy.deepcopy(call))], type_ignores=[])), "actual-state-flag", "exec"), env)
        args = parser.parse_args(["--state-path", override] if explicit else [])
        assert str(args.state_path) == (override if explicit else selected)


def test_state_flags_without_override_keep_default(monkeypatch):
    monkeypatch.delenv("MNEMOSTACK_STATE_PATH", raising=False)
    tree = ast.parse(inspect.getsource(cli.build_parser))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument" and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--state-path"]
    for call in calls:
        default = next(k.value for k in call.keywords if k.arg == "default")
        if isinstance(default, ast.Constant) and default.value is None:
            continue
        value = eval(compile(ast.fix_missing_locations(ast.Expression(copy.deepcopy(default))), "actual-state-default", "eval"), {"os": os, "default_state_path": default_state_path})
        assert value == default_state_path()


def _patch_recall(monkeypatch, tmp_path, *, fail=True):
    calls = []
    rows = [object()]
    for name, value in {"_parse_filters": lambda args: {"project": "p"}, "_effective_token_budget": lambda args: None, "_payload_schema": lambda: ("text", "timestamp", "iso"), "FileStateStore": lambda *a: object(), "default_state_path": lambda: tmp_path / "state.json", "build_full_pipeline": lambda **k: object(), "resolve_access_bonus_max": lambda v: 0.25, "_graph_auth": lambda args: {}, "_llm_build_kwargs": lambda *a: {}, "Reranker": lambda **k: object()}.items():
        monkeypatch.setattr(cli, name, value)

    def get_llm(*args, **kwargs):
        if fail:
            raise RuntimeError("PRIVATE_SYNTHETIC_MARKER")
        return object()

    def flow(*args, **kwargs):
        calls.append(kwargs)
        return rows

    monkeypatch.setattr(cli, "get_llm", get_llm)
    monkeypatch.setattr(cli, "recall_flow", flow)
    return calls, rows


def test_optional_reranker_failure_warns_on_stderr_not_stdout(monkeypatch, tmp_path, capsys):
    calls, rows = _patch_recall(monkeypatch, tmp_path)
    result = cli._recall_for_cli(SimpleNamespace(), object(), "query", 3)
    output = capsys.readouterr()
    assert result is rows and len(calls) == 1
    assert calls[0]["reranker"] is None and calls[0]["filters"] == {"project": "p"}
    assert "reranker unavailable" in output.err and output.out == ""
    assert "PRIVATE_SYNTHETIC_MARKER" not in output.err


def test_required_reranker_failure_still_stops_before_recall(monkeypatch, tmp_path, capsys):
    calls, _ = _patch_recall(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli._recall_for_cli(SimpleNamespace(require_rerank=True), object(), "query", 3)
    assert exc.value.code == 2 and not calls
    assert "--require-rerank" in capsys.readouterr().err


def test_healthy_reranker_path_remains_quiet(monkeypatch, tmp_path, capsys):
    calls, rows = _patch_recall(monkeypatch, tmp_path, fail=False)
    assert cli._recall_for_cli(SimpleNamespace(), object(), "query", 3) is rows
    assert calls[0]["reranker"] is not None
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("override", [None, "", "selected-state.json"])
def test_cli_recall_reads_the_same_state_path_as_feedback(monkeypatch, tmp_path, override):
    calls, rows = _patch_recall(monkeypatch, tmp_path, fail=False)
    if override is None:
        monkeypatch.delenv("MNEMOSTACK_STATE_PATH", raising=False)
    else:
        monkeypatch.setenv("MNEMOSTACK_STATE_PATH", str(tmp_path / override) if override else "")
    state_paths = []
    monkeypatch.setattr(cli, "FileStateStore", lambda path: state_paths.append(str(path)) or object())
    assert cli._recall_for_cli(SimpleNamespace(), object(), "query", 3) is rows
    assert len(calls) == 1
    tree = ast.parse(inspect.getsource(cli.build_parser))
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "p_feedback" and n.func.attr == "add_argument" and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--state-path")
    parser = argparse.ArgumentParser(add_help=False)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(copy.deepcopy(call))], type_ignores=[])), "actual-feedback-state-flag", "exec"), {"p_feedback": parser, "os": os, "default_state_path": cli.default_state_path})
    assert state_paths == [str(parser.parse_args([]).state_path)]
