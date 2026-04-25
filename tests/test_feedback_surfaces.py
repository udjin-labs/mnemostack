"""Tests for CLI/MCP feedback surfaces."""
from __future__ import annotations

import argparse
import json

from mnemostack.cli import cmd_feedback
from mnemostack.feedback import apply_feedback
from mnemostack.recall.pipeline import FileStateStore, build_full_pipeline


def test_apply_feedback_updates_file_state(tmp_path):
    state_path = tmp_path / "state.json"
    pipeline = build_full_pipeline(state_store=FileStateStore(state_path), graph_uri=None)

    outcome = apply_feedback(
        pipeline,
        hit_id="mem-1",
        signal="clicked",
        query="how to configure nginx",
        sources=["vector", "bm25", "vector"],
    )

    assert outcome.ok is True
    assert outcome.reward == 0.7
    assert outcome.query_type == "technical"
    assert outcome.ior_recorded is True
    assert outcome.q_learning_updates == 2

    state = json.loads(state_path.read_text())
    assert state["ior_log"][0]["id"] == "mem-1"
    assert state["q_table"]["vector"]["technical"]["n"] == 1
    assert state["q_table"]["bm25"]["technical"]["n"] == 1


def test_cli_feedback_records_state(tmp_path, capsys):
    state_path = tmp_path / "state.json"
    args = argparse.Namespace(
        hit_id="mem-2",
        signal="useful",
        query=None,
        query_type="person",
        source="graph",
        source_list=[],
        reward=None,
        state_path=str(state_path),
        json=False,
    )

    rc = cmd_feedback(args)

    assert rc == 0
    assert "q_updates=1" in capsys.readouterr().out
    state = json.loads(state_path.read_text())
    assert state["q_table"]["graph"]["person"]["n"] == 1
    assert "ior_log" not in state


def test_mcp_server_registers_feedback_tool():
    import asyncio

    import pytest

    pytest.importorskip("fastmcp")

    from mnemostack.mcp import build_server

    mcp = build_server(collection="test", embedding_provider="ollama")
    tools = asyncio.run(mcp.list_tools())

    assert "mnemostack_feedback" in {tool.name for tool in tools}
