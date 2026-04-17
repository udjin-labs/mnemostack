"""Tests for MCP server — verifies build_server wires up tools correctly."""
import asyncio

import pytest

_fastmcp = pytest.importorskip("fastmcp")

from mnemostack.mcp import build_server  # noqa: E402


def _list_tool_names(mcp) -> set[str]:
    """Extract registered tool names from a FastMCP instance."""
    tools = asyncio.run(mcp.list_tools())
    return {t.name for t in tools}


def test_build_server_returns_fastmcp():
    mcp = build_server(collection="test", embedding_provider="ollama")
    assert mcp is not None
    assert mcp.name == "mnemostack"


def test_build_server_registers_core_tools():
    mcp = build_server(collection="test", embedding_provider="ollama")
    names = _list_tool_names(mcp)
    assert "mnemostack_health" in names
    assert "mnemostack_search" in names
    assert "mnemostack_answer" in names


def test_build_server_without_memgraph_skips_graph_tools():
    mcp = build_server(collection="test", embedding_provider="ollama")
    names = _list_tool_names(mcp)
    assert "mnemostack_graph_query" not in names
    assert "mnemostack_graph_add_triple" not in names


def test_build_server_with_memgraph_adds_graph_tools():
    mcp = build_server(
        collection="test",
        embedding_provider="ollama",
        memgraph_uri="bolt://localhost:7687",
    )
    names = _list_tool_names(mcp)
    assert "mnemostack_graph_query" in names
    assert "mnemostack_graph_add_triple" in names
