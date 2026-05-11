"""Tests for MCP server — verifies build_server wires up tools correctly."""
import asyncio

import pytest
from pydantic import ValidationError

_fastmcp = pytest.importorskip("fastmcp")

from mnemostack.mcp import build_server  # noqa: E402
from mnemostack.mcp.server import (  # noqa: E402
    FeedbackInput,
    GraphAddTripleInput,
    GraphQueryInput,
    SearchInput,
    _validate_tool_input,
    _validation_error,
)


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


def test_mcp_search_input_accepts_dict_and_typed_input():
    params = _validate_tool_input(SearchInput, {"query": "needle", "limit": 3})
    assert params.query == "needle"
    assert params.limit == 3

    same = _validate_tool_input(SearchInput, params)
    assert same is params


def test_mcp_validation_rejects_negative_limit_with_clear_message():
    with pytest.raises(Exception) as exc:
        _validate_tool_input(SearchInput, {"query": "needle", "limit": -1})

    payload = _validation_error(exc.value)
    assert payload["ok"] is False
    assert "limit" in payload["error"]
    assert "greater than or equal to 1" in payload["error"]


def test_mcp_validation_rejects_invalid_feedback_signal():
    with pytest.raises(Exception) as exc:
        _validate_tool_input(FeedbackInput, {"hit_id": "h1", "signal": "bad"})

    payload = _validation_error(exc.value)
    assert payload["ok"] is False
    assert "signal" in payload["error"]


def test_mcp_graph_models_validate_limits_and_required_fields():
    query = _validate_tool_input(GraphQueryInput, {"limit": 50})
    assert query.limit == 50

    triple = _validate_tool_input(
        GraphAddTripleInput,
        {"subject": "a", "predicate": "knows", "obj": "b"},
    )
    assert triple.subject == "a"

    with pytest.raises(ValidationError):
        _validate_tool_input(GraphQueryInput, {"limit": 0})
