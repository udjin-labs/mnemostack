"""
MCP (Model Context Protocol) server for mnemostack.

Exposes memory stack tools to MCP-compatible clients (Claude Desktop,
ChatGPT, Cursor, etc.). Clients call tools via stdio JSON-RPC.

Requirements: install mnemostack with MCP extras
    pip install 'mnemostack[mcp]'

Usage:
    # Start server (stdio transport for desktop clients)
    python -m mnemostack.mcp.server

    # Or via CLI
    mnemostack mcp-serve

Tools exposed:
    - mnemostack_health — check all components
    - mnemostack_search — hybrid recall (BM25 + vector + RRF)
    - mnemostack_answer — inference answer with confidence
    - mnemostack_graph_query — query knowledge graph (optional, requires Memgraph)
    - mnemostack_graph_add_triple — add a fact to the graph
"""

from .server import build_server

__all__ = ["build_server"]
