"""Shared construction of a graph connection from resolved config.

Every CLI / MCP / HTTP entry point builds its ``GraphStore`` through
:func:`make_graph_store`, so ``graph.user`` / ``graph.password`` /
``graph.database`` reach every graph write path uniformly. Without this a
Memgraph/Neo4j deployment that requires auth or a non-default database is
unusable from the CLI and MCP even though ``GraphStore`` itself supports it.
"""

from __future__ import annotations

from typing import Any


def make_graph_store(
    uri: str,
    *,
    timeout: float = 5.0,
    user: str = "",
    password: str = "",
    database: str | None = None,
) -> Any:
    """Build a ``GraphStore`` with auth threaded from config.

    The one place graph auth is mapped onto the ``GraphStore`` constructor.
    Imported lazily so importing this module never pulls in the neo4j driver.
    """
    from .store import GraphStore

    return GraphStore(
        uri=uri, user=user, password=password, database=database, timeout=timeout
    )
