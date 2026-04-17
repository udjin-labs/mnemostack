"""
Knowledge graph (Memgraph / Neo4j) with temporal validity.

Facts have valid_from and valid_until timestamps — you can query
point-in-time state:

    graph.query_at("X works_on Y", as_of="2024-06-01")

Invalidating a fact means setting valid_until, not deleting it:

    graph.invalidate(subject="alice", predicate="works_on", obj="project-X",
                     ended="2024-06-15")

Usage:
    from mnemostack.graph import GraphStore
    store = GraphStore(uri="bolt://localhost:7687", user="", password="")
    store.add_triple("alice", "works_on", "project-X", valid_from="2024-01-01")
"""

from .extractor import TripleExtractor
from .store import GraphStore, Triple

__all__ = ["GraphStore", "Triple", "TripleExtractor"]
