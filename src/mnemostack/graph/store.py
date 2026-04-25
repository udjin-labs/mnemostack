"""Memgraph/Neo4j wrapper with temporal validity."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable

    _AVAILABLE = True
except ImportError:  # pragma: no cover
    _AVAILABLE = False


@dataclass
class Triple:
    """A temporal fact in the graph.

    subject and obj are node names (nodes are created on demand).
    predicate is the relationship type.
    valid_from / valid_until are ISO date strings. The marker "current" means
    still valid.
    """

    subject: str
    predicate: str
    obj: str
    valid_from: str | None = None
    valid_until: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)


def _to_iso(value: str | date | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


class GraphStore:
    """Connection wrapper for Memgraph/Neo4j.

    Uses generic Entity label for all nodes so Memgraph's Cypher compatibility
    works out of the box. Callers can add additional labels via `add_label`.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "",
        password: str = "",
        database: str | None = None,
        timeout: float = 5.0,
    ):
        if not _AVAILABLE:
            raise ImportError(
                "GraphStore requires neo4j driver (already in core deps)"
            )
        self.uri = uri
        self.database = database
        self.timeout = timeout
        auth = (user, password) if user else None
        self.driver = GraphDatabase.driver(
            uri,
            auth=auth,
            connection_timeout=timeout,
            connection_acquisition_timeout=timeout,
        )

    def close(self) -> None:
        self.driver.close()

    def health_check(self) -> tuple[bool, str]:
        try:
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1").single()
            return True, "ok"
        except ServiceUnavailable as e:  # pragma: no cover
            return False, f"unreachable: {e}"
        except Exception as e:  # noqa: BLE001  # pragma: no cover
            return False, str(e)

    # ---------- write ----------

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_from: str | date | datetime | None = None,
        valid_until: str | date | datetime | None = None,
        subject_label: str = "Entity",
        obj_label: str = "Entity",
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Add a temporal fact. Nodes are created on demand."""
        props = {
            "valid_from": _to_iso(valid_from),
            "valid_until": _to_iso(valid_until) or "current",
            **(properties or {}),
        }
        # Filter out None values (neo4j property can be null but cleaner to omit)
        props = {k: v for k, v in props.items() if v is not None}
        # Sanitize labels and relationship type (only alphanumerics + underscore)
        s_label = self._safe_label(subject_label)
        o_label = self._safe_label(obj_label)
        rel = self._safe_rel(predicate)
        query = (
            f"MERGE (s:{s_label} {{name: $subject}}) "
            f"MERGE (o:{o_label} {{name: $obj}}) "
            f"SET s.valid_until = coalesce(s.valid_until, 'current'), "
            f"    o.valid_until = coalesce(o.valid_until, 'current') "
            f"MERGE (s)-[r:{rel}]->(o) "
            f"SET r += $props"
        )
        with self.driver.session(database=self.database) as session:
            session.run(query, subject=subject, obj=obj, props=props)

    def invalidate(
        self,
        subject: str,
        predicate: str,
        obj: str,
        ended: str | date | datetime,
    ) -> int:
        """Mark a fact as no longer valid. Returns number of edges updated."""
        rel = self._safe_rel(predicate)
        query = (
            f"MATCH (s {{name: $subject}})-[r:{rel}]->(o {{name: $obj}}) "
            f"WHERE r.valid_until = 'current' OR r.valid_until IS NULL "
            f"SET r.valid_until = $ended "
            f"RETURN count(r) AS n"
        )
        with self.driver.session(database=self.database) as session:
            rec = session.run(
                query, subject=subject, obj=obj, ended=_to_iso(ended)
            ).single()
            return rec["n"] if rec else 0

    # ---------- read ----------

    def query_triples(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
        as_of: str | date | datetime | None = None,
        limit: int = 100,
    ) -> list[Triple]:
        """Query triples with optional SPO filters and point-in-time constraint.

        If `as_of` is provided, returns only facts valid at that date
        (valid_from <= as_of < valid_until, with valid_until "current" treated
        as indefinite future). Legacy NULL markers are also treated as current.
        """
        where_parts = []
        params: dict[str, Any] = {"limit": limit}
        if subject:
            where_parts.append("s.name = $subject")
            params["subject"] = subject
        if obj:
            where_parts.append("o.name = $obj")
            params["obj"] = obj
        if as_of:
            params["as_of"] = _to_iso(as_of)
            where_parts.append(
                "(r.valid_from IS NULL OR r.valid_from <= $as_of) "
                "AND (r.valid_until = 'current' OR r.valid_until IS NULL OR r.valid_until > $as_of)"
            )

        rel_pattern = f":{self._safe_rel(predicate)}" if predicate else ""
        where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
        query = (
            f"MATCH (s)-[r{rel_pattern}]->(o) "
            f"{where_clause} "
            f"RETURN s.name AS subject, type(r) AS predicate, o.name AS obj, "
            f"r.valid_from AS valid_from, r.valid_until AS valid_until, "
            f"properties(r) AS props "
            f"LIMIT $limit"
        )

        triples = []
        with self.driver.session(database=self.database) as session:
            for rec in session.run(query, **params):
                props = dict(rec["props"] or {})
                props.pop("valid_from", None)
                props.pop("valid_until", None)
                triples.append(
                    Triple(
                        subject=rec["subject"],
                        predicate=rec["predicate"],
                        obj=rec["obj"],
                        valid_from=rec["valid_from"],
                        valid_until=rec["valid_until"],
                        properties=props,
                    )
                )
        return triples

    def neighbors(
        self,
        node: str,
        as_of: str | date | datetime | None = None,
        limit: int = 50,
    ) -> list[Triple]:
        """All outgoing edges from a node, optionally filtered by point-in-time."""
        return self.query_triples(subject=node, as_of=as_of, limit=limit)

    def count_nodes(self) -> int:
        with self.driver.session(database=self.database) as session:
            rec = session.run("MATCH (n) RETURN count(n) AS n").single()
            return rec["n"] if rec else 0

    def count_edges(self) -> int:
        with self.driver.session(database=self.database) as session:
            rec = session.run("MATCH ()-[r]->() RETURN count(r) AS n").single()
            return rec["n"] if rec else 0

    def backfill_current_markers(self, dry_run: bool = False) -> dict[str, int]:
        """Backfill legacy NULL validity markers to the explicit "current" marker."""
        with self.driver.session(database=self.database) as session:
            if dry_run:
                nodes = session.run(
                    "MATCH (n) WHERE n.valid_until IS NULL RETURN count(n) AS n"
                ).single()
                rels = session.run(
                    "MATCH ()-[r]->() WHERE r.valid_until IS NULL RETURN count(r) AS n"
                ).single()
            else:
                nodes = session.run(
                    "MATCH (n) WHERE n.valid_until IS NULL "
                    "SET n.valid_until = 'current' RETURN count(n) AS n"
                ).single()
                rels = session.run(
                    "MATCH ()-[r]->() WHERE r.valid_until IS NULL "
                    "SET r.valid_until = 'current' RETURN count(r) AS n"
                ).single()
        return {
            "nodes": int(nodes["n"] if nodes else 0),
            "relationships": int(rels["n"] if rels else 0),
        }

    # ---------- helpers ----------

    @staticmethod
    def _safe_rel(predicate: str) -> str:
        """Sanitize predicate for use as Cypher relationship type."""
        # Allow letters, digits, underscore; uppercase for Neo4j convention
        cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in predicate)
        if not cleaned or not cleaned[0].isalpha():
            cleaned = "_" + cleaned
        return cleaned.upper()

    @staticmethod
    def _safe_label(label: str) -> str:
        cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in label)
        if not cleaned or not cleaned[0].isalpha():
            cleaned = "Node"
        return cleaned

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
