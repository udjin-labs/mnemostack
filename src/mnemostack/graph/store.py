"""Memgraph/Neo4j wrapper with temporal validity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from ..recall.validity import graph_as_of_predicate, to_utc_instant, to_utc_iso

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


#: Server-owned graph property carrying the tenant a node/edge belongs to.
#: Mirrors the vector store's ``tenant_id`` payload key so a graph result can be
#: stamped and survive the recall ``filter_by_tenant`` backstop.
TENANT_KEY = "tenant"


def _to_iso(value: str | date | datetime | None) -> str | None:
    """ISO-8601 string for a validity bound, UTC-normalized.

    Validity predicates compare these as raw strings in Cypher, so a
    timezone-bearing instant is canonicalized to UTC (via ``to_utc_iso``) on
    write; bare dates and the ``current`` marker pass through unchanged. This
    keeps stored bounds comparable with a UTC-normalized ``as_of`` at query
    time. Pre-existing offset-bearing data would need the same normalization.
    """
    if value is None:
        return None
    text = value.isoformat() if isinstance(value, (date, datetime)) else str(value)
    return to_utc_iso(text)


class GraphStore:
    """Connection wrapper for Memgraph/Neo4j.

    Uses generic Entity label for all nodes by default so Memgraph's Cypher
    compatibility works out of the box. Callers can choose labels per write via
    `subject_label` and `obj_label`.
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
            raise ImportError("GraphStore requires neo4j driver (already in core deps)")
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
        *,
        tenant: str | None = None,
    ) -> None:
        """Add a temporal fact. Nodes are created on demand.

        With ``tenant`` set, the ``tenant`` property is folded into the node
        MERGE key and stamped on both nodes and the edge, so two tenants writing
        the same ``(subject, predicate, obj)`` get distinct, isolated nodes — a
        scoped read (``tenant=`` on the query side) can never reach another
        tenant's node. With ``tenant=None`` the Cypher is byte-for-byte the
        legacy single-tenant form, so existing graphs are untouched.
        """
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
        tk = self._tenant_key_frag(tenant)  # ", tenant: $tenant" when scoped
        params: dict[str, Any] = {"subject": subject, "obj": obj, "props": props}
        if tenant is not None:
            props[TENANT_KEY] = tenant  # stamp the edge too
            params["tenant"] = tenant
            set_tenant = ", s.tenant = $tenant, o.tenant = $tenant"
            # Scoped: fold the tenant into the relationship MERGE key so a scoped
            # write only ever matches/creates ITS OWN edge — it never reuses (and
            # then relabels) a foreign-tenant/untenanted edge that happens to sit
            # between these nodes in a partially migrated or hand-edited graph.
            edge_merge = f"MERGE (s)-[r:{rel} {{tenant: $tenant}}]->(o) "
            edge_set = "SET r += $props"
        else:
            set_tenant = ""
            # Unscoped: a name-only MERGE can subset-match a tenant-owned node and
            # its edge, so only write props on a tenant-LESS edge — an unscoped
            # add_triple must never overwrite a tenant's edge validity/properties.
            # On a single-tenant graph every edge is tenant-less, so this always
            # runs (equivalent to the legacy unconditional SET). The node SETs use
            # coalesce, so they're idempotent even if they bind a tenant node.
            edge_merge = f"MERGE (s)-[r:{rel}]->(o) "
            edge_set = (
                "FOREACH (_ IN CASE WHEN r.tenant IS NULL THEN [1] ELSE [] END | SET r += $props)"
            )
        query = (
            f"MERGE (s:{s_label} {{name: $subject{tk}}}) "
            f"MERGE (o:{o_label} {{name: $obj{tk}}}) "
            f"SET s.valid_until = coalesce(s.valid_until, 'current'), "
            f"    o.valid_until = coalesce(o.valid_until, 'current'){set_tenant} "
            f"{edge_merge}"
            f"{edge_set}"
        )
        with self.driver.session(database=self.database) as session:
            session.run(query, **params)

    def sync_file_links(
        self,
        source: str,
        targets: list[str],
        *,
        index_root: str | None = None,
        tenant: str | None = None,
    ) -> int:
        """Replace a file's outgoing ``LINKS_TO`` edges with ``targets``.

        Deletes the file's existing ``LINKS_TO`` relationships first, then
        re-creates one per target — so a re-index accurately reflects the
        current links (links removed from the file are dropped, not left
        dangling). ``:File`` nodes are keyed by ``(name, index_root)`` so two
        corpora that share a relative filename (both have ``index.md``) don't
        collide: re-indexing one root never touches the other's edges. Returns
        the number of edges written.

        Each node also gets a Python-lowercased ``name_lower`` so
        ``MemgraphRetriever.search`` (which probes
        ``coalesce(n.name_lower, toLower(n.name))``) can find files with
        non-ASCII names — Memgraph's ``toLower`` only folds ASCII.
        """
        root = index_root or ""
        # One managed write transaction: the delete + re-creates commit together
        # or roll back together, so a mid-way graph failure never leaves the file
        # with its edges deleted-but-not-recreated (silent link loss). The driver
        # also auto-retries the whole transaction on a transient error.
        with self.driver.session(database=self.database) as session:
            session.execute_write(
                self._sync_file_links_tx, source, list(targets), root, tenant
            )
        return len(targets)

    @staticmethod
    def _sync_file_links_tx(
        tx: Any, source: str, targets: list[str], root: str, tenant: str | None
    ) -> None:
        # The ``tenant`` property joins ``(name, index_root)`` in the :File key
        # when scoped, so a tenant's link graph is isolated exactly like its
        # nodes. Unscoped (tenant=None) keeps the legacy key untouched.
        tk = GraphStore._tenant_key_frag(tenant)
        set_tenant = ", s.tenant = $tenant, o.tenant = $tenant" if tenant is not None else ""
        common = {"root": root}
        if tenant is not None:
            common["tenant"] = tenant
        # Confine the edge-clearing DELETE. When scoped, pin the far end AND the
        # edge to the tenant. When UNSCOPED, gate on the SOURCE NODE's tenant
        # (`s.tenant IS NULL`), not the edge's: a :File property map subset-matches
        # a tenant-stamped node, so an unscoped re-index (index-markdown, no
        # --tenant) must not clear a *migrated* file's edges — but it must still
        # clear a genuine single-tenant file's stale links even if those edges
        # carry a legacy `tenant` property of their own. On an unmigrated graph the
        # source node has no tenant, so this deletes exactly as before.
        if tenant is not None:
            del_target = "{tenant: $tenant}"
            del_where = " WHERE r.tenant = $tenant"
        else:
            del_target = ""
            del_where = " WHERE s.tenant IS NULL"
        tx.run(
            f"MATCH (s:File {{name: $src, index_root: $root{tk}}})"
            f"-[r:LINKS_TO]->({del_target}){del_where} DELETE r",
            src=source,
            **common,
        )
        # Fold the tenant into the LINKS_TO MERGE key when scoped, so a scoped
        # resync only matches/creates its own edge and never reuses (then relabels)
        # a foreign-tenant/untenanted edge between these nodes — mirroring the DELETE
        # guard above, which deliberately leaves such an edge alone.
        rel_merge = (
            "MERGE (s)-[r:LINKS_TO {tenant: $tenant}]->(o) "
            if tenant is not None
            else "MERGE (s)-[r:LINKS_TO]->(o) "
        )
        for target in targets:
            tx.run(
                f"MERGE (s:File {{name: $src, index_root: $root{tk}}}) "
                f"MERGE (o:File {{name: $dst, index_root: $root{tk}}}) "
                f"SET s.valid_until = coalesce(s.valid_until, 'current'), "
                f"    o.valid_until = coalesce(o.valid_until, 'current'), "
                f"    s.name_lower = $src_lower, o.name_lower = $dst_lower{set_tenant} "
                f"{rel_merge}"
                f"SET r.valid_until = coalesce(r.valid_until, 'current')",
                src=source,
                dst=target,
                src_lower=source.lower(),
                dst_lower=target.lower(),
                **common,
            )

    def referrers_of_dangling(
        self,
        name_keys: list[str],
        *,
        index_root: str | None = None,
        tenant: str | None = None,
    ) -> list[str]:
        """Sources linking to a dangling ``:File`` node a new note now satisfies.

        When a note ``b.md`` is created, a pre-existing ``[[B]]`` in ``a.md`` that
        was stored as a dangling edge to a node named ``B`` should now resolve to
        ``b.md``. This finds each source ``a.md`` whose ``LINKS_TO`` target's
        ``name_lower`` is one of the new note's name keys (its stem / relative
        path without ``.md``); the caller re-resolves those sources' links.
        Case-insensitive via ``name_lower``. Only name-based dangling targets are
        matched — the new note's own ``:File`` node is named by its rel (e.g.
        ``b.md``), which is never a bare name key, so it can't match itself.
        """
        root = index_root or ""
        keys = [k.lower() for k in name_keys]
        tk = self._tenant_key_frag(tenant)
        # Bind the edge and require r.tenant when scoped, so this lookup honors the
        # same nodes-AND-edges boundary as the other scoped reads.
        rel = "[r:LINKS_TO]" if tenant is not None else "[:LINKS_TO]"
        rtenant = " AND r.tenant = $tenant" if tenant is not None else ""
        params: dict[str, Any] = {"root": root, "keys": keys}
        if tenant is not None:
            params["tenant"] = tenant
        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"MATCH (x:File {{index_root: $root{tk}}})-{rel}->(d:File {{index_root: $root{tk}}}) "
                f"WHERE d.name_lower IN $keys{rtenant} "
                "RETURN DISTINCT x.name AS name",
                **params,
            )
            return [rec["name"] for rec in result if rec["name"] is not None]

    def file_link_sources(
        self, *, index_root: str | None = None, tenant: str | None = None
    ) -> list[str]:
        """Names of ``:File`` nodes with outgoing ``LINKS_TO`` edges in a root.

        Lets a re-index discover files it linked from previously: any source no
        longer present on disk can then have its stale edges cleared via
        ``sync_file_links(name, [])``.
        """
        root = index_root or ""
        tk = self._tenant_key_frag(tenant)
        # Require the edge to be tenant-owned when scoped, so this list (which
        # drives stale-link cleanup) only reports files with an owned LINKS_TO
        # edge — honoring the same nodes-AND-edges boundary as the other reads.
        rel = "[r:LINKS_TO]" if tenant is not None else "[:LINKS_TO]"
        # Bind the edge AND the target to the tenant when scoped, so a source is
        # only reported when it owns the whole link (source, edge, target) — a
        # stale-link reconcile must not act on a link the tenant doesn't own.
        target = "(d:File {tenant: $tenant})" if tenant is not None else "()"
        rwhere = " WHERE r.tenant = $tenant" if tenant is not None else ""
        params: dict[str, Any] = {"root": root}
        if tenant is not None:
            params["tenant"] = tenant
        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"MATCH (f:File {{index_root: $root{tk}}})-{rel}->{target}{rwhere} "
                "RETURN DISTINCT f.name AS name",
                **params,
            )
            return [rec["name"] for rec in result if rec["name"] is not None]

    def invalidate(
        self,
        subject: str,
        predicate: str,
        obj: str,
        ended: str | date | datetime,
        *,
        tenant: str | None = None,
    ) -> int:
        """Mark a fact as no longer valid. Returns number of edges updated.

        With ``tenant`` set, both endpoints are pinned to that tenant, so a
        caller can only close its own tenant's edges — never another's.
        """
        rel = self._safe_rel(predicate)
        tk = self._tenant_key_frag(tenant)
        params: dict[str, Any] = {"subject": subject, "obj": obj, "ended": _to_iso(ended)}
        # Confine the closed edge by tenant. Scoped: only this tenant's edge.
        # Unscoped: only tenant-LESS edges (`r.tenant IS NULL`), so an unscoped
        # invalidate can't close a tenant-owned edge after the graph was migrated
        # (the name-only match subset-binds tenant endpoints). On a single-tenant
        # graph every edge is tenant-less, so it behaves as before.
        if tenant is not None:
            rtenant = " AND r.tenant = $tenant"
            params["tenant"] = tenant
        else:
            rtenant = " AND r.tenant IS NULL"
        query = (
            f"MATCH (s {{name: $subject{tk}}})-[r:{rel}]->(o {{name: $obj{tk}}}) "
            f"WHERE (r.valid_until = 'current' OR r.valid_until IS NULL){rtenant} "
            f"SET r.valid_until = $ended "
            f"RETURN count(r) AS n"
        )
        with self.driver.session(database=self.database) as session:
            rec = session.run(query, **params).single()
            return rec["n"] if rec else 0

    # ---------- read ----------

    def query_triples(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
        as_of: str | date | datetime | None = None,
        limit: int = 100,
        *,
        tenant: str | None = None,
    ) -> list[Triple]:
        """Query triples with optional SPO filters and point-in-time constraint.

        If `as_of` is provided, returns only facts valid at that date
        (valid_from <= as_of < valid_until, with valid_until "current" treated
        as indefinite future). Legacy NULL markers are also treated as current.

        With ``tenant`` set, both endpoints are confined to that tenant
        (``s.tenant = $tenant AND o.tenant = $tenant``), so a scoped query can
        never read across the tenant boundary.
        """
        where_parts = []
        params: dict[str, Any] = {"limit": limit}
        if subject:
            where_parts.append("s.name = $subject")
            params["subject"] = subject
        if obj:
            where_parts.append("o.name = $obj")
            params["obj"] = obj
        if tenant is not None:
            # Confine both endpoints AND the relationship to the tenant — the
            # boundary lives on nodes and edges, so an edge missing/mismatched on
            # tenant (partial or hand-edited migration) is excluded, not returned.
            where_parts.append(
                "s.tenant = $tenant AND o.tenant = $tenant AND r.tenant = $tenant"
            )
            params["tenant"] = tenant
        if as_of:
            # Expand a bare-date as_of to a full midnight-UTC instant, then
            # compare bounds as PARSED instants (datetime(...)) so mixed
            # sub-second precision orders correctly (shared predicate, verified
            # against Memgraph — handles bare-date bounds and the markers).
            params["as_of"] = to_utc_instant(as_of)
            where_parts.append(graph_as_of_predicate("r"))

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
        *,
        tenant: str | None = None,
    ) -> list[Triple]:
        """All outgoing edges from a node, optionally filtered by point-in-time."""
        return self.query_triples(subject=node, as_of=as_of, limit=limit, tenant=tenant)

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

    def stamp_tenant(
        self, tenant: str, *, only_missing: bool = True, dry_run: bool = False
    ) -> dict[str, int]:
        """Assign the ``tenant`` property to graph nodes/edges (multi-tenant migration).

        The graph twin of ``VectorStore.stamp_tenant``: adopt tenancy on an
        existing single-tenant graph by stamping every node and relationship with
        ``tenant``. With ``only_missing`` (default) only records lacking a
        ``tenant`` are touched, so it's idempotent and safe to re-run; pass
        ``only_missing=False`` to force-relabel (the cleanup path for a graph that
        already used ``tenant`` for something else). ``dry_run`` counts without
        writing. Returns ``{"nodes": n, "relationships": m}``.

        Like the ``:File`` ``index_root`` migration, this is unscoped over the
        whole database — point it at a graph mnemostack owns.
        """
        node_where = "WHERE n.tenant IS NULL" if only_missing else ""
        rel_where = "WHERE r.tenant IS NULL" if only_missing else ""
        with self.driver.session(database=self.database) as session:
            if dry_run:
                nodes = session.run(
                    f"MATCH (n) {node_where} RETURN count(n) AS n"
                ).single()
                rels = session.run(
                    f"MATCH ()-[r]->() {rel_where} RETURN count(r) AS n"
                ).single()
            else:
                nodes = session.run(
                    f"MATCH (n) {node_where} SET n.tenant = $tenant RETURN count(n) AS n",
                    tenant=tenant,
                ).single()
                rels = session.run(
                    f"MATCH ()-[r]->() {rel_where} SET r.tenant = $tenant RETURN count(r) AS n",
                    tenant=tenant,
                ).single()
        return {
            "nodes": int(nodes["n"] if nodes else 0),
            "relationships": int(rels["n"] if rels else 0),
        }

    # ---------- helpers ----------

    @staticmethod
    def _tenant_key_frag(tenant: str | None) -> str:
        """MERGE/MATCH node-key fragment pinning a node to ``tenant``.

        Returns ``", tenant: $tenant"`` for a scoped write/read (folds the tenant
        into the node identity) or ``""`` when unscoped — so ``tenant=None`` keeps
        the exact legacy Cypher and never rekeys an existing single-tenant graph.
        """
        return ", tenant: $tenant" if tenant is not None else ""

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
