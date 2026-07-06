"""GraphResurrection — spreading-activation stage via Memgraph/Neo4j.

Ported from workspace/scripts/enhanced-recall.py:resurrection_boost.

For each query, take query terms as seeds and walk 1-hop neighbors in the
knowledge graph. Neighbors connected to MULTIPLE seeds are scored higher.
Adds resurrected results as new RecallResult entries with capped score
(max 0.30 — should never outrank strong direct hits).

Fails soft: if the graph driver is unavailable or the connection fails,
the stage is a no-op. This matches legacy behaviour.
"""

from __future__ import annotations

import logging
from typing import Any

from ..recaller import RecallResult
from ..retrievers import graph_valid_clause
from ..validity import to_utc_instant
from .base import PipelineContext, Stage
from .stages import STOPWORDS

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class GraphResurrection(Stage):
    """Resurrect forgotten memories via 1-hop graph walk.

    Args:
        uri: Memgraph/Neo4j bolt URI (default bolt://localhost:7687)
        user/password: auth if needed
        limit: max resurrected nodes to add per query
        min_seed_len: ignore query tokens shorter than this
        max_seeds: cap on how many seeds we probe (keeps Cypher cheap)
        max_per_seed: how many neighbors per seed
        driver: injected driver for testing (bypasses URI connection)
    """

    name = "graph_resurrection"

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "",
        password: str = "",
        limit: int = 3,
        min_seed_len: int = 4,
        max_seeds: int = 8,
        max_per_seed: int = 5,
        driver: Any = None,
        timeout: float = 5.0,
        # database appended at the tail to preserve positional back-compat.
        database: str | None = None,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.limit = limit
        self.min_seed_len = min_seed_len
        self.max_seeds = max_seeds
        self.max_per_seed = max_per_seed
        self.timeout = timeout
        self._driver = driver
        self._own_driver = driver is None

    def _get_driver(self):
        if self._driver is not None:
            return self._driver
        if not _AVAILABLE:
            return None
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password) if self.user else None,
                connection_timeout=self.timeout,
                connection_acquisition_timeout=self.timeout,
            )
            return self._driver
        except Exception:
            return None

    def close(self) -> None:
        if self._driver is not None and self._own_driver:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None

    def _seeds(self, query: str) -> set[str]:
        tokens = {t for t in query.lower().split() if t not in STOPWORDS}
        return {t for t in tokens if len(t) >= self.min_seed_len}

    def apply(
        self,
        context: PipelineContext,
        results: list[RecallResult],
    ) -> list[RecallResult]:
        query = context.query
        seeds = self._seeds(query)
        if not seeds:
            return results
        driver = self._get_driver()
        if driver is None:
            return results

        # Match the recall's validity view: resurrect graph neighbors valid at
        # `as_of` (point-in-time), and don't suppress closed edges when the
        # caller asked to include invalidated facts. Both ride in via
        # PipelineContext.extras; `as_of` is UTC-normalized for string compare.
        as_of = to_utc_instant(context.extras.get("as_of"))
        include_invalidated = bool(context.extras.get("include_invalidated", False))
        n_valid = graph_valid_clause("n", as_of, include_invalidated)
        m_valid = graph_valid_clause("m", as_of, include_invalidated)
        r_valid = graph_valid_clause("r1", as_of, include_invalidated)
        extra_params: dict[str, Any] = {"as_of": as_of} if as_of is not None else {}
        # Tenant scope (rides in via context.extras, like as_of): confine the
        # seed, the neighbor, and the edge to the tenant so a resurrected node
        # can never come from another tenant's subgraph. Unscoped (tenant=None)
        # adds nothing — a single-tenant graph walk is unchanged.
        tenant = context.extras.get("tenant")
        tpred = (
            " AND n.tenant = $tenant AND m.tenant = $tenant AND r1.tenant = $tenant"
            if tenant is not None
            else ""
        )
        if tenant is not None:
            extra_params["tenant"] = tenant

        existing = " ".join(
            (r.text or "") + " " + (r.payload.get("text", "") if r.payload else "") for r in results
        ).lower()

        seed_match: dict[str, dict[str, Any]] = {}
        # Only pass database= when a non-default DB is configured, so an injected
        # driver whose session() takes no args (fakes/wrappers) still works.
        session_kwargs = {"database": self.database} if self.database else {}
        try:
            with driver.session(**session_kwargs) as session:
                for seed in list(seeds)[: self.max_seeds]:
                    rows = session.run(
                        f"""
                        MATCH (n)-[r1]-(m)
                        WHERE toLower(n.name) = $seed
                          AND {n_valid}
                          AND {m_valid}
                          AND {r_valid}{tpred}
                        RETURN DISTINCT m.name AS name, labels(m)[0] AS type,
                               m.memory_class AS mc, type(r1) AS rel
                        LIMIT $lim
                        """,
                        seed=seed,
                        lim=self.max_per_seed,
                        **extra_params,
                    ).data()
                    for nb in rows:
                        name = nb.get("name") or ""
                        if not name:
                            continue
                        key = name.lower()
                        slot = seed_match.setdefault(
                            key,
                            {"data": nb, "seeds": set(), "rels": set()},
                        )
                        slot["seeds"].add(seed)
                        slot["rels"].add(nb.get("rel") or "")
        except Exception:
            # Fail soft (graph optional), but log — a malformed stored bound or
            # driver error skips resurrection for this call, which was silent.
            logger.warning("graph resurrection failed, skipping", exc_info=True)
            return results

        resurrected: list[tuple[RecallResult, float]] = []
        for key, info in seed_match.items():
            nb = info["data"]
            name = nb.get("name") or ""
            if key in existing:
                continue
            overlap = len(info["seeds"]) / max(len(seeds), 1)
            score = min(0.10 + 0.15 * overlap, 0.30)
            rels = ", ".join(sorted(r for r in info["rels"] if r))
            text = f"[Graph] {nb.get('type', '')}: {name} (rel: {rels})"
            payload: dict[str, Any] = {
                "text": text,
                "source": "memgraph",
                "resurrected": True,
                "resurrection_seed": ",".join(sorted(info["seeds"])),
                "memory_class": nb.get("mc") or "",
            }
            # Stamp the tenant so the resurrected node survives the recall
            # `filter_by_tenant` backstop (which drops any result lacking a
            # matching tenant_id). The seed walk above already confined every
            # neighbor to this tenant. Namespace the id by tenant too, so two
            # tenants' same-named graph nodes don't collide in the stateful
            # pipeline's IoR/feedback state (keyed on str(result.id)) — matching
            # the retriever and the already-tenant-scoped vector ids.
            if tenant is not None:
                payload["tenant_id"] = tenant
                result_id = f"graph:{tenant}:{name}"
            else:
                result_id = f"graph:{name}"
            rr = RecallResult(
                id=result_id,
                text=text,
                score=score,
                payload=payload,
                sources=["memgraph"],
            )
            resurrected.append((rr, score))

        resurrected.sort(key=lambda x: -x[1])
        for rr, _ in resurrected[: self.limit]:
            results.append(rr)
        return results
