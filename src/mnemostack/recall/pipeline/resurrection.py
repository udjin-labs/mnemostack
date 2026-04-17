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

from typing import Any

from ..recaller import RecallResult
from .base import PipelineContext, Stage
from .stages import STOPWORDS

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
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.limit = limit
        self.min_seed_len = min_seed_len
        self.max_seeds = max_seeds
        self.max_per_seed = max_per_seed
        self._driver = driver
        self._own_driver = driver is None

    def _get_driver(self):
        if self._driver is not None:
            return self._driver
        if not _AVAILABLE:
            return None
        try:
            self._driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password) if self.user else None
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

        existing = " ".join(
            (r.text or "") + " " + (r.payload.get("text", "") if r.payload else "")
            for r in results
        ).lower()

        seed_match: dict[str, dict[str, Any]] = {}
        try:
            with driver.session() as session:
                for seed in list(seeds)[: self.max_seeds]:
                    rows = session.run(
                        """
                        MATCH (n)-[r1]-(m)
                        WHERE toLower(n.name) = $seed
                          AND n.valid_until = 'current'
                          AND m.valid_until = 'current'
                        RETURN DISTINCT m.name AS name, labels(m)[0] AS type,
                               m.memory_class AS mc, type(r1) AS rel
                        LIMIT $lim
                        """,
                        seed=seed, lim=self.max_per_seed,
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
            text = f"[Graph] {nb.get('type','')}: {name} (rel: {rels})"
            rr = RecallResult(
                id=f"graph:{name}",
                text=text,
                score=score,
                payload={
                    "text": text,
                    "source": "memgraph",
                    "resurrected": True,
                    "resurrection_seed": ",".join(sorted(info["seeds"])),
                    "memory_class": nb.get("mc") or "",
                },
                sources=["memgraph"],
            )
            resurrected.append((rr, score))

        resurrected.sort(key=lambda x: -x[1])
        for rr, _ in resurrected[: self.limit]:
            results.append(rr)
        return results
