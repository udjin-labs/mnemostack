"""Built-in consolidation phases.

These are generic phases that work out of the box. Users can subclass Phase
to add custom logic (e.g. extract facts from documents, sync graph from text).
"""
from __future__ import annotations

from ..embeddings.base import EmbeddingProvider
from ..graph.store import GraphStore
from ..vector.qdrant import VectorStore
from .runtime import Phase, PhaseResult


class EmbeddingHealthPhase(Phase):
    """Ping embedding provider with a small probe."""

    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider

    @property
    def name(self) -> str:
        return "embedding_health"

    def run(self) -> PhaseResult:
        ok, msg = self.provider.health_check()
        return PhaseResult(
            name=self.name,
            ok=ok,
            duration_ms=0,  # will be filled by runtime
            message=msg,
            data={"provider": self.provider.name, "dim": self.provider.dimension},
        )


class VectorHealthPhase(Phase):
    """Check vector store collection exists and report size."""

    def __init__(self, store: VectorStore):
        self.store = store

    @property
    def name(self) -> str:
        return "vector_health"

    def run(self) -> PhaseResult:
        try:
            exists = self.store.collection_exists()
            count = self.store.count() if exists else 0
            return PhaseResult(
                name=self.name,
                ok=exists,
                duration_ms=0,
                message="ok" if exists else "collection missing",
                data={"collection": self.store.collection, "points": count, "exists": exists},
            )
        except Exception as e:  # noqa: BLE001
            return PhaseResult(name=self.name, ok=False, duration_ms=0, message=str(e))


class GraphHealthPhase(Phase):
    """Ping graph store + report node/edge counts."""

    def __init__(self, store: GraphStore):
        self.store = store

    @property
    def name(self) -> str:
        return "graph_health"

    def run(self) -> PhaseResult:
        ok, msg = self.store.health_check()
        data: dict = {"message": msg}
        if ok:
            try:
                data["nodes"] = self.store.count_nodes()
                data["edges"] = self.store.count_edges()
            except Exception as e:  # noqa: BLE001
                ok = False
                msg = f"count failed: {e}"
        return PhaseResult(name=self.name, ok=ok, duration_ms=0, message=msg, data=data)


class CallablePhase(Phase):
    """Wrap a plain callable as a phase. Callable should return (ok, message, data?)."""

    def __init__(self, name: str, func):
        self._name = name
        self._func = func

    @property
    def name(self) -> str:
        return self._name

    def run(self) -> PhaseResult:
        try:
            result = self._func()
        except Exception as e:  # noqa: BLE001
            return PhaseResult(name=self._name, ok=False, duration_ms=0, message=str(e))
        # Accept tuple (ok, msg) or (ok, msg, data)
        if isinstance(result, tuple):
            ok = result[0]
            msg = result[1] if len(result) > 1 else ""
            data = result[2] if len(result) > 2 else {}
        else:
            ok = bool(result)
            msg = "ok" if ok else "failed"
            data = {}
        return PhaseResult(name=self._name, ok=ok, duration_ms=0, message=str(msg), data=data)
