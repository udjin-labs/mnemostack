"""
Consolidation runtime — memory lifecycle (ingest, decay, graph-sync, health).

The Runtime orchestrator runs the supplied phases in order. Failures are recorded
and do not block subsequent phases unless stop_on_error=True.

Usage:
    from mnemostack.consolidation import Runtime, CallablePhase
    runtime = Runtime(phases=[CallablePhase("ingest", lambda: (True, "ingested"))])
    state = runtime.run()
    print(state.to_dict())
"""

from .phases import CallablePhase, EmbeddingHealthPhase, GraphHealthPhase, VectorHealthPhase
from .runtime import Phase, PhaseResult, Runtime, RuntimeState

__all__ = [
    "Runtime",
    "RuntimeState",
    "Phase",
    "PhaseResult",
    "CallablePhase",
    "EmbeddingHealthPhase",
    "VectorHealthPhase",
    "GraphHealthPhase",
]
