"""
Consolidation runtime — memory lifecycle (ingest, decay, graph-sync, health).

The Runtime orchestrator runs phases in order; each phase can be enabled/disabled
via config. Failures in one phase are logged but do not block subsequent phases.

Usage:
    from mnemostack.consolidation import Runtime, Phase
    runtime = Runtime(phases=[MyIngestPhase(), DecayPhase(), HealthPhase()])
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
