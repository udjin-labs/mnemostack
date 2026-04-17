"""Consolidation runtime — phase orchestrator.

Extensible design: users register phases (callables) and the runtime
executes them in order, tracking success/failure/duration for each.
Inspired by Kairos (internal prototype) but generalized for any memory stack.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class PhaseResult:
    """Outcome of a single phase run."""

    name: str
    ok: bool
    duration_ms: int
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "duration_ms": self.duration_ms,
            "message": self.message,
            "data": self.data,
        }


class Phase(ABC):
    """Base class for a runtime phase. Override .run()."""

    @property
    def name(self) -> str:
        """Phase identifier used in logs/state. Override if class name is awkward."""
        return self.__class__.__name__.replace("Phase", "").lower() or "phase"

    @abstractmethod
    def run(self) -> PhaseResult:
        """Do the work. Must return a PhaseResult with ok + message."""


@dataclass
class RuntimeState:
    """State from a runtime execution."""

    started_at: str
    finished_at: str | None = None
    phases: list[PhaseResult] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and all(p.ok for p in self.phases)

    @property
    def total_duration_ms(self) -> int:
        return sum(p.duration_ms for p in self.phases)

    def phase(self, name: str) -> PhaseResult | None:
        for p in self.phases:
            if p.name == name:
                return p
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "ok": self.ok,
            "total_duration_ms": self.total_duration_ms,
            "error": self.error,
            "phases": [p.to_dict() for p in self.phases],
        }


class Runtime:
    """Phase orchestrator.

    Runs phases in order; each phase gets a timer. If a phase raises, the
    error is caught and recorded as PhaseResult(ok=False), and the runtime
    continues with the next phase unless stop_on_error=True.

    Args:
        phases: list of Phase instances to execute in order
        stop_on_error: if True, halt on first phase failure
        dry_run: if True, skip phase execution, just record plan
    """

    def __init__(
        self,
        phases: list[Phase],
        stop_on_error: bool = False,
        dry_run: bool = False,
    ):
        self.phases = phases
        self.stop_on_error = stop_on_error
        self.dry_run = dry_run

    def run(self) -> RuntimeState:
        state = RuntimeState(started_at=_now_iso())
        for phase in self.phases:
            result = self._run_phase(phase)
            state.phases.append(result)
            if not result.ok and self.stop_on_error:
                state.error = f"stopped at phase {phase.name}: {result.message}"
                break
        state.finished_at = _now_iso()
        return state

    def _run_phase(self, phase: Phase) -> PhaseResult:
        if self.dry_run:
            return PhaseResult(name=phase.name, ok=True, duration_ms=0, message="dry-run")
        start = time.monotonic()
        try:
            result = phase.run()
            # Ensure name + duration are set if phase didn't populate them
            if not result.name:
                result.name = phase.name
            if result.duration_ms == 0:
                result.duration_ms = int((time.monotonic() - start) * 1000)
            return result
        except Exception as e:  # noqa: BLE001
            return PhaseResult(
                name=phase.name,
                ok=False,
                duration_ms=int((time.monotonic() - start) * 1000),
                message=f"exception: {e}",
            )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
