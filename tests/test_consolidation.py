"""Tests for consolidation runtime and built-in phases."""
import pytest

from mnemostack.consolidation import (
    CallablePhase,
    Phase,
    PhaseResult,
    Runtime,
    RuntimeState,
)


class GoodPhase(Phase):
    @property
    def name(self):
        return "good"

    def run(self):
        return PhaseResult(name="good", ok=True, duration_ms=0, message="works")


class BadPhase(Phase):
    @property
    def name(self):
        return "bad"

    def run(self):
        return PhaseResult(name="bad", ok=False, duration_ms=0, message="broken")


class RaisingPhase(Phase):
    @property
    def name(self):
        return "raising"

    def run(self):
        raise RuntimeError("kaboom")


def test_runtime_all_good():
    rt = Runtime(phases=[GoodPhase(), GoodPhase()])
    state = rt.run()
    assert state.ok
    assert len(state.phases) == 2
    assert all(p.ok for p in state.phases)
    assert state.error is None
    assert state.finished_at is not None


def test_runtime_continues_on_failure():
    rt = Runtime(phases=[GoodPhase(), BadPhase(), GoodPhase()])
    state = rt.run()
    assert not state.ok
    assert len(state.phases) == 3  # all ran
    assert state.phases[1].message == "broken"


def test_runtime_stop_on_error():
    rt = Runtime(phases=[GoodPhase(), BadPhase(), GoodPhase()], stop_on_error=True)
    state = rt.run()
    assert not state.ok
    assert len(state.phases) == 2  # stopped after failure
    assert state.error is not None
    assert "bad" in state.error


def test_runtime_catches_exceptions():
    rt = Runtime(phases=[GoodPhase(), RaisingPhase(), GoodPhase()])
    state = rt.run()
    raising_result = state.phase("raising")
    assert raising_result is not None
    assert not raising_result.ok
    assert "kaboom" in raising_result.message
    # Runtime continued
    assert len(state.phases) == 3


def test_runtime_dry_run():
    rt = Runtime(phases=[RaisingPhase()], dry_run=True)
    state = rt.run()
    assert state.ok  # dry-run skips actual execution
    assert state.phases[0].message == "dry-run"


def test_state_to_dict():
    rt = Runtime(phases=[GoodPhase()])
    state = rt.run()
    d = state.to_dict()
    assert "started_at" in d
    assert "finished_at" in d
    assert d["ok"] is True
    assert len(d["phases"]) == 1
    assert d["phases"][0]["name"] == "good"


def test_state_phase_lookup():
    rt = Runtime(phases=[GoodPhase(), BadPhase()])
    state = rt.run()
    assert state.phase("good").ok
    assert not state.phase("bad").ok
    assert state.phase("nonexistent") is None


def test_callable_phase_bool_return():
    p = CallablePhase("test", lambda: True)
    result = p.run()
    assert result.ok
    assert result.message == "ok"

    p2 = CallablePhase("test", lambda: False)
    assert not p2.run().ok


def test_callable_phase_tuple_return():
    p = CallablePhase("test", lambda: (True, "all good", {"count": 42}))
    result = p.run()
    assert result.ok
    assert result.message == "all good"
    assert result.data == {"count": 42}


def test_callable_phase_catches_exceptions():
    def boom():
        raise ValueError("nope")

    p = CallablePhase("test", boom)
    result = p.run()
    assert not result.ok
    assert "nope" in result.message


def test_runtime_state_duration():
    import time

    class SlowPhase(Phase):
        @property
        def name(self):
            return "slow"

        def run(self):
            time.sleep(0.02)  # 20ms
            return PhaseResult(name="slow", ok=True, duration_ms=0)

    rt = Runtime(phases=[SlowPhase()])
    state = rt.run()
    assert state.phases[0].duration_ms >= 10  # at least ~20ms
    assert state.total_duration_ms >= 10
