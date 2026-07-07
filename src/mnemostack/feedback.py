"""Shared feedback recording helpers for stateful recall stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .recall import ClassifyQuery, PipelineContext

FeedbackSignal = Literal["useful", "irrelevant", "clicked"]


@dataclass
class FeedbackHit:
    id: str


@dataclass
class FeedbackOutcome:
    hit_id: str
    signal: str
    reward: float
    query_type: str
    ior_recorded: bool
    q_learning_updates: int

    @property
    def ok(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "hit_id": self.hit_id,
            "signal": self.signal,
            "reward": self.reward,
            "query_type": self.query_type,
            "ior_recorded": self.ior_recorded,
            "q_learning_updates": self.q_learning_updates,
        }


def feedback_reward(signal: str, override: float | None = None) -> float:
    if signal not in {"useful", "clicked", "irrelevant"}:
        raise ValueError("signal must be one of: useful, irrelevant, clicked")
    if override is not None:
        reward = float(override)
        if not 0.0 <= reward <= 1.0:
            raise ValueError("reward must be in [0, 1]")
        return reward
    return {
        "useful": 1.0,
        "clicked": 0.7,
        "irrelevant": 0.0,
    }[signal]


def feedback_query_type(query: str | None = None, query_type: str | None = None) -> str:
    if query_type:
        return query_type
    if not query:
        return "general"
    ctx = PipelineContext(query=query)
    ClassifyQuery().apply(ctx, [])
    return ctx.query_type


def feedback_sources(
    source: str | None = None,
    sources: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in [source, *(sources or [])]:
        if not item:
            continue
        item = str(item)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def record_recall_events(pipeline, results, tenant: str | None = None) -> bool:
    # Pass tenant only when set, so a custom stage with the legacy
    # record_recall(memory_id) signature still works in a single-tenant deployment
    # (mirrors the graph-adapter back-compat pattern). A tenant-scoped deployment
    # requires tenant-aware stages, which the built-in ones are.
    tkw: dict[str, Any] = {"tenant": tenant} if tenant is not None else {}
    recorded = False
    for stage in pipeline:
        record_recall = getattr(stage, "record_recall", None)
        if record_recall is None:
            continue
        for result in results:
            record_recall(str(result.id), **tkw)
            recorded = True
    return recorded


def record_feedback_events(
    pipeline,
    sources: list[str],
    query_type: str,
    reward: float,
    tenant: str | None = None,
) -> int:
    tkw: dict[str, Any] = {"tenant": tenant} if tenant is not None else {}
    updates = 0
    for stage in pipeline:
        record_feedback = getattr(stage, "record_feedback", None)
        if record_feedback is None:
            continue
        for source in sources:
            record_feedback(source, query_type, reward, **tkw)
            updates += 1
    return updates


def apply_feedback(
    pipeline,
    hit_id: str,
    signal: str,
    query: str | None = None,
    query_type: str | None = None,
    source: str | None = None,
    sources: list[str] | tuple[str, ...] | None = None,
    reward: float | None = None,
    tenant: str | None = None,
) -> FeedbackOutcome:
    """Record feedback into the pipeline's learning state.

    With ``tenant`` set, the Q-table and IoR updates land in that tenant's own
    partition of the shared state store, so one tenant's feedback can't shift
    another's ranking. ``tenant=None`` writes the unscoped (single-tenant) state.
    """
    resolved_reward = feedback_reward(signal, reward)
    resolved_query_type = feedback_query_type(query=query, query_type=query_type)
    resolved_sources = feedback_sources(source=source, sources=sources)
    q_updates = record_feedback_events(
        pipeline,
        resolved_sources,
        resolved_query_type,
        resolved_reward,
        tenant,
    )
    ior_recorded = False
    if signal == "clicked":
        ior_recorded = record_recall_events(pipeline, [FeedbackHit(hit_id)], tenant)
    return FeedbackOutcome(
        hit_id=hit_id,
        signal=signal,
        reward=resolved_reward,
        query_type=resolved_query_type,
        ior_recorded=ior_recorded,
        q_learning_updates=q_updates,
    )
