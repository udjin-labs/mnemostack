"""Ready-to-use pipeline presets.

Predefined stage compositions for common use-cases. Users who want a single
line for reasonable defaults should import from here; advanced users can
compose their own pipelines via `Pipeline([...])`.
"""
from __future__ import annotations

from typing import Any

from .base import Pipeline
from .resurrection import GraphResurrection
from .stages import (
    ClassifyQuery,
    CuriosityBoost,
    ExactTokenRescue,
    FreshnessBlend,
    GravityDampen,
    HubDampen,
    InhibitionOfReturn,
    QLearningReranker,
)
from .state import InMemoryStateStore, StateStore


def build_full_pipeline(
    state_store: StateStore | None = None,
    hub_degrees: dict[Any, int] | None = None,
    freshness_weight: float = 0.2,
    gravity_penalty: float = 0.5,
    rescue_boost: float = 0.5,
    enable_q_learning: bool = True,
    enable_ior: bool = True,
    enable_curiosity: bool = True,
    graph_uri: str | None = None,
    graph_user: str = "",
    graph_password: str = "",
    graph_limit: int = 3,
) -> Pipeline:
    """Build the full 8-stage reranking pipeline.

    Stage order (applied left-to-right):
        1. ClassifyQuery       — determine query_type for downstream stages
        2. ExactTokenRescue    — boost exact-match sources for infra queries
        3. GravityDampen       — penalize results missing query key terms
        4. HubDampen           — penalize highly-connected graph hubs
        5. FreshnessBlend      — blend similarity with recency
        6. InhibitionOfReturn  — penalize recently-recalled memories
        7. CuriosityBoost      — surface old, rarely-recalled memories
        8. QLearningReranker   — boost scores from historically-good sources

    Args:
        state_store: persistent state for stateful stages. If None, uses
                     in-memory store (state lost on restart).
        hub_degrees: graph node degree map for hub dampening (pass None to skip).
        enable_*: toggle individual stateful stages.
    """
    store = state_store or InMemoryStateStore()

    # Stage order mirrors legacy enhanced-recall.py pipeline:
    #   Gravity → Hub → Q-learning → Curiosity → Freshness → IOR → Graph
    # Q-learning applies BEFORE Freshness so BM25/graph results get the
    # source-utility boost while scores are still in raw RRF range.
    stages = [
        ClassifyQuery(),
        ExactTokenRescue(boost=rescue_boost),
        GravityDampen(penalty=gravity_penalty),
    ]

    if hub_degrees:
        stages.append(HubDampen(hub_degrees=hub_degrees))

    if enable_q_learning:
        stages.append(QLearningReranker(state_store=store))

    if enable_curiosity:
        stages.append(CuriosityBoost(state_store=store))

    stages.append(FreshnessBlend(weight=freshness_weight))

    if enable_ior:
        stages.append(InhibitionOfReturn(state_store=store))

    if graph_uri:
        stages.append(GraphResurrection(
            uri=graph_uri, user=graph_user, password=graph_password,
            limit=graph_limit,
        ))

    return Pipeline(stages)


def build_stateless_pipeline(
    hub_degrees: dict[Any, int] | None = None,
    freshness_weight: float = 0.2,
    gravity_penalty: float = 0.5,
    rescue_boost: float = 0.5,
) -> Pipeline:
    """Minimal pipeline with stateless stages only.

    No state persistence, no Q-learning, no IOR. Good for:
    - Short-lived scripts
    - Unit tests
    - Stateless web handlers that can't share state
    """
    stages = [
        ClassifyQuery(),
        ExactTokenRescue(boost=rescue_boost),
        GravityDampen(penalty=gravity_penalty),
    ]
    if hub_degrees:
        stages.append(HubDampen(hub_degrees=hub_degrees))
    stages.append(FreshnessBlend(weight=freshness_weight))
    return Pipeline(stages)
