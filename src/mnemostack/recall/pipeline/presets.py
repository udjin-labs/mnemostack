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
    DEFAULT_ACCESS_BONUS_MAX,
    CalibrateRelevance,
    ClassifyQuery,
    CuriosityBoost,
    ExactTokenProtection,
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
    enable_stateful_stages: bool = True,
    enable_q_learning: bool = True,
    enable_ior: bool = True,
    enable_curiosity: bool = True,
    graph_uri: str | None = None,
    graph_user: str = "",
    graph_password: str = "",
    graph_limit: int = 3,
    graph_timeout: float = 5.0,
    # graph_database appended at the tail to preserve positional back-compat for
    # library callers passing graph_limit/graph_timeout positionally.
    graph_database: str | None = None,
    # Keyword-only: the positional surface above is frozen.
    *,
    text_key: str = "text",
    timestamp_key: str = "timestamp",
    timestamp_format: str = "iso",
    access_bonus_max: float = DEFAULT_ACCESS_BONUS_MAX,
) -> Pipeline:
    """Build the full reranking pipeline.

    Stage order (applied left-to-right):
        1. ClassifyQuery: determine query_type for downstream stages
        2. CalibrateRelevance: put rank scores on a bounded utility scale
        3. ExactTokenRescue: boost exact-match sources for infra queries
        4. HubDampen: SDK-only with nonempty hub_degrees; absent on CLI, HTTP and MCP
        5. QLearningReranker: conditional on stateful and Q-learning toggles
        6. CuriosityBoost: conditional on stateful and curiosity toggles
        7. FreshnessBlend: blend similarity with recency
        8. GravityDampen: penalize results missing query key terms
        9. InhibitionOfReturn: conditional on stateful and IoR toggles
        10. ExactTokenProtection: protect strong exact-token vector hits
        11. GraphResurrection: conditional on graph_uri

    Args:
        state_store: persistent state for stateful stages. If None, uses
                     in-memory store (state lost on restart).
        hub_degrees: graph node degree map for hub dampening (pass None to skip).
        enable_stateful_stages: master toggle for Q-learning, IoR, and curiosity.
                                Disable for deterministic benchmarks.
        enable_*: toggle individual stateful stages.
    """
    store = state_store or InMemoryStateStore()
    enable_q_learning = enable_stateful_stages and enable_q_learning
    enable_ior = enable_stateful_stages and enable_ior
    enable_curiosity = enable_stateful_stages and enable_curiosity

    # Calibration precedes the unit-scale source, exploration and age signals.
    stages = [
        ClassifyQuery(),
        CalibrateRelevance(),
        ExactTokenRescue(boost=rescue_boost),
    ]

    if hub_degrees:
        stages.append(HubDampen(hub_degrees=hub_degrees))

    # Calibrate rank-score units before mixing source, exploration and age utility.

    if enable_q_learning:
        stages.append(QLearningReranker(state_store=store))

    if enable_curiosity:
        stages.append(
            CuriosityBoost(
                state_store=store,
                timestamp_key=timestamp_key,
                timestamp_format=timestamp_format,
            )
        )

    stages.append(
        FreshnessBlend(
            weight=freshness_weight,
            timestamp_key=timestamp_key,
            timestamp_format=timestamp_format,
            access_bonus_max=access_bonus_max,
        )
    )

    # Apply the lexical penalty to the blended utility, not only its relevance
    # component. Otherwise a uniform penalty changes the recency weight.
    stages.append(GravityDampen(penalty=gravity_penalty))

    if enable_ior:
        stages.append(InhibitionOfReturn(state_store=store))

    stages.append(ExactTokenProtection())

    if graph_uri:
        stages.append(
            GraphResurrection(
                uri=graph_uri,
                user=graph_user,
                password=graph_password,
                database=graph_database,
                limit=graph_limit,
                timeout=graph_timeout,
                text_key=text_key,
            )
        )

    return Pipeline(stages)


def build_stateless_pipeline(
    hub_degrees: dict[Any, int] | None = None,
    freshness_weight: float = 0.2,
    gravity_penalty: float = 0.5,
    rescue_boost: float = 0.5,
    *,
    timestamp_key: str = "timestamp",
    timestamp_format: str = "iso",
    access_bonus_max: float = DEFAULT_ACCESS_BONUS_MAX,
) -> Pipeline:
    """Minimal pipeline with stateless stages only.

    Hub dampening is SDK-only with nonempty hub_degrees; CLI, HTTP and MCP
    paths do not supply it.

    No state persistence, no Q-learning, no IOR. Good for:
    - Short-lived scripts
    - Unit tests
    - Stateless web handlers that can't share state
    """
    stages = [
        ClassifyQuery(),
        CalibrateRelevance(),
        ExactTokenRescue(boost=rescue_boost),
    ]
    if hub_degrees:
        stages.append(HubDampen(hub_degrees=hub_degrees))
    stages.append(
        FreshnessBlend(
            weight=freshness_weight,
            timestamp_key=timestamp_key,
            timestamp_format=timestamp_format,
            access_bonus_max=access_bonus_max,
        )
    )
    stages.append(GravityDampen(penalty=gravity_penalty))
    stages.append(ExactTokenProtection())
    return Pipeline(stages)
