"""
Staged recall pipeline — composable list of Stage objects.

Each Stage transforms a list of RecallResult objects. Stages can be:
- Stateless (gravity_dampen, freshness_blend, classify_query)
- Stateful (QLearning, InhibitionOfReturn, MoodCongruent) — take a StateStore

Usage:
    from mnemostack.recall.pipeline import Pipeline, Stage
    from mnemostack.recall.pipeline.stages import (
        GravityDampen, HubDampen, FreshnessBlend,
    )

    pipeline = Pipeline([
        GravityDampen(),
        HubDampen(hub_degrees={'alice': 42}),
        FreshnessBlend(weight=0.15),
    ])
    results = pipeline.apply(query, results)
"""

from .base import Pipeline, PipelineContext, Stage
from .presets import build_full_pipeline, build_stateless_pipeline
from .stages import (
    ClassifyQuery,
    CuriosityBoost,
    ExactTokenRescue,
    FreshnessBlend,
    GravityDampen,
    HubDampen,
    InhibitionOfReturn,
    QLearningReranker,
    is_exact_token_query,
)
from .state import FileStateStore, InMemoryStateStore, StateStore

__all__ = [
    "Pipeline",
    "PipelineContext",
    "Stage",
    "StateStore",
    "InMemoryStateStore",
    "FileStateStore",
    "ClassifyQuery",
    "ExactTokenRescue",
    "GravityDampen",
    "HubDampen",
    "FreshnessBlend",
    "InhibitionOfReturn",
    "CuriosityBoost",
    "QLearningReranker",
    "is_exact_token_query",
    "build_full_pipeline",
    "build_stateless_pipeline",
]
