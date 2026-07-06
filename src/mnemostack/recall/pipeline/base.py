"""Pipeline and Stage base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..recaller import RecallResult


@dataclass
class PipelineContext:
    """Shared context passed through all stages.

    Carries the query, derived query metadata (classification, tokens),
    and arbitrary extras that upstream stages can populate for downstream.
    """

    query: str
    query_type: str = "general"  # filled by ClassifyQueryStage
    query_tokens: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


class Stage(ABC):
    """Pipeline stage — transforms a list of RecallResult.

    Stages should be pure with respect to their config: given the same
    (context, results), always produce the same output. State (like Q-learning
    or IOR logs) lives in a StateStore injected via constructor.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def apply(
        self,
        context: PipelineContext,
        results: list[RecallResult],
    ) -> list[RecallResult]:
        """Return transformed results. Empty input → empty output."""


class Pipeline:
    """Ordered list of stages. Apply in sequence.

    Args:
        stages: list of Stage instances, applied left-to-right
        stop_on_empty: if True, skip remaining stages once results become empty
    """

    def __init__(self, stages: list[Stage], stop_on_empty: bool = True):
        self.stages = stages
        self.stop_on_empty = stop_on_empty

    def apply(
        self,
        query: str,
        results: list[RecallResult],
        *,
        as_of: str | None = None,
        include_invalidated: bool = False,
        tenant: str | None = None,
    ) -> list[RecallResult]:
        context = PipelineContext(query=query)
        # Validity + tenant context for stages that reach back to the graph
        # (GraphResurrection) so they match the recall's view and scope.
        if as_of is not None:
            context.extras["as_of"] = as_of
        if include_invalidated:
            context.extras["include_invalidated"] = True
        if tenant is not None:
            context.extras["tenant"] = tenant
        for stage in self.stages:
            if self.stop_on_empty and not results:
                break
            results = stage.apply(context, results)
        return results

    def apply_with_context(
        self,
        context: PipelineContext,
        results: list[RecallResult],
    ) -> list[RecallResult]:
        """Run pipeline with pre-built context (for integration tests)."""
        for stage in self.stages:
            if self.stop_on_empty and not results:
                break
            results = stage.apply(context, results)
        return results

    def __iter__(self):
        return iter(self.stages)

    def __len__(self):
        return len(self.stages)

    def add(self, stage: Stage) -> Pipeline:
        """Append stage (chainable)."""
        self.stages.append(stage)
        return self
