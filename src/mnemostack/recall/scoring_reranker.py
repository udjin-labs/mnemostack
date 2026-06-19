"""Score-based reranker abstraction.

Unlike the generative LLM ``Reranker`` that asks a model to emit IDs, this
reranker only requires a backend to return one numeric relevance score per
candidate. Cross-encoders and dedicated rerank services fit this interface
directly; generative LLM scoring can be wrapped too, but is usually less stable
because it still depends on prompt/output parsing.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import Protocol

from .recaller import RecallResult

logger = logging.getLogger(__name__)


class RelevanceScorer(Protocol):
    """Backend that scores query/document relevance.

    Implementations must return exactly one score for every input document in
    the same order. Scores are compared only relative to each other.
    """

    def score(self, query: str, documents: list[str]) -> list[float]: ...


class ScoringReranker:
    """Rerank by numeric relevance scores from an injected backend.

    Args:
        scorer: backend returning one score per candidate document.
        max_items: rerank only this prefix; later results keep their order.

    If scoring fails or returns malformed output, the original order is kept.
    """

    def __init__(self, scorer: RelevanceScorer, max_items: int = 200):
        if max_items < 1:
            raise ValueError("max_items must be >= 1")
        self.scorer = scorer
        self.max_items = max_items
        self.last_fallback_reason: str | None = None

    def rerank(self, query: str, results: list[RecallResult]) -> list[RecallResult]:
        self.last_fallback_reason = None
        if not results:
            return []
        head = results[: self.max_items]
        tail = results[self.max_items :]
        try:
            scores = _materialize_scores(self.scorer.score(query, [r.text for r in head]))
        except Exception as exc:  # noqa: BLE001 - reranking must stay fail-open
            logger.warning("scoring rerank failed, keeping original order: %s", exc)
            self.last_fallback_reason = "reranker:fallback"
            return results
        if len(scores) != len(head):
            logger.warning(
                "scoring rerank returned %d scores for %d candidates; keeping original order",
                len(scores),
                len(head),
            )
            self.last_fallback_reason = "reranker:fallback"
            return results
        try:
            scored = [
                (result, _finite_score_or_floor(float(score)), idx)
                for idx, (result, score) in enumerate(zip(head, scores, strict=True))
            ]
        except (TypeError, ValueError):
            logger.warning("scoring rerank returned non-numeric scores; keeping original order")
            self.last_fallback_reason = "reranker:fallback"
            return results
        scored.sort(key=lambda item: (-item[1], item[2]))
        return [result for result, _score, _idx in scored] + tail


def _finite_score_or_floor(score: float) -> float:
    """Use non-finite scorer output as the lowest possible relevance score."""
    return score if math.isfinite(score) else -math.inf


def _materialize_scores(scores: object) -> list[float]:
    if scores is None or isinstance(scores, (str, bytes)) or not isinstance(scores, Iterable):
        raise TypeError("score backend must return an iterable of numeric scores")
    return list(scores)
