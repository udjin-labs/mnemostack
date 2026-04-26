"""Cat_3 inference retry — query decomposition for hard inference questions.

When the answer generator returns 'Not in memory' or low-confidence response
for an inference question, this module:
1. Decomposes the question into 2-4 evidence-seeking sub-queries via LLM.
2. Runs each sub-query through the recaller.
3. Fuses results via RRF.
4. Re-runs answer generation on the merged context.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..llm.base import LLMProvider
    from .recaller import RecallResult

_DECOMPOSE_PROMPT = """Generate 2-4 evidence-seeking search queries to help answer this inference question.

QUESTION: {query}

EXAMPLES:
- 'What would X's political leaning be?' → ['X political views', 'X causes activism', 'X social issues opinions', 'X vote elections']
- 'Would Y be religious?' → ['Y religion church', 'Y faith spiritual', 'Y prayer worship', 'Y church holidays']
- 'Does Z prefer beach or mountains?' → ['Z beach vacation', 'Z mountain hiking', 'Z home location climate']

RULES:
1. Each sub-query must be 2-5 words, focused on a SPECIFIC piece of evidence.
2. Cover different angles — don't repeat the same idea.
3. Output JSON: {{"queries": ["q1", "q2", "q3"]}}

JSON_OUTPUT:"""


def decompose_query(query: str, llm: LLMProvider) -> list[str]:
    """Generate sub-queries for evidence retrieval. Returns [] on failure."""
    try:
        response = llm.generate(
            _DECOMPOSE_PROMPT.format(query=query),
            max_tokens=200,
        )
        # Parse JSON, allow whitespace/code-fence noise.
        text = response.text if hasattr(response, "text") else str(response)
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
        data = json.loads(text)
        queries = data.get("queries", [])
        if not isinstance(queries, list):
            return []
        return [q.strip() for q in queries if isinstance(q, str) and q.strip()][:4]
    except (json.JSONDecodeError, AttributeError, KeyError, TypeError) as e:
        logger.warning("decompose_query failed: %s", e)
        return []
    except Exception as e:
        logger.warning("decompose_query unexpected error: %s", e)
        return []


def should_retry(
    draft_answer_text: str,
    draft_confidence: float,
    low_confidence_threshold: float = 0.4,
) -> bool:
    """Decide if cat_3 inference question needs a retry."""
    if not draft_answer_text:
        return True
    if "not in memory" in draft_answer_text.lower():
        return True
    return draft_confidence < low_confidence_threshold


def merge_results(
    original_results: list[RecallResult],
    *result_lists: list[RecallResult],
) -> list[RecallResult]:
    """Fuse original + sub-query results via reciprocal rank fusion."""
    from .fusion import reciprocal_rank_fusion

    ranked_lists = [
        [(result, result.score) for result in results]
        for results in (original_results, *result_lists)
    ]
    fused = reciprocal_rank_fusion(ranked_lists)
    return [replace(result, score=score) for result, score in fused]
