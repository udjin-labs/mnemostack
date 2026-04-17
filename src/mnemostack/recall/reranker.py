"""LLM-based reranker — reorders top retrieval results by relevance to query.

Uses a ranking prompt instead of embedding similarity. Works well for
queries where semantic similarity alone is too broad (e.g. 'when did X happen'
can match any mention of X, but only specific dates are actually relevant).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..llm.base import LLMProvider
from .recaller import RecallResult

logger = logging.getLogger(__name__)

_RERANK_PROMPT = """Rank these memories by how well they answer a query.

QUERY: {query}

MEMORIES:
{memories}

RULES:
1. Output ONLY a space-separated list of memory IDs, most relevant first.
2. Include only memories that are actually relevant (skip irrelevant ones).
3. If none are relevant, output: NONE
4. No explanation, no prose, just the IDs.

RELEVANT_IDS:"""


@dataclass
class RerankResult:
    """Reranked memory with new position."""

    result: RecallResult
    new_rank: int  # 0-indexed
    original_rank: int


class Reranker:
    """Rerank retrieval results using an LLM.

    Args:
        llm: LLM provider (Gemini Flash works well with thinkingBudget=0)
        max_items: how many top-K to rerank (reranking all is expensive)
        max_tokens: LLM output budget (rerank returns just IDs, 200 is enough)
    """

    def __init__(
        self,
        llm: LLMProvider,
        max_items: int = 20,
        max_tokens: int = 200,
    ):
        self.llm = llm
        self.max_items = max_items
        self.max_tokens = max_tokens

    def rerank(
        self,
        query: str,
        results: list[RecallResult],
    ) -> list[RecallResult]:
        """Rerank top results by LLM-judged relevance.

        Items beyond max_items stay at their original positions after reranked
        prefix. If LLM fails, original order is preserved.
        """
        if not results:
            return []
        head = results[: self.max_items]
        tail = results[self.max_items :]

        prompt = self._build_prompt(query, head)
        resp = self.llm.generate(prompt, max_tokens=self.max_tokens, temperature=0.0)
        if not resp.ok:
            logger.warning("rerank failed, keeping original order: %s", resp.error)
            return results  # graceful fallback

        ranked_ids = self._parse_ids(resp.text)
        if not ranked_ids:
            logger.debug("rerank returned no relevant ids, keeping original order")
            return results

        # Build id → result map. Also keep a prefix-indexed fallback so
        # LLMs that drop trailing segments (e.g. `MEMORY.md` instead of
        # `MEMORY.md:45`) still resolve to the right result.
        id_map = {str(r.id): r for r in head}

        def _resolve(rid: str) -> RecallResult | None:
            if rid in id_map:
                return id_map[rid]
            # Fuzzy fallback: longest full-id that starts with the candidate,
            # or that the candidate starts with. This catches composite ids
            # (paths / namespaced keys) when the LLM emits a shorter form.
            matches = [
                full for full in id_map
                if full.startswith(rid) or rid.startswith(full)
            ]
            if len(matches) == 1:
                return id_map[matches[0]]
            if len(matches) > 1:
                # Prefer the longest match (most specific)
                matches.sort(key=len, reverse=True)
                return id_map[matches[0]]
            return None

        reordered: list[RecallResult] = []
        seen: set[str] = set()
        for rid in ranked_ids:
            r = _resolve(rid)
            if r is not None and str(r.id) not in seen:
                reordered.append(r)
                seen.add(str(r.id))
        # Append results the LLM didn't rank (keeps them available)
        for r in head:
            if str(r.id) not in seen:
                reordered.append(r)
        reordered.extend(tail)
        return reordered

    def _build_prompt(self, query: str, results: list[RecallResult]) -> str:
        lines = []
        for r in results:
            text = r.text.strip().replace("\n", " ")[:300]
            lines.append(f"ID={r.id}: {text}")
        memories_str = "\n".join(lines)
        return _RERANK_PROMPT.format(query=query, memories=memories_str)

    @staticmethod
    def _parse_ids(raw: str) -> list[str]:
        """Extract IDs from LLM output like '3 7 1' or 'NONE'.

        IDs can be integers, strings, or composite forms (paths with slashes
        and colons, e.g. `notes/MEMORY.md:45` or `graph:alice-works-on-x`).

        The regex captures any run of non-whitespace, non-comma characters,
        which covers all forms we emit from Retrievers.
        """
        text = raw.strip()
        if text.upper().startswith("NONE"):
            return []
        # Take only the first line (LLM might add prose despite instructions)
        first_line = text.split("\n", 1)[0]
        # Split on whitespace / commas; keep composite tokens intact
        tokens = [t.strip().rstrip(".") for t in re.split(r"[\s,]+", first_line) if t.strip()]
        return [t for t in tokens if t.upper() not in {"RELEVANT_IDS", "NONE", ""}]
