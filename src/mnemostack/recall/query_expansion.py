"""LLM-backed query expansion for improving recall."""
from __future__ import annotations

from ..llm.base import LLMProvider


def expand_query(query: str, llm: LLMProvider, n_variants: int = 3) -> list[str]:
    """Generate alternative phrasings of a query for broader retrieval."""
    prompt = f"""Rewrite this question in {n_variants} different ways to improve search recall.
Keep the same meaning but use different words, include likely answer terms.
Return one variant per line, no numbering.

Question: {query}

Variants:"""
    resp = llm.generate(prompt, max_tokens=200)
    if not resp.ok:
        return []
    variants = [v.strip() for v in resp.text.strip().split("\n") if v.strip()]
    return variants[:n_variants]
