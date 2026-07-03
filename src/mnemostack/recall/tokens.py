"""Token estimation and token-budget trimming for recall results.

Recall callers that assemble LLM prompts need to know roughly how many
tokens the returned memories will occupy, and often need a hard ceiling
("give me the best results that fit in 2000 tokens"). Real tokenizers are
model-specific and heavy, so the default counter here is a dependency-free
estimate; callers that need exact counts pass their own ``token_counter``
(e.g. ``lambda s: len(tiktoken_encoding.encode(s))``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .recaller import RecallResult

#: A callable that maps text to a token count.
TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` without a tokenizer dependency.

    Heuristic calibrated against BPE-family tokenizers: ASCII text averages
    ~4 characters per token, while non-ASCII scripts (Cyrillic, CJK, ...)
    encode denser — counted at ~2 characters per token so multilingual
    content is not under-budgeted. Non-empty text always counts as >= 1.
    """
    if not text:
        return 0
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    other_chars = len(text) - ascii_chars
    return max(1, round(ascii_chars / 4 + other_chars / 2))


def sum_tokens(results: list[RecallResult], counter: TokenCounter | None = None) -> int:
    """Total token count of the results' text under ``counter``."""
    count = counter or estimate_tokens
    return sum(count(r.text) for r in results)


def apply_token_budget(
    results: list[RecallResult],
    token_budget: int,
    counter: TokenCounter | None = None,
) -> tuple[list[RecallResult], int]:
    """Trim a ranked result list so its total text tokens fit ``token_budget``.

    Keeps the ranked prefix whose cumulative token count stays within the
    budget and stops at the first result that would exceed it — a hard cap,
    never a best-effort fill, so ranking order is preserved and the budget
    is never overshot (an oversized top result yields an empty list rather
    than a blown budget). Only ``RecallResult.text`` is counted; payload
    metadata and any prompt formatting overhead are the caller's margin.

    Returns ``(kept_results, tokens_of_kept)``.
    """
    if token_budget <= 0:
        raise ValueError(f"token_budget must be positive, got {token_budget}")
    count = counter or estimate_tokens
    kept: list[RecallResult] = []
    total = 0
    for result in results:
        tokens = count(result.text)
        if total + tokens > token_budget:
            break
        kept.append(result)
        total += tokens
    return kept, total
