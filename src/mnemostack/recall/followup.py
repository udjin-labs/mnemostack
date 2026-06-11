"""Follow-up question rewriting — resolve pronouns and ellipses before recall.

Conversational RAG stumbles on context-dependent questions ("and who wrote
that?"): the recall query carries none of the conversation, so retrieval
misses. This helper rewrites a follow-up into a standalone question using
the caller's conversation history.

mnemostack deliberately holds no dialog state — the history lives with the
caller and is passed per call. The non-trivial part is NOT the LLM call but
deciding when to make it: rewriting a question that is already
self-contained wastes a call and can corrupt a good query. The default
delegates that judgement to the LLM inside one call (rule: output unchanged
if self-contained); pass `needs_rewrite=` to skip the call entirely for
queries your own heuristic considers standalone — that heuristic is
language-dependent, so core ships none.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from ..llm.base import LLMProvider
from ..observability import counter

logger = logging.getLogger(__name__)

_FOLLOWUP_PROMPT = """You resolve conversational follow-up questions into standalone questions.

CONVERSATION SO FAR (oldest first):
{history}

FOLLOW-UP QUESTION: {query}

RULES:
1. If the question is already self-contained (understandable without the conversation), output it EXACTLY as is.
2. Otherwise rewrite it into ONE standalone question: resolve pronouns and ellipses using the conversation. Keep the question's language and intent; do not add facts the conversation doesn't contain.
3. Do not answer the question. Output ONLY the question text.

STANDALONE_QUESTION:"""

_REQUIRED_PLACEHOLDERS = ("{history}", "{query}")


def rewrite_followup(
    query: str,
    history: Sequence[tuple[str, str] | str],
    llm: LLMProvider,
    *,
    needs_rewrite: Callable[[str], bool] | None = None,
    prompt_template: str | None = None,
    max_history: int = 4,
) -> str:
    """Return *query* rewritten into a standalone question, or unchanged.

    *history* is the caller's conversation, oldest first: `(question,
    answer)` pairs or plain strings; only the last *max_history* entries are
    shown to the LLM. *needs_rewrite* is an optional trigger — when it
    returns False the LLM is not called at all (use it to skip queries your
    heuristic considers self-contained). *prompt_template* must contain
    `{history}` and `{query}`.

    Fail-open: no history, an LLM error, or an empty response all return the
    original query.
    """
    if prompt_template is not None:
        missing = [ph for ph in _REQUIRED_PLACEHOLDERS if ph not in prompt_template]
        if missing:
            raise ValueError(f"prompt_template must contain {missing} placeholders")
    if max_history <= 0:
        # -0 == 0 in Python, so a [-max_history:] slice would send the WHOLE
        # history; this parameter gates privacy and token budget, so zero
        # must mean "show none" — and with no history there is nothing to
        # resolve against.
        return query
    if not query.strip() or not history:
        return query
    if needs_rewrite is not None and not needs_rewrite(query):
        return query

    template = prompt_template or _FOLLOWUP_PROMPT
    prompt = template.format(history=_format_history(history, max_history), query=query)
    try:
        resp = llm.generate(prompt, max_tokens=200)
    except Exception as exc:  # noqa: BLE001 — rewriting must never break recall
        logger.warning("rewrite_followup failed (%s) — using the original query", exc)
        counter("mnemostack.recall.followup_rewrite_failed", 1)
        return query
    if not resp.ok or not resp.text.strip():
        counter("mnemostack.recall.followup_rewrite_failed", 1)
        return query
    rewritten = resp.text.strip()
    # LLMs occasionally echo the prompt's final label back; strip it so the
    # marker never leaks into the recall query.
    if rewritten.upper().startswith("STANDALONE_QUESTION:"):
        rewritten = rewritten[len("STANDALONE_QUESTION:") :].strip()
    if not rewritten:
        return query
    if rewritten != query:
        counter("mnemostack.recall.followup_rewritten", 1)
    return rewritten


def _format_history(history: Sequence[tuple[str, str] | str], max_history: int) -> str:
    lines: list[str] = []
    for entry in list(history)[-max_history:]:
        if isinstance(entry, str):
            lines.append(entry.strip())
            continue
        question, answer = entry
        lines.append(f"Q: {question.strip()}")
        lines.append(f"A: {answer.strip()}")
    return "\n".join(lines)
