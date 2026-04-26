"""Specificity resolver for replacing generic answer placeholders.

This module is intentionally conservative: it only triggers on short generic
phrases that commonly hurt factual QA scoring, then asks the LLM to rewrite the
already-generated answer using exact names present in the retrieved memories.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from ..llm.base import LLMProvider
from .recaller import RecallResult

_GENERIC_NOUNS = (
    "country",
    "home",
    "pet",
    "dog",
    "cat",
    "book",
    "job",
    "friend",
    "colleague",
    "spouse",
    "sister",
    "brother",
    "son",
    "daughter",
    "company",
    "school",
    "university",
    "city",
    "town",
)
_ARTICLE_NOUNS = (
    "book",
    "movie",
    "film",
    "game",
    "song",
    "place",
    "colleague",
    "friend",
    "city",
    "town",
    "country",
    "someone",
    "something",
)
_CATEGORY_PHRASES = (
    "video games",
    "social media",
)

_POSSESSIVE_RE = re.compile(
    rf"\b(?:her|his|their)\s+(?:(?:home|best)\s+)?(?:{'|'.join(_GENERIC_NOUNS)})\b",
    re.IGNORECASE,
)
_ARTICLE_RE = re.compile(
    rf"\b(?:a|an|the|that)\s+(?:{'|'.join(_ARTICLE_NOUNS)})\b",
    re.IGNORECASE,
)
_CATEGORY_RE = re.compile(
    rf"\b(?:{'|'.join(re.escape(phrase) for phrase in _CATEGORY_PHRASES)})\b",
    re.IGNORECASE,
)
_SPECIFIC_AFTER_RE = re.compile(r"^\s+[A-Z][\w'’-]+")

_QUERY_BY_NOUN = {
    "country": "specific name of country where the person lives or is from",
    "home": "specific home country or hometown",
    "pet": "specific pet name",
    "dog": "specific dog name",
    "cat": "specific cat name",
    "book": "specific book title",
    "movie": "specific movie title",
    "film": "specific film title",
    "game": "specific game title or platform",
    "friend": "specific person's name",
    "colleague": "specific colleague's name",
    "someone": "specific person's name",
    "spouse": "specific spouse's name",
    "sister": "specific sister's name",
    "brother": "specific brother's name",
    "son": "specific son's name",
    "daughter": "specific daughter's name",
    "company": "specific company name",
    "school": "specific school name",
    "university": "specific university name",
    "city": "specific city name",
    "town": "specific town name",
    "place": "specific place or location name",
    "song": "specific song title",
    "video games": "specific video game titles or platforms",
    "social media": "specific social media platform",
}


def detect_placeholders(text: str) -> list[tuple[str, str]]:
    """Return generic placeholders in *text* with targeted lookup queries.

    Avoids phrases that are already followed by an apparent proper name, e.g.
    ``her brother John``.
    """
    if not text:
        return []

    matches: list[tuple[int, int, str]] = []
    for regex in (_POSSESSIVE_RE, _ARTICLE_RE, _CATEGORY_RE):
        for match in regex.finditer(text):
            if _has_specific_name_after(text, match.end()):
                continue
            matches.append((match.start(), match.end(), match.group(0)))

    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    out: list[tuple[str, str]] = []
    seen_spans: list[tuple[int, int]] = []
    seen_text: set[str] = set()
    for start, end, placeholder in matches:
        if any(start < prev_end and end > prev_start for prev_start, prev_end in seen_spans):
            continue
        key = placeholder.lower()
        if key in seen_text:
            continue
        noun = _placeholder_key(key)
        out.append((placeholder, _QUERY_BY_NOUN.get(noun, f"specific name for {placeholder}")))
        seen_spans.append((start, end))
        seen_text.add(key)
    return out


def resolve_specificity(
    query: str,
    draft_answer: str,
    candidate_memories: Iterable[RecallResult | str],
    llm: LLMProvider,
) -> str:
    """Rewrite a draft answer with exact names from candidate memories.

    Returns the original draft unchanged when there are no placeholders, the LLM
    fails, or no replacement is supported by the memories.
    """
    placeholders = detect_placeholders(draft_answer)
    if not placeholders:
        return draft_answer

    prompt = _build_specificity_prompt(
        query=query,
        draft_answer=draft_answer,
        placeholders=placeholders,
        candidate_memories=candidate_memories,
    )
    try:
        resp = llm.generate(prompt, max_tokens=200)
    except Exception:  # noqa: BLE001 - resolver must never break answer generation
        return draft_answer
    if not resp.ok:
        return draft_answer
    rewritten = resp.text.strip()
    return rewritten or draft_answer


def _has_specific_name_after(text: str, end: int) -> bool:
    return bool(_SPECIFIC_AFTER_RE.match(text[end : end + 40]))


def _placeholder_key(placeholder: str) -> str:
    for phrase in _CATEGORY_PHRASES:
        if phrase in placeholder:
            return phrase
    words = placeholder.split()
    return words[-1] if words else placeholder


def _build_specificity_prompt(
    query: str,
    draft_answer: str,
    placeholders: list[tuple[str, str]],
    candidate_memories: Iterable[RecallResult | str],
) -> str:
    memories = _format_memories(candidate_memories)
    placeholder_lines = "\n".join(
        f"- {placeholder}: {candidate_query}"
        for placeholder, candidate_query in placeholders
    )
    return f"""REWRITE the answer below to replace generic placeholders with exact specific names found in the memories.

ORIGINAL QUESTION: {query}
DRAFT ANSWER: {draft_answer}
PLACEHOLDERS DETECTED:
{placeholder_lines}

MEMORIES:
{memories}

RULES:
1. Replace each placeholder with the EXACT name/title/location from memories.
2. If memories don't contain the specific name — keep the placeholder.
3. Don't invent. Don't generalize.
4. Output only the rewritten answer.

REWRITTEN_ANSWER:"""


def _format_memories(candidate_memories: Iterable[RecallResult | str]) -> str:
    lines: list[str] = []
    for i, memory in enumerate(candidate_memories, 1):
        if isinstance(memory, str):
            text = memory.strip().replace("\n", " ")
            lines.append(f"[{i}] {text[:500]}")
            continue
        text = memory.text.strip().replace("\n", " ")
        source = memory.payload.get("source", "")
        ts = memory.payload.get("timestamp", "")
        prefix = f"[{i}]"
        if ts:
            prefix = f"{prefix} [{ts[:10]}]"
        if source:
            prefix = f"{prefix} ({source})"
        lines.append(f"{prefix} {text[:500]}")
    return "\n".join(lines)
