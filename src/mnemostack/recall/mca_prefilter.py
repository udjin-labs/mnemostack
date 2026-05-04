"""Exact-token prefilter for technical identifiers (MCA-lite)."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .bm25 import BM25

if TYPE_CHECKING:
    from .recaller import RecallResult

_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PATH_RE = re.compile(r"(?<!\w)(?:~?/|\.\.?/)[^\s:;,'\"()<>]+")
_REL_PATH_RE = re.compile(r"\b[\w.-]+/[\w./-]+\b")
_DOTTED_MODULE_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]*(?:\.[a-zA-Z][a-zA-Z0-9]*){2,}\b")
_COMMON_BRANCH_RE = re.compile(r"\b(?:main|develop)\b")
_SNAKE_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]*\b")
_HYPHEN_DIGIT_RE = re.compile(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*-\d+[A-Za-z0-9-]*\b|\b[A-Za-z0-9]*\d[A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b")
_CAMEL_RE = re.compile(r"\b[A-Za-z]+[a-z][A-Z][A-Za-z0-9]*\b")
_COMMON_REL_PATH_FALSE_POSITIVES = {"и/или"}
_COMMON_BRANCHES = {"main", "develop"}


def _is_noise_exact_token(token: str) -> bool:
    token_key = token.lower()
    return token_key not in _COMMON_BRANCHES and (
        len(token) < 5 or token_key in _COMMON_REL_PATH_FALSE_POSITIVES
    )


def extract_exact_tokens(query: str) -> list[str]:
    """Extract technical tokens worth giving an exact-match lexical boost."""
    if not query:
        return []
    tokens: list[str] = []
    seen: set[str] = set()
    for regex in (_UUID_RE, _IP_RE, _PATH_RE, _REL_PATH_RE, _DOTTED_MODULE_RE, _COMMON_BRANCH_RE, _SNAKE_RE, _HYPHEN_DIGIT_RE, _CAMEL_RE):
        for match in regex.finditer(query):
            token = match.group(0).strip(".,;:!?)]}>")
            if not token or _is_noise_exact_token(token):
                continue
            key = token.lower()
            if key not in seen:
                seen.add(key)
                tokens.append(token)
    return tokens


def mca_prefilter(query: str, bm25: BM25, limit: int = 10) -> list[RecallResult]:
    """Run BM25 over extracted technical tokens and return RecallResult hits."""
    tokens = extract_exact_tokens(query)
    if not tokens:
        return []

    from .recaller import RecallResult

    hits = bm25.search(" ".join(tokens), limit=limit)
    return [
        RecallResult(
            id=doc.id,
            text=doc.text,
            score=score,
            payload=doc.payload or {},
            sources=["mca"],
        )
        for doc, score in hits
    ]
