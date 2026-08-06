"""Syntax-aware chunking for source-code corpora.

Fixed-size character windows — right for prose — routinely cut source files
mid-function, which hurts retrieval badly: the half that matched the query
often lacks the signature, and the half with the signature lacks the body.
This module splits code at TOP-LEVEL definition boundaries instead, so a
chunk is usually one function/class/const with its doc comment.

The design contract (deliberately narrow, so the heuristics can be simple):

- **Boundaries are hints, correctness is the partition.** Detection is a
  per-language-family line regex for top-level definitions — no brace
  counting, no string/comment tracking, no AST. A missed boundary merely
  merges two units into one chunk; a segment that outgrows ``max_chars`` is
  split at character offsets exactly like the classic chunker. The worst
  case IS the status quo, never worse.
- **Chunks partition the file.** Every chunk's ``offset`` is the exact
  character offset of its text in the file, and chunks cover the file in
  order without overlap — so ``stable_chunk_id``, ``--prune``, payload
  refresh and `mnemostack resolve` (hash + position) all work unchanged.
- **Symbols are best-effort.** When the boundary line yields a name it is
  recorded in the payload; when it doesn't, the chunk simply has no symbol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "CODE_EXTENSIONS",
    "CodeChunk",
    "chunk_code",
    "identifier_tokens",
    "language_for",
]

#: Extension → language. A curated set of widely-used languages; unknown
#: extensions are simply not treated as code (the caller keeps its normal
#: handling). Lowercase keys, leading dot included.
CODE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "sql",
    ".lua": "lua",
}

# One boundary regex per language family. Anchored at column 0 (top-level):
# indentation-based languages nest by indent, brace-based ones almost always
# indent members — so column-0 definition keywords are a high-precision,
# low-recall signal, which is exactly the right trade-off under the
# "boundaries are hints" contract.
_PY_BOUNDARY = re.compile(
    r"^(?:@\w|(?:async\s+)?def\s+(?P<name>\w+)|class\s+(?P<cname>\w+))"
)
_BRACE_BOUNDARY = re.compile(
    r"^(?:export\s+(?:default\s+)?)?"
    r"(?:pub(?:\([^)]*\))?\s+|public\s+|private\s+|protected\s+|internal\s+|"
    r"static\s+|final\s+|abstract\s+|sealed\s+|partial\s+|unsafe\s+|"
    r"(?:async\s+))*"
    r"(?:function\s*\*?\s*(?P<fname>\w+)?|class\s+(?P<cname>\w+)|"
    r"interface\s+(?P<iname>\w+)|struct\s+(?P<sname>\w+)|enum\s+(?P<ename>\w+)|"
    r"trait\s+(?P<tname>\w+)|impl\b|func\s+(?:\([^)]*\)\s*)?(?P<gname>\w+)|"
    r"fn\s+(?P<rname>\w+)|def\s+(?P<dname>\w+)|"
    r"(?:const|let|var)\s+(?P<vname>\w+)\s*=|"
    r"(?:module|namespace|object|type)\s+(?P<mname>\w+))"
)
_SHELL_BOUNDARY = re.compile(r"^(?:function\s+(?P<name>\w+)|(?P<fname>[\w.-]+)\s*\(\)\s*\{)")
_SQL_BOUNDARY = re.compile(
    r"^(?:CREATE|ALTER|DROP|INSERT|UPDATE|DELETE|WITH|SELECT)\b", re.IGNORECASE
)
# C/C++ functions are TYPE-prefixed, not keyword-led: `static int parse(...)`.
# One-or-more type tokens, then the name, then an open paren with no `;` on
# the line (a `;` means a prototype, not a definition).
_C_FUNC_PATTERN = (
    r"^(?:[A-Za-z_][\w:<>,\*&\[\]]*[ \t]+)+[\*&]*(?P<cfn>[A-Za-z_]\w*)\s*\([^;]*$"
)
_C_BOUNDARY = re.compile(
    "(?:" + _BRACE_BOUNDARY.pattern + ")|(?:" + _C_FUNC_PATTERN + ")"
)

_BOUNDARIES: dict[str, re.Pattern[str]] = {
    "python": _PY_BOUNDARY,
    "ruby": _PY_BOUNDARY,  # def/class at column 0, same shape
    "shell": _SHELL_BOUNDARY,
    "sql": _SQL_BOUNDARY,
    "c": _C_BOUNDARY,
    "cpp": _C_BOUNDARY,
}
# Everything else in CODE_EXTENSIONS uses the brace-family regex.
_DEFAULT_BOUNDARY = _BRACE_BOUNDARY

#: Line prefixes (after indentation) treated as comments when deciding
#: whether a trailing block belongs to the NEXT definition — a doc comment
#: directly above a boundary must travel with the definition it documents.
_COMMENT_PREFIXES = ("#", "//", "/*", "*", "--")


def _is_comment_line(line: str) -> bool:
    stripped = line.lstrip()
    return bool(stripped) and stripped.startswith(_COMMENT_PREFIXES)

#: Below this many characters a segment keeps accumulating even across a
#: boundary line — one chunk per two-line helper would fragment retrieval
#: (and the vector count) for no gain.
MIN_SEGMENT_CHARS = 200

#: Cap on the identifier subtoken string written to the payload — a bound,
#: not a ranking decision (the lexical arm gates on exact tokens anyway).
MAX_IDENTIFIER_TOKENS = 256

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


@dataclass
class CodeChunk:
    """One code chunk: exact file offset, text, and best-effort metadata."""

    offset: int
    text: str
    language: str
    symbol: str | None = None
    #: Deduplicated, order-preserving identifier subtokens (see
    #: :func:`identifier_tokens`) — written to the payload so a lexical arm
    #: can gate on ``snake_case``/``camelCase`` parts.
    tokens: list[str] = field(default_factory=list)


def language_for(filename: str) -> str | None:
    """Language for a filename by extension, or None when not code."""
    dot = filename.rfind(".")
    if dot < 0:
        return None
    return CODE_EXTENSIONS.get(filename[dot:].lower())


def identifier_tokens(text: str, *, limit: int = MAX_IDENTIFIER_TOKENS) -> list[str]:
    """Identifier subtokens for lexical gating, lowercased and deduplicated.

    ``parseHttpRequest`` / ``parse_http_request`` both yield
    ``["parse", "http", "request", "parsehttprequest", ...]`` — the full
    identifier is kept alongside its parts so exact-name queries still gate.
    Order of first appearance is preserved; single characters are dropped
    (pure noise for a token gate).
    """
    out: list[str] = []
    seen: set[str] = set()
    for match in _IDENTIFIER.finditer(text):
        ident = match.group(0)
        parts = [p for chunk in ident.split("_") for p in _CAMEL_SPLIT.split(chunk) if p]
        for token in (*parts, ident):
            low = token.lower()
            if len(low) < 2 or low in seen:
                continue
            seen.add(low)
            out.append(low)
            if len(out) >= limit:
                return out
    return out


def _symbol_of(match: re.Match[str]) -> str | None:
    for value in match.groupdict().values():
        if value:
            return value
    return None


def chunk_code(
    text: str,
    language: str,
    *,
    max_chars: int = 2000,
    min_chars: int = MIN_SEGMENT_CHARS,
) -> list[CodeChunk]:
    """Split source text at top-level definition boundaries.

    Returns chunks whose offsets exactly partition ``text`` (in order, no
    overlap, whitespace-only chunks dropped — mirroring the classic
    chunker). A segment larger than ``max_chars`` is split at character
    offsets, so pathological files degrade to today's behavior instead of
    producing unbounded chunks.
    """
    boundary = _BOUNDARIES.get(language, _DEFAULT_BOUNDARY)
    lines = text.splitlines(keepends=True)
    # A --chunk-size below the merge minimum must still win: otherwise every
    # boundary would be merged past it and pass 2 would char-split anyway,
    # silently defeating the boundary alignment the caller asked for.
    min_chars = min(min_chars, max_chars)

    # Pass 1: segment at boundary lines (respecting the minimum size).
    segments: list[tuple[int, str, str | None]] = []  # (offset, text, symbol)
    seg_start = 0
    seg_parts: list[str] = []
    seg_len = 0
    seg_symbol: str | None = None
    offset = 0
    for line in lines:
        match = boundary.match(line)
        if match is not None:
            if seg_parts and seg_len >= min_chars:
                # A doc comment directly above the definition belongs to the
                # definition, not to the previous chunk: peel the trailing
                # comment run off and carry it into the new segment.
                carry_idx = len(seg_parts)
                while carry_idx > 0 and _is_comment_line(seg_parts[carry_idx - 1]):
                    carry_idx -= 1
                head = seg_parts[:carry_idx]
                carried = seg_parts[carry_idx:]
                if head:
                    segments.append((seg_start, "".join(head), seg_symbol))
                    carried_len = sum(len(part) for part in carried)
                    seg_start = offset - carried_len
                    seg_parts = carried
                    seg_len = carried_len
                    seg_symbol = _symbol_of(match)
                elif seg_symbol is None:
                    # The whole segment was comments — merge it into the
                    # definition it precedes (and adopt the name).
                    seg_symbol = _symbol_of(match)
            elif seg_symbol is None:
                # A boundary merged into a still-small unnamed segment (file
                # preamble, tiny helpers) names it — the first definition is
                # what a reader would call this chunk.
                seg_symbol = _symbol_of(match)
            # else: a boundary inside a still-small NAMED segment is merged
            # into it (MIN_SEGMENT_CHARS) and the opener's name stays.
        seg_parts.append(line)
        seg_len += len(line)
        offset += len(line)
    if seg_parts:
        segments.append((seg_start, "".join(seg_parts), seg_symbol))

    # Pass 2: enforce max size + drop whitespace-only chunks.
    chunks: list[CodeChunk] = []
    for seg_offset, seg_text, symbol in segments:
        pieces: list[tuple[int, str, str | None]]
        if len(seg_text) <= max_chars:
            pieces = [(seg_offset, seg_text, symbol)]
        else:
            pieces = [
                (seg_offset + i, seg_text[i : i + max_chars], symbol if i == 0 else None)
                for i in range(0, len(seg_text), max_chars)
            ]
        for piece_offset, piece_text, piece_symbol in pieces:
            if not piece_text.strip():
                continue
            chunks.append(
                CodeChunk(
                    offset=piece_offset,
                    text=piece_text,
                    language=language,
                    symbol=piece_symbol,
                    tokens=identifier_tokens(piece_text),
                )
            )
    return chunks
