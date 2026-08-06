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
- **The recognized-construct set is deliberately FROZEN.** The boundary
  regexes cover the common top-level shapes of each family (definitions,
  types, module/namespace headers, C-family type-prefixed and qualified
  functions, declaration prefixes like doc comments / annotations /
  attributes / templates). Every language has further corners; chasing
  them is explicitly out of scope for a line heuristic — an unrecognized
  construct degrades to character chunking, which the contract above
  already prices in. Corpus-specific needs extend ``_BOUNDARIES``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

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
    r"local\s+|(?:async\s+))*"
    r"(?:function\s*\*?\s*(?P<fname>\w+)?|class\s+(?P<cname>\w+)|"
    r"interface\s+(?P<iname>\w+)|struct\s+(?P<sname>\w+)|"
    r"enum\s+(?:class\s+|struct\s+)?(?P<ename>\w+)|"
    r"trait\s+(?P<tname>\w+)|impl\b|func\s+(?:\([^)]*\)\s*)?(?P<gname>\w+)|"
    r"fn\s+(?P<rname>\w+)|fun\s+(?P<ktname>\w+)|def\s+(?P<dname>\w+)|"
    r"(?:const|let|var)\s+(?P<vname>\w+)\s*(?::[^=\n]*)?=|"
    r"(?:module|namespace|object|type)\s+(?P<mname>\w+))"
)
_RUBY_BOUNDARY = re.compile(
    r"^(?:module\s+(?P<mname>\w+)|class\s+(?P<cname>\w+)|"
    r"def\s+(?:self\.)?(?P<name>\w+[?!]?))"
)
_SHELL_BOUNDARY = re.compile(r"^(?:function\s+(?P<name>\w+)|(?P<fname>[\w.-]+)\s*\(\)\s*\{)")
# No SELECT here on purpose: a column-0 SELECT is as often the final query
# of a WITH CTE as an independent statement, and a line heuristic cannot
# tell them apart — splitting a CTE from its SELECT is strictly worse than
# merging two statements (the hints contract prices merges in). WITH and
# the DML/DDL keywords still open statements.
_SQL_BOUNDARY = re.compile(
    r"^(?:CREATE|ALTER|DROP|INSERT|UPDATE|DELETE|WITH)\b", re.IGNORECASE
)
# C/C++ functions are TYPE-prefixed, not keyword-led: `static int parse(...)`.
# One-or-more type tokens, then the name, then an open paren — on a line NOT
# ending in `;` (optionally followed by a trailing comment): a terminating
# semicolon means a prototype (or a declaration with an initializer call),
# while a one-line body `{ return n; }` is a definition despite containing
# semicolons.
_C_FUNC_PATTERN = (
    r"^(?:[A-Za-z_][\w:<>,\*&\[\]]*[ \t]+)+[\*&]*"
    r"(?P<cfn>[A-Za-z_]\w*(?:::~?\w+)*)\s*\((?!.*;\s*(?://.*|/\*.*)?$)"
)
_C_BOUNDARY = re.compile(
    "(?:" + _BRACE_BOUNDARY.pattern + ")|(?:" + _C_FUNC_PATTERN + ")"
)

_BOUNDARIES: dict[str, re.Pattern[str]] = {
    "python": _PY_BOUNDARY,
    "ruby": _RUBY_BOUNDARY,
    "shell": _SHELL_BOUNDARY,
    "sql": _SQL_BOUNDARY,
    "c": _C_BOUNDARY,
    "cpp": _C_BOUNDARY,
}
# Everything else in CODE_EXTENSIONS uses the brace-family regex.
_DEFAULT_BOUNDARY = _BRACE_BOUNDARY

#: Comment prefixes are LANGUAGE-FAMILY-specific when deciding whether a
#: trailing block belongs to the NEXT definition: `#` is a comment in the
#: hash family but a PREPROCESSOR DIRECTIVE in C/C++ — peeling an `#endif`
#: off a conditional block and gluing it to the next chunk would be worse
#: than not carrying at all.
_HASH_COMMENTS = ("#",)
_DASH_COMMENTS = ("--",)
_SLASH_COMMENTS = ("//", "/*", "*")
_CARRY_COMMENT_PREFIXES: dict[str, tuple[str, ...]] = {
    "python": _HASH_COMMENTS,
    "ruby": _HASH_COMMENTS,
    "shell": _HASH_COMMENTS,
    "sql": _DASH_COMMENTS,
    "lua": _DASH_COMMENTS,
}
#: Everything else (the brace family, incl. C/C++) uses slash comments.
_DEFAULT_CARRY_COMMENTS = _SLASH_COMMENTS

_TEMPLATE_PREFIX = re.compile(r"template\s*<")


def _is_carry_line(line: str, language: str) -> bool:
    """A line that belongs to the DEFINITION below it, not the chunk above.

    Doc comments (per family), annotations/decorators (``@Deprecated``,
    ``@Component``) and C++ ``template <...>`` prefixes are declaration
    prefixes: leaving them in the previous chunk would attach them to an
    unrelated symbol and strip the definition of its own metadata.
    Annotations/templates only exist in the brace family — a hash-family
    ``@`` line (a Ruby ivar, a Python decorator, which is a BOUNDARY there)
    is never carried.
    """
    stripped = line.lstrip()
    if not stripped:
        return False
    prefixes = _CARRY_COMMENT_PREFIXES.get(language, _DEFAULT_CARRY_COMMENTS)
    if stripped.startswith(prefixes):
        return True
    if language in _CARRY_COMMENT_PREFIXES:
        return False  # non-brace family: comments only
    return (
        stripped.startswith("@")
        or stripped.startswith("#[")  # Rust attributes: #[derive(...)], #[cfg(...)]
        or _TEMPLATE_PREFIX.match(stripped) is not None
    )

#: Below this many characters a segment keeps accumulating even across a
#: boundary line — one chunk per two-line helper would fragment retrieval
#: (and the vector count) for no gain.
MIN_SEGMENT_CHARS = 200

#: Cap on the identifier subtoken string written to the payload — a bound,
#: not a ranking decision (the lexical arm gates on exact tokens anyway).
MAX_IDENTIFIER_TOKENS = 256

#: The CLOSED set of payload keys the code indexer owns. The `_code_keys`
#: ownership record is validated against this set on read, so a legacy or
#: tampered record can never mark an unrelated payload field for deletion.
CODE_OWNED_KEYS = ("chunk_kind", "code_tokens", "language", "symbol")

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
    Order of first appearance is preserved. Tokens shorter than three
    characters are dropped: the lexical retriever's default
    ``min_token_len=3`` strips them from every QUERY gate anyway, so
    emitting them would create tokens that can never match — the producer
    minimum is deliberately aligned with the consumer's.
    """
    out: list[str] = []
    seen: set[str] = set()
    for match in _IDENTIFIER.finditer(text):
        ident = match.group(0)
        parts = [p for chunk in ident.split("_") for p in _CAMEL_SPLIT.split(chunk) if p]
        for token in (*parts, ident):
            low = token.lower()
            if len(low) < 3 or low in seen:
                continue
            seen.add(low)
            out.append(low)
            if len(out) >= limit:
                return out
    return out


def _symbol_of(match: re.Match[str]) -> str | None:
    for value in match.groupdict().values():
        if value:
            # A qualified C++ name (`Widget::render`) records the member name.
            return value.rsplit("::", 1)[-1]
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

    # Pass 1: segment at boundary lines (respecting the minimum size). Each
    # segment also records the boundaries MERGED into it (`internal`), so
    # pass 2 can split an oversized segment at real definition starts
    # instead of arbitrary character offsets.
    segments: list[tuple[int, str, str | None, list[tuple[int, str | None]]]] = []
    seg_start = 0
    seg_parts: list[str] = []
    seg_len = 0
    seg_symbol: str | None = None
    # Internal bounds as [offset, symbol, tail_line_start]: a nameless bound
    # (a decorator line — `@app.route` matches the boundary regex but yields
    # no name) extends through ADJACENT boundary lines instead of standing
    # alone, so pass 2 never cuts a decorator away from its definition.
    seg_internal: list[list[Any]] = []

    def _note_internal(line_start: int, prev_start: int, sym: str | None) -> None:
        if seg_internal and seg_internal[-1][1] is None and seg_internal[-1][2] == prev_start:
            seg_internal[-1][1] = sym
            seg_internal[-1][2] = line_start
        else:
            seg_internal.append([line_start, sym, line_start])

    offset = 0
    prev_line_start = 0
    for line in lines:
        match = boundary.match(line)
        if match is not None:
            if seg_parts and seg_len >= min_chars:
                # A doc comment / annotation / template prefix directly above
                # the definition belongs to the definition, not the previous
                # chunk: peel the trailing run off and carry it forward.
                carry_idx = len(seg_parts)
                while carry_idx > 0 and _is_carry_line(seg_parts[carry_idx - 1], language):
                    carry_idx -= 1
                head = seg_parts[:carry_idx]
                carried = seg_parts[carry_idx:]
                if head:
                    segments.append(
                        (seg_start, "".join(head), seg_symbol,
                         [(o, s) for o, s, _t in seg_internal])
                    )
                    carried_len = sum(len(part) for part in carried)
                    seg_start = offset - carried_len
                    seg_parts = carried
                    seg_len = carried_len
                    seg_symbol = _symbol_of(match)
                    seg_internal = []
                elif seg_symbol is None:
                    # The whole segment was comments — merge it into the
                    # definition it precedes (and adopt the name).
                    seg_symbol = _symbol_of(match)
                else:
                    _note_internal(offset, prev_line_start, _symbol_of(match))
            elif seg_symbol is None:
                # A boundary merged into a still-small unnamed segment (file
                # preamble, tiny helpers) names it — the first definition is
                # what a reader would call this chunk.
                seg_symbol = _symbol_of(match)
            else:
                # A boundary inside a still-small NAMED segment is merged in
                # (MIN_SEGMENT_CHARS), the opener's name stays — but pass 2
                # remembers where the definition started.
                _note_internal(offset, prev_line_start, _symbol_of(match))
        seg_parts.append(line)
        seg_len += len(line)
        prev_line_start = offset
        offset += len(line)
    if seg_parts:
        segments.append(
            (seg_start, "".join(seg_parts), seg_symbol,
             [(o, s) for o, s, _t in seg_internal])
        )

    # Pass 2: enforce max size + drop whitespace-only chunks. An oversized
    # segment first re-splits at its recorded internal definition starts —
    # a small helper merged with a large following function must not force
    # both into one arbitrary character split.
    chunks: list[CodeChunk] = []
    for seg_offset, seg_text, symbol, internal in segments:
        subsegments: list[tuple[int, str, str | None]]
        if len(seg_text) <= max_chars or not internal:
            subsegments = [(seg_offset, seg_text, symbol)]
        else:
            cuts = [seg_offset, *[b for b, _s in internal], seg_offset + len(seg_text)]
            names = [symbol, *[s for _b, s in internal]]
            subsegments = [
                (cuts[j], seg_text[cuts[j] - seg_offset : cuts[j + 1] - seg_offset], names[j])
                for j in range(len(cuts) - 1)
            ]
        pieces: list[tuple[int, str, str | None]] = []
        for sub_offset, sub_text, sub_symbol in subsegments:
            if len(sub_text) <= max_chars:
                pieces.append((sub_offset, sub_text, sub_symbol))
            else:
                pieces.extend(
                    (sub_offset + i, sub_text[i : i + max_chars], sub_symbol if i == 0 else None)
                    for i in range(0, len(sub_text), max_chars)
                )
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
