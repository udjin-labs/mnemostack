"""Shared code-block detection for markdown chunking and link extraction.

Both the markdown chunker (heading detection) and the markdown link extractor
must ignore code — fenced blocks (``` / ~~~), indented code blocks (>=4 spaces),
and, for links, inline spans. Keeping the fence/indent logic here means the two
callers agree on what counts as code instead of drifting apart.
"""

from __future__ import annotations

import re

# A fence line: >=3 backticks or tildes at line start, indented up to 3 spaces,
# then the rest of the line (an info string on the opener; must be blank on the
# closer). Group 1 is the delimiter run, group 2 the remainder of the line.
_FENCE_RE = re.compile(r"^ {0,3}([`~]{3,})([^\n]*)$", re.MULTILINE)


def code_fence_ranges(text: str) -> list[tuple[int, int]]:
    """(start, end) spans of fenced code blocks.

    A fence closes only on a *matching bare* delimiter — the same character,
    length >= the opener, and nothing but whitespace after it (a line like
    ``` ``` not a close ``` is code, not a closer). An opener may carry an info
    string. An unterminated fence runs to EOF (CommonMark).
    """
    ranges: list[tuple[int, int]] = []
    open_start: int | None = None
    open_fence = ""
    for m in _FENCE_RE.finditer(text):
        token, rest = m.group(1), m.group(2)
        if open_start is None:
            open_start, open_fence = m.start(), token
        elif token[0] == open_fence[0] and len(token) >= len(open_fence) and not rest.strip():
            ranges.append((open_start, m.start()))
            open_start, open_fence = None, ""
    if open_start is not None:
        ranges.append((open_start, len(text)))
    return ranges


def _is_indented(line: str) -> bool:
    return line.startswith("    ") or line.startswith("\t")


def indented_code_ranges(text: str) -> list[tuple[int, int]]:
    """(start, end) spans of indented (>=4 space / tab) code blocks.

    A block starts on an indented non-blank line that follows a blank line (an
    indented code block cannot interrupt a paragraph) and extends over the
    following indented-or-blank lines. Trailing blank lines are not included.
    An approximation of CommonMark good enough to keep sample links/headings in
    indented code out of the graph and out of ``heading_path``.
    """
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)

    ranges: list[tuple[int, int]] = []
    n = len(lines)
    i = 0
    prev_blank = True  # start of document counts as "after a blank line"
    while i < n:
        line = lines[i]
        if prev_blank and _is_indented(line) and line.strip():
            start = offsets[i]
            last = i
            j = i
            while j < n and (not lines[j].strip() or _is_indented(lines[j])):
                if lines[j].strip():
                    last = j
                j += 1
            ranges.append((start, offsets[last] + len(lines[last])))
            i = j
            prev_blank = False
            continue
        prev_blank = not line.strip()
        i += 1
    return ranges


def code_ranges(text: str) -> list[tuple[int, int]]:
    """Fenced + indented code spans (the union the chunker masks for headings)."""
    return code_fence_ranges(text) + indented_code_ranges(text)
