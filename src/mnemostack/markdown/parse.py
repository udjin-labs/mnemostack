"""Parsing helpers for the markdown indexer: frontmatter + links.

Frontmatter is split with ``pyyaml`` (a core dependency). Links are extracted by
a real CommonMark parser (``markdown-it-py``) rather than hand-rolled regex, so
code spans/blocks, HTML comments, escapes, balanced brackets/parens, titles, and
the like are handled to spec. ``[[wikilinks]]`` (not CommonMark) are recognized
by a small inline plugin, so they too are correctly excluded from code/comments.
Both helpers are fail-open: malformed frontmatter yields no metadata (the body
is still indexed).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote

import yaml
from markdown_it import MarkdownIt
from markdown_it.rules_inline import StateInline

logger = logging.getLogger(__name__)

# A frontmatter block is a YAML document fenced by --- lines at the very top of
# the file. The content may be empty (``---\n---``). The closing fence must be
# on its own line at column 0 (``re.MULTILINE`` anchors it), so a scalar value
# ending in ``---`` inside the YAML isn't mistaken for the fence. It may be the
# last line of the file (no trailing newline).
_FRONTMATTER_RE = re.compile(
    r"^---[ \t]*\r?\n(.*?)^(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)", re.DOTALL | re.MULTILINE
)


def _wikilink_rule(state: StateInline, silent: bool) -> bool:
    """Inline rule: ``[[Target]]`` → a ``wikilink`` token (content = raw inside).

    Runs before the CommonMark ``link`` rule but *after* code-span parsing, so a
    ``[[...]]`` inside inline code (or a fenced/indented block, or an HTML
    comment) is never seen here — code exclusion comes for free from the parser.
    An Obsidian embed ``![[...]]`` (leading ``!``) is left to fall through to
    normal parsing (it becomes literal text, i.e. no edge).
    """
    pos = state.pos
    src = state.src
    if src[pos : pos + 2] != "[[":
        return False
    if pos > 0 and src[pos - 1] == "!":  # ![[...]] embed → not a wikilink
        return False
    end = src.find("]]", pos + 2)
    if end < 0:
        return False
    content = src[pos + 2 : end]
    if "[" in content or "]" in content:  # malformed / nested — not a wikilink
        return False
    if not silent:
        token = state.push("wikilink", "", 0)
        token.content = content
        token.markup = "[["
    state.pos = end + 2
    return True


def _make_md() -> MarkdownIt:
    md = MarkdownIt("commonmark")
    md.inline.ruler.before("link", "wikilink", _wikilink_rule)
    return md


_MD = _make_md()


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split leading YAML frontmatter from the markdown body.

    Returns ``(metadata, body)``. When there is no frontmatter at all, returns
    ``({}, text)``. When a frontmatter fence *is* present but its YAML is
    malformed or not a mapping (including an empty block), returns ``({}, body)``
    — the recognized fence is stripped so it is never embedded as chunk text.
    """
    # Files saved with a UTF-8 BOM start with U+FEFF, which would defeat the
    # anchored ^--- match; drop a leading BOM before looking for frontmatter.
    if text.startswith("\ufeff"):
        text = text[1:]
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    body = text[match.end() :]
    try:
        loaded = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        logger.warning("skipping malformed frontmatter: %s", exc)
        return {}, body
    if not isinstance(loaded, dict):
        return {}, body
    return loaded, body


def _strip_target(raw: str) -> str:
    """Normalize a link target to a note key: drop query/anchor/alias and .md."""
    target = raw.split("#", 1)[0].split("?", 1)[0].split("|", 1)[0].strip()
    target = unquote(target)  # decode %20 etc. so "My%20Note" == "My Note"
    target = target.rstrip("/")
    if target.lower().endswith(".md"):
        target = target[:-3]
    return target


# Common attachment/asset extensions. A link to one of these is not a note, so
# it never becomes a graph edge. A denylist (rather than "any dotted segment")
# keeps note basenames that merely contain dots — a daily note ``2026.07.04`` or
# ``v1.2.0`` — as real notes.
_ASSET_EXTS = frozenset(
    {
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".ico", ".tif",
        ".tiff", ".pdf", ".mp4", ".mov", ".webm", ".mkv", ".mp3", ".wav", ".ogg",
        ".flac", ".zip", ".gz", ".tar", ".7z", ".rar", ".docx", ".xlsx", ".pptx",
        ".doc", ".xls", ".ppt", ".csv", ".json", ".yaml", ".yml", ".toml",
    }
)


def _is_note_target(raw: str) -> bool:
    """True when ``raw`` points at a note, not an attachment/asset.

    A target whose final path segment ends with a known asset extension (image,
    pdf, archive, ...) is not a note, so it never becomes a graph edge. Bare
    targets, ``.md`` targets, and note names that merely contain dots
    (``2026.07.04``) are all notes. A ``?query`` (and ``#anchor``) is stripped
    first so ``paper.pdf?download=1`` is still recognized as an asset.
    """
    anchorless = raw.split("#", 1)[0].split("?", 1)[0].split("|", 1)[0]
    last = anchorless.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].lower()
    dot = last.rfind(".")
    return dot < 0 or last[dot:] not in _ASSET_EXTS


@dataclass(frozen=True)
class Link:
    """An outgoing link target and how it should be resolved.

    ``target`` is normalized (no anchor/alias/``.md``). ``is_wikilink`` is True
    for ``[[...]]`` links (resolved by note name, corpus-wide) and False for
    inline ``[text](path)`` links (resolved relative to the source file first).
    """

    target: str
    is_wikilink: bool


def _iter_inline_tokens(text: str):
    """Yield the inline child tokens of every block in ``text``."""
    for block in _MD.parse(text):
        if block.type == "inline" and block.children:
            yield from block.children


def _is_external(dest: str) -> bool:
    low = dest.lower()
    return low.startswith(("http://", "https://", "mailto:", "//", "#")) or "://" in low


def extract_links(text: str) -> list[Link]:
    """Extract outgoing link targets (inline markdown links + ``[[wikilinks]]``).

    Uses a CommonMark parser, so link syntax inside code (fenced, indented, or
    inline spans) and inside HTML comments is not a reference and is skipped;
    escapes, balanced brackets/parens, titles, and angle-bracketed/percent-
    encoded destinations are handled to spec. External links (``http(s)://``,
    ``mailto:``, protocol-relative ``//``) and pure anchors are dropped, as are
    image embeds and non-note file targets (``.png``, ``.pdf``, ...). De-dup is
    keyed by ``(style, target)`` so a wikilink and an inline link to the same
    name both survive — they resolve differently (corpus-wide vs. relative).
    """
    seen: dict[tuple[bool, str], Link] = {}
    for tok in _iter_inline_tokens(text):
        if tok.type == "link_open":
            dest = tok.attrGet("href") or ""
            if _is_external(dest) or not _is_note_target(dest):
                continue
            key = _strip_target(dest)
            if key:
                seen.setdefault((False, key), Link(target=key, is_wikilink=False))
        elif tok.type == "wikilink":
            raw = tok.content
            if not _is_note_target(raw):
                continue
            key = _strip_target(raw)
            if key:
                seen.setdefault((True, key), Link(target=key, is_wikilink=True))
    return list(seen.values())
