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

logger = logging.getLogger(__name__)

# A frontmatter block is a YAML document fenced by --- lines at the very top of
# the file. The content may be empty (``---\n---``). The closing fence must be
# on its own line at column 0 (``re.MULTILINE`` anchors it), so a scalar value
# ending in ``---`` inside the YAML isn't mistaken for the fence. It may be the
# last line of the file (no trailing newline).
_FRONTMATTER_RE = re.compile(
    r"^---[ \t]*\r?\n(.*?)^(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)", re.DOTALL | re.MULTILINE
)


_MD = MarkdownIt("commonmark")

# ``[[Target]]`` / ``[[Target|alias]]`` / ``[[Target#anchor]]`` — captures the
# target. Applied only to the parser's plain-text tokens (never code, comments,
# or the label of a real markdown link), so it inherits correct exclusion.
_WIKILINK_RE = re.compile(r"\[\[([^\]|#\n]+)(?:[|#][^\]\n]*)?\]\]")


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
    # A masked escaped bracket (possibly percent-encoded in an href, decoded by
    # unquote above) becomes a literal ``[`` — its CommonMark meaning.
    target = target.replace(_ESC_BRACKET, "[")
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
    and the path is percent-decoded first, so ``paper.pdf?download=1`` and
    ``paper%2Epdf`` are both recognized as assets.
    """
    anchorless = unquote(raw.split("#", 1)[0].split("?", 1)[0].split("|", 1)[0])
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


# A URI scheme per RFC 3986: a letter, then letters/digits/+/-/. then a colon.
# Any such prefix (``http:``, ``mailto:``, ``tel:``, ``data:``, ...) marks an
# external link, not an intra-corpus note.
_URI_SCHEME_RE = re.compile(r"[a-z][a-z0-9+.\-]*:", re.IGNORECASE)


def _is_external(dest: str) -> bool:
    # Protocol-relative (``//host``), pure anchor (``#x``), or any URI scheme.
    return dest.startswith(("//", "#")) or bool(_URI_SCHEME_RE.match(dest))


# Private-use sentinel that a backslash-escaped ``[`` is swapped to before
# parsing, so an escaped opener can never match ``[[`` or start a link. Restored
# to a literal ``[`` when a real destination/target is read back.
_ESC_BRACKET = "\ue000"


def _mask_escaped_brackets(text: str) -> str:
    """Replace a backslash-escaped ``[`` with a private-use sentinel.

    An escaped ``\\[`` is literal text in CommonMark, and markdown-it drops the
    backslash — so a rendered ``\\[[Draft]]`` or ``\\[x](note.md)`` looks like a
    real wikilink/link in the parsed text tokens. Neutralizing the escaped ``[``
    *before* parsing (on a scan-copy only, never the chunked body) means an
    escaped opener can't match — no code/comment/label region-detection needed,
    since the parser already excludes those. An escaped backslash ``\\\\`` is
    preserved so a genuine following ``[`` still forms a link.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == "[":
                out.append(_ESC_BRACKET)  # escaped [ → sentinel (won't match [[ )
            else:
                out.append(c)  # keep the escape for markdown-it
                out.append(nxt)
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def extract_links(text: str) -> list[Link]:
    """Extract outgoing link targets (inline markdown links + ``[[wikilinks]]``).

    Uses a CommonMark parser, so link syntax inside code (fenced, indented, or
    inline spans) and inside HTML comments/blocks is not a reference and is
    skipped; balanced brackets/parens, titles, and angle-bracketed/percent-
    encoded destinations are handled to spec. Backslash-escaped openers
    (``\\[x](...)``, ``\\[[...]]``) are neutralized before parsing so they can't
    become edges. Inline links come from ``link_open`` hrefs; ``[[wikilinks]]``
    are matched only on the parser's plain-text tokens (so a ``[[B]]`` inside
    another link's label is not a separate edge). Any URI scheme, protocol-
    relative ``//``, pure anchors, image embeds (``![[...]]``), and non-note file
    targets are dropped. De-dup is keyed by ``(style, target)`` so a wikilink
    and an inline link to the same name both survive — they resolve differently.
    """
    seen: dict[tuple[bool, str], Link] = {}
    link_depth = 0
    for tok in _iter_inline_tokens(_mask_escaped_brackets(text)):
        if tok.type == "link_open":
            if link_depth == 0:
                # An escaped bracket in the destination (``foo\[bar\].md``) was
                # masked to the sentinel; ``_strip_target`` restores it after
                # decoding, so the real sibling resolves.
                dest = tok.attrGet("href") or ""
                if not _is_external(dest) and _is_note_target(dest):
                    key = _strip_target(dest)
                    if key:
                        seen.setdefault((False, key), Link(target=key, is_wikilink=False))
            link_depth += 1
        elif tok.type == "link_close":
            link_depth = max(0, link_depth - 1)
        elif tok.type == "text" and link_depth == 0:
            for m in _WIKILINK_RE.finditer(tok.content):
                if m.start() > 0 and tok.content[m.start() - 1] == "!":
                    continue  # ![[...]] embed, not a wikilink
                if not _is_note_target(m.group(1)):
                    continue
                key = _strip_target(m.group(1))
                if key:
                    seen.setdefault((True, key), Link(target=key, is_wikilink=True))
    return list(seen.values())
