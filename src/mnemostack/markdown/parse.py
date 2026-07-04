"""Parsing helpers for the markdown indexer: frontmatter + links.

Kept dependency-light — YAML frontmatter uses ``pyyaml`` (already a core
dependency); link extraction is plain regex. Both are fail-open: malformed
frontmatter yields no metadata (the body is still indexed), and link
extraction never raises.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote

import yaml

logger = logging.getLogger(__name__)

# A frontmatter block is a YAML document fenced by --- lines at the very top of
# the file. The closing fence is a line that is exactly --- (or ...); it may be
# the last line of the file (no trailing newline) — common for frontmatter-only
# notes — so accept either a newline or end-of-string after it.
_FRONTMATTER_RE = re.compile(
    r"^---[ \t]*\r?\n(.*?)\r?\n(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)", re.DOTALL
)

# ``[[Target]]`` / ``[[Target|alias]]`` / ``[[Target#heading]]`` — capture the
# target. Matched with finditer so we can see a leading ``!`` (embed) and skip it.
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")

# ``[text](target)`` inline links. The destination is either angle-bracketed
# (``<My Note.md>`` — allows spaces) or a bare run of non-space chars; capture
# both forms (groups 1 and 2). A leading ``!`` (image embed) is excluded.
_MDLINK_RE = re.compile(
    r"(?<!\!)\[[^\]]*\]\(\s*(?:<([^>]*)>|([^)\s]+))(?:\s+\"[^\"]*\")?\s*\)"
)


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split leading YAML frontmatter from the markdown body.

    Returns ``(metadata, body)``. When there is no frontmatter, or it is
    malformed, or it does not parse to a mapping, returns ``({}, text)`` — the
    body is always the text that should be chunked and indexed.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    try:
        loaded = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        logger.warning("skipping malformed frontmatter: %s", exc)
        return {}, text
    if not isinstance(loaded, dict):
        return {}, text
    body = text[match.end() :]
    return loaded, body


def _strip_target(raw: str) -> str:
    """Normalize a link target to a note key: drop anchor, alias, and .md."""
    target = raw.split("#", 1)[0].split("|", 1)[0].strip()
    target = unquote(target)  # decode %20 etc. so "My%20Note" == "My Note"
    target = target.rstrip("/")
    if target.lower().endswith(".md"):
        target = target[:-3]
    return target


def _is_note_target(raw: str) -> bool:
    """True when ``raw`` points at a note (``.md`` or extensionless), not an asset.

    A target whose final path segment has a non-``.md`` file extension (an
    image, pdf, ...) is not a note, so it never becomes a graph edge. Bare and
    ``.md`` targets are notes.
    """
    anchorless = raw.split("#", 1)[0].split("|", 1)[0]
    last = anchorless.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return not ("." in last and not last.lower().endswith(".md"))


@dataclass(frozen=True)
class Link:
    """An outgoing link target and how it should be resolved.

    ``target`` is normalized (no anchor/alias/``.md``). ``is_wikilink`` is True
    for ``[[...]]`` links (resolved by note name, corpus-wide) and False for
    inline ``[text](path)`` links (resolved relative to the source file first).
    """

    target: str
    is_wikilink: bool


def extract_links(text: str) -> list[Link]:
    """Extract outgoing link targets (wikilinks + inline markdown links).

    Returns normalized, de-duplicated ``Link`` records in first-seen order.
    External links (``http://``, ``https://``, ``mailto:``) and pure anchors
    (``#section``) are skipped — only intra-corpus references become edges.
    Image embeds (``![...](...)`` and ``![[...]]``) and non-note file targets
    (``.png``, ``.pdf``, ...) are ignored for both link styles.
    """
    seen: dict[str, Link] = {}
    for m in _WIKILINK_RE.finditer(text):
        # Skip Obsidian embeds ``![[...]]`` and non-note targets (``[[img.png]]``).
        if m.start() > 0 and text[m.start() - 1] == "!":
            continue
        raw = m.group(1)
        if not _is_note_target(raw):
            continue
        key = _strip_target(raw)
        if key:
            seen.setdefault(key, Link(target=key, is_wikilink=True))
    for m in _MDLINK_RE.finditer(text):
        raw = m.group(1) if m.group(1) is not None else m.group(2)
        low = raw.lower()
        if low.startswith(("http://", "https://", "mailto:", "#")) or "://" in low:
            continue
        if not _is_note_target(raw):
            continue
        key = _strip_target(raw)
        if key:
            seen.setdefault(key, Link(target=key, is_wikilink=False))
    return list(seen.values())
