"""Generic markdown indexer — frontmatter → payload, links → graph edges.

Indexes a folder of markdown files: YAML frontmatter becomes payload fields
(usable as recall filters), the markdown-aware chunker carries heading paths,
and ``[[wikilinks]]`` / inline ``[text](target.md)`` links become graph edges.
Built for arbitrary markdown folders; Obsidian vaults work as a side effect.
"""

from __future__ import annotations

from .indexer import LinkEdge, MarkdownChunk, MarkdownCollection, collect_markdown
from .parse import extract_links, parse_frontmatter
from .sync import FileSyncResult, MarkdownSyncer

__all__ = [
    "collect_markdown",
    "MarkdownChunk",
    "LinkEdge",
    "MarkdownCollection",
    "parse_frontmatter",
    "extract_links",
    "MarkdownSyncer",
    "FileSyncResult",
]
