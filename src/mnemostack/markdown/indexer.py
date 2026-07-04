"""Collect chunks and link edges from a folder of markdown files.

Pure and I/O-light: reads the files, but does not embed, upsert, or touch the
graph — it returns the chunks (with frontmatter folded into the payload) and
the resolved wikilink/markdown-link edges, so the caller owns embedding,
upserting, and graph writes (and can reuse the existing skip-unchanged / prune
plumbing). Built for arbitrary markdown folders; an Obsidian vault is just one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from ..chunking import MarkdownChunker
from ..ingest import stable_chunk_id
from .parse import extract_links, parse_frontmatter


@dataclass
class MarkdownChunk:
    """One indexed chunk: a stable id, its text, and the payload to store."""

    id: str
    text: str
    payload: dict[str, Any]


@dataclass
class LinkEdge:
    """A resolved outgoing link: ``source`` file → ``target`` note.

    ``source`` is always a corpus-relative path. ``target`` is the relative
    path of the linked note when it resolves within the corpus, or the raw
    link key when it does not (a dangling link — still a useful edge).
    """

    source: str
    target: str
    resolved: bool = True


@dataclass
class MarkdownCollection:
    chunks: list[MarkdownChunk] = field(default_factory=list)
    edges: list[LinkEdge] = field(default_factory=list)
    #: Corpus-relative paths of every file visited, including ones that
    #: produced no chunks (empty / frontmatter-only) — the caller needs these
    #: to prune their old chunks and re-sync their (now absent) links.
    sources: list[str] = field(default_factory=list)

    @property
    def files(self) -> int:
        return len(self.sources)


def _rel(path: Path, base: Path) -> str:
    """Corpus-relative path with forward slashes (stable across platforms)."""
    return path.relative_to(base).as_posix()


def _norm_target(raw: str) -> str:
    """Normalize a link key for lookup: forward slashes, no leading ``./``."""
    key = raw.replace("\\", "/")
    while key.startswith("./"):
        key = key[2:]
    return key


def _resolve_relative(source_rel: str, target: str, all_rels: set[str]) -> str | None:
    """Resolve an inline link relative to its source file's directory.

    ``[c](../index.md)`` from ``sub/page.md`` resolves to ``index.md`` — so
    normal markdown relative links (and same-directory links in folders with
    duplicate basenames) point at the right note. Returns the matched
    corpus-relative path, or None if nothing resolves.
    """
    combined = PurePosixPath(source_rel).parent / _norm_target(target)
    parts: list[str] = []
    for part in combined.parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part not in (".", ""):
            parts.append(part)
    normalized = "/".join(parts)
    for candidate in (f"{normalized}.md", normalized):
        if candidate in all_rels:
            return candidate
    return None


def collect_markdown(
    root: str | Path,
    *,
    chunk_size: int = 1200,
    index_root: str | None = None,
) -> MarkdownCollection:
    """Walk ``root`` for ``*.md`` files and return their chunks + link edges.

    Frontmatter maps to payload fields (usable as recall filters); protected
    keys (``text``/``source``/``offset``/``index_root``) always win over a
    frontmatter key of the same name. The markdown-aware chunker carries the
    heading path into each chunk's payload. Links (``[[wikilinks]]`` and inline
    ``[text](target.md)``) resolve against the corpus by note name or relative
    path; unresolved targets become dangling edges.
    """
    root = Path(root)
    files = sorted(root.rglob("*.md")) if root.is_dir() else [root]
    base = root if root.is_dir() else root.parent

    # Resolution maps: note name (basename) and relative path (without .md)
    # both point at the canonical relative path. Keys are lower-cased so link
    # resolution is case-insensitive (Obsidian-style); path keys win over
    # bare names. `all_rels` is the set of real files for relative resolution.
    key_to_rel: dict[str, str] = {}
    all_rels: set[str] = set()
    for f in files:
        rel = _rel(f, base)
        all_rels.add(rel)
        rel_key = rel[:-3] if rel.lower().endswith(".md") else rel
        key_to_rel.setdefault(f.stem.lower(), rel)
        key_to_rel[rel_key.lower()] = rel

    chunker = MarkdownChunker(chunk_size=chunk_size)
    out = MarkdownCollection()
    for f in files:
        rel = _rel(f, base)
        text = f.read_text(encoding="utf-8", errors="ignore")
        meta, body = parse_frontmatter(text)
        out.sources.append(rel)

        for link in extract_links(body):
            norm = _norm_target(link.target)
            resolved: str | None = None
            if not link.is_wikilink:
                # Inline markdown link: resolve relative to the source file
                # first (handles ../ and same-dir links, and duplicate
                # basenames), then fall back to a corpus-wide name/path match.
                resolved = _resolve_relative(rel, link.target, all_rels)
            resolved = resolved or key_to_rel.get(link.target.lower()) or key_to_rel.get(norm.lower())
            out.edges.append(
                LinkEdge(
                    source=rel,
                    target=resolved or norm,
                    resolved=resolved is not None,
                )
            )

        for chunk in chunker.chunk(body):
            payload: dict[str, Any] = {
                **meta,
                "text": chunk.text,
                "source": rel,
                "offset": chunk.offset,
            }
            if index_root is not None:
                payload["index_root"] = index_root
            heading_path = chunk.metadata.get("heading_path")
            if heading_path:
                payload["heading_path"] = heading_path
            out.chunks.append(
                MarkdownChunk(
                    id=stable_chunk_id(rel, chunk.offset, chunk.text),
                    text=chunk.text,
                    payload=payload,
                )
            )
    return out
