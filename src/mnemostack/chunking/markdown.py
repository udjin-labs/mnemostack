"""Markdown chunker — splits on headers, preserves heading hierarchy."""

from __future__ import annotations

from markdown_it import MarkdownIt

from .base import Chunk, Chunker

# A CommonMark parser finds headings to spec — ATX (incl. 0-3 space indent) and
# Setext (``Title``/``====``) — and never mistakes a ``#`` inside code (fenced,
# indented, inline) for a heading.
_MD = MarkdownIt("commonmark")


def _line_start_offsets(text: str) -> list[int]:
    """Char offset of the start of each line (index = 0-based line number)."""
    offsets = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            offsets.append(i + 1)
    return offsets


class MarkdownChunker(Chunker):
    """Split markdown on headers, carrying heading path in metadata.

    Each chunk includes the full heading hierarchy (h1 > h2 > h3) so that
    semantically the chunk knows where it sits in the document tree. This
    dramatically improves retrieval for queries about specific sections.

    Code blocks (``` fences) are kept intact — we don't split inside them.

    Args:
        chunk_size: target char count per chunk (sections larger than this
                    are further split by CharChunker-style windows)
        include_heading_in_text: if True, prepend heading path to chunk text
                                 for better embedding quality
    """

    def __init__(
        self,
        chunk_size: int = 1200,
        include_heading_in_text: bool = True,
    ):
        self.chunk_size = chunk_size
        self.include_heading_in_text = include_heading_in_text

    def chunk(self, text: str) -> list[Chunk]:
        if self.chunk_size <= 0:
            # A non-positive size would wedge the large-section split loop
            # (sub_offset never advances) — fail fast instead of hanging.
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if not text.strip():
            return []

        # Find headings via the CommonMark parser (ATX + Setext; code excluded).
        tokens = _MD.parse(text)
        line_starts = _line_start_offsets(text)
        headers: list[tuple[int, int, str]] = []  # (start, level, title)
        for k, tok in enumerate(tokens):
            if tok.type != "heading_open" or not tok.map:
                continue
            level = int(tok.tag[1:])  # "h2" -> 2
            inline = tokens[k + 1] if k + 1 < len(tokens) else None
            title = inline.content.strip() if inline and inline.type == "inline" else ""
            start = line_starts[tok.map[0]] if tok.map[0] < len(line_starts) else 0
            headers.append((start, level, title))

        if not headers:
            # No headers — split by size so a large headingless note (e.g. a
            # plain README) still respects chunk_size and never overruns the
            # embedding provider's input limit.
            pieces: list[Chunk] = []
            self._emit_plain_windows(text.strip(), pieces)
            return pieces

        # Build sections: each section spans from one header to the next
        chunks: list[Chunk] = []
        heading_stack: list[tuple[int, str]] = []  # (level, title)

        # Lead-in text before the first header is real content — emit it (with an
        # empty heading path) so it is not silently dropped from the index.
        self._emit_plain_windows(text[: headers[0][0]].strip(), chunks)

        for i, (start, level, title) in enumerate(headers):
            # Maintain heading hierarchy
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))

            end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
            body = text[start:end].strip()
            if not body:
                continue

            heading_path = [t for _, t in heading_stack]

            # If section is small enough, emit as single chunk
            if len(body) <= self.chunk_size:
                chunk_text = body
                if self.include_heading_in_text and len(heading_path) > 1:
                    # Prepend the outer path as context (inner heading is already at top of body)
                    parent_path = " > ".join(heading_path[:-1])
                    chunk_text = f"[{parent_path}]\n{body}"
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        offset=start,
                        metadata={"heading_path": list(heading_path)},
                    )
                )
            else:
                # Too big — split further, keep heading context on each piece
                sub_offset = 0
                while sub_offset < len(body):
                    piece = body[sub_offset : sub_offset + self.chunk_size]
                    piece_text = piece
                    if self.include_heading_in_text:
                        path_str = " > ".join(heading_path)
                        piece_text = f"[{path_str}]\n{piece}"
                    chunks.append(
                        Chunk(
                            text=piece_text,
                            offset=start + sub_offset,
                            metadata={"heading_path": list(heading_path)},
                        )
                    )
                    sub_offset += self.chunk_size

        return chunks

    def _emit_plain_windows(self, body: str, chunks: list[Chunk]) -> None:
        """Append ``body`` as headingless chunks, windowed to ``chunk_size``.

        Used for a headingless note and for the lead-in text before the first
        header. Empty ``body`` appends nothing. Offsets are body-relative, which
        is enough for stable ids and ordering and never collides with the
        header-anchored section offsets that follow.
        """
        if not body:
            return
        if len(body) <= self.chunk_size:
            chunks.append(Chunk(text=body, offset=0, metadata={"heading_path": []}))
            return
        sub_offset = 0
        while sub_offset < len(body):
            chunks.append(
                Chunk(
                    text=body[sub_offset : sub_offset + self.chunk_size],
                    offset=sub_offset,
                    metadata={"heading_path": []},
                )
            )
            sub_offset += self.chunk_size
