"""Markdown chunker — splits on headers, preserves heading hierarchy."""
from __future__ import annotations

import re

from .base import Chunk, Chunker

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"^```", re.MULTILINE)


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
        if not text.strip():
            return []

        # Find header positions, respecting code fences (don't match # inside ```)
        code_ranges = self._code_block_ranges(text)
        headers: list[tuple[int, int, str]] = []  # (start, level, title)
        for match in _HEADER_RE.finditer(text):
            if not self._in_ranges(match.start(), code_ranges):
                level = len(match.group(1))
                title = match.group(2).strip()
                headers.append((match.start(), level, title))

        if not headers:
            # No headers — return the whole text (or split by size)
            return [Chunk(text=text.strip(), offset=0, metadata={"heading_path": []})]

        # Build sections: each section spans from one header to the next
        chunks: list[Chunk] = []
        heading_stack: list[tuple[int, str]] = []  # (level, title)

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

    @staticmethod
    def _code_block_ranges(text: str) -> list[tuple[int, int]]:
        """Find (start, end) ranges of fenced code blocks."""
        fences = [m.start() for m in _CODE_FENCE_RE.finditer(text)]
        ranges = []
        i = 0
        while i + 1 < len(fences):
            ranges.append((fences[i], fences[i + 1]))
            i += 2
        return ranges

    @staticmethod
    def _in_ranges(pos: int, ranges: list[tuple[int, int]]) -> bool:
        return any(start <= pos < end for start, end in ranges)
