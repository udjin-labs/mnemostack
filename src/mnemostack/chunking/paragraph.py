"""Paragraph-based chunker — splits on blank lines, merges short paragraphs."""
from __future__ import annotations

import re

from .base import Chunk, Chunker

_PARA_SPLIT = re.compile(r"\n\s*\n")


class ParagraphChunker(Chunker):
    """Split on blank lines. Merge small paragraphs up to target size.

    Good default for prose. Keeps sentences together.

    Args:
        chunk_size: target char count per chunk
        min_chunk: don't emit chunks smaller than this unless end of text
    """

    def __init__(self, chunk_size: int = 800, min_chunk: int = 200):
        self.chunk_size = chunk_size
        self.min_chunk = min_chunk

    def chunk(self, text: str) -> list[Chunk]:
        if not text.strip():
            return []
        # Find all paragraph boundaries to preserve offsets
        paragraphs: list[tuple[int, str]] = []
        offset = 0
        for part in _PARA_SPLIT.split(text):
            if part.strip():
                # Find the actual offset in original text
                idx = text.find(part, offset)
                if idx == -1:
                    idx = offset
                paragraphs.append((idx, part.strip()))
                offset = idx + len(part)

        chunks: list[Chunk] = []
        buffer: list[str] = []
        buffer_offset: int | None = None
        buffer_len = 0

        def flush():
            nonlocal buffer, buffer_offset, buffer_len
            if buffer:
                joined = "\n\n".join(buffer)
                chunks.append(Chunk(text=joined, offset=buffer_offset or 0))
                buffer = []
                buffer_offset = None
                buffer_len = 0

        for para_offset, para in paragraphs:
            para_len = len(para)
            if buffer_offset is None:
                buffer_offset = para_offset

            # If single paragraph is already too large, flush current buffer then emit alone
            if para_len >= self.chunk_size:
                flush()
                chunks.append(Chunk(text=para, offset=para_offset))
                continue

            # If adding would exceed target, flush first
            if buffer_len + para_len + 2 > self.chunk_size and buffer_len >= self.min_chunk:
                flush()
                buffer_offset = para_offset

            buffer.append(para)
            buffer_len += para_len + 2

        flush()
        return chunks
