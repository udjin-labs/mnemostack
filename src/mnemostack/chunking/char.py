"""Fixed-size character chunker — baseline, language-agnostic."""
from __future__ import annotations

from .base import Chunk, Chunker


class CharChunker(Chunker):
    """Split text into fixed-size character chunks with overlap.

    Simple and fast. Doesn't respect sentence or word boundaries, which can
    cut tokens awkwardly — prefer ParagraphChunker or MarkdownChunker for
    natural-language content.

    Args:
        chunk_size: chars per chunk
        overlap: chars shared between consecutive chunks
    """

    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[Chunk]:
        if not text:
            return []
        chunks = []
        step = self.chunk_size - self.overlap
        offset = 0
        while offset < len(text):
            piece = text[offset : offset + self.chunk_size]
            if piece.strip():
                chunks.append(Chunk(text=piece, offset=offset))
            offset += step
        return chunks
