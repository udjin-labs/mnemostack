"""
Text chunking strategies for indexing.

Chunkers take text and return a list of Chunk objects with text + metadata
(e.g. heading path for markdown). Different strategies fit different content:

- CharChunker — fixed-size character split (baseline, works on any text)
- MarkdownChunker — splits on headers, preserves heading hierarchy
- ParagraphChunker — splits on blank lines, merges short paragraphs

Usage:
    from mnemostack.chunking import MarkdownChunker
    chunker = MarkdownChunker(chunk_size=800, overlap=100)
    for chunk in chunker.chunk(text):
        print(chunk.text, chunk.metadata)
"""

from .base import Chunk, Chunker
from .char import CharChunker
from .markdown import MarkdownChunker
from .messages import MessagePairChunker
from .paragraph import ParagraphChunker

__all__ = [
    "Chunk",
    "Chunker",
    "CharChunker",
    "MarkdownChunker",
    "MessagePairChunker",
    "ParagraphChunker",
]
