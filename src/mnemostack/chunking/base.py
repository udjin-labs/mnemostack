"""Chunker interface and Chunk dataclass."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A piece of text with positional and semantic metadata."""

    text: str
    offset: int = 0                     # character offset in source document
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)


class Chunker(ABC):
    """Base class for chunking strategies."""

    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]:
        """Split text into chunks."""
