"""Abstract base class for embedding providers."""
from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Base interface for embedding providers.

    Subclasses must implement `embed`, `embed_batch`, and the `dimension` / `name` properties.
    Providers should handle their own errors gracefully and return an empty list on failure.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return embedding vector for a single text. Empty list on failure."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for multiple texts. Empty list for failed items."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension returned by this provider/model."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier like 'gemini:embedding-001' — used in logs and config."""

    def health_check(self) -> tuple[bool, str]:
        """Lightweight reachability check. Returns (is_healthy, message)."""
        try:
            v = self.embed("healthcheck")
            if v and len(v) == self.dimension:
                return True, f"ok, dim={len(v)}"
            return False, f"unexpected dim: got {len(v)}, expected {self.dimension}"
        except Exception as e:  # noqa: BLE001
            return False, f"error: {e}"
