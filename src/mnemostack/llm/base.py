"""Abstract base class for LLM providers (used for answer generation, reranking)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

DEFAULT_IMAGE_PROMPT = (
    "Describe this image for a memory index in 2-3 dense sentences: visible objects "
    "and their attributes, any text or signs VERBATIM, colors, setting, people and "
    "what they are doing. No speculation beyond what is visible."
)


@dataclass
class LLMResponse:
    """Structured LLM output."""

    text: str
    tokens_used: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class LLMProvider(ABC):
    """Base interface for LLM providers.

    Providers should handle their own errors gracefully — set `error` field
    in LLMResponse rather than raising. Thinking budget control is optional
    (for models that support it, like Gemini 2.5 Pro/Flash).
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate text from a prompt. Return LLMResponse with .text or .error."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier like 'gemini:2.5-flash' — used in logs."""

    def health_check(self) -> tuple[bool, str]:
        """Lightweight reachability check."""
        resp = self.generate("Reply with just OK.", max_tokens=10)
        if resp.ok:
            return True, f"ok — {resp.text[:30]}"
        return False, resp.error or "unknown error"

    def describe_image(
        self,
        image: bytes,
        mime_type: str = "image/jpeg",
        prompt: str = DEFAULT_IMAGE_PROMPT,
        max_tokens: int = 250,
    ) -> LLMResponse:
        """Describe an image for indexing (multimodal ingest). Optional capability.

        Providers with vision support override this; the default reports the
        gap as a normal LLMResponse error (fail-open, never raises), so
        callers can branch on `.ok` uniformly.
        """
        return LLMResponse(
            text="",
            error=f"{self.name} does not support image description",
        )
