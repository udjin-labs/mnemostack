"""
Embedding providers for memvault.

Provider registry pattern — user selects provider via config.
Recommended: Gemini (best quality, needs API key) or Ollama (local, no key).

Usage:
    from memvault.embeddings import get_provider
    provider = get_provider('gemini', model='gemini-embedding-001')
    vec = provider.embed('some text')
"""

from .base import EmbeddingProvider
from .registry import get_provider, list_providers, register_provider

__all__ = ["EmbeddingProvider", "get_provider", "list_providers", "register_provider"]
