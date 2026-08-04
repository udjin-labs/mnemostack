"""
Embedding providers for mnemostack.

Provider registry pattern — user selects provider via config.
Recommended: Gemini (best quality, needs API key) or Ollama (local, no key).

Usage:
    from mnemostack.embeddings import get_provider
    provider = get_provider('gemini', model='gemini-embedding-001')
    query_vec = provider.embed_query('some question')
    doc_vecs = provider.embed_documents(['chunk one', 'chunk two'])

`embed`/`embed_batch` remain the neutral primitives; the role methods apply
the model family's query/document input conventions (embedding profiles)
exactly once before delegating to them.
"""

from .base import EmbeddingCapabilities, EmbeddingProvider, ProviderProbeError
from .profiles import (
    IDENTITY_PROFILE,
    EmbeddingProfile,
    apply_transform,
    register_embedding_profile,
    resolve_profile,
)
from .registry import get_provider, list_providers, register_provider

__all__ = [
    "IDENTITY_PROFILE",
    "EmbeddingCapabilities",
    "EmbeddingProfile",
    "EmbeddingProvider",
    "ProviderProbeError",
    "apply_transform",
    "get_provider",
    "list_providers",
    "register_embedding_profile",
    "register_provider",
    "resolve_profile",
]
