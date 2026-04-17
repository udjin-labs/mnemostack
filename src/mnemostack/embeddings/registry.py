"""Provider registry — factory pattern for embedding providers."""
from __future__ import annotations

from typing import Any

from .base import EmbeddingProvider

_REGISTRY: dict[str, type[EmbeddingProvider]] = {}


def register_provider(name: str, cls: type[EmbeddingProvider]) -> None:
    """Register a provider class under a short name."""
    _REGISTRY[name.lower()] = cls


def list_providers() -> list[str]:
    """Return list of registered provider names."""
    return sorted(_REGISTRY.keys())


def get_provider(name: str, **kwargs: Any) -> EmbeddingProvider:
    """Instantiate a provider by name with optional keyword args.

    Example:
        provider = get_provider('gemini', api_key='...', model='gemini-embedding-001')
        provider = get_provider('ollama', host='http://10.0.0.5:11434')
    """
    key = name.lower()
    if key not in _REGISTRY:
        _lazy_register_builtins()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown embedding provider: {name!r}. Registered: {list_providers()}"
        )
    return _REGISTRY[key](**kwargs)


def _lazy_register_builtins() -> None:
    """Register built-in providers on first use. Import lazily so optional deps don't break."""
    try:
        from .gemini import GeminiProvider

        register_provider("gemini", GeminiProvider)
    except ImportError:
        pass
    try:
        from .ollama import OllamaProvider

        register_provider("ollama", OllamaProvider)
    except ImportError:
        pass
    try:
        from .huggingface import HuggingFaceProvider

        register_provider("huggingface", HuggingFaceProvider)
        register_provider("hf", HuggingFaceProvider)
    except ImportError:
        # Optional: needs `pip install mnemostack[huggingface]`
        pass


_lazy_register_builtins()
