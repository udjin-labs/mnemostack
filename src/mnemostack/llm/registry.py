"""LLM provider registry."""
from __future__ import annotations

from typing import Any

from .base import LLMProvider

_REGISTRY: dict[str, type[LLMProvider]] = {}


def register_llm(name: str, cls: type[LLMProvider]) -> None:
    _REGISTRY[name.lower()] = cls


def list_llms() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_llm(name: str, **kwargs: Any) -> LLMProvider:
    key = name.lower()
    if key not in _REGISTRY:
        _lazy_register_builtins()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown LLM provider: {name!r}. Registered: {list_llms()}"
        )
    return _REGISTRY[key](**kwargs)


def _lazy_register_builtins() -> None:
    try:
        from .gemini import GeminiLLM

        register_llm("gemini", GeminiLLM)
        register_llm("gemini-flash", GeminiLLM)  # alias, default model is flash
    except ImportError:
        pass
    try:
        from .ollama import OllamaLLM

        register_llm("ollama", OllamaLLM)
    except ImportError:
        pass


_lazy_register_builtins()
