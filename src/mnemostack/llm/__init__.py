"""
LLM providers for inference (answer generation, reranking).

Provider registry pattern — same idea as embeddings module.
Recommended: Gemini Flash (fast, cheap, strong) or Ollama (local).

Usage:
    from mnemostack.llm import get_llm
    llm = get_llm('gemini-flash', api_key='...')
    text = llm.generate('What is the capital of France?')
"""

from .base import LLMProvider, LLMResponse
from .registry import get_llm, list_llms, register_llm

__all__ = ["LLMProvider", "LLMResponse", "get_llm", "list_llms", "register_llm"]
