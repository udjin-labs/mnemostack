"""
mnemostack — Memory stack for AI agents.

Core components:
- BM25 + Qdrant + RRF + 8-stage recall pipeline
- Pluggable embedding providers (Gemini, Ollama, HuggingFace)
- Memgraph knowledge graph with temporal validity
- Consolidation/decay runtime (Kairos)
- Inference layer via Gemini Flash

See ARCHITECTURE.md for design, README.md for usage.
"""

from .config import Config

__version__ = "0.1.0a9"
__all__ = ["Config"]
