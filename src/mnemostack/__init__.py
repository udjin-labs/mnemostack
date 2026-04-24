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
from .ingest import IngestItem, Ingestor, IngestStats, stable_chunk_id

__version__ = "0.2.0a1"
__all__ = ["Config", "IngestItem", "IngestStats", "Ingestor", "stable_chunk_id"]
