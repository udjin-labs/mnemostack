"""mnemostack — Memory stack for AI agents.

Keep the package import light. Most users and tools import submodules such as
``mnemostack.recall.recaller`` or only need ``mnemostack.__version__``; pulling
Qdrant/httpx through the ingest API at package-import time makes those paths
slow and can leave async test runners waiting on unrelated imports.
"""

__version__ = "0.2.0a3"

__all__ = ["Config", "IngestItem", "IngestStats", "Ingestor", "stable_chunk_id"]


def __getattr__(name: str):
    if name == "Config":
        from .config import Config

        return Config
    if name in {"IngestItem", "IngestStats", "Ingestor", "stable_chunk_id"}:
        from .ingest import IngestItem, Ingestor, IngestStats, stable_chunk_id

        return {
            "IngestItem": IngestItem,
            "IngestStats": IngestStats,
            "Ingestor": Ingestor,
            "stable_chunk_id": stable_chunk_id,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
