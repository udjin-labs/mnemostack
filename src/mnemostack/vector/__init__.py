"""
Vector store (Qdrant) operations for mnemostack.

Provides indexing, chunking, search with payload filtering, and collection management.

Usage:
    from mnemostack.vector import VectorStore
    store = VectorStore(host='http://localhost:6333', collection='my-memory', dimension=3072)
    store.ensure_collection()
    store.upsert(id=1, vector=vec, payload={'text': '...'})
    hits = store.search(query_vector=q, limit=10)
"""

from .qdrant import VectorStore, Hit

__all__ = ["VectorStore", "Hit"]
